"""
TRT engine runtime wrapper.

Provides a Python interface to load and run a TRT engine,
managing CUDA memory allocation and binding setup.
"""

import logging
import os
import json
from typing import Dict, Optional

import numpy as np
import torch
import tensorrt as trt

logger = logging.getLogger(__name__)


class TRTEngineWrapper:
    """
    Runtime wrapper for a serialized TensorRT engine.
    
    Manages:
    - Engine deserialization
    - Execution context creation
    - CUDA device memory for input/output bindings
    - Dynamic shape support via set_input_shape
    
    Usage:
        wrapper = TRTEngineWrapper("engine.engine")
        outputs = wrapper.infer({
            'x': tensor_x,
            'timestep': tensor_t,
            ...
        })
    """
    
    def __init__(
        self,
        engine_path: str,
        device: torch.device = None,
        metadata_path: Optional[str] = None,
    ):
        """
        Load and initialize a TRT engine.
        
        Args:
            engine_path: Path to serialized .engine file
            device: CUDA device to use
            metadata_path: Optional path to engine metadata JSON.
                          If None, looks for {engine_path}_metadata.json
        """
        self.device = device or torch.device("cuda")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = engine_path.replace('.engine', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            logger.info(f"Engine metadata loaded: {metadata_path}")
        else:
            self.metadata = {}
            logger.warning(f"No metadata found at {metadata_path}")
        
        # Load engine
        logger.info(f"Loading TRT engine: {engine_path}")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine from {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Discover I/O tensor names and types
        self.input_names = []
        self.output_names = []
        self.io_dtypes = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
            
            self.io_dtypes[name] = self._trt_dtype_to_torch(dtype)
        
        logger.info(
            f"Engine loaded: {len(self.input_names)} inputs, "
            f"{len(self.output_names)} outputs"
        )
        logger.info(f"  Inputs: {self.input_names}")
        logger.info(f"  Outputs: {self.output_names}")
        
        # Pre-allocated output buffers (will be resized as needed)
        self._output_buffers: Dict[str, torch.Tensor] = {}
        
        # CUDA stream for async execution
        self._stream = torch.cuda.Stream(device=self.device)
    
    @staticmethod
    def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
        """Convert TRT dtype to PyTorch dtype."""
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BF16: torch.bfloat16,
        }
        return mapping.get(trt_dtype, torch.float32)
    
    def infer(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference with the TRT engine.
        
        Args:
            inputs: Dict mapping input tensor names to GPU tensors.
                   Tensors must be contiguous and on the correct device.
                   
        Returns:
            Dict mapping output tensor names to GPU tensors.
        """
        # Set input shapes and bind input tensors
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Missing input tensor: {name}")
            
            tensor = inputs[name].contiguous()
            if not tensor.is_cuda:
                tensor = tensor.to(self.device)
            
            # Set input shape for dynamic dimensions
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        # Allocate output buffers
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self.io_dtypes[name]
            
            # Check if we can reuse existing buffer
            shape_tuple = tuple(shape)
            if (name in self._output_buffers and
                self._output_buffers[name].shape == shape_tuple and
                self._output_buffers[name].dtype == dtype):
                out_tensor = self._output_buffers[name]
            else:
                out_tensor = torch.empty(
                    shape_tuple, dtype=dtype, device=self.device
                )
                self._output_buffers[name] = out_tensor
            
            self.context.set_tensor_address(name, out_tensor.data_ptr())
            outputs[name] = out_tensor
        
        # Execute
        with torch.cuda.stream(self._stream):
            success = self.context.execute_async_v3(
                stream_handle=self._stream.cuda_stream
            )
        
        self._stream.synchronize()
        
        if not success:
            raise RuntimeError("TRT engine execution failed")
        
        return outputs
    
    def __del__(self):
        """Clean up TRT resources."""
        if hasattr(self, 'context') and self.context:
            del self.context
        if hasattr(self, 'engine') and self.engine:
            del self.engine
        if hasattr(self, 'runtime') and self.runtime:
            del self.runtime
