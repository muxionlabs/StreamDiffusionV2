# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT engine builder for StreamDiffusionV2 models.
"""

import os
import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .utilities import Engine, export_onnx, optimize_onnx, build_engine
from .models import CausalWanModelTRT, VAEEncoderTRT, VAEDecoderTRT, T5EncoderTRT

logger = logging.getLogger(__name__)


class EngineBuilder:
    """
    Orchestrates the full ONNX export → optimize → TensorRT build pipeline.
    
    Usage:
        builder = EngineBuilder(
            engine_dir="./trt_engines",
            model_path="./wan_models/Wan2.1-T2V-1.3B",
        )
        builder.build_all(height=480, width=832, num_frames=21)
    """
    
    def __init__(
        self,
        engine_dir: str,
        model_path: str,
        model_type: str = "T2V-1.3B",
        fp16: bool = True,
        device: str = "cuda",
        force_rebuild: bool = False,
    ):
        self.engine_dir = Path(engine_dir)
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.fp16 = fp16
        self.device = device
        self.force_rebuild = force_rebuild
        
        # Create directory structure
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_dir = self.engine_dir / "onnx"
        self.onnx_dir.mkdir(exist_ok=True)
    
    def _get_engine_path(self, component: str) -> Path:
        return self.engine_dir / f"{component}.engine"
    
    def _get_onnx_path(self, component: str) -> Path:
        return self.onnx_dir / f"{component}.onnx"
    
    def _get_onnx_opt_path(self, component: str) -> Path:
        return self.onnx_dir / f"{component}.opt.onnx"
    
    def _engine_exists(self, component: str) -> bool:
        return self._get_engine_path(component).exists() and not self.force_rebuild
    
    def build_dit(
        self,
        pipeline,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 21,
        skip_onnx_optimize: bool = False,
    ) -> Optional[Engine]:
        """
        Build TensorRT engine for DiT model.
        
        Args:
            pipeline: CausalStreamInferencePipeline containing the model
            batch_size: Optimization batch size
            height: Video height
            width: Video width
            num_frames: Number of frames
        
        Returns:
            Built TensorRT Engine or None if already exists
        """
        component = "dit"
        engine_path = self._get_engine_path(component)
        
        if self._engine_exists(component):
            logger.info(f"DiT engine exists: {engine_path}")
            return None
        
        logger.info("Building DiT TensorRT engine...")
        
        # Get the underlying model and convert to TRT-compatible version
        from .causal_model_trt import CausalWanModelTRTExport
        
        original_model = pipeline.generator.model
        trt_model = CausalWanModelTRTExport.from_pretrained_model(original_model)
        
        # CRITICAL: Delete original model to free GPU memory before export
        # The ONNX export needs a lot of memory for intermediate tensors
        del original_model
        pipeline.generator.model = None  # Prevent access to deleted model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Freed original model memory for ONNX export")
        
        trt_model.eval().to(self.device)
        
        # Convert to fp16 if specified
        if self.fp16:
            trt_model = trt_model.half()
        
        # Create export wrapper that calls forward_export (simpler, no KV cache lists)
        class DiTExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x, timestep, context, grid_sizes):
                return self.model.forward_export(x, timestep, context, grid_sizes)
        
        export_model = DiTExportWrapper(trt_model).eval()
        
        # Model definition for profiles (simplified without caches)
        model_def = CausalWanModelTRT(
            model_type=self.model_type,
            fp16=self.fp16,
            device=self.device,
        )
        
        # Get simplified sample inputs (without caches)
        lat_h, lat_w = height // 8, width // 8
        dtype = torch.float16 if self.fp16 else torch.float32
        
        sample_inputs = {
            "x": torch.randn(batch_size, num_frames, 16, lat_h, lat_w, device=self.device, dtype=dtype),
            "timestep": torch.randint(0, 1000, (batch_size, num_frames), device=self.device),
            "context": torch.randn(batch_size, 512, 4096, device=self.device, dtype=dtype),
            "grid_sizes": torch.tensor([[num_frames, lat_h // 2, lat_w // 2]] * batch_size, device=self.device, dtype=torch.long),
        }
        
        # Simplified input/output names
        input_names = ["x", "timestep", "context", "grid_sizes"]
        output_names = ["output"]
        
        # Dynamic axes for variable batch and resolution
        dynamic_axes = {
            "x": {0: "batch", 1: "frames", 3: "height", 4: "width"},
            "timestep": {0: "batch", 1: "frames"},
            "context": {0: "batch"},
            "grid_sizes": {0: "batch"},
            "output": {0: "batch", 2: "frames", 3: "height", 4: "width"},
        }
        
        # Export to ONNX - use dynamo_export for memory efficiency
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            export_model,
            onnx_path,
            tuple(sample_inputs.values()),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            use_dynamo=True,  # More memory efficient
        )
        
        # Optimize ONNX (optional - skip to save RAM)
        if skip_onnx_optimize:
            logger.info("Skipping ONNX optimization (--skip_onnx_optimize)")
            onnx_opt_path = onnx_path  # Use unoptimized ONNX
        else:
            onnx_opt_path = str(self._get_onnx_opt_path(component))
            optimize_onnx(onnx_path, onnx_opt_path)
        
        # Build TensorRT engine with simplified profiles
        # Note: grid_sizes is folded as constant during ONNX tracing, so only x, timestep, context are inputs
        # Use max(1, ...) to ensure MIN <= OPT <= MAX even with small frame counts
        opt_frames = max(1, num_frames)
        max_frames = max(2, num_frames * 2)
        input_profile = {
            "x": (
                (1, 1, 16, lat_h // 2, lat_w // 2),  # min
                (batch_size, opt_frames, 16, lat_h, lat_w),  # opt
                (batch_size * 2, max_frames, 16, lat_h * 2, lat_w * 2),  # max
            ),
            "timestep": (
                (1, 1),
                (batch_size, opt_frames),
                (batch_size * 2, max_frames),
            ),
            "context": (
                (1, 512, 4096),
                (batch_size, 512, 4096),
                (batch_size * 2, 512, 4096),
            ),
            # grid_sizes is folded as constant during ONNX export (not a dynamic input)
        }
        
        engine = build_engine(
            str(engine_path),
            onnx_opt_path,
            input_profile,
            fp16=self.fp16,
        )
        
        # Cleanup
        del trt_model, export_model, sample_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return engine
    
    def build_dit_streaming(
        self,
        pipeline,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 1,  # Streaming chunk size
        max_seq_len: int = 150000,
        skip_onnx_optimize: bool = False,
    ) -> Optional[Engine]:
        """Build TensorRT engine for DiT model with Streaming KV Cache support."""
        component = "dit_streaming"
        engine_path = self._get_engine_path(component)
        
        if self._engine_exists(component):
            logger.info(f"DiT Streaming engine exists: {engine_path}")
            return None
            
        logger.info("Building DiT Streaming TensorRT engine...")
        
        from .causal_model_trt import CausalWanModelTRTExport
        
        # We need the model structure
        original_model = pipeline.generator.model
        trt_model = CausalWanModelTRTExport.from_pretrained_model(original_model, max_seq_len=max_seq_len)
        
        # Clean up original model
        del original_model
        pipeline.generator.model = None
        gc.collect()
        torch.cuda.empty_cache()
        
        trt_model.eval().to(self.device)
        if self.fp16:
            trt_model = trt_model.half()
            
        class DiTStreamingWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x, t, c, kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13, kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25, kv26, kv27, kv28, kv29, cs, sf):
                return self.model.forward_export_streaming(x, t, c, kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13, kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25, kv26, kv27, kv28, kv29, cs, sf)

        export_model = DiTStreamingWrapper(trt_model).eval()
        
        # Prepare sample inputs (Split buffer into 5 chunks of 6 layers)
        lat_h, lat_w = height // 8, width // 8
        dtype = torch.float16 if self.fp16 else torch.float32
        
        # KV Cache dims
        total_layers = 30 # T2V-1.3B
        chunk_layers = 1 # 1 layer per chunk (Safest for TRT < 1GB)
        num_heads = 12 
        head_dim = 128
        
        export_seq_len = 128 
        
        sample_inputs = {
            "x": torch.randn(batch_size, num_frames, 16, lat_h, lat_w, device=self.device, dtype=dtype),
            "timestep": torch.randint(0, 1000, (batch_size, num_frames), device=self.device),
            "context": torch.randn(batch_size, 512, 4096, device=self.device, dtype=dtype),
            # 30 Chunks of cache
            "kv_cache_0": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_1": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_2": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_3": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_4": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_5": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_6": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_7": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_8": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_9": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_10": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_11": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_12": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_13": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_14": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_15": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_16": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_17": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_18": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_19": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_20": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_21": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_22": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_23": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_24": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_25": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_26": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_27": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_28": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "kv_cache_29": torch.randn(chunk_layers, 2, batch_size, export_seq_len, num_heads, head_dim, device=self.device, dtype=dtype),
            "current_start": torch.tensor([1], device=self.device, dtype=torch.long),
            "start_frame_idx": torch.tensor([0], device=self.device, dtype=torch.long),
        }
        
        input_names = [
            "x", "timestep", "context", 
            "kv_cache_0", "kv_cache_1", "kv_cache_2", "kv_cache_3", "kv_cache_4",
            "kv_cache_5", "kv_cache_6", "kv_cache_7", "kv_cache_8", "kv_cache_9",
            "kv_cache_10", "kv_cache_11", "kv_cache_12", "kv_cache_13", "kv_cache_14",
            "kv_cache_15", "kv_cache_16", "kv_cache_17", "kv_cache_18", "kv_cache_19",
            "kv_cache_20", "kv_cache_21", "kv_cache_22", "kv_cache_23", "kv_cache_24",
            "kv_cache_25", "kv_cache_26", "kv_cache_27", "kv_cache_28", "kv_cache_29",
            "current_start", "start_frame_idx"
        ]
        output_names = [
            "output", 
            "new_kv_cache_0", "new_kv_cache_1", "new_kv_cache_2", "new_kv_cache_3", "new_kv_cache_4",
            "new_kv_cache_5", "new_kv_cache_6", "new_kv_cache_7", "new_kv_cache_8", "new_kv_cache_9",
            "new_kv_cache_10", "new_kv_cache_11", "new_kv_cache_12", "new_kv_cache_13", "new_kv_cache_14",
            "new_kv_cache_15", "new_kv_cache_16", "new_kv_cache_17", "new_kv_cache_18", "new_kv_cache_19",
            "new_kv_cache_20", "new_kv_cache_21", "new_kv_cache_22", "new_kv_cache_23", "new_kv_cache_24",
            "new_kv_cache_25", "new_kv_cache_26", "new_kv_cache_27", "new_kv_cache_28", "new_kv_cache_29"
        ]
        
        dynamic_axes = {
            "x": {0: "batch", 1: "frames", 3: "height", 4: "width"},
            "timestep": {0: "batch", 1: "frames"},
            "context": {0: "batch"},
            "kv_cache_0": {2: "batch", 3: "seq_len"},
            "kv_cache_1": {2: "batch", 3: "seq_len"},
            "kv_cache_2": {2: "batch", 3: "seq_len"},
            "kv_cache_3": {2: "batch", 3: "seq_len"},
            "kv_cache_4": {2: "batch", 3: "seq_len"},
            "kv_cache_5": {2: "batch", 3: "seq_len"},
            "kv_cache_6": {2: "batch", 3: "seq_len"},
            "kv_cache_7": {2: "batch", 3: "seq_len"},
            "kv_cache_8": {2: "batch", 3: "seq_len"},
            "kv_cache_9": {2: "batch", 3: "seq_len"},
            "kv_cache_10": {2: "batch", 3: "seq_len"},
            "kv_cache_11": {2: "batch", 3: "seq_len"},
            "kv_cache_12": {2: "batch", 3: "seq_len"},
            "kv_cache_13": {2: "batch", 3: "seq_len"},
            "kv_cache_14": {2: "batch", 3: "seq_len"},
            "kv_cache_15": {2: "batch", 3: "seq_len"},
            "kv_cache_16": {2: "batch", 3: "seq_len"},
            "kv_cache_17": {2: "batch", 3: "seq_len"},
            "kv_cache_18": {2: "batch", 3: "seq_len"},
            "kv_cache_19": {2: "batch", 3: "seq_len"},
            "kv_cache_20": {2: "batch", 3: "seq_len"},
            "kv_cache_21": {2: "batch", 3: "seq_len"},
            "kv_cache_22": {2: "batch", 3: "seq_len"},
            "kv_cache_23": {2: "batch", 3: "seq_len"},
            "kv_cache_24": {2: "batch", 3: "seq_len"},
            "kv_cache_25": {2: "batch", 3: "seq_len"},
            "kv_cache_26": {2: "batch", 3: "seq_len"},
            "kv_cache_27": {2: "batch", 3: "seq_len"},
            "kv_cache_28": {2: "batch", 3: "seq_len"},
            "kv_cache_29": {2: "batch", 3: "seq_len"},
            "current_start": {0: "batch"},
            "start_frame_idx": {0: "batch"},
            "output": {0: "batch", 2: "frames", 3: "height", 4: "width"},
            "new_kv_cache_0": {2: "batch", 3: "seq_len"},
            "new_kv_cache_1": {2: "batch", 3: "seq_len"},
            "new_kv_cache_2": {2: "batch", 3: "seq_len"},
            "new_kv_cache_3": {2: "batch", 3: "seq_len"},
            "new_kv_cache_4": {2: "batch", 3: "seq_len"},
            "new_kv_cache_5": {2: "batch", 3: "seq_len"},
            "new_kv_cache_6": {2: "batch", 3: "seq_len"},
            "new_kv_cache_7": {2: "batch", 3: "seq_len"},
            "new_kv_cache_8": {2: "batch", 3: "seq_len"},
            "new_kv_cache_9": {2: "batch", 3: "seq_len"},
            "new_kv_cache_10": {2: "batch", 3: "seq_len"},
            "new_kv_cache_11": {2: "batch", 3: "seq_len"},
            "new_kv_cache_12": {2: "batch", 3: "seq_len"},
            "new_kv_cache_13": {2: "batch", 3: "seq_len"},
            "new_kv_cache_14": {2: "batch", 3: "seq_len"},
            "new_kv_cache_15": {2: "batch", 3: "seq_len"},
            "new_kv_cache_16": {2: "batch", 3: "seq_len"},
            "new_kv_cache_17": {2: "batch", 3: "seq_len"},
            "new_kv_cache_18": {2: "batch", 3: "seq_len"},
            "new_kv_cache_19": {2: "batch", 3: "seq_len"},
            "new_kv_cache_20": {2: "batch", 3: "seq_len"},
            "new_kv_cache_21": {2: "batch", 3: "seq_len"},
            "new_kv_cache_22": {2: "batch", 3: "seq_len"},
            "new_kv_cache_23": {2: "batch", 3: "seq_len"},
            "new_kv_cache_24": {2: "batch", 3: "seq_len"},
            "new_kv_cache_25": {2: "batch", 3: "seq_len"},
            "new_kv_cache_26": {2: "batch", 3: "seq_len"},
            "new_kv_cache_27": {2: "batch", 3: "seq_len"},
            "new_kv_cache_28": {2: "batch", 3: "seq_len"},
            "new_kv_cache_29": {2: "batch", 3: "seq_len"},
        }
        
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            export_model,
            onnx_path,
            tuple(sample_inputs.values()),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            use_dynamo=True, 
        )
        
        if skip_onnx_optimize:
            onnx_opt_path = onnx_path
        else:
            onnx_opt_path = str(self._get_onnx_opt_path(component))
            optimize_onnx(onnx_path, onnx_opt_path)
            
        # Profiles for streaming
        input_profile = {
            "x": ((1, 1, 16, lat_h, lat_w), (batch_size, num_frames, 16, lat_h, lat_w), (batch_size, num_frames, 16, lat_h, lat_w)),
            "timestep": ((1, 1), (batch_size, num_frames), (batch_size, num_frames)),
            "context": ((1, 512, 4096), (batch_size, 512, 4096), (batch_size, 512, 4096)),
            # Split profiling (1 layer per chunk)
            "kv_cache_0": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_1": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_2": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_3": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_4": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_5": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_6": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_7": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_8": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_9": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_10": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_11": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_12": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_13": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_14": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_15": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_16": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_17": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_18": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_19": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_20": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_21": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_22": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_23": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_24": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_25": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_26": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_27": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_28": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "kv_cache_29": ((chunk_layers, 2, 1, 1, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim), (chunk_layers, 2, batch_size, max_seq_len, num_heads, head_dim)),
            "current_start": ((1,), (batch_size,), (batch_size,)),
            "start_frame_idx": ((1,), (batch_size,), (batch_size,)),
        }
        
        engine = build_engine(
            str(engine_path),
            onnx_opt_path,
            input_profile,
            fp16=self.fp16,
        )
        
        return engine
    
    def build_vae_encoder(
        self,
        vae_model,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 21,
    ) -> Optional[Engine]:
        """Build TensorRT engine for VAE encoder."""
        component = "vae_encoder"
        engine_path = self._get_engine_path(component)
        
        if self._engine_exists(component):
            logger.info(f"VAE encoder engine exists: {engine_path}")
            return None
        
        logger.info("Building VAE encoder TensorRT engine...")
        
        # Create encoder-only wrapper
        class VAEEncoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.encoder = vae.model.encoder
                self.conv1 = vae.model.conv1
                
            def forward(self, video):
                x = self.encoder(video)
                mu, _ = self.conv1(x).chunk(2, dim=1)
                return mu
        
        encoder = VAEEncoderWrapper(vae_model).eval().to(self.device)
        if self.fp16:
            encoder = encoder.half()
        
        # Patch Upsample layers to use 'nearest' instead of 'nearest-exact' for ONNX compatibility
        for module in encoder.modules():
            if isinstance(module, torch.nn.Upsample) and module.mode == 'nearest-exact':
                module.mode = 'nearest'
        
        model_def = VAEEncoderTRT(fp16=self.fp16, device=self.device)
        sample_inputs = model_def.get_sample_input(batch_size, height, width, num_frames)
        
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            encoder,
            onnx_path,
            sample_inputs,
            input_names=model_def.get_input_names(),
            output_names=model_def.get_output_names(),
            dynamic_axes=model_def.get_dynamic_axes(),
        )
        
        onnx_opt_path = str(self._get_onnx_opt_path(component))
        optimize_onnx(onnx_path, onnx_opt_path)
        
        input_profile = model_def.get_input_profile(batch_size, height, width, num_frames)
        engine = build_engine(str(engine_path), onnx_opt_path, input_profile, fp16=self.fp16)
        
        del encoder, sample_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"VAE encoder engine built: {engine_path}")
        return engine
    
    def build_vae_decoder(
        self,
        vae_model,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 21,
    ) -> Optional[Engine]:
        """Build TensorRT engine for VAE decoder."""
        component = "vae_decoder"
        engine_path = self._get_engine_path(component)
        
        if self._engine_exists(component):
            logger.info(f"VAE decoder engine exists: {engine_path}")
            return None
        
        logger.info("Building VAE decoder TensorRT engine...")
        
        class VAEDecoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.decoder = vae.model.decoder
                self.conv2 = vae.model.conv2
                
            def forward(self, latent):
                x = self.conv2(latent)
                return self.decoder(x)
        
        decoder = VAEDecoderWrapper(vae_model).eval().to(self.device)
        if self.fp16:
            decoder = decoder.half()
        
        # Patch Upsample layers to use 'nearest' instead of 'nearest-exact' for ONNX compatibility
        for module in decoder.modules():
            if isinstance(module, torch.nn.Upsample) and module.mode == 'nearest-exact':
                module.mode = 'nearest'
        
        model_def = VAEDecoderTRT(fp16=self.fp16, device=self.device)
        sample_inputs = model_def.get_sample_input(batch_size, height, width, num_frames)
        
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            decoder,
            onnx_path,
            sample_inputs,
            input_names=model_def.get_input_names(),
            output_names=model_def.get_output_names(),
            dynamic_axes=model_def.get_dynamic_axes(),
        )
        
        onnx_opt_path = str(self._get_onnx_opt_path(component))
        optimize_onnx(onnx_path, onnx_opt_path)
        
        input_profile = model_def.get_input_profile(batch_size, height, width, num_frames)
        engine = build_engine(str(engine_path), onnx_opt_path, input_profile, fp16=self.fp16)
        
        del decoder, sample_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"VAE decoder engine built: {engine_path}")
        return engine
    
    def build_t5_encoder(
        self,
        text_encoder,
        batch_size: int = 1,
    ) -> Optional[Engine]:
        """Build TensorRT engine for T5 text encoder."""
        component = "t5_encoder"
        engine_path = self._get_engine_path(component)
        
        if self._engine_exists(component):
            logger.info(f"T5 encoder engine exists: {engine_path}")
            return None
        
        logger.info("Building T5 encoder TensorRT engine...")
        
        class T5EncoderWrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder.text_encoder
                
            def forward(self, input_ids, attention_mask):
                return self.encoder(input_ids, attention_mask)
        
        encoder = T5EncoderWrapper(text_encoder).eval().to(self.device)
        
        model_def = T5EncoderTRT(fp16=self.fp16, device=self.device)
        sample_inputs = model_def.get_sample_input(batch_size)
        
        onnx_path = str(self._get_onnx_path(component))
        export_onnx(
            encoder,
            onnx_path,
            sample_inputs,
            input_names=model_def.get_input_names(),
            output_names=model_def.get_output_names(),
            dynamic_axes=model_def.get_dynamic_axes(),
        )
        
        onnx_opt_path = str(self._get_onnx_opt_path(component))
        optimize_onnx(onnx_path, onnx_opt_path)
        
        input_profile = model_def.get_input_profile(batch_size)
        engine = build_engine(str(engine_path), onnx_opt_path, input_profile, fp16=self.fp16)
        
        del encoder, sample_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"T5 encoder engine built: {engine_path}")
        return engine
    
    def build_all(
        self,
        pipeline,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 21,
        skip_onnx_optimize: bool = False,
        skip_t5: bool = False,
        skip_vae: bool = True,  # Default True - TRT VAE has 3D conv issues
        streaming: bool = True, # Default True for streaming support
        max_seq_len: int = 150000, # Default max seq len
    ) -> Dict[str, Path]:
        """
        Build all TensorRT engines.
        
        Args:
            pipeline: CausalStreamInferencePipeline
            batch_size: Optimization batch size
            height: Video height
            width: Video width
            num_frames: Number of frames
            skip_onnx_optimize: Skip ONNX optimization step
            skip_t5: Skip T5 encoder engine
            skip_vae: Skip VAE engines
            streaming: Build streaming-optimized DiT engine (dit_streaming) instead of standard dit
        
        Returns:
            Dictionary of component name -> engine path
        """
        logger.info(f"Building all TensorRT engines in {self.engine_dir}")
        logger.info(f"Configuration: {batch_size}x{height}x{width}, {num_frames} frames, streaming={streaming}")
        
        engines = {}
        
        # Free up memory by deleting unused pipeline components before DiT export
        vae = pipeline.vae  # Save reference
        text_encoder = pipeline.text_encoder  # Save reference
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Build DiT (most memory intensive)
        if streaming:
            # Streaming engine requires large KV cache but fixed small batch/frame
            # For streaming, num_frames usually 1 (chunk size), but we keep original arg for compatibility
            self.build_dit_streaming(
                pipeline, 
                batch_size, 
                height, 
                width, 
                num_frames=1, # Streaming uses 1 frame chunks
                max_seq_len=max_seq_len,
                skip_onnx_optimize=skip_onnx_optimize
            )
            engines["dit"] = self._get_engine_path("dit_streaming")
        else:
            self.build_dit(pipeline, batch_size, height, width, num_frames, skip_onnx_optimize)
            engines["dit"] = self._get_engine_path("dit")
        
        # Build VAE
        if not skip_vae:
            # Reassign VAE to pipeline just in case (though we saved ref)
            self.build_vae_encoder(vae, batch_size, height, width, num_frames)
            self.build_vae_decoder(vae, batch_size, height, width, num_frames)
            engines["vae_encoder"] = self._get_engine_path("vae_encoder")
            engines["vae_decoder"] = self._get_engine_path("vae_decoder")
        
        # Build T5
        if not skip_t5:
            self.build_t5_encoder(text_encoder, batch_size)
            engines["t5_encoder"] = self._get_engine_path("t5_encoder")
        
        logger.info("All TensorRT engines built successfully!")
        return engines

