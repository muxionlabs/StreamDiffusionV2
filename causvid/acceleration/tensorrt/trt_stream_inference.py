"""
TRT-accelerated CausalStreamInferencePipeline.

This is a drop-in replacement for CausalStreamInferencePipeline that uses
a TensorRT engine for the transformer forward pass. All pipeline logic
(KV cache management, denoising steps, hidden state batching) stays identical.
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from causvid.acceleration.tensorrt.trt_wrapper import TRTWanDiffusionWrapper

logger = logging.getLogger(__name__)


class TRTCausalStreamInferencePipeline(CausalStreamInferencePipeline):
    """
    TRT-accelerated streaming inference pipeline.
    
    Inherits from CausalStreamInferencePipeline but replaces the PyTorch
    generator (WanDiffusionWrapper) with the TRT-accelerated wrapper
    (TRTWanDiffusionWrapper).
    
    All other logic (KV cache init/management, denoising steps, hidden state
    batching, text encoding, VAE) remains identical.
    """
    
    def __init__(self, args, device: str, engine_path: str = None):
        """
        Initialize TRT pipeline.
        
        Args:
            args: OmegaConf config (same as parent)
            device: CUDA device string
            engine_path: Path to serialized TRT .engine file.
                        If None, looks for args.engine_path
        """
        # Initialize parent — this sets up text_encoder, vae, and the
        # PyTorch generator. We'll replace the generator below.
        super().__init__(args=args, device=device)
        
        # Determine engine path
        if engine_path is None:
            engine_path = getattr(args, 'engine_path', None)
        
        if engine_path is None:
            raise ValueError(
                "engine_path must be provided either directly or via args.engine_path"
            )
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")
        
        logger.info(f"Loading TRT engine from {engine_path}")
        
        # Replace the PyTorch generator with TRT wrapper
        # Keep reference to original for weight extraction if needed
        original_generator = self.generator
        
        self.generator = TRTWanDiffusionWrapper(
            engine_path=engine_path,
            config=args,
            device=torch.device(device),
        )
        
        # Copy scheduler from original
        self.generator.scheduler = original_generator.scheduler
        
        # Clean up original model to free GPU memory
        del original_generator
        torch.cuda.empty_cache()
        
        logger.info("TRT pipeline initialized — generator replaced with TRT engine")
    
    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize KV cache — same format as parent.
        
        The TRT wrapper expects the same dict-based cache format.
        It converts to flat tensors internally before engine calls.
        """
        super()._initialize_kv_cache(batch_size, dtype, device)
    
    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize cross-attention cache — same as parent."""
        super()._initialize_crossattn_cache(batch_size, dtype, device)
    
    def prepare(self, *args, **kwargs):
        """Prepare the pipeline — same as parent."""
        return super().prepare(*args, **kwargs)
    
    def inference_stream(self, noise, current_start, current_end,
                        current_step=None):
        """Stream inference — same as parent."""
        return super().inference_stream(
            noise=noise,
            current_start=current_start,
            current_end=current_end,
            current_step=current_step,
        )


def create_trt_pipeline(
    config,
    device: str = "cuda",
    engine_path: str = None,
) -> TRTCausalStreamInferencePipeline:
    """
    Factory function to create a TRT-accelerated pipeline.
    
    Args:
        config: OmegaConf config
        device: CUDA device
        engine_path: Path to TRT engine
        
    Returns:
        TRTCausalStreamInferencePipeline
    """
    return TRTCausalStreamInferencePipeline(
        args=config,
        device=device,
        engine_path=engine_path,
    )
