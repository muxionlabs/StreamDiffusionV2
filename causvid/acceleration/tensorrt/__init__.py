"""
TensorRT acceleration for WanCausalDiT (StreamDiffusionV2).

This package provides:
- TRT-safe model wrappers for ONNX export
- ONNX export and TRT engine build utilities
- TRT engine runtime wrapper
- Drop-in pipeline replacement for streaming inference
"""

from causvid.acceleration.tensorrt.engine_wrapper import TRTEngineWrapper
from causvid.acceleration.tensorrt.trt_wrapper import TRTWanDiffusionWrapper

__all__ = [
    "TRTEngineWrapper",
    "TRTWanDiffusionWrapper",
]
