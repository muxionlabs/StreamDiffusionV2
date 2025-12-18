# TensorRT Acceleration for StreamDiffusionV2

This module provides ONNX export, engine building, and runtime integration for TensorRT acceleration in StreamDiffusionV2. 

- `builder.py`: Handles ONNX export and TensorRT engine creation.
- `engine_manager.py`: Manages engine caching, loading, and runtime selection.
- `utilities.py`: Utility functions for ONNX and TensorRT workflows.

To add support for a new model, implement an ONNX export and engine wrapper, then register it with the engine manager.
