"""
Phase 3: The Engine Builder (480p Baked)
Function: Compiles the 480p-baked ONNX graph into a TensorRT Engine.
"""

import tensorrt as trt
import os
import argparse
import sys

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_file_path, engine_file_path, fp16=True):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)

    print("ONNX Parse successful. Configuring Optimization Profile...")

    # MATCHING SHAPES (60x104 Latents)
    profile = builder.create_optimization_profile()
    
    # x: Fixed H=60, W=104
    profile.set_shape("x", (1, 16, 1, 60, 104), (1, 16, 1, 60, 104), (1, 16, 1, 60, 104))
    
    # t
    profile.set_shape("t", (1, 2), (1, 2), (1, 2))
    
    # context
    profile.set_shape("context", (1, 512, 4096), (1, 512, 4096), (1, 512, 4096))
    
    # kv_cache: Fixed SeqLen=6240 (1 * 60 * 104)
    profile.set_shape("kv_cache", (1, 30, 6240, 12, 256), (1, 30, 6240, 12, 256), (1, 30, 6240, 12, 256))
    
    # cross_cache
    profile.set_shape("cross_cache", (1, 30, 512, 12, 256), (1, 30, 512, 12, 256), (1, 30, 512, 12, 256))

    config.add_optimization_profile(profile)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 Enabled.")

    print("Building serialized network... (5-15 mins)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Engine build failed!")
        sys.exit(1)
        
    print(f"Saving engine to {engine_file_path}")
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print("Engine build successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    build_engine(args.onnx, args.output, args.fp16)
