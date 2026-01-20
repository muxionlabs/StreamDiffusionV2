import tensorrt as trt
import argparse
import sys
import os

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_file, engine_file, fp16=True):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Limit Workspace to 8GB (Helps prevent OOM)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (1 << 30))

    if not os.path.exists(onnx_file):
        print(f"E: ONNX file not found: {onnx_file}")
        sys.exit(1)

    print(f"I: Parsing ONNX file: {onnx_file}")
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print("E: Failed to parse the ONNX file.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)

    print("I: Configuring Optimization Profile...")
    profile = builder.create_optimization_profile()

    num_inputs = network.num_inputs
    for i in range(num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        print(f"  -> Configuring input: {name} | Is Shape Tensor: {tensor.is_shape_tensor}")

        if "x" == name:
            profile.set_shape(name, (1, 16, 1, 60, 104), (1, 16, 1, 60, 104), (1, 16, 1, 60, 104))
        elif "t" == name:
            profile.set_shape(name, (1, 2), (1, 2), (1, 2))
        elif "context" == name:
            profile.set_shape(name, (1, 512, 4096), (1, 512, 4096), (1, 512, 4096))
        
        # CONSERVATIVE LIMIT: 25,000 tokens (~16 Frames / 16 seconds)
        # This drastically lowers VRAM reservation to ~15GB total.
        # This is the "Just Make It Work" setting.
        elif "kv_cache" == name:
            profile.set_shape(name, (1, 30, 1560, 12, 256), (1, 30, 6240, 12, 256), (1, 30, 25000, 12, 256))
            
        elif "cross_cache" == name:
            profile.set_shape(name, (1, 30, 512, 12, 256), (1, 30, 512, 12, 256), (1, 30, 512, 12, 256))
        elif "cross_init" == name:
            profile.set_shape(name, (1, 30), (1, 30), (1, 30))
            
        elif "k_starts" == name or "k_ends" == name:
            if tensor.is_shape_tensor:
                profile.set_shape_input(name, (0,), (1560,), (25000,))
                print(f"     -> Set as SHAPE INPUT (Range: 0 - 25000)")
            else:
                profile.set_shape(name, (1,), (1,), (1,))
                print(f"     -> Set as EXECUTION INPUT (Dim: 1)")

    config.add_optimization_profile(profile)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("I: FP16 Enabled.")

    print("I: Building serialized network... (This may take 5-15 mins)")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("E: Engine build failed! Check logs above.")
        sys.exit(1)

    with open(engine_file, "wb") as f:
        f.write(engine)

    print(f"TRT engine built successfully: {engine_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    build_engine(args.onnx, args.output, args.fp16)