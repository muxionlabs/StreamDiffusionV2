import tensorrt as trt
import argparse
import sys
import os

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    parser_onnx = trt.OnnxParser(network, TRT_LOGGER)

    # Workspace (8GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (1 << 30))

    print(f"I: Parsing ONNX file: {args.onnx}")
    with open(args.onnx, "rb") as f:
        if not parser_onnx.parse(f.read()):
            for i in range(parser_onnx.num_errors):
                print(parser_onnx.get_error(i))
            sys.exit(1)

    print("I: Creating Optimization Profile...")
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = tensor.shape

        print(f"  -> Input: {name} | Shape: {shape}")

        # ---- Stateless model inputs ----
        if name == "x":
            # [B, C, T, H, W]  (latent video)
            profile.set_shape(name, (1,16,1,60,104),
                                    (1,16,1,60,104),
                                    (1,16,1,60,104))

        elif name == "t":
            # [B, num_frames]
            profile.set_shape(name, (1,2), (1,2), (1,2))

        elif name == "context":
            # [B, text_len, text_dim]
            profile.set_shape(name, (1,512,4096),
                                    (1,512,4096),
                                    (1,512,4096))

        elif name == "seq_len":
            # [B]
            profile.set_shape(name, (1,), (1,), (1,))

        else:
            print(f"Unknown input '{name}', leaving default shape")

    config.add_optimization_profile(profile)

    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("I: FP16 Enabled.")

    print("I: Building TensorRT engine...")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("Engine build failed.")
        sys.exit(1)

    with open(args.output, "wb") as f:
        f.write(engine)

    print(f"Engine saved to: {args.output}")

if __name__ == "__main__":
    build_engine()
