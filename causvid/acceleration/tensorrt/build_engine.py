"""
TensorRT engine builder.

Builds a TRT engine from ONNX with optimization profiles for the
WanCausalDiT streaming inference workload.

Usage:
    python -m causvid.acceleration.tensorrt.build_engine \
        --onnx_path engines/wan_causal_dit.onnx \
        --engine_path engines/wan_causal_dit.engine \
        --fp16
"""

import argparse
import json
import logging
import os
import time

import tensorrt as trt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    max_workspace_size: int = 8 * (1 << 30),  # 8 GB
    # Profile dimensions
    # For 480x832 -> H_p=30, W_p=52 -> frame_seq_len=1560
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    min_num_frames: int = 1,
    opt_num_frames: int = 1,
    max_num_frames: int = 5,  # first chunk has up to 5 frames
    min_cache_len: int = 1560,
    opt_cache_len: int = 9360,    # 6 frames worth
    max_cache_len: int = 15600,   # 10 frames worth
    height: int = 480,
    width: int = 832,
    num_layers: int = 30,
    num_heads: int = 12,
    head_dim: int = 128,
    text_len: int = 512,
    text_dim: int = 4096,
):
    """
    Build TRT engine from ONNX with optimization profiles.
    
    The engine supports dynamic shapes within the profile range:
    - batch_size: [min_batch, max_batch]
    - num_frames: [min_num_frames, max_num_frames]
    - cache_len: [min_cache_len, max_cache_len]
    """
    H_latent = height // 8
    W_latent = width // 8
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Set workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    # Enable FP16
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled")
        else:
            logger.warning("FP16 not supported on this platform, using FP32")
    
    # Parse ONNX — use parse_from_file so TRT can locate external weight files
    logger.info(f"Parsing ONNX: {onnx_path}")
    onnx_path_abs = os.path.abspath(onnx_path)
    success = parser.parse_from_file(onnx_path_abs)
    if not success:
        for i in range(parser.num_errors):
            logger.error(f"ONNX parse error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model")
    logger.info(f"ONNX parsed: {network.num_inputs} inputs, {network.num_outputs} outputs")
    
    # Create optimization profile
    profile = builder.create_optimization_profile()
    
    # Define shape ranges for each input
    # x: [batch, 16, num_frames, H_latent, W_latent]
    profile.set_shape('x',
        min=(min_batch, 16, min_num_frames, H_latent, W_latent),
        opt=(opt_batch, 16, opt_num_frames, H_latent, W_latent),
        max=(max_batch, 16, max_num_frames, H_latent, W_latent))
    
    # timestep: [batch, num_frames]
    profile.set_shape('timestep',
        min=(min_batch, min_num_frames),
        opt=(opt_batch, opt_num_frames),
        max=(max_batch, max_num_frames))
    
    # context: [batch, text_len, text_dim]
    profile.set_shape('context',
        min=(min_batch, text_len, text_dim),
        opt=(opt_batch, text_len, text_dim),
        max=(max_batch, text_len, text_dim))
    
    # current_start, current_end: [batch]
    profile.set_shape('current_start',
        min=(min_batch,), opt=(opt_batch,), max=(max_batch,))
    profile.set_shape('current_end',
        min=(min_batch,), opt=(opt_batch,), max=(max_batch,))
    
    # KV caches: [batch, num_layers, cache_len, num_heads, head_dim]
    profile.set_shape('all_kv_k',
        min=(min_batch, num_layers, min_cache_len, num_heads, head_dim),
        opt=(opt_batch, num_layers, opt_cache_len, num_heads, head_dim),
        max=(max_batch, num_layers, max_cache_len, num_heads, head_dim))
    profile.set_shape('all_kv_v',
        min=(min_batch, num_layers, min_cache_len, num_heads, head_dim),
        opt=(opt_batch, num_layers, opt_cache_len, num_heads, head_dim),
        max=(max_batch, num_layers, max_cache_len, num_heads, head_dim))
    
    # KV seq lens: [batch, num_layers]
    profile.set_shape('all_kv_seq_lens',
        min=(min_batch, num_layers),
        opt=(opt_batch, num_layers),
        max=(max_batch, num_layers))
    profile.set_shape('all_local_start_indices',
        min=(min_batch, num_layers),
        opt=(opt_batch, num_layers),
        max=(max_batch, num_layers))
    
    # Cross-attn caches: [batch, num_layers, text_len, num_heads, head_dim]
    profile.set_shape('all_crossattn_k',
        min=(min_batch, num_layers, text_len, num_heads, head_dim),
        opt=(opt_batch, num_layers, text_len, num_heads, head_dim),
        max=(max_batch, num_layers, text_len, num_heads, head_dim))
    profile.set_shape('all_crossattn_v',
        min=(min_batch, num_layers, text_len, num_heads, head_dim),
        opt=(opt_batch, num_layers, text_len, num_heads, head_dim),
        max=(max_batch, num_layers, text_len, num_heads, head_dim))

    # text_mask: [batch, 1, 1, text_len]
    profile.set_shape('text_mask',
        min=(min_batch, 1, 1, text_len),
        opt=(opt_batch, 1, 1, text_len),
        max=(max_batch, 1, 1, text_len))
    
    # RoPE inputs: [freq_len, dim]
    # We used freq_len=1024 in export. 
    # Shapes must match the split logic:
    half_dim = head_dim // 2
    c_t = half_dim - 2 * (half_dim // 3)
    c_h = half_dim // 3
    c_w = half_dim // 3
    
    rope_len = 1024
    # We can allow dynamic length or fix it. Let's fix min to 1 and max to 4096 (safe).
    min_rope = 1
    opt_rope = 1024
    max_rope = 4096
    
    for name, dim_size in [
        ('rope_cos_t', c_t), ('rope_sin_t', c_t),
        ('rope_cos_h', c_h), ('rope_sin_h', c_h),
        ('rope_cos_w', c_w), ('rope_sin_w', c_w)
    ]:
        profile.set_shape(name,
            min=(min_rope, dim_size),
            opt=(opt_rope, dim_size),
            max=(max_rope, dim_size))
    
    config.add_optimization_profile(profile)
    
    # Build engine
    logger.info("Building TRT engine (this may take 15-30 minutes)...")
    t_start = time.time()
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TRT engine")
    
    elapsed = time.time() - t_start
    logger.info(f"Engine built in {elapsed:.1f}s")
    
    # Save engine
    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(bytes(serialized_engine))
    
    engine_size_mb = serialized_engine.nbytes / (1 << 20)
    logger.info(f"Engine saved: {engine_path} ({engine_size_mb:.1f} MB)")
    
    # Save engine metadata
    engine_meta = {
        'onnx_path': onnx_path,
        'engine_path': engine_path,
        'fp16': fp16,
        'height': height,
        'width': width,
        'min_batch': min_batch,
        'max_batch': max_batch,
        'min_num_frames': min_num_frames,
        'max_num_frames': max_num_frames,
        'min_cache_len': min_cache_len,
        'opt_cache_len': opt_cache_len,
        'max_cache_len': max_cache_len,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'text_len': text_len,
        'build_time_seconds': elapsed,
        'engine_size_mb': engine_size_mb,
    }
    meta_path = engine_path.replace('.engine', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(engine_meta, f, indent=2)
    logger.info(f"Engine metadata saved: {meta_path}")
    
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Build TRT engine from ONNX")
    parser.add_argument("--onnx_path", type=str, required=True,
                       help="Path to ONNX model")
    parser.add_argument("--engine_path", type=str, required=True,
                       help="Output TRT engine path")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Enable FP16 (default: True)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--max_workspace_gb", type=int, default=8,
                       help="Max workspace size in GB")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max_cache_len", type=int, default=15600,
                       help="Max KV cache length (10 frames * 1560)")
    parser.add_argument("--max_num_frames", type=int, default=5,
                       help="Max frames per inference call")
    args = parser.parse_args()
    
    build_engine(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        fp16=args.fp16,
        max_workspace_size=args.max_workspace_gb * (1 << 30),
        height=args.height,
        width=args.width,
        max_cache_len=args.max_cache_len,
        max_num_frames=args.max_num_frames,
    )


if __name__ == "__main__":
    main()
