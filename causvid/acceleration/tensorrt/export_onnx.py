"""
ONNX export for TRTCausalWanModel.

Exports the TRT-safe model to ONNX format with dynamic axes for:
- batch_size (dim 0)
- seq_len (dim 1 of hidden states)  
- kv_cache_len (dim 2 of KV cache)

Usage:
    python -m causvid.acceleration.tensorrt.export_onnx \
        --config_path configs/wan_causal_dmd_v2v.yaml \
        --checkpoint_folder ckpts/wan_causal_dmd_v2v \
        --output_path engines/wan_causal_dit.onnx
"""

import argparse
import os
import json
import logging
import torch
from omegaconf import OmegaConf

from causvid.acceleration.tensorrt.trt_model import TRTCausalWanModel
from causvid.acceleration.tensorrt.weight_converter import load_checkpoint_for_trt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default model config for T2V-1.3B
T2V_1_3B_CONFIG = dict(
    model_type='t2v',
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=12,
    num_layers=30,
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)


def create_dummy_inputs(
    batch_size: int = 1,
    num_frames: int = 1,
    height: int = 480,
    width: int = 832,
    num_layers: int = 30,
    max_cache_len: int = 15600,
    text_len: int = 512,
    num_heads: int = 12,
    head_dim: int = 128,
    text_dim: int = 4096,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Create dummy inputs matching the TRTCausalWanModel.forward() signature."""
    
    # Patch grid dimensions
    H_p = height // 2   # patch_size[1] = 2
    W_p = width // 2    # patch_size[2] = 2
    frame_seq_len = H_p * W_p  # 240 * 416 / 4 = 24960? No, H/2 * W/2
    # Wait: height=480, patch_size_h=2, so H_p = 480/2 = 240. But that's spatial.
    # Actually the latent is H/8, W/8 from VAE, then patch /2 each.
    # So H_latent = 480/8 = 60, W_latent = 832/8 = 104
    # After patch embedding (stride 2 in H,W): H_p = 60/2 = 30, W_p = 104/2 = 52
    H_latent = height // 8
    W_latent = width // 8
    H_p = H_latent // 2  # 30
    W_p = W_latent // 2  # 52
    frame_seq_len = H_p * W_p  # 1560
    
    inputs = {
        'x': torch.randn(batch_size, 16, num_frames, H_latent, W_latent,
                         device=device, dtype=dtype),
        'timestep': torch.tensor([[500] * num_frames], device=device, dtype=torch.long)
                    .expand(batch_size, -1),
        'context': torch.randn(batch_size, text_len, text_dim,
                               device=device, dtype=dtype),
        'current_start': torch.tensor([0], device=device, dtype=torch.long)
                        .expand(batch_size),
        'current_end': torch.tensor([frame_seq_len * num_frames], device=device,
                                    dtype=torch.long).expand(batch_size),
        'all_kv_k': torch.zeros(batch_size, num_layers, max_cache_len,
                                num_heads, head_dim, device=device, dtype=dtype),
        'all_kv_v': torch.zeros(batch_size, num_layers, max_cache_len,
                                num_heads, head_dim, device=device, dtype=dtype),
        'all_kv_seq_lens': torch.zeros(batch_size, num_layers, device=device,
                                       dtype=torch.long),
        'all_local_start_indices': torch.zeros(batch_size, num_layers,
                                               device=device, dtype=torch.long),
        'all_crossattn_k': torch.randn(batch_size, num_layers, text_len,
                                       num_heads, head_dim, device=device, dtype=dtype),
        'all_crossattn_v': torch.randn(batch_size, num_layers, text_len,
                                       num_heads, head_dim, device=device, dtype=dtype),
    }
    
    return inputs


def export_to_onnx(
    model: TRTCausalWanModel,
    output_path: str,
    batch_size: int = 1,
    num_frames: int = 1,
    height: int = 480,
    width: int = 832,
    max_cache_len: int = 15600,
    opset_version: int = 17,
):
    """
    Export the TRT-safe model to ONNX with external data format.
    
    For models > 2GB (like T2V-1.3B at ~2.6GB in FP16), weights are stored
    in a separate .bin file alongside the .onnx file to avoid protobuf's
    2GB message size limit.
    
    Args:
        model: TRTCausalWanModel with loaded weights
        output_path: Path to save .onnx file
        batch_size: Batch size for dummy inputs
        num_frames: Number of frames per chunk
        height: Video height (before VAE)
        width: Video width (before VAE)
        max_cache_len: Maximum KV cache length
        opset_version: ONNX opset version
    """
    import onnx
    import numpy as np
    
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Log model size
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    logger.info(f"Model: {param_count/1e6:.1f}M params, {param_bytes/1e9:.2f} GB")
    
    # Create dummy inputs
    inputs = create_dummy_inputs(
        batch_size=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        num_layers=model.num_layers,
        max_cache_len=max_cache_len,
        text_len=model.text_len,
        num_heads=model.num_heads,
        head_dim=model.head_dim,
        text_dim=model.text_dim,
        device=str(device),
        dtype=dtype,
    )
    
    # Input/output names for ONNX
    input_names = list(inputs.keys())
    output_names = [
        'output',
        'out_kv_k',
        'out_kv_v', 
        'out_kv_seq_lens',
    ]
    
    # Dynamic axes
    H_latent = height // 8
    W_latent = width // 8
    H_p = H_latent // 2
    W_p = W_latent // 2
    frame_seq_len = H_p * W_p
    
    dynamic_axes = {
        # Input latent: batch and frames are dynamic
        'x': {0: 'batch', 2: 'num_frames'},
        'timestep': {0: 'batch', 1: 'num_frames'},
        'context': {0: 'batch'},
        'current_start': {0: 'batch'},
        'current_end': {0: 'batch'},
        # KV cache: batch and cache_len are dynamic
        'all_kv_k': {0: 'batch', 2: 'cache_len'},
        'all_kv_v': {0: 'batch', 2: 'cache_len'},
        'all_kv_seq_lens': {0: 'batch'},
        'all_local_start_indices': {0: 'batch'},
        'all_crossattn_k': {0: 'batch'},
        'all_crossattn_v': {0: 'batch'},
        # Outputs
        'output': {0: 'batch', 2: 'out_frames'},
        'out_kv_k': {0: 'batch', 2: 'cache_len'},
        'out_kv_v': {0: 'batch', 2: 'cache_len'},
        'out_kv_seq_lens': {0: 'batch'},
    }
    
    logger.info(f"Exporting ONNX to {output_path}")
    logger.info(f"  Input shapes: {', '.join(f'{k}={v.shape}' for k,v in inputs.items())}")
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Use tuple of values in the order defined by input_names
    input_tuple = tuple(inputs[name] for name in input_names)
    
    # Step 1: Export to ONNX (initial export — may be incomplete for large models)
    logger.info("Step 1/3: Running torch.onnx.export...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_tuple,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=False,  # MUST be False to preserve PyTorch param names
            verbose=False,
        )
    
    initial_size = os.path.getsize(output_path)
    logger.info(f"  Initial ONNX file: {initial_size / 1e6:.1f} MB")
    
    # Step 2: Inject weights from PyTorch model into ONNX graph
    # torch.onnx.export drops weight data for models > 2GB (protobuf limit).
    # The graph structure and initializer names are correct — we just need
    # to fill in the actual tensor data from the PyTorch model.
    logger.info("Step 2/3: Injecting weights into ONNX graph...")
    
    onnx_model = onnx.load(output_path, load_external_data=False)
    
    # Build lookup from PyTorch model (state_dict includes buffers)
    pt_state = {}
    for name, param in model.named_parameters():
        pt_state[name] = param.detach().cpu()
    for name, buf in model.named_buffers():
        pt_state[name] = buf.detach().cpu()
    
    # Determine ONNX data type
    if dtype == torch.float16:
        onnx_dtype = onnx.TensorProto.FLOAT16
        np_dtype = np.float16
    elif dtype == torch.bfloat16:
        # ONNX doesn't have bfloat16 — use float16
        onnx_dtype = onnx.TensorProto.FLOAT16
        np_dtype = np.float16
    else:
        onnx_dtype = onnx.TensorProto.FLOAT
        np_dtype = np.float32
    
    # Count name types for debugging
    onnx_auto = [i.name for i in onnx_model.graph.initializer if i.name.startswith('onnx::')]
    pt_named = [i.name for i in onnx_model.graph.initializer if not i.name.startswith('onnx::')]
    logger.info(f"  Initializers: {len(onnx_model.graph.initializer)} total, "
                f"{len(pt_named)} PyTorch-named, {len(onnx_auto)} onnx::-auto-named")
    
    injected = 0
    skipped = 0
    skipped_names = []
    
    for initializer in onnx_model.graph.initializer:
        name = initializer.name
        tensor = pt_state.get(name)
        if tensor is None:
            skipped += 1
            skipped_names.append(name)
            continue
        
        # Convert to numpy (handle bfloat16 which numpy doesn't support)
        if tensor.dtype == torch.bfloat16:
            tensor_np = tensor.float().numpy().astype(np_dtype)
        elif tensor.dtype == torch.float16:
            tensor_np = tensor.numpy()
        else:
            tensor_np = tensor.numpy().astype(np_dtype)
        
        initializer.raw_data = tensor_np.tobytes()
        initializer.data_type = onnx_dtype
        injected += 1
    
    if skipped_names:
        logger.warning(f"  {skipped} initializers not matched. First 10:")
        for n in skipped_names[:10]:
            # Find shape
            init = next(i for i in onnx_model.graph.initializer if i.name == n)
            logger.warning(f"    {n}: shape={list(init.dims)}")
    
    
    logger.info(f"  Injected {injected} weights, skipped {skipped}")
    
    # Now save with external data format (weights in separate .bin file)
    weight_file = os.path.basename(output_path).replace('.onnx', '_weights.bin')
    output_dir = os.path.dirname(output_path) or '.'
    weight_path = os.path.join(output_dir, weight_file)
    
    # Remove old weight file if exists
    if os.path.exists(weight_path):
        os.remove(weight_path)
    
    logger.info(f"  Saving with external data format...")
    onnx.save_model(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weight_file,
        size_threshold=1024,  # tensors > 1KB go to external file
    )
    
    final_onnx_size = os.path.getsize(output_path)
    if os.path.exists(weight_path):
        weight_size = os.path.getsize(weight_path)
        logger.info(f"  ONNX graph: {final_onnx_size / 1e6:.1f} MB")
        logger.info(f"  External weights: {weight_size / 1e9:.2f} GB")
        total_size = final_onnx_size + weight_size
    else:
        total_size = final_onnx_size
        logger.warning(f"  External weight file not found at {weight_path}")
    
    # Validate: total size should be close to model parameter size
    if total_size < param_bytes * 0.5:
        logger.error(
            f"  EXPORT LIKELY FAILED: total file size ({total_size/1e9:.2f} GB) "
            f"is much smaller than model params ({param_bytes/1e9:.2f} GB)"
        )
    else:
        logger.info(f"  Total ONNX size: {total_size / 1e9:.2f} GB ✓")
    
    # Step 3: Save metadata alongside
    logger.info("Step 3/3: Saving metadata...")
    metadata = {
        'model_type': 'T2V-1.3B',
        'height': height,
        'width': width,
        'H_p': H_p,
        'W_p': W_p,
        'frame_seq_len': frame_seq_len,
        'max_cache_len': max_cache_len,
        'num_layers': model.num_layers,
        'num_heads': model.num_heads,
        'head_dim': model.head_dim,
        'dim': model.dim,
        'text_len': model.text_len,
        'patch_size': list(model.patch_size),
        'opset_version': opset_version,
        'input_names': input_names,
        'output_names': output_names,
        'weight_file': weight_file,
        'dtype': str(dtype),
        'param_count': param_count,
    }
    meta_path = output_path.replace('.onnx', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {meta_path}")
    logger.info("ONNX export complete ✓")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export TRT-safe model to ONNX")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to YAML config (e.g. configs/wan_causal_dmd_v2v.yaml)")
    parser.add_argument("--checkpoint_folder", type=str, required=True,
                       help="Path to checkpoint folder")
    parser.add_argument("--output_path", type=str, default="engines/wan_causal_dit.onnx",
                       help="Output ONNX file path")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max_cache_len", type=int, default=15600,
                       help="Max KV cache length (default: 10 frames * 1560 tokens/frame)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Export in FP16 precision (default: True)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                       help="Export in FP32 precision")
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config_path)
    
    # Create TRT model
    model_config = T2V_1_3B_CONFIG.copy()
    model = TRTCausalWanModel(**model_config)
    
    dtype = torch.float16 if args.fp16 else torch.float32
    model = model.to(device=args.device, dtype=dtype)
    
    # Load weights
    ckpt_path = os.path.join(args.checkpoint_folder, "model.pt")
    load_checkpoint_for_trt(ckpt_path, model, strict=False)
    
    # Export
    export_to_onnx(
        model=model,
        output_path=args.output_path,
        height=args.height,
        width=args.width,
        max_cache_len=args.max_cache_len,
    )


if __name__ == "__main__":
    main()
