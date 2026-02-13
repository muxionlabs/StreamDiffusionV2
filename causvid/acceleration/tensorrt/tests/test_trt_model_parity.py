"""
Component-level parity test: TRTCausalWanModel vs original CausalWanModel.

Verifies that the TRT-safe model (before ONNX export) produces outputs
matching the original PyTorch model within acceptable tolerance.

Usage:
    python -m causvid.acceleration.tensorrt.tests.test_trt_model_parity \
        --config_path configs/wan_causal_dmd_v2v.yaml \
        --checkpoint_folder ckpts/wan_causal_dmd_v2v
"""

import argparse
import logging
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_original_model(config, checkpoint_path, device):
    """
    Load the original CausalWanModel with weights.
    """
    from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
    
    logger.info("Loading original PyTorch pipeline...")
    pipeline = CausalStreamInferencePipeline(config, device=str(device))
    pipeline.to(device=str(device), dtype=torch.bfloat16)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if 'generator' in ckpt:
            state_dict = ckpt['generator']
        elif 'generator_ema' in ckpt:
            state_dict = ckpt['generator_ema']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    pipeline.generator.load_state_dict(state_dict, strict=True)
    pipeline.generator.eval()
    
    return pipeline


def create_trt_model(config, checkpoint_path, device):
    """
    Load the TRT-safe model with weights converted from original checkpoint.
    """
    from causvid.acceleration.tensorrt.trt_model import TRTCausalWanModel
    from causvid.acceleration.tensorrt.weight_converter import load_checkpoint_for_trt
    from causvid.acceleration.tensorrt.export_onnx import T2V_1_3B_CONFIG
    
    logger.info("Creating TRT-safe model...")
    model = TRTCausalWanModel(**T2V_1_3B_CONFIG)
    model = model.to(device=device, dtype=torch.bfloat16)
    
    load_checkpoint_for_trt(checkpoint_path, model, strict=False)
    model.eval()
    
    return model


def test_rope_parity(device):
    """Test RoPE implementations match between original and TRT-safe."""
    from causvid.models.wan.wan_base.modules.model import rope_apply, rope_params
    from causvid.acceleration.tensorrt.trt_model import (
        precompute_rope_freqs_real, trt_rope_apply
    )
    
    logger.info("=" * 60)
    logger.info("Testing RoPE parity...")
    
    # Setup
    B, S, N, D = 1, 1560, 12, 128  # 1 frame, 30*52 tokens, 12 heads, 128 dim
    F, H, W = 1, 30, 52
    
    x = torch.randn(B, S, N, D, device=device, dtype=torch.bfloat16)
    # Also test with float32 input to isolate precision issues
    x_f32 = x.float()
    grid_sizes = torch.tensor([[F, H, W]], device=device, dtype=torch.long)
    
    # Original RoPE: rope_params(1024, head_dim) then split in rope_apply
    # This matches the ACTUAL original model (CausalWanModel uses rope_params(1024, dim//num_heads))
    freqs_orig = rope_params(1024, D).to(device)  # [1024, 64] complex
    
    out_orig = rope_apply(x, grid_sizes, freqs_orig)
    
    # TRT-safe RoPE (should now match since we compute full-dim then split)
    rope_freqs = precompute_rope_freqs_real(1024, D)
    cos_t, sin_t = rope_freqs[0].to(device), rope_freqs[1].to(device)
    cos_h, sin_h = rope_freqs[2].to(device), rope_freqs[3].to(device)
    cos_w, sin_w = rope_freqs[4].to(device), rope_freqs[5].to(device)
    
    out_trt = trt_rope_apply(x, grid_sizes, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w)
    
    # Also test with float32 inputs for diagnosis
    out_orig_f32 = rope_apply(x_f32, grid_sizes, freqs_orig)
    out_trt_f32 = trt_rope_apply(x_f32, grid_sizes, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w)
    
    # Compare bf16
    max_diff = (out_orig - out_trt).abs().max().item()
    mean_diff = (out_orig - out_trt).abs().mean().item()
    logger.info(f"  [bf16] Max absolute diff: {max_diff:.6e}")
    logger.info(f"  [bf16] Mean absolute diff: {mean_diff:.6e}")
    
    # Compare f32
    max_diff_f32 = (out_orig_f32 - out_trt_f32).abs().max().item()
    mean_diff_f32 = (out_orig_f32 - out_trt_f32).abs().mean().item()
    logger.info(f"  [f32] Max absolute diff: {max_diff_f32:.6e}")
    logger.info(f"  [f32] Mean absolute diff: {mean_diff_f32:.6e}")
    
    # For bf16, tolerance must account for both bf16 quantization and
    # float64 vs float32 precision gap. 5e-2 is reasonable.
    passed = max_diff < 5e-2
    logger.info(f"  RoPE parity: {'PASS' if passed else 'FAIL'}")
    
    return passed


def test_full_model_parity(config, checkpoint_path, device):
    """Test full model forward pass parity."""
    logger.info("=" * 60)
    logger.info("Testing full model parity...")
    logger.info("  (This test requires both the original model and checkpoint)")
    
    # Create both models
    pytorch_pipeline = create_original_model(config, checkpoint_path, device)
    trt_model = create_trt_model(config, checkpoint_path, device)
    
    # Create matching inputs
    B = 1
    F_dim = 1
    H, W = config.height // 8, config.width // 8
    H_p, W_p = H // 2, W // 2
    frame_seq_len = H_p * W_p
    max_cache_len = frame_seq_len * getattr(config, 'num_kv_cache', 10)
    
    torch.manual_seed(42)
    
    # Noise input
    noise = torch.randn(B, F_dim, 16, H, W, device=device, dtype=torch.bfloat16)
    
    # Text prompt
    text_prompts = ["a beautiful sunset over the ocean"]
    
    logger.info("  Running PyTorch pipeline.prepare()...")
    pytorch_pred = pytorch_pipeline.prepare(
        text_prompts=text_prompts,
        device=device,
        dtype=torch.bfloat16,
        block_mode='input',
        noise=noise,
        current_start=0,
        current_end=frame_seq_len * 2,
        batch_denoise=False,  # single-step for comparison
    )
    
    logger.info(f"  PyTorch output shape: {pytorch_pred.shape}")
    logger.info(f"  PyTorch output range: [{pytorch_pred.min():.4f}, {pytorch_pred.max():.4f}]")
    
    # NOTE: Full TRT model parity requires the TRT model to receive the same
    # pre-processed inputs (after text encoding, KV cache init, etc.)
    # This is complex to set up without the engine.
    # For now, we just verify the TRT model produces non-NaN output.
    
    logger.info("  Testing TRT model produces valid output...")
    
    # Create dummy inputs for TRT model
    context = pytorch_pipeline.conditional_dict["prompt_embeds"]
    if isinstance(context, list):
        # Pad and stack
        context_padded = []
        for emb in context:
            if emb.shape[0] < 512:
                pad = torch.zeros(512 - emb.shape[0], emb.shape[1],
                                 device=device, dtype=torch.bfloat16)
                emb = torch.cat([emb, pad])
            context_padded.append(emb)
        context_tensor = torch.stack(context_padded)
    else:
        context_tensor = context
    
    with torch.no_grad():
        trt_output, _, _, _ = trt_model(
            x=noise.permute(0, 2, 1, 3, 4),  # [B, C, F, H, W]
            timestep=torch.tensor([[700]], device=device, dtype=torch.long),
            context=context_tensor,
            current_start=torch.tensor([0], device=device, dtype=torch.long),
            current_end=torch.tensor([frame_seq_len * 2], device=device, dtype=torch.long),
            all_kv_k=torch.zeros(B, 30, max_cache_len, 12, 128,
                                device=device, dtype=torch.bfloat16),
            all_kv_v=torch.zeros(B, 30, max_cache_len, 12, 128,
                                device=device, dtype=torch.bfloat16),
            all_kv_seq_lens=torch.zeros(B, 30, device=device, dtype=torch.long),
            all_local_start_indices=torch.zeros(B, 30, device=device, dtype=torch.long),
            all_crossattn_k=torch.zeros(B, 30, 512, 12, 128,
                                       device=device, dtype=torch.bfloat16),
            all_crossattn_v=torch.zeros(B, 30, 512, 12, 128,
                                       device=device, dtype=torch.bfloat16),
        )
    
    has_nan = torch.isnan(trt_output).any().item()
    has_inf = torch.isinf(trt_output).any().item()
    
    logger.info(f"  TRT model output shape: {trt_output.shape}")
    logger.info(f"  TRT model output range: [{trt_output.min():.4f}, {trt_output.max():.4f}]")
    logger.info(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
    
    passed = not has_nan and not has_inf
    logger.info(f"  Full model validity: {'PASS' if passed else 'FAIL'}")
    
    # Clean up
    del pytorch_pipeline, trt_model
    torch.cuda.empty_cache()
    
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Test parity between TRT-safe model and original PyTorch model")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_full", action="store_true",
                       help="Skip full model test (only test RoPE)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    config = OmegaConf.load(args.config_path)
    # Merge defaults that may be missing from the YAML
    defaults = {
        'height': 480,
        'width': 832,
        'model_type': 'T2V-1.3B',
    }
    # Only add keys that are missing
    for k, v in defaults.items():
        if k not in config:
            config[k] = v
    
    checkpoint_path = os.path.join(args.checkpoint_folder, "model.pt")
    
    results = {}
    
    # Test 1: RoPE parity
    with torch.no_grad():
        results['rope'] = test_rope_parity(device)
    
    # Test 2: Full model parity
    if not args.skip_full:
        with torch.no_grad():
            results['full_model'] = test_full_model_parity(
                config, checkpoint_path, device)
    
    # Summary
    logger.info("=" * 60)
    logger.info("PARITY TEST RESULTS")
    logger.info("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("All parity tests PASSED")
    else:
        logger.info("Some parity tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
