"""
Direct TRT engine parity test.

Loads both the PyTorch TRT-safe model and the TRT engine,
feeds identical inputs, and compares outputs.

Usage:
    python -m causvid.acceleration.tensorrt.tests.test_engine_parity \
        --config_path configs/wan_causal_dmd_v2v.yaml \
        --checkpoint_folder ckpts/wan_causal_dmd_v2v \
        --engine_path engines/wan_causal_dit.engine
"""

import argparse
import logging
import sys

import torch
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_engine_parity(config, checkpoint_path, engine_path, device):
    """Compare TRT engine output against PyTorch TRT-safe model output."""
    from causvid.acceleration.tensorrt.trt_model import TRTCausalWanModel
    from causvid.acceleration.tensorrt.weight_converter import load_checkpoint_for_trt
    from causvid.acceleration.tensorrt.export_onnx import T2V_1_3B_CONFIG
    from causvid.acceleration.tensorrt.engine_wrapper import TRTEngineWrapper

    # === 1. Load PyTorch model ===
    logger.info("=" * 60)
    logger.info("Loading PyTorch TRT-safe model...")
    pt_model = TRTCausalWanModel(**T2V_1_3B_CONFIG)
    pt_model = pt_model.to(device=device, dtype=torch.float16)
    load_checkpoint_for_trt(checkpoint_path, pt_model, strict=False)
    pt_model.eval()
    logger.info("  PyTorch model loaded")

    # === 2. Load TRT engine ===
    logger.info("Loading TRT engine...")
    engine = TRTEngineWrapper(engine_path, device=device)
    logger.info("  TRT engine loaded")

    # === 3. Create identical inputs ===
    logger.info("Creating test inputs...")
    H = config.height // 8
    W = config.width // 8
    H_p, W_p = H // 2, W // 2
    frame_seq_len = H_p * W_p  # 30*52 = 1560
    num_layers = T2V_1_3B_CONFIG['num_layers']  # 30
    num_heads = T2V_1_3B_CONFIG['num_heads']    # 12
    head_dim = T2V_1_3B_CONFIG['dim'] // T2V_1_3B_CONFIG['num_heads']  # 128
    text_len = T2V_1_3B_CONFIG['text_len']      # 512
    text_dim = T2V_1_3B_CONFIG['text_dim']      # 4096
    max_cache_len = frame_seq_len * 6  # 6 frames in cache

    torch.manual_seed(42)
    
    # 1 frame of latent input
    x = torch.randn(1, 16, 1, H, W, device=device, dtype=torch.float16)
    timestep = torch.tensor([[500]], device=device, dtype=torch.long)
    context = torch.randn(1, text_len, text_dim, device=device, dtype=torch.float16)
    current_start = torch.tensor([0], device=device, dtype=torch.long)
    current_end = torch.tensor([frame_seq_len], device=device, dtype=torch.long)
    
    # KV caches (start empty)
    all_kv_k = torch.zeros(1, num_layers, max_cache_len, num_heads, head_dim,
                           device=device, dtype=torch.float16)
    all_kv_v = torch.zeros(1, num_layers, max_cache_len, num_heads, head_dim,
                           device=device, dtype=torch.float16)
    all_kv_seq_lens = torch.zeros(1, num_layers, device=device, dtype=torch.long)
    all_local_start_indices = torch.zeros(1, num_layers, device=device, dtype=torch.long)
    
    # Cross-attn caches (simulate pre-computed context KV)
    all_crossattn_k = torch.randn(1, num_layers, text_len, num_heads, head_dim,
                                   device=device, dtype=torch.float16)
    all_crossattn_v = torch.randn(1, num_layers, text_len, num_heads, head_dim,
                                   device=device, dtype=torch.float16)

    logger.info(f"  x: {x.shape}, timestep: {timestep.shape}, context: {context.shape}")
    logger.info(f"  KV cache: {all_kv_k.shape}, cross-attn: {all_crossattn_k.shape}")

    # === 4. Run PyTorch model ===
    logger.info("Running PyTorch model...")
    with torch.no_grad():
        pt_output, pt_kv_k, pt_kv_v, pt_kv_seq = pt_model(
            x, timestep, context, current_start, current_end,
            all_kv_k.clone(), all_kv_v.clone(),
            all_kv_seq_lens.clone(), all_local_start_indices.clone(),
            all_crossattn_k.clone(), all_crossattn_v.clone(),
        )
    
    has_nan_pt = torch.isnan(pt_output).any().item()
    has_inf_pt = torch.isinf(pt_output).any().item()
    logger.info(f"  PT output: shape={pt_output.shape}, "
                f"range=[{pt_output.min():.4f}, {pt_output.max():.4f}], "
                f"nan={has_nan_pt}, inf={has_inf_pt}")

    # === 5. Run TRT engine ===
    logger.info("Running TRT engine...")
    # TRT engine input names must match ONNX export names
    trt_inputs = {
        'x': x.clone(),
        'timestep': timestep.clone(),
        'context': context.clone(),
        'current_start': current_start.clone(),
        'current_end': current_end.clone(),
        'all_kv_k': all_kv_k.clone(),
        'all_kv_v': all_kv_v.clone(),
        'all_kv_seq_lens': all_kv_seq_lens.clone(),
        'all_local_start_indices': all_local_start_indices.clone(),
        'all_crossattn_k': all_crossattn_k.clone(),
        'all_crossattn_v': all_crossattn_v.clone(),
    }
    
    try:
        trt_outputs = engine.infer(trt_inputs)
    except Exception as e:
        logger.error(f"  TRT engine inference FAILED: {e}")
        return False
    
    # Get output tensor (name should be 'output' or first output)
    output_name = engine.output_names[0]
    trt_output = trt_outputs[output_name]
    
    has_nan_trt = torch.isnan(trt_output).any().item()
    has_inf_trt = torch.isinf(trt_output).any().item()
    logger.info(f"  TRT output: shape={trt_output.shape}, "
                f"range=[{trt_output.min():.4f}, {trt_output.max():.4f}], "
                f"nan={has_nan_trt}, inf={has_inf_trt}")

    # === 6. Compare outputs ===
    logger.info("=" * 60)
    logger.info("Comparing outputs...")
    
    if pt_output.shape != trt_output.shape:
        logger.error(f"  Shape mismatch! PT={pt_output.shape}, TRT={trt_output.shape}")
        return False
    
    # Cast both to float32 for comparison
    pt_f32 = pt_output.float()
    trt_f32 = trt_output.float()
    
    diff = (pt_f32 - trt_f32).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Relative diff
    pt_abs = pt_f32.abs()
    rel_diff = (diff / (pt_abs + 1e-8)).mean().item()
    
    # Cosine similarity
    pt_flat = pt_f32.flatten()
    trt_flat = trt_f32.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        pt_flat.unsqueeze(0), trt_flat.unsqueeze(0)).item()

    logger.info(f"  Max absolute diff:  {max_diff:.6e}")
    logger.info(f"  Mean absolute diff: {mean_diff:.6e}")
    logger.info(f"  Mean relative diff: {rel_diff:.6e}")
    logger.info(f"  Cosine similarity:  {cos_sim:.6f}")
    
    # Also compare KV cache updates
    if 'out_kv_k' in trt_outputs:
        kv_k_diff = (pt_kv_k.float() - trt_outputs['out_kv_k'].float()).abs().max().item()
        kv_v_diff = (pt_kv_v.float() - trt_outputs['out_kv_v'].float()).abs().max().item()
        logger.info(f"  KV cache K max diff: {kv_k_diff:.6e}")
        logger.info(f"  KV cache V max diff: {kv_v_diff:.6e}")
    
    # Pass criteria
    # FP16 quantization + different attention kernels = some diff is expected
    # Cosine similarity > 0.99 is good; > 0.95 is acceptable for FP16
    passed_cosine = cos_sim > 0.95
    passed_validity = not (has_nan_pt or has_inf_pt or has_nan_trt or has_inf_trt)
    passed = passed_cosine and passed_validity
    
    logger.info("=" * 60)
    logger.info(f"  Validity check: {'PASS' if passed_validity else 'FAIL'}")
    logger.info(f"  Cosine similarity: {'PASS' if passed_cosine else 'FAIL'} (threshold: 0.95)")
    logger.info(f"  Overall: {'PASS' if passed else 'FAIL'}")
    
    # Cleanup
    del pt_model, engine
    torch.cuda.empty_cache()
    
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Direct TRT engine vs PyTorch model parity test")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--engine_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config_path)
    # Merge defaults
    for k, v in {'height': 480, 'width': 832, 'model_type': 'T2V-1.3B'}.items():
        if k not in config:
            config[k] = v
    
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint_folder + "/model.pt"
    
    with torch.no_grad():
        passed = test_engine_parity(config, checkpoint_path, args.engine_path, device)
    
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
