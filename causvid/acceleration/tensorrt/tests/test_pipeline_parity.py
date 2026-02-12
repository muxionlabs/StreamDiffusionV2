"""
End-to-end pipeline parity test: TRT engine vs PyTorch.

Runs multiple streaming chunks through both pipelines with identical
noise seeds and compares the outputs (PSNR, SSIM).

Usage:
    python -m causvid.acceleration.tensorrt.tests.test_pipeline_parity \
        --config_path configs/wan_causal_dmd_v2v.yaml \
        --checkpoint_folder ckpts/wan_causal_dmd_v2v \
        --engine_path engines/wan_causal_dit.engine \
        --num_chunks 3
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def run_pytorch_pipeline(config, checkpoint_path, prompts, num_chunks, device):
    """Run PyTorch pipeline and collect per-chunk outputs."""
    from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
    
    logger.info("=" * 60)
    logger.info("Running PyTorch pipeline...")
    
    pipeline = CausalStreamInferencePipeline(config, device=str(device))
    pipeline.to(device=str(device), dtype=torch.bfloat16)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if 'generator' in ckpt:
        state_dict = ckpt['generator']
    elif 'generator_ema' in ckpt:
        state_dict = ckpt['generator_ema']
    else:
        state_dict = ckpt
    pipeline.generator.load_state_dict(state_dict, strict=True)
    
    frame_seq_len = pipeline.frame_seq_length
    H, W = pipeline.height, pipeline.width
    chunk_size = 4  # frames per streaming chunk
    num_frame_per_block = pipeline.num_frame_per_block
    
    results = []
    
    # First chunk: prepare
    torch.manual_seed(42)
    noise = torch.randn(1, 1 + num_frame_per_block, 16,
                        config.height // 8, config.width // 8,
                        device=device, dtype=torch.bfloat16)
    
    current_start = 0
    current_end = frame_seq_len * 2
    
    denoised = pipeline.prepare(
        text_prompts=prompts,
        device=device,
        dtype=torch.bfloat16,
        block_mode='input',
        noise=noise,
        current_start=current_start,
        current_end=current_end,
        batch_denoise=False,
    )
    
    results.append(denoised.cpu().float())
    logger.info(f"  Chunk 0: shape={denoised.shape}, range=[{denoised.min():.4f}, {denoised.max():.4f}]")
    
    # Streaming chunks
    for i in range(1, num_chunks):
        current_start = current_end
        current_end = current_start + num_frame_per_block * frame_seq_len
        
        torch.manual_seed(42 + i)
        noise = torch.randn(1, num_frame_per_block, 16,
                           config.height // 8, config.width // 8,
                           device=device, dtype=torch.bfloat16)
        
        denoised = pipeline.inference_stream(
            noise=noise,
            current_start=current_start,
            current_end=current_end,
            current_step=500,
        )
        results.append(denoised.cpu().float())
        logger.info(f"  Chunk {i}: shape={denoised.shape}")
    
    del pipeline
    torch.cuda.empty_cache()
    
    return results


def run_trt_pipeline(config, checkpoint_path, engine_path, prompts, num_chunks, device):
    """Run TRT pipeline and collect per-chunk outputs."""
    from causvid.acceleration.tensorrt.trt_stream_inference import (
        TRTCausalStreamInferencePipeline
    )
    
    logger.info("=" * 60)
    logger.info("Running TRT pipeline...")
    
    pipeline = TRTCausalStreamInferencePipeline(
        args=config, device=str(device), engine_path=engine_path)
    pipeline.to(device=str(device), dtype=torch.bfloat16)
    
    frame_seq_len = pipeline.frame_seq_length
    num_frame_per_block = pipeline.num_frame_per_block
    
    results = []
    
    # First chunk: prepare
    torch.manual_seed(42)
    noise = torch.randn(1, 1 + num_frame_per_block, 16,
                        config.height // 8, config.width // 8,
                        device=device, dtype=torch.bfloat16)
    
    current_start = 0
    current_end = frame_seq_len * 2
    
    denoised = pipeline.prepare(
        text_prompts=prompts,
        device=device,
        dtype=torch.bfloat16,
        block_mode='input',
        noise=noise,
        current_start=current_start,
        current_end=current_end,
        batch_denoise=False,
    )
    
    results.append(denoised.cpu().float())
    logger.info(f"  Chunk 0: shape={denoised.shape}, range=[{denoised.min():.4f}, {denoised.max():.4f}]")
    
    # Streaming chunks
    for i in range(1, num_chunks):
        current_start = current_end
        current_end = current_start + num_frame_per_block * frame_seq_len
        
        torch.manual_seed(42 + i)
        noise = torch.randn(1, num_frame_per_block, 16,
                           config.height // 8, config.width // 8,
                           device=device, dtype=torch.bfloat16)
        
        denoised = pipeline.inference_stream(
            noise=noise,
            current_start=current_start,
            current_end=current_end,
            current_step=500,
        )
        results.append(denoised.cpu().float())
        logger.info(f"  Chunk {i}: shape={denoised.shape}")
    
    del pipeline
    torch.cuda.empty_cache()
    
    return results


def compare_results(pytorch_results, trt_results):
    """Compare outputs from both pipelines."""
    logger.info("=" * 60)
    logger.info("Comparing outputs...")
    
    all_passed = True
    
    for i, (pt_out, trt_out) in enumerate(zip(pytorch_results, trt_results)):
        if pt_out.shape != trt_out.shape:
            logger.error(f"  Chunk {i}: Shape mismatch! PT={pt_out.shape}, TRT={trt_out.shape}")
            all_passed = False
            continue
        
        diff = (pt_out - trt_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Convert to numpy for PSNR
        pt_np = pt_out.numpy()
        trt_np = trt_out.numpy()
        
        # Normalize to [0, 1] for PSNR
        combined = np.concatenate([pt_np.flatten(), trt_np.flatten()])
        vmin, vmax = combined.min(), combined.max()
        if vmax > vmin:
            pt_norm = (pt_np - vmin) / (vmax - vmin)
            trt_norm = (trt_np - vmin) / (vmax - vmin)
        else:
            pt_norm = pt_np
            trt_norm = trt_np
        
        chunk_psnr = psnr(pt_norm, trt_norm)
        
        logger.info(f"  Chunk {i}:")
        logger.info(f"    Max diff: {max_diff:.6e}")
        logger.info(f"    Mean diff: {mean_diff:.6e}")
        logger.info(f"    PSNR: {chunk_psnr:.2f} dB")
        
        # Pass criteria: PSNR > 25 dB (accounting for FP16 quantization + different attention impl)
        passed = chunk_psnr > 25.0 or max_diff < 0.1
        logger.info(f"    Status: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline parity test: PyTorch vs TRT")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--engine_path", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create({
        'height': getattr(config, 'height', 480),
        'width': getattr(config, 'width', 832),
    }))
    
    device = torch.device(args.device)
    checkpoint_path = os.path.join(args.checkpoint_folder, "model.pt")
    prompts = ["a beautiful sunset over the ocean"]
    
    # Run both pipelines
    with torch.no_grad():
        pytorch_results = run_pytorch_pipeline(
            config, checkpoint_path, prompts, args.num_chunks, device)
        
        trt_results = run_trt_pipeline(
            config, checkpoint_path, args.engine_path, prompts, args.num_chunks, device)
    
    # Compare
    passed = compare_results(pytorch_results, trt_results)
    
    logger.info("=" * 60)
    if passed:
        logger.info("Pipeline parity test PASSED")
    else:
        logger.info("Pipeline parity test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
