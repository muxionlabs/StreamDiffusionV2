"""
TRT-Accelerated Inference Script for StreamDiffusionV2
"""
import sys
import os
import argparse
import torch
import time
import numpy as np
import logging
from PIL import Image
from omegaconf import OmegaConf
from diffusers.utils import export_to_video

# --- TRT IMPORT ---
from trt_integration import replace_model_with_trt
from inference import SingleGPUInferencePipeline, load_mp4_as_tensor, load_image_as_video_tensor

def main():
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--trt_engine_path", type=str, required=True, help="Path to wan_stream.engine")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, required=True)
    
    # Inputs
    parser.add_argument("--video_path", type=str, required=False, default=None)
    parser.add_argument("--image_path", type=str, required=False, default=None)
    
    # Parameters (Synced with inference.py)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--noise_scale", type=float, default=0.7)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    parser.add_argument("--num_frames", type=int, default=81)
    
    # Flags (kept for compatibility)
    parser.add_argument("--fixed_noise_scale", action="store_true", default=False)
    parser.add_argument("--img2img", action="store_true", default=False)

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    # 1. Load Config
    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    
    # 2. Logic for Denoising Steps (Copied from inference.py)
    full_denoising_list = [700, 600, 500, 400, 0]
    step_value = int(args.step)
    if step_value <= 1:
        config.denoising_step_list = [700, 0]
    elif step_value == 2:
        config.denoising_step_list = [700, 500, 0]
    elif step_value == 3:
        config.denoising_step_list = [700, 600, 400, 0]
    else:
        config.denoising_step_list = full_denoising_list

    # 3. Init Pipeline (Standard)
    pipeline_manager = SingleGPUInferencePipeline(config, device)
    pipeline_manager.load_model(args.checkpoint_folder)
    
    # 4. --- TRT INJECTION ---
    # This swaps the slow PyTorch DiT for your fast TRT Engine
    replace_model_with_trt(pipeline_manager.pipeline, args.trt_engine_path)
    # ------------------------

    # 5. Load Data
    input_video = None
    t = args.num_frames
    
    if args.video_path:
        input_video = load_mp4_as_tensor(args.video_path, resize_hw=(args.height, args.width)).unsqueeze(0)
        # Ensure correct BFloat16 input for pipeline compatibility
        input_video = input_video.to(device=device, dtype=torch.bfloat16)
        t = input_video.shape[2]
        print(f"Loaded Video: {input_video.shape}")

    from causvid.data import TextDataset
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]

    # 6. Run Inference
    chunck_size = 4
    num_chuncks = (t - 1) // chunck_size
    num_steps = len(pipeline_manager.pipeline.denoising_step_list)

    try:
        print(">>> Starting TRT Inference...")
        pipeline_manager.run_inference(
            input_video, prompts, 
            num_chuncks, chunck_size, 
            args.noise_scale, args.output_folder, args.fps, num_steps
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()