"""
TeaCache Accelerated Inference for StreamDiffusionV2
Strategy: DEEP-LAYER FFN SKIPPING (The "Noise-Safe" Strategy)
"""

import os
import torch
import torch.nn as nn
import time
import numpy as np
import logging
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from omegaconf import OmegaConf
import argparse

# --- VANILLA IMPORTS ---
from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset

# ==============================================================================
#  TEA CACHE MANAGER
# ==============================================================================
class TeaCacheManager:
    def __init__(self, threshold=0.15): 
        self.threshold = threshold
        self.cache = {} 
        self.frame_counter = 0

    def step_frame(self):
        self.frame_counter += 1

# ==============================================================================
#  DEEP FFN WRAPPER
# ==============================================================================
class TeaCacheFFNWrapper(nn.Module):
    def __init__(self, original_ffn, manager, layer_id):
        super().__init__()
        self.ffn = original_ffn
        self.manager = manager
        self.layer_id = layer_id
        
    def forward(self, x, *args, **kwargs):
        cache_key = f"FFN_{self.layer_id}"
        cached_input = self.manager.cache.get(f"{cache_key}_input", None)
        
        should_calc = True
        
        # 1. Check Input Similarity
        if cached_input is not None and x.shape == cached_input.shape:
            # Simple L1 Metric (Robust)
            abs_diff = torch.mean(torch.abs(x - cached_input)).item()
            prev_mag = torch.mean(torch.abs(cached_input)).item()
            if prev_mag < 1e-6: prev_mag = 1e-6
            diff = abs_diff / prev_mag
            
            # 2. Decision
            if diff < self.manager.threshold:
                should_calc = False
        
        # 3. Execution
        if should_calc:
            out = self.ffn(x, *args, **kwargs)
            self.manager.cache[f"{cache_key}_input"] = x.detach() 
            self.manager.cache[f"{cache_key}_output"] = out.detach() 
            return out
        else:
            return self.manager.cache[f"{cache_key}_output"]

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.ffn, name)

# ==============================================================================
#  INJECTOR (Deep Layers Only)
# ==============================================================================
def inject_teacache_deep_only(model, threshold=0.15):
    print(f"Injecting TeaCache into DEEP FFN Layers (Threshold: {threshold})...")
    manager = TeaCacheManager(threshold)
    
    injected_count = 0
    total_blocks = 0
    
    # 1. Identify Structure
    blocks = None
    if hasattr(model, "blocks"): blocks = model.blocks
    elif hasattr(model, "transformer_blocks"): blocks = model.transformer_blocks
    
    if blocks is None:
        print("Error: Could not find blocks.")
        return model, None
        
    total_blocks = len(blocks)
    # CRITICAL: Start skipping only in the second half (Layer 20+)
    start_index = int(total_blocks * 0.5) 
    print(f"Total Blocks: {total_blocks}. Skipping optimization for first {start_index} blocks (Safety Zone).")

    # 2. Wrap Deep Layers
    def wrap_block_ffn(block, idx):
        if idx < start_index: return False # IGNORE EARLY LAYERS
        
        if hasattr(block, 'mlp'):
            block.mlp = TeaCacheFFNWrapper(block.mlp, manager, idx)
            return True
        elif hasattr(block, 'ffn'):
            block.ffn = TeaCacheFFNWrapper(block.ffn, manager, idx)
            return True
        return False

    for i, block in enumerate(blocks):
        if wrap_block_ffn(block, i): injected_count += 1
             
    print(f"Successfully wrapped {injected_count} Deep FFN layers.")
    return model, manager

# ==============================================================================
#  PIPELINE
# ==============================================================================

def load_mp4_as_tensor(video_path, max_frames=None, resize_hw=None, normalize=True):
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW")
    if max_frames is not None: video = video[:max_frames]
    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([TF.resize(video[:, i], resize_hw, antialias=True) for i in range(t)], dim=1)
    if video.dtype != torch.float32: video = video.float()
    if normalize: video = video / 127.5 - 1.0
    return video

def compute_noise_scale_and_step(input_video_original, end_idx, chunck_size, noise_scale, init_noise_scale):
    l2_dist = (input_video_original[:,:,end_idx-chunck_size:end_idx]-input_video_original[:,:,end_idx-chunck_size-1:end_idx-1])**2
    l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
    new_noise_scale = (init_noise_scale-0.1*l2_dist.item())*0.9+noise_scale*0.1
    current_step = int(1000*new_noise_scale)-100
    return new_noise_scale, current_step

class InferencePipelineTeaCacheDeep:
    def __init__(self, config, device: torch.device, teacache_thresh=0.15):
        self.config = config
        self.device = device
        self.teacache_thresh = teacache_thresh
        self.manager = None
        
        print(f"Initializing Vanilla Pipeline on {device}...")
        self.pipeline = CausalStreamInferencePipeline(config, device=str(device))
        self.pipeline.to(device=str(device), dtype=torch.bfloat16) 
        
        self.processed = 0
        self.fixed_noise = None 

    def load_model(self, checkpoint_folder: str):
        ckpt_path = os.path.join(checkpoint_folder, "model.pt")
        print(f"Loading weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if 'generator' in ckpt: state_dict = ckpt['generator']
            elif 'generator_ema' in ckpt: state_dict = ckpt['generator_ema']
            elif 'state_dict' in ckpt: state_dict = ckpt['state_dict']
            else: state_dict = ckpt
        else:
            state_dict = ckpt

        try:
            self.pipeline.generator.load_state_dict(state_dict, strict=False)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Load warning: {e}")
            
        self.pipeline.generator.model, self.manager = inject_teacache_deep_only(
            self.pipeline.generator.model, 
            threshold=self.teacache_thresh
        )

    def run_inference(self, input_video_original, prompts, num_chuncks, chunck_size, noise_scale, output_folder, fps, num_steps):
        os.makedirs(output_folder, exist_ok=True)
        results = {}
        save_results = 0
        start_idx = 0
        end_idx = 5
        current_start = 0
        current_end = self.pipeline.frame_seq_length * 2
        torch.cuda.empty_cache()

        # INIT
        if input_video_original is not None:
            inp = input_video_original[:, :, start_idx:end_idx]
            latents = self.pipeline.vae.stream_encode(inp)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16) 
            noise = torch.randn_like(latents) 
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
        else:
            noise = torch.randn(1, 1, 16, self.pipeline.height, self.pipeline.width, device=self.device, dtype=torch.bfloat16)
            noisy_latents = noise

        denoised_pred = self.pipeline.prepare(
            text_prompts=prompts,
            device=self.device,
            dtype=torch.bfloat16,
            block_mode='input',
            noise=noisy_latents,
            current_start=current_start,
            current_end=current_end
        )
        
        if self.manager: self.manager.step_frame()

        with torch.no_grad():
            video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        results[save_results] = video.cpu().float().numpy()
        save_results += 1
        init_noise_scale = noise_scale
        
        print(f"Starting Inference Stream...")
        
        while self.processed < num_chuncks + num_steps - 1:
            start_idx = end_idx
            end_idx = end_idx + chunck_size
            current_start = current_end
            current_end = current_end + (chunck_size // 4) * self.pipeline.frame_seq_length

            if input_video_original is not None and end_idx <= input_video_original.shape[2]:
                inp = input_video_original[:, :, start_idx:end_idx]
                noise_scale, current_step = compute_noise_scale_and_step(
                    input_video_original, end_idx, chunck_size, noise_scale, init_noise_scale
                )
                latents = self.pipeline.vae.stream_encode(inp)
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                
                if (self.fixed_noise is None or self.fixed_noise.shape != latents.shape):
                    self.fixed_noise = torch.randn_like(latents)
                noisy_latents = self.fixed_noise * noise_scale + latents * (1 - noise_scale)
            else:
                if self.fixed_noise is None:
                     self.fixed_noise = torch.randn(1, 16, 1, self.pipeline.height, self.pipeline.width, device=self.device, dtype=torch.bfloat16)
                noisy_latents = self.fixed_noise
                current_step = None 

            if self.manager: self.manager.step_frame()

            iter_start = time.time()
            denoised_pred = self.pipeline.inference_stream(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )
            self.processed += 1
            
            if self.processed >= num_steps:
                with torch.no_grad():
                    video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(0, 2, 3, 1).contiguous()
                results[save_results] = video.cpu().float().numpy()
                save_results += 1
                
                torch.cuda.synchronize()
                t = time.time() - iter_start
                fps_val = chunck_size / t if t > 0 else 0
                print(f"Processed {self.processed}, step time: {t:.4f} s, FPS: {fps_val:.2f}")
        
        if len(results) > 0:
            video_list = [results[i] for i in sorted(results.keys())]
            video = np.concatenate(video_list, axis=0)
            output_path = os.path.join(output_folder, f"teacache_deep.mp4")
            export_to_video(video, output_path, fps=fps)
            print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=False, default=None)
    parser.add_argument("--image_path", type=str, required=False, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.700)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B")
    parser.add_argument("--num_frame_per_block", type=int, default=1)
    parser.add_argument("--warp_denoising_step", action="store_true", default=False)
    # Threshold for DEEP layers
    parser.add_argument("--teacache_thresh", type=float, default=0.2) 
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    
    device = torch.device("cuda")
    config = OmegaConf.load(args.config_path)
    cmd_conf = OmegaConf.create(vars(args))
    config = OmegaConf.merge(config, cmd_conf)
    config.text_encoder_device = "cpu" 

    if args.step == 1:
        config.denoising_step_list = [700, 0]
    elif args.step == 2:
        config.denoising_step_list = [700, 500, 0]
    else:
        config.denoising_step_list = [700, 600, 500, 400, 0] 

    pipeline = InferencePipelineTeaCacheDeep(config, device, teacache_thresh=args.teacache_thresh)
    pipeline.load_model(args.checkpoint_folder)
    
    input_video = None
    if args.video_path:
        print(f"Loading video: {args.video_path}")
        input_video = load_mp4_as_tensor(args.video_path, resize_hw=(480, 832))
        input_video = input_video.unsqueeze(0).to(device, dtype=torch.bfloat16)
    
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    
    pipeline.run_inference(
        input_video, prompts, (args.num_frames - 1) // 4, 4, 
        args.noise_scale, args.output_folder, args.fps, len(config.denoising_step_list)
    )

if __name__ == "__main__":
    main()
