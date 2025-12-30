"""
Full-Featured Single GPU TensorRT Inference Pipeline
Strategy: DiT on GPU (Fast) + VAE on CPU (Safe & Manual)
"""

import os
# Fixes memory fragmentation for large VAE decodes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import time
import numpy as np
import logging
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

# Import our TRT Backend
from src.trt_backend import WanTRTBackend

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

class SingleGPUInferencePipelineTRT:
    def __init__(self, config, device: torch.device, engine_path: str):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("TRTInference")
        self.logger.setLevel(logging.INFO)
        
        print(f"Initializing Pipeline on {device}...")
        self.pipeline = CausalStreamInferencePipeline(config, device=str(device))
        self.pipeline.to(device=str(device), dtype=torch.bfloat16) 
        
        print(f"Swapping PyTorch Generator for TensorRT Engine: {engine_path}")
        self.trt_model = WanTRTBackend(engine_path, device=str(device))
        self.pipeline.generator = self.trt_model
        
        self.processed = 0
        self.vae_on_cpu = False
        self.logger.info("TRT Pipeline Ready.")

    def load_model(self, checkpoint_folder: str):
        ckpt_path = os.path.join(checkpoint_folder, "model.pt")
        print(f"Loading VAE/TextEncoder weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        try:
            temp_gen = self.pipeline.generator
            self.pipeline.generator = None 
            self.pipeline.load_state_dict(ckpt, strict=False)
        except Exception as e:
            print(f"Partial weight load (Expected for TRT): {e}")
        finally:
            self.pipeline.generator = temp_gen 

    def _switch_vae_to_cpu(self):
        """
        Helper to forcefully move the VAE to CPU (float32) when OOM occurs.
        PERFORMS DEEP SCAN to move hidden 'feat_cache' lists.
        """
        if self.vae_on_cpu: return
        print("(!) Critical Memory Pressure: Moving VAE to CPU (Float32)...")
        self.vae_on_cpu = True
        torch.cuda.empty_cache()
        
        # 1. Move main wrapper
        self.pipeline.vae = self.pipeline.vae.cpu().float()
        
        # 2. Move inner model
        if hasattr(self.pipeline.vae, 'model'):
            self.pipeline.vae.model = self.pipeline.vae.model.cpu().float()
            
            # --- DEEP SCAN FOR CACHES ---
            print("    Deep Scan: Migrating internal VAE caches to CPU...")
            moved_count = 0
            # Recursively walk every single module in the VAE
            for name, module in self.pipeline.vae.model.named_modules():
                # Check for 'feat_cache' (Encoder/Decoder cache)
                if hasattr(module, 'feat_cache'):
                    cache = getattr(module, 'feat_cache')
                    if isinstance(cache, list):
                        # Move every tensor in the list
                        new_cache = []
                        for item in cache:
                            if isinstance(item, torch.Tensor):
                                new_cache.append(item.detach().cpu().float())
                            else:
                                new_cache.append(item)
                        module.feat_cache = new_cache
                        moved_count += 1
                    elif isinstance(cache, torch.Tensor):
                        module.feat_cache = cache.detach().cpu().float()
                        moved_count += 1
                        
                # Check for 'res_cache' (Residual cache)
                if hasattr(module, 'res_cache'):
                    cache = getattr(module, 'res_cache')
                    if isinstance(cache, torch.Tensor):
                        module.res_cache = cache.detach().cpu().float()
            print(f"    Moved {moved_count} internal cache buffers.")

        # 3. Force 'mean' and 'std' attributes to CPU
        if hasattr(self.pipeline.vae, 'mean'): 
            self.pipeline.vae.mean = self.pipeline.vae.mean.cpu().float()
        if hasattr(self.pipeline.vae, 'std'): 
            self.pipeline.vae.std = self.pipeline.vae.std.cpu().float()

    def _manual_cpu_encode(self, pixel_chunk):
        """Bypasses WanVAEWrapper logic to ensure CPU execution."""
        x_cpu = pixel_chunk.detach().cpu().float()
        mean = getattr(self.pipeline.vae, 'mean', torch.tensor(0.0)).cpu().float()
        std = getattr(self.pipeline.vae, 'std', torch.tensor(1.0)).cpu().float()
        scale = [mean, std]

        with torch.no_grad():
            latents = self.pipeline.vae.model.stream_encode(x_cpu, scale)
            
        return latents.to(self.device, dtype=torch.float16)

    def _manual_cpu_decode(self, z):
        """Bypasses WanVAEWrapper logic to ensure CPU execution."""
        z_cpu = z.detach().cpu().float()
        mean = getattr(self.pipeline.vae, 'mean', torch.tensor(0.0)).cpu().float()
        std = getattr(self.pipeline.vae, 'std', torch.tensor(1.0)).cpu().float()
        scale = [mean, std]
        
        with torch.no_grad():
            output = self.pipeline.vae.model.stream_decode(z_cpu, scale)
        
        return output.float().clamp(-1, 1)

    def safe_encode(self, pixel_chunk):
        """Safely encodes video chunk, handling CPU fallback if active."""
        if self.vae_on_cpu:
            return self._manual_cpu_encode(pixel_chunk)
        
        try:
            return self.pipeline.vae.stream_encode(pixel_chunk)
        except (torch.OutOfMemoryError, RuntimeError):
            self._switch_vae_to_cpu()
            return self._manual_cpu_encode(pixel_chunk)

    def safe_decode(self, latent_slice):
        """Safely decodes latent slice, handling CPU fallback if active."""
        if self.vae_on_cpu:
            return self._manual_cpu_decode(latent_slice)

        try:
            return self.pipeline.vae.stream_decode_to_pixel(latent_slice.to(torch.bfloat16))
        except (torch.OutOfMemoryError, RuntimeError):
            self._switch_vae_to_cpu()
            return self._manual_cpu_decode(latent_slice)

    def run_inference(self, input_video_original, prompts, num_chuncks, chunck_size, noise_scale, output_folder, fps, num_steps):
        os.makedirs(output_folder, exist_ok=True)
        results = {}
        save_results = 0
        
        start_idx = 0
        end_idx = 5
        current_start = 0
        current_end = self.pipeline.frame_seq_length * 2
        
        print("Warming up TRT Engine...")
        dummy_x = torch.randn(1, 16, 1, 60, 106, dtype=torch.float16, device=self.device)
        dummy_t = torch.tensor([1.0], dtype=torch.float16, device=self.device)
        dummy_ctx = torch.randn(1, 512, 4096, dtype=torch.float16, device=self.device)
        self.trt_model(dummy_x, dummy_t, dummy_ctx)
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache() 
        
        if input_video_original is not None:
            inp = input_video_original[:, :, start_idx:end_idx]
            latents = self.safe_encode(inp)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.float16) 
            noise = torch.randn_like(latents)
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
        else:
            noisy_latents = torch.randn(1, 1, 16, 60, 106, device=self.device, dtype=torch.float16)

        denoised_pred = self.pipeline.prepare(
            text_prompts=prompts,
            device=self.device,
            dtype=torch.float16,
            block_mode='input',
            noise=noisy_latents,
            current_start=current_start,
            current_end=current_end
        )
        
        # Decode First Frame
        torch.cuda.empty_cache()
        with torch.no_grad():
            video = self.safe_decode(denoised_pred[[-1]])

        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        results[save_results] = video.cpu().float().numpy()
        save_results += 1
        
        init_noise_scale = noise_scale
        
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
                
                # --- Safe Encode Call ---
                latents = self.safe_encode(inp)
                # -----------------------
                
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.float16)
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
            else:
                noisy_latents = torch.randn(1, 1, 16, 60, 106, device=self.device, dtype=torch.float16)
                current_step = None 

            iter_start = time.time()
            noisy_latents = noisy_latents.to(self.device)
            
            denoised_pred = self.pipeline.inference_stream(
                noise=noisy_latents,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )
            self.processed += 1
            
            if self.processed >= num_steps:
                torch.cuda.empty_cache() 
                with torch.no_grad():
                    video = self.safe_decode(denoised_pred[[-1]])

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
            output_path = os.path.join(output_folder, f"trt_output.mp4")
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
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B")
    parser.add_argument("--num_frame_per_block", type=int, default=1)
    parser.add_argument("--warp_denoising_step", action="store_true", default=False)
    
    args = parser.parse_args()
    
    if args.width != 848:
        args.width = 848

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

    pipeline = SingleGPUInferencePipelineTRT(config, device, "wan_dit.engine")
    pipeline.load_model(args.checkpoint_folder)
    
    input_video = None
    if args.video_path:
        print(f"Loading video: {args.video_path}")
        input_video = load_mp4_as_tensor(args.video_path, resize_hw=(480, 848))
        input_video = input_video.unsqueeze(0).to(device, dtype=torch.bfloat16)
    
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    
    pipeline.run_inference(
        input_video, prompts, (args.num_frames - 1) // 4, 4, 
        args.noise_scale, args.output_folder, args.fps, len(config.denoising_step_list)
    )

if __name__ == "__main__":
    main()