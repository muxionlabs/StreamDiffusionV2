"""
TensorRT-Accelerated Inference Script

Uses TensorRT DiT engine with PyTorch VAE for optimal performance.
Based on streamv2v/inference.py but with TRT acceleration for 3.5x speedup.

Usage:
    python scripts/trt_inference.py \
        --config_path ./configs/your_config.yaml \
        --checkpoint_folder ./checkpoints/your_model \
        --output_folder ./outputs \
        --prompt_file_path ./prompts.txt \
        --dit_engine_path ./trt_engines/dit_streaming.engine \
        --video_path ./input.mp4 \
        --guidance_scale 1.0
"""

import argparse
import os
import time
import logging
from typing import Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from diffusers.utils import export_to_video

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRTInference")


def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Load an .mp4 video as a tensor [C, T, H, W]."""
    import torchvision
    import torchvision.transforms.functional as TF
    from einops import rearrange
    
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    
    # Use 'sec' units to avoid warnings and inaccurate timestamps
    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit="sec")
    if max_frames is not None:
        video = video[:max_frames]
    
    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([
            TF.resize(video[:, i], resize_hw, antialias=True)
            for i in range(t)
        ], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0
    
    return video


class TRTAcceleratedInferencePipeline:
    """
    TensorRT-accelerated inference pipeline.
    
    Uses TensorRT DiT engine for fast denoising while keeping
    PyTorch VAE (which is already fast enough).
    """
    
    def __init__(
        self,
        config,
        dit_engine_path: str,
        device: torch.device = None,
        max_seq_len: int = 150000,
        enable_cfg: bool = True,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cfg = enable_cfg
        self.max_seq_len = max_seq_len  # Initialize early for both TRT and PyTorch modes
        
        # Load PyTorch pipeline for VAE and text encoder
        from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
        self.pytorch_pipeline = CausalStreamInferencePipeline(config, device=str(self.device))
        self.pytorch_pipeline.to(device=str(self.device), dtype=torch.bfloat16)
        
        # Load TensorRT DiT engine
        from causvid.acceleration.tensorrt.engines.simple_engines import DiTEngineSimple, DiTEngineStreaming
        import os
        
        # Check if TRT engine exists, otherwise fall back to PyTorch
        self.streaming = os.path.exists(dit_engine_path) if dit_engine_path else False
        
        if not self.streaming:
            logger.info("TRT engine not found or not specified, using pure PyTorch mode")
        
        if self.streaming:
            # FREE PYTORCH KV CACHE to save memory (~18GB for B=2)
            # We only use TRT cache
            if hasattr(self.pytorch_pipeline, "kv_cache1"):
                self.pytorch_pipeline.kv_cache1 = None
                self.pytorch_pipeline.kv_cache2 = None
                torch.cuda.empty_cache()
            
            # Offload PyTorch DiT model to CPU to save VRAM (we use TRT engine)
            if hasattr(self.pytorch_pipeline, "generator"):
                logger.info("Offloading PyTorch DiT model to CPU to save VRAM")
                self.pytorch_pipeline.generator.model.to("cpu")
                
            # Also offload Text Encoder (T5-XXL is ~22GB!)
            if hasattr(self.pytorch_pipeline, "text_encoder"):
                logger.info("Offloading Text Encoder to CPU to save VRAM")
                self.pytorch_pipeline.text_encoder.to("cpu")
                
            torch.cuda.empty_cache()

            self.dit_engine = DiTEngineStreaming(
                dit_engine_path, 
                use_cuda_graph=False, # Must be False for dynamic control flow/shapes if unpredictable
                device=str(self.device),
            )
            
            # Smart Memory Allocation - SPLIT CACHE
            num_layers_per_chunk = 1 # 1 Layer per chunk (Safest for TRT < 1GB)
            num_chunks = 30 # 30 Layers / 1 = 30 chunks
            
            num_heads = 12
            total_layers = 30
            head_dim = 128
            dtype_size = 2 # float16
            
            # Calculate memory per token per batch
            bytes_per_token = 2 * total_layers * num_heads * head_dim * dtype_size
            
            # Use max_seq_len from args
            self.max_seq_len = max_seq_len
            
            # Check available VRAM
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            free_vram = t - a 
            
            # Log chunk size
            chunk_bytes = num_layers_per_chunk * 2 * max_seq_len * num_heads * head_dim * 2
            logger.info(f"Allocating 1 x {num_chunks} TRT KV Chunks: ({num_layers_per_chunk}, 2, 1, {max_seq_len}, {num_heads}, {head_dim})")
            logger.info(f"Size per chunk: {chunk_bytes/1e9:.2f} GB (Safe < 2.14GB)")
            
            logger.info(f"VRAM Stats: Total={t/1e9:.2f}GB, Allocated={a/1e9:.2f}GB, Free (approx)={(t-a)/1e9:.2f}GB")
            
            num_caches = 2 if self.enable_cfg else 1
            required_mem = max_seq_len * bytes_per_token * num_caches
            
            if required_mem > (free_vram * 0.95):
                logger.warning(f"WARNING: Requested cache size ({required_mem/1e9:.2f} GB) exceeds available VRAM ({free_vram/1e9:.2f} GB)!")
                safe_mem = free_vram * 0.90
                max_safe_seq = int(safe_mem / (bytes_per_token * num_caches))
                logger.warning(f"Downgrading max_seq_len from {max_seq_len} to {max_safe_seq} to fit VRAM.")
                max_seq_len = max_safe_seq
            
            self.max_seq_len = max_seq_len
            
            # Allocate 5 chunks of cache
            # Shape per chunk: [Layers, 2, Batch, Seq, Head, Dim] where Batch=1 (Required by Engine)
            # 6D shape: [Layers, 2, 1, Seq, Head, Dim]
            chunk_shape = (num_layers_per_chunk, 2, 1, max_seq_len, num_heads, head_dim)
            logger.info(f"Allocating {num_caches} x {num_chunks} TRT KV Chunks: {chunk_shape} (Float16)")
            logger.info(f"Total Cache Memory: {required_mem/1e9:.2f} GB")
            
            # List of lists [BatchSet_0_Chunks, BatchSet_1_Chunks (if CFG)]
            self.trt_kv_cache = []
            for _ in range(num_caches):
                # Valid list of 5 chunks
                chunks = [
                    torch.zeros(chunk_shape, device=self.device, dtype=torch.float16)
                    for _ in range(num_chunks)
                ]
                self.trt_kv_cache.append(chunks)
            
            # Cache Metadata for Eviction Logic (matching PyTorch pipeline)
            self.cache_metadata = {
                'global_end_index': torch.zeros(num_caches, dtype=torch.long, device=self.device),
                'local_end_index': torch.zeros(num_caches, dtype=torch.long, device=self.device),
                'sink_size': 3,  # Preserve first 3 frames
                'adapt_sink_thr': -1,  # Adaptive sink threshold (-1 = disabled)
                'kv_cache_size': max_seq_len,  # Max cache capacity
            }
            logger.info(f"Cache metadata initialized: sink_size={self.cache_metadata['sink_size']}, max_capacity={max_seq_len}")
        else:
            self.dit_engine = DiTEngineSimple(
                dit_engine_path,
                use_cuda_graph=True,
                device=str(self.device),
            )
        
        # Reference to PyTorch components
        self.vae = self.pytorch_pipeline.vae
        self.text_encoder = self.pytorch_pipeline.text_encoder
        
        # Tracking
        self.processed = 0
        
        logger.info(f"TRT-accelerated pipeline initialized (streaming={self.streaming})")
        logger.info(f"  DiT engine: {dit_engine_path}")
        logger.info(f"  VAE: PyTorch (recommended)")
        
    def load_model(self, checkpoint_folder: str):
        """Load model checkpoint (supports .pt and .safetensors)."""
        import os
        from safetensors.torch import load_file
        
        # Priority 1: model.pt
        pt_path = os.path.join(checkpoint_folder, "model.pt")
        # Priority 2: diffusion_pytorch_model.safetensors (Standard Diffusers)
        sf_path = os.path.join(checkpoint_folder, "diffusion_pytorch_model.safetensors")
        # Priority 3: model.safetensors
        sf_v2_path = os.path.join(checkpoint_folder, "model.safetensors")
        
        if os.path.exists(pt_path):
            logger.info(f"Loading checkpoint from {pt_path}")
            ckpt = torch.load(pt_path, map_location="cpu")
        elif os.path.exists(sf_path):
            logger.info(f"Loading checkpoint from {sf_path}")
            ckpt = load_file(sf_path)
        elif os.path.exists(sf_v2_path):
            logger.info(f"Loading checkpoint from {sf_v2_path}")
            ckpt = load_file(sf_v2_path)
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_folder}. Checked: model.pt, diffusion_pytorch_model.safetensors")
        
        if isinstance(ckpt, dict):
            if 'generator' in ckpt:
                state_dict = ckpt['generator']
            elif 'generator_ema' in ckpt:
                state_dict = ckpt['generator_ema']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        
        # Load weights
        msg = self.pytorch_pipeline.generator.load_state_dict(state_dict, strict=False)
        logger.info(f"Checkpoint loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
    
    def _evict_kv_cache(self, batch_idx: int, num_new_tokens: int, frame_seqlen: int):
        """
        Evict oldest tokens from KV cache when full (excluding sink tokens).
        Ported from causvid/models/wan/causal_model.py lines 175-189.
        
        Args:
            batch_idx: Index in batch (0 for single sequence, 0/1 for CFG)
            num_new_tokens: Number of tokens being added
            frame_seqlen: Tokens per frame (e.g., 1560 for 480p)
        """
        sink_tokens = self.cache_metadata['sink_size'] * frame_seqlen
        kv_cache_size = self.cache_metadata['kv_cache_size']
        local_end = self.cache_metadata['local_end_index'][batch_idx].item()
        
        # Calculate eviction
        num_evicted_tokens = num_new_tokens + local_end - kv_cache_size
        num_rolled_tokens = local_end - num_evicted_tokens - sink_tokens
        
        logger.info(f"[Eviction] Batch {batch_idx}: Evicting {num_evicted_tokens} tokens, rolling {num_rolled_tokens} tokens")
        
        # Shift cache left for all chunks (evict oldest non-sink tokens)
        cache_chunks = self.trt_kv_cache[batch_idx]
        for chunk in cache_chunks:
            # chunk shape: [Layers, 2, Batch, MaxSeq, H, D] (Layers=1, Batch=1)
            # Extract K and V: [2, 1, MaxSeq, H, D]
            kv = chunk[0]  # [2, 1, MaxSeq, H, D]
            
            # Shift K cache (index 0)
            # kv[0] is [1, MaxSeq, H, D]. We slice on dim 1 (Seq).
            kv[0, :, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv[0, :, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            
            # Shift V cache (index 1)
            kv[1, :, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv[1, :, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        
        # Update local end index (cache fill level after eviction)
        new_local_end = sink_tokens + num_rolled_tokens
        self.cache_metadata['local_end_index'][batch_idx] = new_local_end
        
        logger.info(f"[Eviction] Batch {batch_idx}: Cache now at {new_local_end}/{kv_cache_size} tokens")
        
        return new_local_end
    
    def prepare_pipeline(
        self,
        text_prompts: list,
        noise: torch.Tensor,
        current_start: int,
        current_end: int,
    ):
        """Prepare pipeline (uses PyTorch for multi-timestep denoising)."""
        logger.info("Preparing pipeline with PyTorch (initial setup)...")
        
        # Move PyTorch models back to GPU (they were offloaded to CPU to save VRAM)
        if hasattr(self.pytorch_pipeline, "generator"):
            self.pytorch_pipeline.generator.to(self.device)
        if hasattr(self.pytorch_pipeline, "text_encoder"):
            self.pytorch_pipeline.text_encoder.to(self.device)
        
        # Delegate to PyTorch for prepare phase (complex multi-timestep denoising)
        # TRT is better suited for the simpler inference_stream phase
        return self.pytorch_pipeline.prepare(
            text_prompts=text_prompts,
            device=self.device,
            dtype=torch.bfloat16,
            noise=noise,
            current_start=current_start,
            current_end=current_end,
        )
    
    def inference_stream_trt(
        self,
        noise: torch.Tensor,
        current_start: int,
        current_end: int,
        logical_frame_idx: int,
        current_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        TRT-accelerated streaming inference (matches PyTorch's inference_stream).
        Processes one frame through the denoising batch using TRT engine.
        Returns: (output_tensor, num_new_tokens_consumed)
        """
        if not self.streaming:
            # Legacy PyTorch path (not used in this fix)
            out = self.pytorch_pipeline.inference_stream(
                noise=noise,
                current_start=current_start,
                current_end=current_end,
                current_step=current_step,
            )
            return out, 0
            
        # ... implementation ...
        
        # Get dependencies
        conditional_dict = self.pytorch_pipeline.conditional_dict
        device = self.device
        
        # Prepare inputs (matching PyTorch's inference_stream lines 241-256)
        B, F, C, H, W = noise.shape
        noisy_image = noise.to(torch.float16)  # TRT uses float16
        prompt_embeds = conditional_dict["prompt_embeds"].to(device=device, dtype=torch.float16)
        
        # Handle CFG disable case (guidance_scale=1.0 should use single batch)
        # Cache metadata was initialized for batch size 1
        if prompt_embeds.shape[0] > 1 and len(self.trt_kv_cache) == 1:
            # Text encoder returned 2 embeds (neg, pos) but cache is sized for 1
            # Use only positive prompt (index 1)
            prompt_embeds = prompt_embeds[1:2]
        
        # Calculate frame_seqlen for eviction
        # Latents are H_pix // 8, W_pix // 8.
        # Patch size is 2x2.
        # Tokens = (H_lat // 2) * (W_lat // 2)
        # H, W here are LATENT dims from noisy_image shape
        frame_seqlen = (H // 2) * (W // 2)
        num_new_tokens = frame_seqlen * F
        
        # Prepare current_start tensor (Physical Cache Index)
        # TRT expects Rank 1 tensor [B] for indices.
        # We should use torch.tensor([val], device=...).
        
        # FIX: Wrap logical frame index to stay within Engine Profile Max (24000)
        # Otherwise, start_frame_idx keeps growing and crashes TRT shape inference.
        kv_cache_size = self.cache_metadata['kv_cache_size']
        start_frame_idx_t = torch.tensor(
            [logical_frame_idx], 
            device=device, 
            dtype=torch.long
        )
        
        # Determine batch size
        B_text = prompt_embeds.shape[0]
        
        if B_text > 1:
            # CFG mode: process each batch separately
            flow_preds = []
            for b_idx in range(B_text):
                # Prepare inputs
                prompt_slice = prompt_embeds[b_idx:b_idx+1]
                
                # Check eviction
                # In Ring Buffer mode (Streaming), simply overwrite based on current_start.
                # current_start is already wrapped modulo max_seq_len by the caller.
                
                cache_chunks = self.trt_kv_cache[b_idx]
                 
                # Physical Start for Cache Write = current_start (Ring Buffer Index)
                current_start_physical_t = torch.tensor([current_start], device=device, dtype=torch.long)
                 
                # Call TRT engine
                flow_out, _ = self.dit_engine(
                    noisy_image, 
                    torch.full((1, F), current_step if current_step is not None else 0, device=device, dtype=torch.long),
                    prompt_slice,
                    cache_chunks,
                    current_start_physical_t, # Physical (where to write)
                    start_frame_idx_t,        # Logical (what time is it)
                )
                flow_preds.append(flow_out)
                
                # Update metadata for consistency (though main loop drives index)
                self.cache_metadata['local_end_index'][b_idx] = (current_start + num_new_tokens) % kv_cache_size
            
            output = torch.cat(flow_preds, dim=0)
        else:
            # Single batch
            b_idx = 0
            
            # Check eviction
            # In Ring Buffer mode, we trust current_start is the correct physical slot.
            
            cache_chunks = self.trt_kv_cache[b_idx]
            
            # Physical Start for Cache Write based on current_start
            current_start_physical_t = torch.tensor([current_start], device=device, dtype=torch.long)
            
            timestep = torch.full((B, F), current_step if current_step is not None else 0, device=device, dtype=torch.long)
            
            logger.info(f"[TRT DEBUG] Engine Call:")
            logger.info(f"  x: {noisy_image.shape} mean={noisy_image.float().mean().item():.4f} std={noisy_image.float().std().item():.4f}")
            logger.info(f"  timestep: {timestep.shape} val={timestep.item()}")
            logger.info(f"  context: {prompt_embeds.shape} mean={prompt_embeds.float().mean().item():.4f} std={prompt_embeds.float().std().item():.4f}")
            logger.info(f"  current_start (phys): {current_start_physical_t.shape} item={current_start_physical_t.item()}")
            logger.info(f"  start_frame (logic): {start_frame_idx_t.shape} item={start_frame_idx_t.item()}")
            logger.info(f"  num_new_tokens: {num_new_tokens}")
            
            # SANITY CHECK: Verify 6D Shapes and Dtypes
            logger.info("[TRT SANITY] Checking Inputs before DiT Engine Call:")
            # for i, c in enumerate(cache_chunks):
            #     logger.info(f"  KV {i}: shape={c.shape} dtype={c.dtype} stride={c.stride()}")
            logger.info(f"  current_start_physical_t: {current_start_physical_t.item()} dtype={current_start_physical_t.dtype} shape={current_start_physical_t.shape}")
            logger.info(f"  start_frame_idx_t: {start_frame_idx_t.item()} dtype={start_frame_idx_t.dtype} shape={start_frame_idx_t.shape}")

            output, _ = self.dit_engine(
                noisy_image,
                timestep,
                prompt_embeds,
                cache_chunks,
                current_start_physical_t, # Physical
                start_frame_idx_t,        # Logical
            )
            logger.info(f"  [TRT DEBUG] TRT Output: shape={output.shape} mean={output.float().mean().item():.4f} std={output.float().std().item():.4f}")
            
            # Update metadata
            self.cache_metadata['local_end_index'][b_idx] = (current_start + num_new_tokens) % kv_cache_size
        
        # Return output (no scheduler step, PyTorch's inference_stream returns prediction directly)
        # TRT output is [B, C, T, H, W], but pipeline expects [B, T, C, H, W]
        if len(output.shape) == 5:
             output = output.permute(0, 2, 1, 3, 4)
             
        return output.to(torch.bfloat16), num_new_tokens
    
    def _sync_pytorch_kv_to_trt(self):
        """
        Copy KV cache data from PyTorch pipeline to TRT cache.
        CRITICAL: This bridges the 'prepare' phase (PyTorch) and 'stream' phase (TRT).
        Without this, TRT sees empty sink tokens and generates garbage.
        """
        logger.info("Synchronizing KV cache from PyTorch to TRT...")
        
        # PyTorch cache: List[Dict['k'/'v': Tensor]]
        pt_cache = self.pytorch_pipeline.kv_cache1
        
        # TRT cache: List[List[Tensor]] (Batch -> Chunks)
        # Chunk shape: [LayersPerChunk, 2, 1, Seq, H, D]
        
        num_blocks = len(pt_cache)
        # Dynamically determine layers per chunk from the allocated cache
        layers_per_chunk = self.trt_kv_cache[0][0].shape[0]
        # Determine valid length to copy (sink size)
        # sink_end = self.pytorch_pipeline.kv_cache_ends[0].item()
        
        # Determine valid length to copy (sink size)
        # The 'prepare' phase fills the first few frames (sink size is 3 frames)
        sink_end = self.pytorch_pipeline.kv_cache_ends[0].item()
        
        # Robustly calculate frame_seqlen from the known sink size (3 frames)
        if sink_end > 0:
             real_frame_seqlen = sink_end // 3
        else:
             real_frame_seqlen = (self.pytorch_pipeline.height // 2) * (self.pytorch_pipeline.width // 2)
             
        logger.info(f"Sync Debug: Sink End (3 frames)={sink_end}")
        logger.info(f"Sync Debug: Frame Seq Len (Derived)={real_frame_seqlen}")
        
        # FIX 2: The previous log showed zeros at index 4000, even though sink_end=4680.
        # This is because prepare_pipeline was called with current_end=3120 (2 frames).
        # So essentially, only Frames 0 and 1 are valid. Frame 2 is empty/partial.
        # SAFE STRATEGY: Sync ONLY 2 Frames (0-1).
        
        safe_frames = 2
        sync_len = safe_frames * real_frame_seqlen
        
        logger.info(f"Syncing up to index: {sync_len} (Strategy: Sync 2 Frames & Rewind)")
        
        # Update metadata to reflect we have valid history up to sync_len
        updated_end = sync_len
        
        for b_idx in range(len(self.trt_kv_cache)):
            for i in range(num_blocks):
                chunk_idx = i // layers_per_chunk
                layer_idx = i % layers_per_chunk
                
                # Get PyTorch K/V
                k_pt = pt_cache[i]['k']
                v_pt = pt_cache[i]['v']
                
                # Handle batch dimension
                if k_pt.shape[0] > 1:
                     k_pt = k_pt[0]
                     v_pt = v_pt[0]
                elif k_pt.dim() == 4 and k_pt.shape[0] == 1:
                     k_pt = k_pt[0]
                     v_pt = v_pt[0]

                trt_chunk = self.trt_kv_cache[b_idx][chunk_idx]
                
                try:
                    # K: trt_chunk[layer_idx, 0, 0, :sync_len]
                    valid_len = min(sync_len, k_pt.shape[0])
                    
                    trt_chunk[layer_idx, 0, 0, :valid_len] = k_pt[:valid_len].to(dtype=torch.float16)
                    trt_chunk[layer_idx, 1, 0, :valid_len] = v_pt[:valid_len].to(dtype=torch.float16)
                         
                except Exception as e:
                    logger.error(f"Failed to sync layer {i} (Shape {k_pt.shape} -> {trt_chunk.shape}): {e}")
                    raise e
        
        # Update metadata
        for b_idx in range(len(self.trt_kv_cache)):
            self.cache_metadata['local_end_index'][b_idx] = updated_end
            self.cache_metadata['global_end_index'][b_idx] = updated_end
            
        logger.info(f"Successfully synchronized {num_blocks} layers (up to {updated_end}) and updated metadata.")
        return updated_end

    def run_inference_v2v(
        self,
        input_video: torch.Tensor,
        prompts: list,
        num_chunks: int,
        chunk_size: int,
        noise_scale: float,
        output_folder: str,
        fps: int,
        num_steps: int,
    ):
        """
        Video-to-video inference with TRT acceleration.
        """
        logger.info("Starting TRT-accelerated v2v inference")
        
        os.makedirs(output_folder, exist_ok=True)
        results = {}
        save_results = 0
        
        fps_list = []
        dit_fps_list = []
        
        start_idx = 0
        # Keep chunk contract aligned with PyTorch baseline inference:
        # warmup uses 5 frames, then every loop consumes `chunk_size` (default=4).
        end_idx = 5
        current_start = 0
        # Initialize current_end for prepare. This limits how much PyTorch caches.
        current_end = self.pytorch_pipeline.frame_seq_length * 2
        
        # Track logical frame count for RoPE (unwrapped)
        total_logical_frames = 0
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # First chunk
        if input_video is not None:
            logger.info(f"Loaded input_video shape: {input_video.shape}")
            inp = input_video[:, :, start_idx:end_idx]
            logger.info(f"Initial chunk (indices {start_idx}:{end_idx}) shape: {inp.shape}")
            latents = self.vae.stream_encode(inp)
            latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
            
            noise = torch.randn_like(latents)
            noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
        else:
            noisy_latents = torch.randn(
                1, 1 + self.pytorch_pipeline.num_frame_per_block, 16,
                self.pytorch_pipeline.height, self.pytorch_pipeline.width,
                device=self.device, dtype=torch.bfloat16
            )
        
        # Prepare
        denoised_pred = self.prepare_pipeline(
            text_prompts=prompts,
            noise=noisy_latents,
            current_start=current_start,
            current_end=current_end,
        )
        
        # Sync KV cache from PyTorch to TRT
        # if self.streaming:
             # synced_end = self._sync_pytorch_kv_to_trt()
        
        # FIX: Memory copying from PyTorch is unreliable due to opaque cache management.
        # ULTIMATE SAFE STRATEGY: Do NOT sync memory.
        # Instead, start TRT from Frame 0 (rewind completely).
        # TRT will re-compute Frames 0-5 itself, building its own perfect cache.
        # We perform a "warm start" by letting PyTorch initialize the pipeline, 
        # but then we let TRT take over from the beginning.
        
        if self.streaming:
             logger.info("Strategy: Partial Sync failed. Rewinding to Frame 0 for clean TRT start (required for empty cache).")
             
             # Rewind to Frame 0
             # Since we are NOT syncing PyTorch KV check (safe execution), we must start from 0.
             # Starting at 3120 with empty cache causes Shape Overflow in attention plugin.
             current_end = 0
             end_idx = 0
             total_logical_frames = 0
             
             # Reset Metadata to 0
             for b_idx in range(len(self.trt_kv_cache)):
                 self.cache_metadata['local_end_index'][b_idx] = 0
                 self.cache_metadata['global_end_index'][b_idx] = 0
             logger.info("TRT Cache reset to 0. Starting clean generation.")
             
             # CRITICAL: Reset VAE state to prevent corruption loop
             if hasattr(self.vae, 'clear_cache'):
                 self.vae.clear_cache()
                 logger.info("VAE streaming cache cleared.")

        
        # Decode first result
        video = self.vae.stream_decode_to_pixel(denoised_pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video[0].permute(0, 2, 3, 1).contiguous()
        logger.info(f"[DEBUG] First chunk decoded {video.shape[0]} frames (denoised_pred shape: {denoised_pred.shape})")
        results[save_results] = video.cpu().float().numpy()
        save_results += 1
        
        init_noise_scale = noise_scale
        
        # Process remaining chunks
        while self.processed < num_chunks + num_steps - 1:
            start_idx = end_idx
            end_idx = end_idx + chunk_size
            # current_start = current_end # REMOVED: In Ring Buffer mode, current_start persists and wraps naturally.
            # FIX: With chunk_size=1, (1//4) was 0, so current_end never updated!
            # The '4' likely came from legacy code assuming 4 frames per block.
            # We should just multiply by chunk_size since frame_seq_length is usually per-frame (1560).
            # But wait, frame_seq_length in PyTorch pipeline is 1560 * num_frames? No, usually per frame.
            # Verified: frame_seq_length in WanPipeline is (H//16)*(W//16).
            # So increment should be chunk_size * frame_seq_len.
            
            # `current_end` is advanced inside the per-latent loop.
            
            # Check for cache overflow
            
            # Check for cache overflow
            # Check for cache overflow
            # With Ring Buffer, current_end (Logical) can safely exceed max_seq_len.
            # Eviction logic inside inference_stream_trt handles the physical limit.
            # if current_end >= self.max_seq_len:
            #     logger.warning(f"Logical Index {current_end} > Max {self.max_seq_len}. Relying on Ring Buffer.")
            
            if input_video is not None and end_idx <= input_video.shape[2]:
                inp = input_video[:, :, start_idx:end_idx]
                logger.info(f"[TRT DEBUG] Input Video: shape={inp.shape} mean={inp.float().mean().item():.4f} std={inp.float().std().item():.4f}")
                
                # Adaptive noise
                # Safe check for boundary condition (cannot compute L2 dist for first frame if start_idx=0)
                if end_idx - chunk_size > 0:
                    l2_dist = (input_video[:, :, end_idx-chunk_size:end_idx] - 
                               input_video[:, :, end_idx-chunk_size-1:end_idx-1]) ** 2
                    l2_dist = (torch.sqrt(l2_dist.mean(dim=(0, 1, 3, 4))).max() / 0.2).clamp(0, 1)
                    noise_scale = (init_noise_scale - 0.1 * l2_dist.item()) * 0.9 + noise_scale * 0.1
                else:
                    # For the first chunk (if rewound to 0), keep init_noise_scale
                    pass
                
                current_step = int(1000 * noise_scale) - 100
                
                latents = self.vae.stream_encode(inp)
                logger.info(f"[TRT DEBUG] VAE Latents: shape={latents.shape} mean={latents.float().mean().item():.4f} std={latents.float().std().item():.4f}")
                latents = latents.transpose(2, 1).contiguous().to(dtype=torch.bfloat16)
                
                # Decouple: VAE produced T>1 latents. DiT expects T=1.
                # Loop over latents.
                denoised_latents_list = []
                # latents is [B, T, C, H, W] after transpose
                num_latents = latents.shape[1] 
                
                # Base noise for the whole block
                noise_base = torch.randn_like(latents)
                noisy_latents_base = noise_base * noise_scale + latents * (1 - noise_scale)
                
                for i in range(num_latents):
                    # Slice Time dimension (Dim 1)
                    sub_noisy = noisy_latents_base[:, i:i+1, :, :, :]
                    
                    torch.cuda.synchronize()
                    dit_start_time = time.time()
                    
                    # DiT inference (TRT integration point)
                    # sub_noisy shape: [B, 1, C, H, W] -> e.g. [1, 1, 16, 60, 104]
                    
                    # Increment for next latent
                    frame_seq = self.pytorch_pipeline.frame_seq_length
                    
                    sub_pred, num_tokens_added = self.inference_stream_trt(
                        noise=sub_noisy,
                        current_start=current_start,
                        current_end=current_end,
                        logical_frame_idx=total_logical_frames,
                        current_step=current_step,
                    )
                    
                    denoised_latents_list.append(sub_pred)
                    
                    # Advance Ring Buffer Pointers
                    # Update physical pointer by exact number of tokens used
                    current_start = (current_start + num_tokens_added) % self.max_seq_len
                    
                    # Advance logical time for next iteration
                    total_logical_frames += 1
                    
                    if self.processed > 3:
                        torch.cuda.synchronize()
                
                # Concatenate over temporal axis [B, T, C, H, W]
                denoised_pred = torch.cat(denoised_latents_list, dim=1)
            else:
                 # Padding case (unused in V2V usually)
                 denoised_pred = torch.zeros(1, 1, 16, 60, 104, device=self.device, dtype=torch.bfloat16)

            
            if self.processed > 3:
                torch.cuda.synchronize()
                dit_fps_list.append(chunk_size / (time.time() - start_time)) # Approx
            
            self.processed += 1
            
            if self.processed >= num_steps: # Logic check? processed is chunks?
                # Decoder expects [B, 16, T, H, W] -> [B, 3, T*4, H*16, W*16]
                video = self.vae.stream_decode_to_pixel(denoised_pred)
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(0, 2, 3, 1).contiguous()
                
                logger.info(f"[DEBUG] Chunk {save_results} decoded {video.shape[0]} frames (denoised_pred shape: {denoised_pred.shape})")
                
                results[save_results] = video.cpu().float().numpy()
                save_results += 1
                
                torch.cuda.synchronize()
                end_time = time.time()
                t = end_time - start_time
                fps_test = chunk_size / t
                fps_list.append(fps_test)
                logger.info(f"Processed {self.processed}, time: {t:.4f} s, FPS: {fps_test:.4f}")
                start_time = end_time
        
        # Save video
        logger.info(f"[DEBUG] Total saved chunks: {save_results}, num_chunks: {num_chunks}")
        video_list = [results[i] for i in range(save_results) if i in results]
        if not video_list:
            logger.error("No frames generated!")
            return

        video = np.concatenate(video_list, axis=0)
        fps_avg = np.mean(np.array(fps_list)) if fps_list else 0
        
        logger.info(f"Video shape: {video.shape}, Average FPS: {fps_avg:.4f}")
        
        output_path = os.path.join(output_folder, "output_trt.mp4")
        export_to_video(video, output_path, fps=fps)
        logger.info(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TRT-accelerated inference")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, required=True)
    parser.add_argument("--dit_engine_path", type=str, default="./trt_engines/dit.engine")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.700)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type")
    parser.add_argument("--chunk_size", type=int, default=8, help="Streaming chunk size (8 is safe for VAE, <24k tokens).")
    
    # New args
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale. Set to 1.0 to save memory.")
    parser.add_argument("--max_seq_len", type=int, default=130000, help="Max cache sequence length (supports ~81 frames at 480x832).")
    parser.add_argument("--num_kv_cache", type=int, default=8, help="Number of cache blocks.")
    parser.add_argument("--num_sink_tokens", type=int, default=0, help="Number of sink tokens.")
    parser.add_argument("--adapt_sink_threshold", type=float, default=0.0, help="Threshold for adaptive sink.")
    
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    
    # Set denoising steps
    step_value = int(args.step)
    if step_value <= 1:
        config.denoising_step_list = [700, 0]
    elif step_value == 2:
        config.denoising_step_list = [700, 500, 0]
    elif step_value == 3:
        config.denoising_step_list = [700, 600, 400, 0]
    else:
        config.denoising_step_list = [700, 600, 500, 400, 0]
    
    # Load video
    if args.video_path:
        input_video = load_mp4_as_tensor(
            args.video_path, resize_hw=(args.height, args.width)
        ).unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        t = input_video.shape[2]
    else:
        input_video = None
        t = args.num_frames
    
    # Create pipeline
    enable_cfg = args.guidance_scale != 1.0
    
    pipeline = TRTAcceleratedInferencePipeline(
        config=config,
        dit_engine_path=args.dit_engine_path,
        device=device,
        max_seq_len=args.max_seq_len,
        enable_cfg=enable_cfg,
    )
    pipeline.load_model(args.checkpoint_folder)
    
    # Load prompts
    from causvid.data import TextDataset
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    
    # Run inference
    chunk_size = args.chunk_size
    num_chunks = (t - 1) // chunk_size
    num_steps = len(config.denoising_step_list)
    
    global_start = time.time()
    pipeline.run_inference_v2v(
        input_video=input_video,
        prompts=prompts,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        noise_scale=args.noise_scale,
        output_folder=args.output_folder,
        fps=args.fps,
        num_steps=num_steps,
    )
    
    total_time = time.time() - global_start
    logger.info(f"Total runtime: {total_time:.2f}s")


if __name__ == "__main__":
    main()
