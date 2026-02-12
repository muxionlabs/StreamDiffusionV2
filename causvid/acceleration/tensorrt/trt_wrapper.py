"""
TRT-accelerated WanDiffusionWrapper — drop-in replacement.

This wrapper replaces the PyTorch CausalWanModel forward pass with a
TensorRT engine call, while keeping all other pipeline logic identical
(flow→x0 conversion, scheduler, KV cache management).

It implements the EXACT SAME forward() signature as WanDiffusionWrapper
so it can be used as self.generator in CausalStreamInferencePipeline.
"""

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from causvid.acceleration.tensorrt.engine_wrapper import TRTEngineWrapper

logger = logging.getLogger(__name__)


class TRTWanDiffusionWrapper(nn.Module):
    """
    Drop-in replacement for WanDiffusionWrapper / CausalWanDiffusionWrapper.
    
    Implements the exact same forward() signature that
    CausalStreamInferencePipeline.inference_stream() calls:
    
        self.generator(
            noisy_image_or_video=...,   # [B, F, C, H, W]
            conditional_dict=...,       # dict with 'prompt_embeds' key
            timestep=...,               # [B, F]
            kv_cache=...,               # list of dicts
            crossattn_cache=...,        # list of dicts
            current_start=...,          # [B] int64 tensor
            current_end=...,            # [B] int64 tensor
        ) -> denoised_pred [B, F, C, H, W]
    """
    
    def __init__(
        self,
        engine_path: str,
        config,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.device = device or torch.device("cuda")
        self.config = config
        
        # Load TRT engine
        self.engine = TRTEngineWrapper(engine_path, device=self.device)
        
        # Model dimensions
        meta = self.engine.metadata
        self.num_layers = meta.get('num_layers', 30)
        self.num_heads = meta.get('num_heads', 12)
        self.head_dim = meta.get('head_dim', 128)
        self.dim = meta.get('dim', 1536)
        self.text_len = meta.get('text_len', 512)
        self.patch_size = tuple(meta.get('patch_size', [1, 2, 2]))
        self.out_dim = 16
        self.in_dim = 16
        self.freq_dim = 256
        self.seq_len = meta.get('frame_seq_len', 1560) * 21  # max seq
        
        self.uniform_timestep = False  # CausalWan uses per-frame timesteps
        
        # Scheduler (same as original)
        from causvid.models.wan.flow_match import FlowMatchScheduler
        self.scheduler = FlowMatchScheduler(
            shift=8.0, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)
        
        # Dummy model attribute for compatibility with pipeline code that
        # accesses self.generator.model.blocks[i].self_attn.sink_size etc.
        self.model = _TRTModelProxy(self)
        
        logger.info(
            f"TRT wrapper initialized: {self.num_layers} layers, "
            f"{self.num_heads} heads, dim={self.dim}"
        )
    
    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        Identical to the original WanDiffusionWrapper implementation.
        
        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        x0 = x_t - sigma_t * pred
        """
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )
        
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)
    
    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[torch.Tensor] = None,
        current_end: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass — EXACT same signature as WanDiffusionWrapper.forward().
        
        Args:
            noisy_image_or_video: [B, F, C, H, W] — noisy latent
            conditional_dict: dict with 'prompt_embeds' -> list of [L, C_text] tensors
            timestep: [B, F] — timestep per frame
            kv_cache: list[dict] per transformer layer
            crossattn_cache: list[dict] per transformer layer
            current_start: [B] int64 — KV cache start position
            current_end: [B] int64 — KV cache end position
        
        Returns:
            pred_x0: [B, F, C, H, W] — denoised prediction
        """
        prompt_embeds = conditional_dict["prompt_embeds"]
        
        # timestep handling (same as original)
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep
        
        B = noisy_image_or_video.shape[0]
        
        # Permute: [B, F, C, H, W] -> [B, C, F, H, W] (model input format)
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4).contiguous()
        
        # Prepare context: stack prompt embeds and pad to text_len
        context_list = []
        for emb in prompt_embeds:
            if emb.shape[0] < self.text_len:
                pad = torch.zeros(
                    self.text_len - emb.shape[0], emb.shape[1],
                    device=emb.device, dtype=emb.dtype
                )
                emb = torch.cat([emb, pad], dim=0)
            context_list.append(emb)
        context = torch.stack(context_list)  # [B, text_len, C_text]
        
        # Convert KV cache from pipeline format to flat tensors
        all_kv_k, all_kv_v, all_kv_seq_lens, all_local_start_indices = \
            self._kv_from_pipeline_cache(kv_cache, B)
        
        # Convert cross-attn cache to flat tensors
        all_cross_k, all_cross_v = self._crossattn_from_pipeline_cache(
            crossattn_cache, B)
        
        # Ensure correct dtypes for TRT engine
        engine_dtype = torch.float16
        
        inputs = {
            'x': x.to(engine_dtype),
            'timestep': input_timestep.to(torch.int64),
            'context': context.to(engine_dtype),
            'current_start': current_start.to(torch.int64),
            'current_end': current_end.to(torch.int64),
            'all_kv_k': all_kv_k.to(engine_dtype),
            'all_kv_v': all_kv_v.to(engine_dtype),
            'all_kv_seq_lens': all_kv_seq_lens.to(torch.int64),
            'all_local_start_indices': all_local_start_indices.to(torch.int64),
            'all_crossattn_k': all_cross_k.to(engine_dtype),
            'all_crossattn_v': all_cross_v.to(engine_dtype),
        }
        
        # Run TRT engine
        outputs = self.engine.infer(inputs)
        
        # Extract flow prediction and permute back: [B, C, F, H, W] -> [B, F, C, H, W]
        flow_pred = outputs['output'].permute(0, 2, 1, 3, 4).to(
            noisy_image_or_video.dtype)
        
        # Write updated KV cache back to pipeline format
        self._kv_to_pipeline_cache(
            kv_cache, outputs['out_kv_k'], outputs['out_kv_v'],
            outputs['out_kv_seq_lens'])
        
        # Mark cross-attn cache as initialized
        if crossattn_cache is not None:
            for cache_entry in crossattn_cache:
                if not cache_entry['is_init']:
                    cache_entry['is_init'] = True
        
        # Convert flow prediction to x0 (stays in PyTorch)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])
        
        return pred_x0
    
    # =========================================================================
    # KV Cache Conversion: Pipeline dict format <-> TRT flat tensor format
    # =========================================================================
    
    def _kv_from_pipeline_cache(self, kv_cache, batch_size):
        """
        Convert pipeline's list-of-dicts KV cache to flat tensors.
        
        Pipeline format per layer: {
            'k': [B, cache_len, N, D],
            'v': [B, cache_len, N, D],
            'global_end_index': [B],
            'local_end_index': [B],
        }
        
        TRT format: all_kv_k[B, num_layers, cache_len, N, D]
        """
        if kv_cache is None:
            max_cache = self.engine.metadata.get('max_cache_len', 15600)
            return (
                torch.zeros(batch_size, self.num_layers, max_cache,
                           self.num_heads, self.head_dim,
                           device=self.device, dtype=torch.float16),
                torch.zeros(batch_size, self.num_layers, max_cache,
                           self.num_heads, self.head_dim,
                           device=self.device, dtype=torch.float16),
                torch.zeros(batch_size, self.num_layers,
                           device=self.device, dtype=torch.int64),
                torch.zeros(batch_size, self.num_layers,
                           device=self.device, dtype=torch.int64),
            )
        
        cache_len = kv_cache[0]['k'].shape[1]
        actual_batch = kv_cache[0]['k'].shape[0]
        
        all_k = torch.stack([c['k'] for c in kv_cache], dim=1)  # [B, L, cache_len, N, D]
        all_v = torch.stack([c['v'] for c in kv_cache], dim=1)
        
        # Local end index is the write position
        all_seq_lens = torch.stack(
            [c.get('global_end_index', torch.zeros(actual_batch, dtype=torch.long, device=self.device))
             for c in kv_cache], dim=1)  # [B, L]
        all_local_starts = torch.stack(
            [c.get('local_end_index', torch.zeros(actual_batch, dtype=torch.long, device=self.device))
             for c in kv_cache], dim=1)  # [B, L]
        
        return all_k, all_v, all_seq_lens, all_local_starts
    
    def _kv_to_pipeline_cache(self, kv_cache, out_kv_k, out_kv_v, out_kv_seq_lens):
        """Write flat TRT KV tensors back to pipeline's list-of-dicts format."""
        if kv_cache is None:
            return
        
        for i, cache_entry in enumerate(kv_cache):
            cache_len = cache_entry['k'].shape[1]
            cache_entry['k'] = out_kv_k[:, i, :cache_len].to(cache_entry['k'].dtype)
            cache_entry['v'] = out_kv_v[:, i, :cache_len].to(cache_entry['v'].dtype)
            if out_kv_seq_lens.dim() >= 2:
                cache_entry['global_end_index'] = out_kv_seq_lens[:, i]
                cache_entry['local_end_index'] = out_kv_seq_lens[:, i]
            else:
                cache_entry['global_end_index'] = out_kv_seq_lens
                cache_entry['local_end_index'] = out_kv_seq_lens
    
    def _crossattn_from_pipeline_cache(self, crossattn_cache, batch_size):
        """Convert pipeline cross-attn cache to flat tensors."""
        if crossattn_cache is None:
            return (
                torch.zeros(batch_size, self.num_layers, self.text_len,
                           self.num_heads, self.head_dim,
                           device=self.device, dtype=torch.float16),
                torch.zeros(batch_size, self.num_layers, self.text_len,
                           self.num_heads, self.head_dim,
                           device=self.device, dtype=torch.float16),
            )
        
        actual_batch = crossattn_cache[0]['k'].shape[0]
        all_k = torch.stack([c['k'] for c in crossattn_cache], dim=1)
        all_v = torch.stack([c['v'] for c in crossattn_cache], dim=1)
        
        return all_k, all_v
    
    def to(self, *args, **kwargs):
        """Override to handle device/dtype moves."""
        for arg in args:
            if isinstance(arg, torch.device):
                self.device = arg
            elif isinstance(arg, str) and 'cuda' in arg:
                self.device = torch.device(arg)
        return self


class _TRTModelProxy:
    """
    Proxy object that mimics CausalWanModel structure for pipeline compatibility.
    
    The pipeline accesses self.generator.model.blocks[i].self_attn.sink_size
    during KV cache initialization. This proxy provides those attributes
    without requiring the actual model to be loaded.
    """
    
    def __init__(self, wrapper: TRTWanDiffusionWrapper):
        self._wrapper = wrapper
        self.num_frame_per_block = 1
        
        # Create proxy blocks  
        self.blocks = [
            _BlockProxy() for _ in range(wrapper.num_layers)
        ]
    
    def to(self, *args, **kwargs):
        return self


class _BlockProxy:
    """Proxy for a single attention block."""
    
    def __init__(self):
        self.self_attn = _SelfAttnProxy()


class _SelfAttnProxy:
    """Proxy for self-attention attributes accessed by the pipeline."""
    
    def __init__(self):
        self.sink_size = 0
        self.adapt_sink_thr = 0
