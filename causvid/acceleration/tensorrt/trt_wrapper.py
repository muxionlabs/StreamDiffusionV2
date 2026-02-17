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
        text_embedding: nn.Module = None,
        crossattn_modules: list = None,
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
        
        # Cross-attention KV pre-computation modules.
        # The TRT engine doesn't have 'context' as an input — cross-attn KV
        # must be pre-computed from the text prompt using these modules.
        # They are extracted from the original PyTorch model at pipeline init.
        self.text_embedding = text_embedding  # MLP: text_dim → dim
        self.crossattn_modules = crossattn_modules  # per-block K,V projections
        
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
        num_input_frames = noisy_image_or_video.shape[1]
        
        # === Multi-frame sequential processing ===
        # The TRT engine was traced with num_frames=1, so internal reshapes
        # have num_frames baked as 1. When prepare() sends 2+ frames, we
        # process each frame sequentially — mathematically equivalent for a
        # causal model since each frame only attends to previous frames via
        # the KV cache.
        if num_input_frames > 1:
            frame_seq_len = self.engine.metadata.get('frame_seq_len', 1560)
            all_pred_x0 = []
            
            for f in range(num_input_frames):
                single_frame = noisy_image_or_video[:, f:f+1]  # [B, 1, C, H, W]
                single_ts = timestep[:, f:f+1]                  # [B, 1]
                frame_start = current_start + f * frame_seq_len
                frame_end = frame_start + frame_seq_len
                
                # Recursive call processes 1 frame, updates KV cache in-place
                pred_x0 = self.forward(
                    single_frame, conditional_dict, single_ts,
                    kv_cache, crossattn_cache,
                    frame_start, frame_end,
                )
                all_pred_x0.append(pred_x0)
            
            return torch.cat(all_pred_x0, dim=1)  # [B, F, C, H, W]
        
        # === Single-frame processing (normal path) ===
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
        
        # Pre-compute cross-attention K,V from text context (one-time per prompt).
        # The TRT engine doesn't have 'context' as an input — cross-attn KV
        # must be pre-computed using the text_embedding MLP + per-block K,V
        # projections extracted from the original PyTorch model.
        if crossattn_cache is not None and not crossattn_cache[0]['is_init']:
            self._precompute_crossattn_kv(context, crossattn_cache)
        
        # === Cache eviction ===
        # The original CausalWanModel evicts old entries when the cache would
        # overflow (causal_model.py:170-188). Sink tokens (first sink_size
        # frames) are always preserved; the oldest non-sink entries are
        # discarded to make room for new entries.
        frame_seq_len = self.engine.metadata.get('frame_seq_len', 1560)
        self._maybe_evict_cache(kv_cache, current_end, B, frame_seq_len)
        
        # Convert KV cache from pipeline format to flat tensors
        all_kv_k, all_kv_v, all_global_end, all_local_end = \
            self._kv_from_pipeline_cache(kv_cache, B)
        
        # Convert cross-attn cache to flat tensors
        all_cross_k, all_cross_v = self._crossattn_from_pipeline_cache(
            crossattn_cache, B)
        
        # === Compute correct KV cache write position ===
        # The original CausalWanModel uses:
        #   new_local_end = local_end + current_end - global_end
        #   local_start   = new_local_end - num_new_tokens
        # This allows OVERWRITING cache positions during multi-step denoising
        # (when current_end == global_end, the delta is 0 → same positions).
        # Without this formula, the cache only grows (append-only) and
        # multi-step denoising corrupts the output.
        frame_seq_len = self.engine.metadata.get('frame_seq_len', 1560)
        # current_end: [B] → [B, 1] for broadcasting with [B, num_layers]
        ce = current_end.unsqueeze(1).to(all_local_end.dtype)
        new_local_end = all_local_end + ce - all_global_end  # [B, L]
        correct_local_start = torch.clamp(new_local_end - frame_seq_len, min=0)
        
        # === Run engine per batch item with TRIMMED KV cache ===
        # The original PyTorch model uses:
        #   flash_attn_with_kvcache(q, k_cache[:, :seq_lens.max()], ...)
        # This attends ONLY to valid cache entries. Our TRT engine's internal
        # attention operates over the entire input cache. By trimming the
        # cache tensor to `new_local_end` entries, we make ALL positions
        # valid — the engine's mask becomes all-zeros, matching FlashAttention.
        engine_dtype = torch.float16
        min_cache = self.engine.metadata.get('min_cache_len', 1560)
        
        batch_flow_preds = []
        for b in range(B):
            # Determine valid cache size for this batch item
            max_valid = int(new_local_end[b].max().item())
            max_valid = max(max_valid, min_cache)  # respect engine minimum
            
            # Prepare RoPE inputs (Runtime configurable!)
            # Dynamic RoPE: Ensure buffer covers the current maximum position index.
            # CRITICAL FIX: RoPE tensors are indexed by (F, H, W) grid positions, NOT flattened token indices.
            # Max index needed is max(current_frame_idx + num_input_frames, H, W).
            # H, W are small (~30, ~50). Frame index grows.
            f_start = (current_start[b] // frame_seq_len).item()
            req_rope_len = int(f_start) + num_input_frames + 1 
            
            # TRT Engine Profile has a HARD LIMIT of 4096 on these inputs.
            # We must clamp the request to 4096 to avoid instant crash.
            # If video > 4096 frames, we will crash anyway (engine limitation), but let's be safe.
            req_rope_len = min(req_rope_len, 4096)
            
            rope_inputs = self._get_rope_inputs(engine_dtype, self.device, req_rope_len)

            single_inputs = {
                'x': x[b:b+1].to(engine_dtype),
                'timestep': input_timestep[b:b+1].to(torch.int64),
                # Convert token index to frame index for RoPE offset.
                'current_start': (current_start[b:b+1] // frame_seq_len).to(torch.int64),
                'all_kv_k': all_kv_k[b:b+1, :, :max_valid].to(engine_dtype),
                'all_kv_v': all_kv_v[b:b+1, :, :max_valid].to(engine_dtype),
                'all_kv_seq_lens': new_local_end[b:b+1].to(torch.int64),
                'all_local_start_indices': correct_local_start[b:b+1].to(torch.int64),
                'all_crossattn_k': all_cross_k[b:b+1].to(engine_dtype),
                'all_crossattn_v': all_cross_v[b:b+1].to(engine_dtype),
                **rope_inputs
            }
            
            # === DIAGNOSTIC LOGGING (remove after debugging) ===
            _frame_idx = int((current_start[b] // frame_seq_len).item())
            _token_idx = int(current_start[b].item())
            _local_start = int(correct_local_start[b].max().item())
            _timestep = int(input_timestep[b].item())
            if not hasattr(self, '_diag_call_count'):
                self._diag_call_count = 0
            self._diag_call_count += 1
            if self._diag_call_count <= 30:  # first 30 calls only
                print(f"[TRT_DIAG] call={self._diag_call_count} b={b} "
                      f"token_start={_token_idx} frame_idx={_frame_idx} "
                      f"cache_size={max_valid} local_start={_local_start} "
                      f"timestep={_timestep}")
            # === END DIAGNOSTIC ===
            
            result = self.engine.infer(single_inputs)
            _flow = result['output']
            batch_flow_preds.append(_flow)
            
            # === OUTPUT TRACKING (remove after debugging) ===
            if self._diag_call_count <= 40:
                _out_norm = _flow.float().norm().item()
                _out_std = _flow.float().std().item()
                _out_min = _flow.min().item()
                _out_max = _flow.max().item()
                print(f"[TRT_OUT] call={self._diag_call_count} b={b} frame={_frame_idx} "
                      f"norm={_out_norm:.1f} std={_out_std:.4f} "
                      f"range=[{_out_min:.3f}, {_out_max:.3f}]")
            # === END OUTPUT TRACKING ===
            
            # Write trimmed KV output back to full-size pipeline cache
            out_k = result['out_kv_k']  # [1, L, max_valid, N, D]
            out_v = result['out_kv_v']
            if kv_cache is not None:
                self._update_pipeline_cache(kv_cache, out_k, out_v, new_local_end[b], B, b)
        
        # Combine flow predictions from all batch items
        flow_pred_combined = torch.cat(batch_flow_preds, dim=0)
        
        # Extract flow prediction and permute back: [B, C, F, H, W] -> [B, F, C, H, W]
        flow_pred = flow_pred_combined.permute(0, 2, 1, 3, 4).to(
            noisy_image_or_video.dtype)
        
        # Convert flow prediction to x0 (stays in PyTorch)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])
        
        return pred_x0

    def _get_rope_inputs(self, dtype, device, min_seq_len):
        """
        Generate RoPE frequency tensors. 
        Refactored to be generated at runtime and resized dynamically.
        """
        # Cache key based on dtype/device
        cache_key = (dtype, device)
        current_cache = getattr(self, '_rope_cache', None)
        
        # Check if cache exists and is large enough
        if current_cache and current_cache.get('key') == cache_key:
             if current_cache['max_len'] >= min_seq_len:
                 return current_cache['tensors']
                 
        # Import RoPE logic (reuse logic from trt_model)
        from causvid.acceleration.tensorrt.trt_model import rope_params
        
        head_dim = self.head_dim
        # Dynamic allocation with buffer to avoid frequent re-generation
        # Round up to next multiple of 4096 or add buffer
        alloc_len = max(min_seq_len + 1024, 1024) # Smaller buffer is fine
        
        # HARD LIMIT: TRT Engine Profile max is 4096.
        # We cannot generate or pass a tensor > 4096.
        alloc_len = min(alloc_len, 4096)
        
        max_seq_len = alloc_len
        
        print(f"[TRT_WRAPPER] Generating RoPE cache for length {max_seq_len} (Req: {min_seq_len})")
        
        half_dim = head_dim // 2
        c_t = half_dim - 2 * (half_dim // 3)
        c_h = half_dim // 3
        c_w = half_dim // 3
        
        # 4. TRUE BASELINE (Original Wan Implementation)
        #    - Generate frequencies for the full head_dim.
        #    - Split into [c_t, c_h, c_w].
        #    - T=High (Start), H=Med (Middle), W=Low (End).
        
        # 6. STUCK DOG REPLICATION (High Freqs Everywhere)
        #    - This configuration WAS stable in the static engine.
        #    - If this works here, Runtime Injection is OK, and Med/Low Freqs are the problem.
        #    - If this FAILS, Runtime Injection is BROKEN.
        
        # 7. SYNTHETIC SCALING STRATEGY
        #    - High Freqs (Time) are STABLE.
        #    - Med/Low Freqs (Space) are UNSTABLE (Green Grass).
        #    - We need lower frequencies for Space to avoid "Disfigured Dog".
        #    - Strategy: Generate Synthetic Med/Low freq values by scaling the Safe Time band.
        #    - If specific values cause crash, we can tune the scale factor.
        
        full_freqs = rope_params(max_seq_len, self.head_dim).to(device) # [Seq, 64] complex
        
        # Base: High Frequencies (Indices 0-22)
        # freqs_base = full_freqs[:, :c_t] 
        
        # 4. TRUE BASELINE (Restore)
        freqs_t = full_freqs[:, :c_t]
        freqs_h = full_freqs[:, c_t:c_t+c_h]
        freqs_w = full_freqs[:, c_t+c_h:]
        
        print(f"[DEBUG_ROPE] TRUE BASELINE ACTIVE")
        print(f"[DEBUG_ROPE] freqs_t shape: {freqs_t.shape}")
        print(f"[DEBUG_ROPE] freqs_h shape: {freqs_h.shape}")
        print(f"[DEBUG_ROPE] freqs_w shape: {freqs_w.shape}")

        '''
        # 7. SYNTHETIC SCALING STRATEGY (DISABLED)
        # ...
        def get_freqs_manually(seq_len, dim, theta=10000.0, scale=1.0):
             freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / self.head_dim))
             freqs = freqs / scale # Apply Manual Scaling
             t = torch.arange(seq_len, device=device).float()
             freqs = torch.outer(t, freqs) # [Seq, Dim/2]
             freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
             return freqs_complex

        # Generate Safe Bands manually
        freqs_t_complex = get_freqs_manually(max_seq_len, c_t * 2, scale=1.0)
        freqs_h_complex = get_freqs_manually(max_seq_len, c_h * 2, scale=10.0) # Approx Med
        freqs_w_complex = get_freqs_manually(max_seq_len, c_w * 2, scale=50.0) # Approx Low
        
        freqs_t = freqs_t_complex
        freqs_h = freqs_h_complex
        freqs_w = freqs_w_complex
        '''

        rope_inputs = {
            'rope_cos_t': freqs_t.real.to(dtype=torch.float32).contiguous(), # Keep float32 for safety
            'rope_sin_t': freqs_t.imag.to(dtype=torch.float32).contiguous(),
            'rope_cos_h': freqs_h.real.to(dtype=torch.float32).contiguous(),
            'rope_sin_h': freqs_h.imag.to(dtype=torch.float32).contiguous(),
            'rope_cos_w': freqs_w.real.to(dtype=torch.float32).contiguous(),
            'rope_sin_w': freqs_w.imag.to(dtype=torch.float32).contiguous(),
        }

        self._rope_cache = {'key': cache_key, 'max_len': max_seq_len, 'tensors': rope_inputs}
        return rope_inputs

    def _update_pipeline_cache(self, kv_cache, out_k, out_v, local_end_indices, B, b_idx):
        """
        Update the pipeline's KV cache with results from the TRT engine.
        
        Args:
            kv_cache: List[dict] - The pipeline's cache structure
            out_k: [1, L, max_valid, N, D] - Updated Keys from engine
            out_v: [1, L, max_valid, N, D] - Updated Values from engine
            local_end_indices: [L] - Valid length for each layer
            B: int - Total batch size (unused but kept for signature consistency)
            b_idx: int - Index of current batch item being processed
        """
        for i, cache_entry in enumerate(kv_cache):
            # Engine output for this layer: [1, max_valid, N, D] -> [max_valid, N, D]
            layer_out_k = out_k[0, i]
            layer_out_v = out_v[0, i]
            
            # Valid length for this layer
            valid_len = int(local_end_indices[i].item())
            
            # Update cache: copy valid portion to the pipeline cache
            # cache_entry['k'] is [B, max_cache_len, N, D]
            # storage is pre-allocated.
            
            # Safety check
            if valid_len > layer_out_k.shape[0]:
                 # This should not happen if engine ran correctly, 
                 # but if valid_len > max_valid, we clamp
                 valid_len = layer_out_k.shape[0]

            cache_entry['k'][b_idx, :valid_len] = layer_out_k[:valid_len].to(
                cache_entry['k'].dtype)
            cache_entry['v'][b_idx, :valid_len] = layer_out_v[:valid_len].to(
                cache_entry['v'].dtype)
                
            # global_end_index is updated by pipeline logic, but we can sync if needed.
            # Here we just ensure the data is correct.
    
    # =========================================================================
    # KV Cache Conversion: Pipeline dict format <-> TRT flat tensor format
    # =========================================================================
    
    def _maybe_evict_cache(self, kv_cache, current_end, batch_size,
                           frame_seq_len):
        """
        Evict old KV cache entries when the cache would overflow.
        
        Matches the original CausalWanModel eviction (causal_model.py:170-188):
        - Sink tokens (first sink_size frames) are always preserved
        - Oldest non-sink entries are discarded to make room
        - local_end_index is adjusted; global_end_index tracks logical position
        
        This operates IN-PLACE on the pipeline-format cache dicts.
        """
        if kv_cache is None:
            return
        
        max_cache_len = kv_cache[0]['k'].shape[1]
        sink_size = 3  # match original CausalWanModel.sink_size
        sink_tokens = sink_size * frame_seq_len
        
        for layer_entry in kv_cache:
            local_end = layer_entry['local_end_index']   # [B]
            global_end = layer_entry['global_end_index']  # [B]
            
            for b in range(batch_size):
                le = local_end[b].item()
                ge = global_end[b].item()
                ce = current_end[b].item() if current_end.dim() > 0 else current_end.item()
                
                # new_local_end if we wrote without eviction
                new_le = le + ce - ge
                
                if new_le > max_cache_len:
                    # Eviction needed: shift cache left by num_evicted,
                    # keeping sink_tokens at the start
                    num_evicted = new_le - max_cache_len + frame_seq_len
                    num_rolled = le - num_evicted - sink_tokens
                    
                    if num_rolled > 0:
                        layer_entry['k'][b, sink_tokens:sink_tokens + num_rolled] = \
                            layer_entry['k'][b, sink_tokens + num_evicted:
                                             sink_tokens + num_evicted + num_rolled].clone()
                        layer_entry['v'][b, sink_tokens:sink_tokens + num_rolled] = \
                            layer_entry['v'][b, sink_tokens + num_evicted:
                                             sink_tokens + num_evicted + num_rolled].clone()
                    
                    # Zero out evicted region
                    new_local_end = le - num_evicted
                    layer_entry['k'][b, new_local_end:] = 0
                    layer_entry['v'][b, new_local_end:] = 0
                    
                    # Update indices
                    layer_entry['local_end_index'][b] = new_local_end
                    # global_end reflects the logical position before eviction
                    # (kept at current value — the formula in forward() uses the
                    # delta current_end - global_end, which stays correct)
    
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
    
    def _kv_to_pipeline_cache(self, kv_cache, out_kv_k, out_kv_v,
                              new_local_end, new_global_end):
        """
        Write flat TRT KV tensors back to pipeline's list-of-dicts format.
        
        Uses correctly computed local_end and global_end indices (matching
        the original CausalWanModel's formula) rather than the engine's
        raw output seq_lens.
        """
        if kv_cache is None:
            return
        
        for i, cache_entry in enumerate(kv_cache):
            cache_len = cache_entry['k'].shape[1]
            cache_entry['k'] = out_kv_k[:, i, :cache_len].to(cache_entry['k'].dtype)
            cache_entry['v'] = out_kv_v[:, i, :cache_len].to(cache_entry['v'].dtype)
            cache_entry['local_end_index'] = new_local_end[:, i]
            cache_entry['global_end_index'] = new_global_end[:, i]
    
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
    
    def _precompute_crossattn_kv(self, context, crossattn_cache):
        """
        Pre-compute cross-attention K,V from text context.
        
        This runs ONCE per prompt (when crossattn_cache is not initialized).
        Uses the text_embedding MLP + per-block K,V projections extracted
        from the original PyTorch model at pipeline init.
        
        Args:
            context: [B, text_len, C_text] — raw text embeddings
            crossattn_cache: list of dicts per block
        """
        if self.text_embedding is None or self.crossattn_modules is None:
            logger.warning(
                "No text_embedding/crossattn_modules provided — "
                "cross-attention KV will be zeros (incorrect for inference)")
            for entry in crossattn_cache:
                entry['is_init'] = True
            return
        
        with torch.no_grad():
            # Apply text embedding MLP: [B, text_len, C_text] → [B, text_len, dim]
            ctx = self.text_embedding(context.to(next(self.text_embedding.parameters()).dtype))
            
            B = ctx.shape[0]
            N, D = self.num_heads, self.head_dim
            
            for i, (mods, entry) in enumerate(zip(self.crossattn_modules, crossattn_cache)):
                # Ensure ctx matches module weight dtype (modules may not have
                # been reached by pipeline.to(bfloat16) since they're plain dicts)
                mod_dtype = next(mods['k'].parameters()).dtype
                ctx_i = ctx.to(mod_dtype)
                
                # K projection + QK norm
                k = mods['norm_k'](mods['k'](ctx_i)).view(B, -1, N, D)
                v = mods['v'](ctx_i).view(B, -1, N, D)
                
                entry['k'] = k.to(torch.float16)
                entry['v'] = v.to(torch.float16)
                entry['is_init'] = True
        
        logger.info(f"Pre-computed cross-attention KV for {len(crossattn_cache)} blocks")
    
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
