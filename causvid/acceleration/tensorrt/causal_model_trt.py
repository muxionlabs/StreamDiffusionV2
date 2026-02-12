# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT-exportable variant of CausalWanModel.

Replaces FlexAttention with standard SDPA for ONNX compatibility.
KV cache is handled as explicit tensor inputs/outputs.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

from causvid.models.wan.wan_base.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    Head,
    MLPProj,
    sinusoidal_embedding_1d,
    # Note: NOT importing rope_params - using our own real-number version
)
from .attention import TRTCausalSelfAttention, TRTCrossAttention, trt_rope_apply


def trt_sinusoidal_embedding_1d(dim, position, dtype=torch.float32):
    """
    TRT-compatible sinusoidal embedding that respects dtype.
    Original uses float64 which breaks FP16/Half export.
    """
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    # Use float32 for high precision calculation then cast
    position = position.float()

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, device=position.device).float().div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(dtype)


def trt_rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Generate RoPE frequency parameters (real-number version for ONNX).
    
    Unlike the original rope_params which returns complex tensors via torch.polar,
    this version returns real-number frequencies that work with ONNX export.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Dimension for frequencies (head_dim // 2)
        theta: Base frequency
    
    Returns:
        Tensor of shape [max_seq_len, dim] containing frequency angles (not complex)
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len).float(),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim))
    )
    return freqs  # Real-valued angles, NOT complex via torch.polar


class TRTCrossAttentionT2V(nn.Module):
    """TensorRT-compatible T2V cross-attention."""
    
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        
        if cache_k is not None and cache_v is not None:
            k, v = cache_k, cache_v
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        
        # SDPA expects [B, heads, L, dim]
        q = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k_t, v_t, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o(out)
        
        return out, k, v


class TRTWanAttentionBlock(nn.Module):
    """TensorRT-compatible attention block."""
    
    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        
        # Layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = TRTCausalSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = TRTCrossAttentionT2V(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        
        # Modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        kv_cache_k: Optional[torch.Tensor] = None,
        kv_cache_v: Optional[torch.Tensor] = None,
        cross_cache_k: Optional[torch.Tensor] = None,
        cross_cache_v: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        current_start: Optional[torch.Tensor] = None,
        current_end: Optional[torch.Tensor] = None,
        start_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with explicit KV cache I/O.
        
        Returns:
            x, new_kv_k, new_kv_v, new_cross_k, new_cross_v
        """
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        
        # Modulation
        e_mod = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        
        # Self-attention
        x_normed = self.norm1(x).unflatten(1, (num_frames, frame_seqlen))
        x_modulated = (x_normed * (1 + e_mod[1]) + e_mod[0]).flatten(1, 2)
        
        y, new_kv_k, new_kv_v = self.self_attn(
            x_modulated, seq_lens, grid_sizes, freqs, causal_mask,
            kv_cache_k, kv_cache_v, cache_seqlens, current_start, current_end,
            start_frame=start_frame
        )
        
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e_mod[2]).flatten(1, 2)
        
        # Cross-attention
        cross_out, new_cross_k, new_cross_v = self.cross_attn(
            self.norm3(x), context, context_lens, cross_cache_k, cross_cache_v
        )
        x = x + cross_out
        
        # FFN
        x_normed2 = self.norm2(x).unflatten(1, (num_frames, frame_seqlen))
        y = self.ffn((x_normed2 * (1 + e_mod[4]) + e_mod[3]).flatten(1, 2))
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e_mod[5]).flatten(1, 2)
        
        return x, new_kv_k, new_kv_v, new_cross_k, new_cross_v


class CausalWanModelTRTExport(nn.Module):
    """
    TensorRT-exportable variant of CausalWanModel.
    
    Key differences from original:
    1. FlexAttention replaced with SDPA
    2. KV cache as explicit tensor inputs/outputs (not dicts)
    3. Causal mask as tensor (not BlockMask)
    4. No dynamic block_mask creation
    """
    
    def __init__(
        self,
        model_type: str = 't2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 1536,
        ffn_dim: int = 8960,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,
        num_layers: int = 30,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        max_seq_len: int = 24000,
    ):
        super().__init__()
        
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )
        
        # Attention blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            TRTWanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps
            )
            for _ in range(num_layers)
        ])
        
        # Head (simplified for export)
        self.head_norm = WanLayerNorm(dim, eps)
        self.head_linear = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.head_modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        
        # RoPE frequencies (real-valued for ONNX compatibility)
        d = dim // num_heads
        self.register_buffer('freqs', torch.cat([
            trt_rope_params(max_seq_len, d - 4 * (d // 6)),
            trt_rope_params(max_seq_len, 2 * (d // 6)),
            trt_rope_params(max_seq_len, 2 * (d // 6))
        ], dim=1))
        
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
    
    @classmethod
    def from_pretrained_model(cls, original_model, max_seq_len: int = 24000) -> 'CausalWanModelTRTExport':
        """Create TRT-exportable model from pretrained CausalWanModel."""
        # Access attributes directly from model (not from config dict)
        # Handle patch_size which might not be in config but on model instance
        patch_size = getattr(original_model, 'patch_size', (1, 2, 2))
        
        trt_model = cls(
            model_type=original_model.model_type,
            patch_size=patch_size,
            text_len=original_model.text_len,
            in_dim=original_model.in_dim,
            dim=original_model.dim,
            ffn_dim=original_model.ffn_dim,
            freq_dim=original_model.freq_dim,
            text_dim=original_model.text_dim,
            out_dim=original_model.out_dim,
            num_heads=original_model.num_heads,
            num_layers=original_model.num_layers,
            window_size=original_model.window_size,
            qk_norm=original_model.qk_norm,
            cross_attn_norm=original_model.cross_attn_norm,

            eps=original_model.eps,
            max_seq_len=max_seq_len,
        )
        
        # Copy weights from original model
        trt_model.patch_embedding.load_state_dict(original_model.patch_embedding.state_dict())
        trt_model.text_embedding.load_state_dict(original_model.text_embedding.state_dict())
        trt_model.time_embedding.load_state_dict(original_model.time_embedding.state_dict())
        trt_model.time_projection.load_state_dict(original_model.time_projection.state_dict())
        
        # Copy attention block weights
        for i, (trt_block, orig_block) in enumerate(zip(trt_model.blocks, original_model.blocks)):
            # Copy normalization layers
            trt_block.norm1.load_state_dict(orig_block.norm1.state_dict())
            trt_block.norm2.load_state_dict(orig_block.norm2.state_dict())
            if hasattr(orig_block.norm3, 'weight'):
                trt_block.norm3.load_state_dict(orig_block.norm3.state_dict())
            
            # Copy self-attention weights
            trt_block.self_attn.q.load_state_dict(orig_block.self_attn.q.state_dict())
            trt_block.self_attn.k.load_state_dict(orig_block.self_attn.k.state_dict())
            trt_block.self_attn.v.load_state_dict(orig_block.self_attn.v.state_dict())
            trt_block.self_attn.o.load_state_dict(orig_block.self_attn.o.state_dict())
            trt_block.self_attn.norm_q.load_state_dict(orig_block.self_attn.norm_q.state_dict())
            trt_block.self_attn.norm_k.load_state_dict(orig_block.self_attn.norm_k.state_dict())
            
            # Copy cross-attention weights
            trt_block.cross_attn.q.load_state_dict(orig_block.cross_attn.q.state_dict())
            trt_block.cross_attn.k.load_state_dict(orig_block.cross_attn.k.state_dict())
            trt_block.cross_attn.v.load_state_dict(orig_block.cross_attn.v.state_dict())
            trt_block.cross_attn.o.load_state_dict(orig_block.cross_attn.o.state_dict())
            trt_block.cross_attn.norm_q.load_state_dict(orig_block.cross_attn.norm_q.state_dict())
            trt_block.cross_attn.norm_k.load_state_dict(orig_block.cross_attn.norm_k.state_dict())
            
            # Copy FFN
            trt_block.ffn.load_state_dict(orig_block.ffn.state_dict())
            
            # Copy modulation
            trt_block.modulation.data.copy_(orig_block.modulation.data)
        
        # Copy head weights
        trt_model.head_norm.load_state_dict(original_model.head.norm.state_dict())
        trt_model.head_linear.load_state_dict(original_model.head.head.state_dict())
        trt_model.head_modulation.data.copy_(original_model.head.modulation.data)
        
        # Note: NOT copying original freqs because:
        # - Original uses complex tensors (via torch.polar) 
        # - Our trt_rope_params generates real-valued freqs for ONNX compatibility
        # The freqs are already initialized correctly in __init__
        
        if hasattr(original_model, 'img_emb'):
            trt_model.img_emb.load_state_dict(original_model.img_emb.state_dict())
        
        return trt_model
    
    def forward_export(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified forward for ONNX export (no KV cache).
        
        This version runs a single forward pass without caching,
        suitable for exporting the core model to ONNX/TensorRT.
        The KV cache logic will be handled in the C++ runtime wrapper.
        
        Args:
            x: [B, F, C, H, W] noisy latents
            timestep: [B, F] timesteps
            context: [B, text_len, text_dim] text embeddings (raw, not embedded)
            grid_sizes: [B, 3] containing (F, H, W)
        
        Returns:
            output: [B, C, F', H', W'] denoised prediction
        """
        device = x.device
        dtype = x.dtype
        b = x.shape[0]
        
        # Embed context (text) - ensure dtype matches
        context_emb = self.text_embedding(context.to(dtype))  # [B, text_len, dim]
        
        # Calculate grid sizes dynamically from input shape [B, F, C, H, W]
        # x is [B, F, C, H, W]
        b_in, f_in, c_in, h_in, w_in = x.shape
        
        # Grid sizes for RoPE [F, H/ph, W/pw] - Must be Token Grid Size
        # patch_size is usually (1, 2, 2)
        pt, ph, pw = self.patch_size
        grid_sizes = torch.tensor([[f_in // pt, h_in // ph, w_in // pw]], device=device, dtype=torch.long).expand(b, -1)
        
        # Embed patches: [B, F, C, H, W] -> [B, dim, F', H', W'] -> [B, L, dim]
        x = self.patch_embedding(x.permute(0, 2, 1, 3, 4))  # [B, C, F, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, L, dim]
        
        # seq_lens = torch.tensor([x.shape[1]] * b, device=device, dtype=torch.long)
        # Use torch.full to support symbolic tracing of shape
        seq_lens = torch.full((b,), x.shape[1], device=device, dtype=torch.long)
        
        # Hardcode removed
        # grid_sizes = torch.tensor([[1, 30, 52]], device=device, dtype=torch.long).expand(b, -1)
        
        # Time embeddings - cast sinusoidal output to model dtype
        from causvid.models.wan.wan_base.modules.model import sinusoidal_embedding_1d
        sin_emb = sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(dtype)
        t_emb = self.time_embedding(sin_emb)
        e = self.time_projection(t_emb).unflatten(1, (6, self.dim))
        e = e.unflatten(0, timestep.shape)  # [B, F, 6, dim]
        
        # Process through blocks (no caching for export)
        for block in self.blocks:
            x, _, _, _, _ = block(
                x, e, seq_lens, grid_sizes, self.freqs, context_emb, None,
                None, None, None, None, None, None, None, None
            )
        
        # Head
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e_head = t_emb.unflatten(0, timestep.shape).unsqueeze(2)  # [B, F, 1, dim]
        e_mod = (self.head_modulation.unsqueeze(1) + e_head).chunk(2, dim=2)
        
        x = self.head_norm(x).unflatten(1, (num_frames, frame_seqlen))
        x = self.head_linear(x * (1 + e_mod[1]) + e_mod[0])
        
        # Unpatchify
        x = self._unpatchify(x, grid_sizes)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
        current_start: torch.Tensor,
        current_end: torch.Tensor,
        kv_caches: List[torch.Tensor],
        cross_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with explicit cache tensors.
        
        Args:
            x: [B, F, C, H, W] noisy latents
            timestep: [B, F] timesteps
            context: [B, text_len, dim] text embeddings (already embedded)
            grid_sizes: [B, 3] containing (F, H, W)
            current_start: [B] cache start positions
            current_end: [B] cache end positions
            kv_caches: List of [k, v] tensors for each layer
            cross_caches: List of [k, v] tensors for each layer
        
        Returns:
            output, new_kv_caches, new_cross_caches
        """
        device = x.device
        b = x.shape[0]
        
        # Embed patches: [B, F, C, H, W] -> [B, dim, F', H', W'] -> [B, L, dim]
        x = self.patch_embedding(x.permute(0, 2, 1, 3, 4))  # [B, C, F, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, L, dim]
        
        seq_lens = torch.tensor([x.shape[1]] * b, device=device, dtype=torch.long)
        
        # Time embeddings
        t_emb = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()))
        e = self.time_projection(t_emb).unflatten(1, (6, self.dim))
        e = e.unflatten(0, timestep.shape)  # [B, F, 6, dim]
        
        # Create causal mask (precomputed for TRT)
        # In practice, this would be a static input for the engine
        causal_mask = None  # Using is_causal=True in SDPA instead
        
        new_kv_caches = []
        new_cross_caches = []
        
        # Process through blocks
        for i, block in enumerate(self.blocks):
            kv_k = kv_caches[i * 2] if kv_caches else None
            kv_v = kv_caches[i * 2 + 1] if kv_caches else None
            cross_k = cross_caches[i * 2] if cross_caches else None
            cross_v = cross_caches[i * 2 + 1] if cross_caches else None
            
            x, new_kv_k, new_kv_v, new_cross_k, new_cross_v = block(
                x, e, seq_lens, grid_sizes, self.freqs, context, None,
                causal_mask, kv_k, kv_v, cross_k, cross_v, None,
                current_start, current_end
            )
            
            new_kv_caches.extend([new_kv_k, new_kv_v])
            new_cross_caches.extend([new_cross_k, new_cross_v])
        
        # Head
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e_head = t_emb.unflatten(0, timestep.shape).unsqueeze(2)  # [B, F, 1, dim]
        e_mod = (self.head_modulation.unsqueeze(1) + e_head).chunk(2, dim=2)
        
        x = self.head_norm(x).unflatten(1, (num_frames, frame_seqlen))
        x = self.head_linear(x * (1 + e_mod[1]) + e_mod[0])
        
        # Unpatchify
        x = self._unpatchify(x, grid_sizes)
        
        return x, new_kv_caches, new_cross_caches
    
    def _unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Reconstruct video from patches. Vectorized for TRT safety."""
        b = x.shape[0]
        c = self.out_dim
        pt, ph, pw = self.patch_size
        
        # Assume uniform grid sizes across batch (valid for standard tensor batches)
        # Use first element to determine shape
        f, h, w = torch.unbind(grid_sizes[0], dim=0)
        
        # Ensure Int64 for shape calculations
        f = f.long()
        h = h.long()
        w = w.long()
        
        # Reshape to [B, F, H, W, Pt, Ph, Pw, C]
        # x is [B, L, C] where L = F*H*W
        x = x.reshape(b, f, h, w, pt, ph, pw, c)
        
        # Permute to [B, C, F, Pt, H, Ph, W, Pw]
        # Input Dims: 0:B, 1:F, 2:H, 3:W, 4:Pt, 5:Ph, 6:Pw, 7:C
        # Target: 0, 7, 1, 4, 2, 5, 3, 6
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        
        # Reshape to final video: [B, C, F_out, H_out, W_out]
        f_out = f * pt
        h_out = h * ph
        w_out = w * pw
        x = x.reshape(b, c, f_out, h_out, w_out)
        
        return x

    def forward_export_streaming(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        kv_cache_0: torch.Tensor,
        kv_cache_1: torch.Tensor,
        kv_cache_2: torch.Tensor,
        kv_cache_3: torch.Tensor,
        kv_cache_4: torch.Tensor,
        kv_cache_5: torch.Tensor,
        kv_cache_6: torch.Tensor,
        kv_cache_7: torch.Tensor,
        kv_cache_8: torch.Tensor,
        kv_cache_9: torch.Tensor,
        kv_cache_10: torch.Tensor,
        kv_cache_11: torch.Tensor,
        kv_cache_12: torch.Tensor,
        kv_cache_13: torch.Tensor,
        kv_cache_14: torch.Tensor,
        kv_cache_15: torch.Tensor,
        kv_cache_16: torch.Tensor,
        kv_cache_17: torch.Tensor,
        kv_cache_18: torch.Tensor,
        kv_cache_19: torch.Tensor,
        kv_cache_20: torch.Tensor,
        kv_cache_21: torch.Tensor,
        kv_cache_22: torch.Tensor,
        kv_cache_23: torch.Tensor,
        kv_cache_24: torch.Tensor,
        kv_cache_25: torch.Tensor,
        kv_cache_26: torch.Tensor,
        kv_cache_27: torch.Tensor,
        kv_cache_28: Optional[torch.Tensor] = None,
        kv_cache_29: Optional[torch.Tensor] = None,
        current_start: Optional[torch.Tensor] = None,
        start_frame_idx: Optional[torch.Tensor] = None, # NEW input for RoPE
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Streaming forward pass (DiT only) - Optimized Interface with SPLIT KV Cache.
        
        Args:
            x: [B, L, C] (L=1560 usually)
            timestep: [B]
            context: [B, text_len, text_dim]
            kv_cache_0..29: [1, 2, B, MaxSeq, H, D] - 30 chunks of 1 layer each
            current_start: [B] Start token index for cache update (Physical memory index)
            start_frame_idx: [B] Start frame index for RoPE (Logical time index)
        """
        device = x.device
        b = x.shape[0]
        
        # Embed patches
        # Input x is [B, C, F, H, W]
        # Calculate grid sizes dynamically from input shape
        # Calculate grid sizes dynamically from input shape [B, F, C, H, W]
        # x is [B, F, C, H, W]
        b_in, f_in, c_in, h_in, w_in = x.shape
        
        # Grid sizes for RoPE [F, H/ph, W/pw] - Must be Token Grid Size
        # patch_size is usually (1, 2, 2)
        pt, ph, pw = self.patch_size
        grid_sizes = torch.tensor([[f_in // pt, h_in // ph, w_in // pw]], device=device, dtype=torch.long).expand(b, -1)
        
        x = self.patch_embedding(x.permute(0, 2, 1, 3, 4))
        x = x.flatten(2).transpose(1, 2)
        
        # Embed context (text) - valid for both export and inference
        context = self.text_embedding(context)
        
        # seq_lens = torch.tensor([x.shape[1]] * b, device=device, dtype=torch.long)
        seq_lens = torch.full((b,), x.shape[1], device=device, dtype=torch.long)
        
        # Hardcode removed
        # grid_sizes = torch.tensor([[1, 30, 52]], device=device, dtype=torch.long).expand(b, -1)
        
        # Time embeddings - use TRT compatible version that respects dtype
        t_emb = self.time_embedding(trt_sinusoidal_embedding_1d(self.freq_dim, timestep.flatten(), dtype=x.dtype))
        e = self.time_projection(t_emb).reshape(b, 6, self.dim)
        
        # Group caches for iteration
        cache_chunks = [
            kv_cache_0, kv_cache_1, kv_cache_2, kv_cache_3, kv_cache_4,
            kv_cache_5, kv_cache_6, kv_cache_7, kv_cache_8, kv_cache_9,
            kv_cache_10, kv_cache_11, kv_cache_12, kv_cache_13, kv_cache_14,
            kv_cache_15, kv_cache_16, kv_cache_17, kv_cache_18, kv_cache_19,
            kv_cache_20, kv_cache_21, kv_cache_22, kv_cache_23, kv_cache_24,
            kv_cache_25, kv_cache_26, kv_cache_27, kv_cache_28, kv_cache_29,
        ]
        
        # Unpack cache tensor: [Layers, 2, B, MaxSeq, Heads, D]
        # We process layer by layer
        new_caches = []
        
        # Helper: unpack modulation
        num_frames = e.shape[1]
        # Ensure Int64 division
        # frame_seqlen = x.shape[1] // num_frames (x is [B, L, C], num_frames is 6?? No. e is [B, 6, dim]).
        # Wait. e shape [B, 6, dim]. num_frames = 6??
        # In `forward`: e is [B, F, 6, dim].
        # In `forward_export_streaming`: timestep is [B]. F=1 usually?
        # If e is [B, 6, dim], e.shape[1] is 6.
        # But `num_frames` usually means F.
        # This looks like a BUG in `forward_export_streaming`?
        # If F=1 (streaming), then we unpack 6 modulation chunks.
        # BUT `num_frames = e.shape[1]` -> 6.
        # `frame_seqlen = x.shape[1] // 6`.
        # If x.shape[1] = 1560. 1560 // 6 = 260.
        # Is this correct?
        # Let's check `forward` (lines 180): e is [B, F, 6, dim].
        # Here e is [B, 6, dim]. It implicitly assumes F=1?
        # If F=1, then e should be [B, 1, 6, dim].
        # `e = self.time_projection(t_emb).reshape(b, 1, 6, self.dim)` might be safer.
        # If so, `num_frames` = e.shape[1] = 1.
        # Let's assume F=1 for streaming.
        
        # Fix: Reshape e to include F=1
        e = e.reshape(b, 1, 6, self.dim)
        num_frames = e.shape[1] # 1
        frame_seqlen = x.shape[1] # 1560
        
        for i, block in enumerate(self.blocks):
            # Determine which chunk and which layer within chunk (1 layer per chunk)
            chunk_idx = i
            layer_idx_in_chunk = 0
            
            # Extract cache for this layer: [2, B, MaxSeq, Heads, D]
            layer_cache = cache_chunks[chunk_idx][layer_idx_in_chunk] 
            k_cache = layer_cache[0]
            v_cache = layer_cache[1]
            
            # Forward block
            x, new_k, new_v, _, _ = block(
                x, e, seq_lens, grid_sizes, self.freqs, context, None, None,
                kv_cache_k=k_cache, kv_cache_v=v_cache,
                cache_seqlens=current_start, # Uses start as current length marker
                current_start=current_start,
                current_end=None, # Calculated internally
                start_frame=start_frame_idx, # Pass explicit frame index for RoPE
            )
            
            # Stack updated K/V for this layer
            new_layer_cache = torch.stack([new_k, new_v])
            new_caches.append(new_layer_cache)
            
        # Head
        x_normed = self.head_norm(x).reshape(b, 1, frame_seqlen, self.dim)
        
        # Head modulation
        # Recompute head modulation since it's hard to pass state for it?
        # Actually head modulation depends on time embedding 'e'.
        # Recalculate e_head logic from forward()
        e_head = t_emb.reshape(b, 1, 1, self.dim) 
        e_mod = (self.head_modulation.unsqueeze(1) + e_head).chunk(2, dim=2)
        
        x = self.head_linear(x_normed * (1 + e_mod[1]) + e_mod[0])
        x = self._unpatchify(x, grid_sizes)
        
        # Re-stack output caches into 10 chunks
        # Re-stack output caches into 30 chunks (1 layer each)
        new_cache_chunks = []
        for c_idx in range(30):
             # Just one layer per chunk now
             new_cache_chunks.append(torch.stack([new_caches[c_idx]]))
        
        return (
            x, 
            new_cache_chunks[0], new_cache_chunks[1], new_cache_chunks[2], new_cache_chunks[3], new_cache_chunks[4],
            new_cache_chunks[5], new_cache_chunks[6], new_cache_chunks[7], new_cache_chunks[8], new_cache_chunks[9],
            new_cache_chunks[10], new_cache_chunks[11], new_cache_chunks[12], new_cache_chunks[13], new_cache_chunks[14],
            new_cache_chunks[15], new_cache_chunks[16], new_cache_chunks[17], new_cache_chunks[18], new_cache_chunks[19],
            new_cache_chunks[20], new_cache_chunks[21], new_cache_chunks[22], new_cache_chunks[23], new_cache_chunks[24],
            new_cache_chunks[25], new_cache_chunks[26], new_cache_chunks[27], new_cache_chunks[28], new_cache_chunks[29]
        )


# =============================================================================
# TensorRT Engine Inference Wrapper
# =============================================================================

class CausalWanModelTRTInference:
    """
    TensorRT-accelerated inference wrapper for CausalWanModel.
    
    Loads the pre-built TensorRT engine and provides a drop-in replacement
    for the PyTorch model's forward method.
    
    Usage:
        # Load engine
        model = CausalWanModelTRTInference("./trt_engines/dit.engine")
        
        # Run inference (same interface as forward_export)
        output = model(x, timestep, context, grid_sizes)
    
    Note:
        This uses the `forward_export` interface (no explicit KV cache).
        For streaming inference with KV cache, use CausalWanModelEngine from
        engines/dit_engine.py instead.
    """
    
    def __init__(
        self,
        engine_path: str,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize TensorRT inference wrapper.
        
        Args:
            engine_path: Path to built TensorRT engine (.engine file)
            use_cuda_graph: Enable CUDA graphs for reduced kernel launch overhead
            device: Device to run inference on
        """
        self.engine_path = engine_path
        self.use_cuda_graph = use_cuda_graph
        self.device = device
        
        # Lazy load engine on first use to avoid import issues
        self._engine = None
        self._stream = None
        self._loaded = False
        self._last_shape_hash = None
    
    def _ensure_loaded(self):
        """Load engine on first inference call."""
        if self._loaded:
            return
        
        from .utilities import Engine
        from polygraphy import cuda
        
        self._engine = Engine(self.engine_path)
        self._engine.load()
        self._engine.activate()
        self._stream = cuda.Stream()
        self._loaded = True
    
    def _shape_hash(self, x: torch.Tensor) -> int:
        """Compute shape hash for buffer reuse checking."""
        return hash((tuple(x.shape), x.dtype))
    
    def __call__(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run TensorRT inference.
        
        Same interface as CausalWanModelTRTExport.forward_export().
        
        Args:
            x: [B, F, C, H, W] noisy latents
            timestep: [B, F] diffusion timesteps
            context: [B, text_len, text_dim] text embeddings
            grid_sizes: [B, 3] containing (F, H', W')
        
        Returns:
            output: [B, C, F', H', W'] denoised prediction
        """
        self._ensure_loaded()
        
        # Check if shapes changed and reallocate buffers if needed
        shape_hash = self._shape_hash(x)
        if shape_hash != self._last_shape_hash:
            shape_dict = {
                "x": tuple(x.shape),
                "timestep": tuple(timestep.shape),
                "context": tuple(context.shape),
                "grid_sizes": tuple(grid_sizes.shape),
            }
            self._engine.allocate_buffers(shape_dict, self.device)
            
            if self.use_cuda_graph:
                self._engine.reset_cuda_graph()
            
            self._last_shape_hash = shape_hash
        
        # Run inference
        outputs = self._engine.infer(
            {
                "x": x.contiguous(),
                "timestep": timestep.contiguous(),
                "context": context.contiguous(),
                "grid_sizes": grid_sizes.contiguous(),
            },
            self._stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        
        # Extract output tensor
        output = outputs.get("output")
        if output is None:
            # Fallback: find first 5D tensor in outputs
            for name, tensor in outputs.items():
                if tensor.dim() == 5:
                    output = tensor
                    break
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for __call__ to match PyTorch module interface."""
        return self(x, timestep, context, grid_sizes)
    
    def reset_cuda_graph(self):
        """Reset CUDA graph for recapture (e.g., after shape change)."""
        if self._engine is not None:
            self._engine.reset_cuda_graph()
        self._last_shape_hash = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if engine is loaded."""
        return self._loaded
    
    def to(self, device):
        """Move to device (no-op, for interface compatibility)."""
        self.device = str(device)
        return self
    
    def eval(self):
        """Set to eval mode (no-op, for interface compatibility)."""
        return self
    
    def half(self):
        """Set to half precision (no-op, TensorRT handles precision)."""
        return self


def load_trt_model(engine_path: str, **kwargs) -> CausalWanModelTRTInference:
    """
    Convenience function to load a TensorRT-accelerated model.
    
    Args:
        engine_path: Path to TensorRT engine file
        **kwargs: Additional arguments for CausalWanModelTRTInference
    
    Returns:
        TensorRT inference wrapper
    
    Example:
        model = load_trt_model("./trt_engines/dit.engine")
        output = model(x, timestep, context, grid_sizes)
    """
    return CausalWanModelTRTInference(engine_path, **kwargs)

