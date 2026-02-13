"""
TRT-safe model definitions for CausalWanDiT.

This module provides TensorRT-compatible replacements for the CausalWanModel
inference path. All Python-level dynamic control flow (loops, .item(), .tolist(),
view_as_complex) is replaced with pure tensor operations that can be traced
through torch.onnx.export.

Key changes from the original:
1. RoPE: Uses sin/cos rotation instead of complex multiplication
2. Attention: Uses SDPA instead of flash_attn_with_kvcache
3. Unpatchify: Uses einops-style tensor ops instead of Python loops
4. All shapes derived from tensor.shape (Int64-safe for TRT 10)
5. No dict inputs — all flattened to tensor arguments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from causvid.models.wan.wan_base.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    sinusoidal_embedding_1d,
    rope_params,
)


# =============================================================================
# TRT-Safe RoPE Implementation
# =============================================================================

def precompute_rope_freqs_real(max_seq_len: int, head_dim: int, theta: float = 10000.0):
    """
    Precompute RoPE frequencies in sin/cos real format (no complex numbers).
    Returns cos and sin tensors of shape [max_seq_len, head_dim // 2].
    
    We split into 3 frequency bands matching the original:
    - temporal: head_dim - 4*(head_dim//6) dims  -> indexed by frame
    - height:   2*(head_dim//6) dims             -> indexed by h
    - width:    2*(head_dim//6) dims             -> indexed by w
    """
    d = head_dim
    c = d // 2  # half head dim for complex pairs
    
    # Match the original rope_params split
    c_t = c - 2 * (c // 3)  # temporal freq dims
    c_h = c // 3             # height freq dims
    c_w = c // 3             # width freq dims
    
    # Compute freqs for each axis
    freqs_t = rope_params(max_seq_len, 2 * c_t)  # [max_seq_len, c_t] complex
    freqs_h = rope_params(max_seq_len, 2 * c_h)  # [max_seq_len, c_h] complex
    freqs_w = rope_params(max_seq_len, 2 * c_w)  # [max_seq_len, c_w] complex
    
    # Convert complex to cos/sin pairs
    # polar form: freqs = cos(theta) + i*sin(theta)
    cos_t = freqs_t.real.float()  # [max_seq_len, c_t]
    sin_t = freqs_t.imag.float()
    cos_h = freqs_h.real.float()  # [max_seq_len, c_h]
    sin_h = freqs_h.imag.float()
    cos_w = freqs_w.real.float()  # [max_seq_len, c_w]
    sin_w = freqs_w.imag.float()
    
    return (cos_t, sin_t, cos_h, sin_h, cos_w, sin_w)


def trt_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    rope_cos_t: torch.Tensor,
    rope_sin_t: torch.Tensor,
    rope_cos_h: torch.Tensor,
    rope_sin_h: torch.Tensor,
    rope_cos_w: torch.Tensor,
    rope_sin_w: torch.Tensor,
    start_frame: int = 0,
):
    """
    TRT-safe RoPE application using real-valued sin/cos rotation.
    
    Key differences from original:
    - No Python loops over batch (batch=1 for TRT during streaming)
    - No view_as_complex / view_as_real
    - No .tolist() / .item()
    - Uses explicit sin/cos rotation: rotate_half pattern
    
    Args:
        x: [B, S, num_heads, head_dim] — query or key tensor
        grid_sizes: [B, 3] — (F, H, W) per batch
        rope_cos/sin_*: precomputed cos/sin for each axis
        start_frame: frame offset for streaming (int, not tensor for tracing)
    
    Returns:
        [B, S, num_heads, head_dim] — rotated tensor
    """
    B, S, N, D = x.shape
    # For TRT export, we assume B=1 (streaming mode processes single batch)
    # and grid_sizes[0] gives (F, H, W)
    F_val = grid_sizes[0, 0]
    H_val = grid_sizes[0, 1]
    W_val = grid_sizes[0, 2]
    seq_len = F_val * H_val * W_val  # actual token count
    
    half_d = D // 2
    c_t = half_d - 2 * (half_d // 3)
    c_h = half_d // 3
    c_w = half_d // 3
    
    # Build 3D position grid: [F, H, W] -> [F*H*W, 1]
    # Temporal indices (offset by start_frame)
    f_idx = torch.arange(F_val, device=x.device) + start_frame  # [F]
    h_idx = torch.arange(H_val, device=x.device)                 # [H]
    w_idx = torch.arange(W_val, device=x.device)                 # [W]
    
    # Gather cos/sin for each position
    # Temporal: each frame position applies to all H*W tokens
    cos_t = rope_cos_t[f_idx]  # [F, c_t]
    sin_t = rope_sin_t[f_idx]  # [F, c_t]
    cos_t = cos_t.unsqueeze(1).unsqueeze(1).expand(-1, H_val, W_val, -1)  # [F,H,W,c_t]
    sin_t = sin_t.unsqueeze(1).unsqueeze(1).expand(-1, H_val, W_val, -1)
    
    # Height: same across frames and widths
    cos_h = rope_cos_h[h_idx]  # [H, c_h]
    sin_h = rope_sin_h[h_idx]
    cos_h = cos_h.unsqueeze(0).unsqueeze(2).expand(F_val, -1, W_val, -1)  # [F,H,W,c_h]
    sin_h = sin_h.unsqueeze(0).unsqueeze(2).expand(F_val, -1, W_val, -1)
    
    # Width: same across frames and heights
    cos_w = rope_cos_w[w_idx]  # [W, c_w]
    sin_w = rope_sin_w[w_idx]
    cos_w = cos_w.unsqueeze(0).unsqueeze(0).expand(F_val, H_val, -1, -1)  # [F,H,W,c_w]
    sin_w = sin_w.unsqueeze(0).unsqueeze(0).expand(F_val, H_val, -1, -1)
    
    # Concatenate all cos/sin: [F*H*W, c_t+c_h+c_w] = [F*H*W, half_d]
    cos_all = torch.cat([cos_t, cos_h, cos_w], dim=-1).reshape(seq_len, 1, half_d)
    sin_all = torch.cat([sin_t, sin_h, sin_w], dim=-1).reshape(seq_len, 1, half_d)
    
    # Apply rotation to the valid portion of x[0] (batch index 0)
    # x shape: [B, S, N, D]
    x_valid = x[0, :seq_len]  # [seq_len, N, D]
    orig_dtype = x_valid.dtype
    
    # Upcast to float32 for precision (original uses float64 complex multiply)
    x_valid = x_valid.float()
    
    # Split into pairs for rotation: [seq_len, N, half_d, 2]
    x_pairs = x_valid.reshape(seq_len, N, half_d, 2)
    x_even = x_pairs[..., 0]  # [seq_len, N, half_d]
    x_odd = x_pairs[..., 1]   # [seq_len, N, half_d]
    
    # Rotary embedding: (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos)
    cos_all = cos_all.float()
    sin_all = sin_all.float()
    
    out_even = x_even * cos_all - x_odd * sin_all
    out_odd = x_even * sin_all + x_odd * cos_all
    
    # Interleave back: [seq_len, N, half_d, 2] -> [seq_len, N, D]
    out = torch.stack([out_even, out_odd], dim=-1).reshape(seq_len, N, D)
    
    # Cast back to original dtype
    out = out.to(orig_dtype)
    
    # Place back into full sequence (padding stays unchanged)
    result = x.clone()
    result[0, :seq_len] = out
    
    return result


# =============================================================================
# TRT-Safe Self-Attention (replaces flash_attn_with_kvcache)
# =============================================================================

class TRTSelfAttention(nn.Module):
    """
    TRT-compatible self-attention with explicit KV cache I/O.
    
    Instead of flash_attn_with_kvcache (which is a CUDA kernel not supported
    by TRT), this uses:
    1. Standard Q/K/V projection
    2. RoPE application
    3. KV cache write (scatter into cache tensor)
    4. torch.nn.functional.scaled_dot_product_attention (SDPA)
    
    The KV cache eviction/sink logic stays OUTSIDE this module (in pipeline).
    This module only does: write to cache → attend over cache → return output.
    """
    
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        
        # Linear projections (same as original)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        rope_cos_t: torch.Tensor,
        rope_sin_t: torch.Tensor,
        rope_cos_h: torch.Tensor,
        rope_sin_h: torch.Tensor,
        rope_cos_w: torch.Tensor,
        rope_sin_w: torch.Tensor,
        kv_k: torch.Tensor,
        kv_v: torch.Tensor,
        kv_seq_len: torch.Tensor,
        local_start_index: torch.Tensor,
        start_frame: int = 0,
    ):
        """
        Args:
            x: [B, S, C] — input hidden states
            grid_sizes: [B, 3] — (F, H, W)
            rope_*: precomputed RoPE cos/sin
            kv_k: [B, max_cache_len, num_heads, head_dim] — key cache (pre-managed)
            kv_v: [B, max_cache_len, num_heads, head_dim] — value cache (pre-managed)
            kv_seq_len: [B] — valid length in cache (Int64)
            local_start_index: [B] — where to write new KV in cache
            start_frame: frame offset for RoPE
            
        Returns:
            output: [B, S, C]
            updated_kv_k, updated_kv_v, updated_kv_seq_len
        """
        B, S, C = x.shape
        N, D = self.num_heads, self.head_dim
        
        # QKV projection
        q = self.norm_q(self.q(x)).reshape(B, S, N, D)
        k = self.norm_k(self.k(x)).reshape(B, S, N, D)
        v = self.v(x).reshape(B, S, N, D)
        
        # Apply RoPE to Q and K
        q = trt_rope_apply(q, grid_sizes, rope_cos_t, rope_sin_t,
                           rope_cos_h, rope_sin_h, rope_cos_w, rope_sin_w,
                           start_frame=start_frame)
        k = trt_rope_apply(k, grid_sizes, rope_cos_t, rope_sin_t,
                           rope_cos_h, rope_sin_h, rope_cos_w, rope_sin_w,
                           start_frame=start_frame)
        
        # Write new K,V to cache at local_start_index
        # For B=1 in streaming mode:
        write_start = local_start_index[0]  # scalar tensor
        write_end = write_start + S
        
        kv_k = kv_k.clone()
        kv_v = kv_v.clone()
        kv_k[0, write_start:write_end] = k[0, :S]
        kv_v[0, write_start:write_end] = v[0, :S]
        
        # Update seq_len
        kv_seq_len = write_end.unsqueeze(0)  # [1]
        cache_len = kv_seq_len[0]
        
        # === TRT-safe attention: full cache + mask (no dynamic tensor sizes) ===
        # Dynamic slicing (kv_k[:, :cache_len]) causes TRT reshape failures
        # because TRT can't handle zero-volume tensors during shape inference.
        # Instead, attend over the FULL cache with an attention mask that
        # zeros out invalid (not-yet-written) positions.
        max_cache_size = kv_k.shape[1]  # fixed at trace time
        
        q_sdpa = q.transpose(1, 2)             # [B, N, S, D]
        k_sdpa = kv_k.transpose(1, 2)          # [B, N, max_cache_size, D]
        v_sdpa = kv_v.transpose(1, 2)           # [B, N, max_cache_size, D]
        
        # Build attention mask: valid where position < cache_len
        # Shape [1, 1, 1, max_cache_size] — broadcasts over [B, N, S, max_cache_size]
        positions = torch.arange(max_cache_size, device=kv_k.device, dtype=torch.long)
        valid = (positions < cache_len)  # [max_cache_size] bool
        # Convert to additive mask: 0.0 for valid, -65504 for invalid (fp16 min)
        attn_mask = torch.where(
            valid.view(1, 1, 1, -1),
            torch.zeros(1, device=kv_k.device, dtype=q_sdpa.dtype),
            torch.full((1,), -65504.0, device=kv_k.device, dtype=q_sdpa.dtype),
        )  # [1, 1, 1, max_cache_size]
        
        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,
            is_causal=False,
            dropout_p=0.0,
        )  # [B, N, S, D]
        
        out = out.transpose(1, 2).reshape(B, S, C)  # [B, S, C]
        out = self.o(out)
        
        return out, kv_k, kv_v, kv_seq_len


# =============================================================================
# TRT-Safe Cross-Attention
# =============================================================================

class TRTCrossAttention(nn.Module):
    """
    TRT-compatible cross-attention for T2V.
    
    Uses SDPA instead of flash_attention. The cross-attention KV is computed
    once and cached — here we always receive the cached K,V as inputs.
    """
    
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        crossattn_k: torch.Tensor,
        crossattn_v: torch.Tensor,
    ):
        """
        Args:
            x: [B, S, C] — query input
            crossattn_k: [B, ctx_len, num_heads, head_dim] — cached context keys
            crossattn_v: [B, ctx_len, num_heads, head_dim] — cached context values
            
        Returns:
            output: [B, S, C]
        """
        B, S, C = x.shape
        N, D = self.num_heads, self.head_dim
        
        q = self.norm_q(self.q(x)).reshape(B, S, N, D)
        
        # Transpose for SDPA
        q_sdpa = q.transpose(1, 2)               # [B, N, S, D]
        k_sdpa = crossattn_k.transpose(1, 2)     # [B, N, ctx_len, D]
        v_sdpa = crossattn_v.transpose(1, 2)      # [B, N, ctx_len, D]
        
        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=None,
            is_causal=False,
            dropout_p=0.0,
        )  # [B, N, S, D]
        
        out = out.transpose(1, 2).reshape(B, S, C)
        out = self.o(out)
        
        return out


# =============================================================================
# TRT-Safe Attention Block
# =============================================================================

class TRTAttentionBlock(nn.Module):
    """
    TRT-compatible version of CausalWanAttentionBlock.
    
    Same architecture but uses TRTSelfAttention and TRTCrossAttention
    instead of the flash_attn-based originals.
    """
    
    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True,
                 cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.eps = eps
        
        # Norms
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        
        # Attention
        self.self_attn = TRTSelfAttention(dim, num_heads, qk_norm, eps)
        self.cross_attn = TRTCrossAttention(dim, num_heads, qk_norm, eps)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        
        # Modulation (same as original)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        grid_sizes: torch.Tensor,
        rope_cos_t: torch.Tensor,
        rope_sin_t: torch.Tensor,
        rope_cos_h: torch.Tensor,
        rope_sin_h: torch.Tensor,
        rope_cos_w: torch.Tensor,
        rope_sin_w: torch.Tensor,
        kv_k: torch.Tensor,
        kv_v: torch.Tensor,
        kv_seq_len: torch.Tensor,
        local_start_index: torch.Tensor,
        crossattn_k: torch.Tensor,
        crossattn_v: torch.Tensor,
        start_frame: int = 0,
    ):
        """
        Args:
            x: [B, S, C]
            e: [B, F, 6, C] — time/modulation embeddings
            grid_sizes: [B, 3]
            rope_*: RoPE params
            kv_k, kv_v: [B, max_cache, N, D] — KV cache for this layer
            kv_seq_len: [B] — valid cache length
            local_start_index: [B] — write position in cache
            crossattn_k, crossattn_v: [B, ctx_len, N, D] — precomputed cross-attn KV
            start_frame: frame offset
            
        Returns:
            x: [B, S, C]
            kv_k, kv_v, kv_seq_len: updated cache
        """
        num_frames = e.shape[1]
        B, S, C = x.shape
        frame_seqlen = S // num_frames
        
        # Modulation: 6 shifts/scales
        e_mod = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # Each e_mod[i] is [B, F, 1, C]
        
        # Self-attention with modulation
        x_normed = self.norm1(x)
        # Reshape for per-frame modulation: [B, F, frame_seqlen, C]
        x_normed = x_normed.reshape(B, num_frames, frame_seqlen, C)
        x_mod = (x_normed * (1 + e_mod[1]) + e_mod[0]).reshape(B, S, C)
        
        y, kv_k, kv_v, kv_seq_len = self.self_attn(
            x_mod, grid_sizes,
            rope_cos_t, rope_sin_t,
            rope_cos_h, rope_sin_h,
            rope_cos_w, rope_sin_w,
            kv_k, kv_v, kv_seq_len, local_start_index,
            start_frame=start_frame,
        )
        
        # Apply self-attn residual with modulation
        y_reshaped = y.reshape(B, num_frames, frame_seqlen, C)
        x = x + (y_reshaped * e_mod[2]).reshape(B, S, C)
        
        # Cross-attention
        x = x + self.cross_attn(self.norm3(x), crossattn_k, crossattn_v)
        
        # FFN with modulation
        x_normed2 = self.norm2(x).reshape(B, num_frames, frame_seqlen, C)
        y_ffn = self.ffn(
            (x_normed2 * (1 + e_mod[4]) + e_mod[3]).reshape(B, S, C)
        )
        y_ffn_reshaped = y_ffn.reshape(B, num_frames, frame_seqlen, C)
        x = x + (y_ffn_reshaped * e_mod[5]).reshape(B, S, C)
        
        return x, kv_k, kv_v, kv_seq_len


# =============================================================================
# TRT-Safe Head
# =============================================================================

class TRTCausalHead(nn.Module):
    """TRT-compatible CausalHead — same logic, no Python loops."""
    
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        
        out_dim_full = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim_full)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
    
    def forward(self, x: torch.Tensor, e: torch.Tensor):
        """
        Args:
            x: [B, S, C]
            e: [B, F, 1, C] — time embedding for head modulation
        Returns:
            [B, S, out_dim_full]
        """
        B, S, C = x.shape
        num_frames = e.shape[1]
        frame_seqlen = S // num_frames
        
        e_mod = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        
        x_normed = self.norm(x).reshape(B, num_frames, frame_seqlen, C)
        x = self.head(x_normed * (1 + e_mod[1]) + e_mod[0])
        # x shape: [B, F, frame_seqlen, out_dim_full]
        return x


# =============================================================================
# TRT-Safe Unpatchify
# =============================================================================

def trt_unpatchify(x: torch.Tensor, grid_sizes: torch.Tensor,
                   patch_size: tuple, out_dim: int):
    """
    TRT-safe unpatchify. Assumes B=1 (streaming mode).
    
    Args:
        x: [B, F, frame_seqlen, out_dim * prod(patch_size)]
           or [B, F, H*W, out_dim * pt*ph*pw]
        grid_sizes: [B, 3] — (F, H, W) in patch grid coordinates
        patch_size: (pt, ph, pw)
        out_dim: number of output channels (16)
    
    Returns:
        [B, out_dim, F*pt, H*ph, W*pw]
    """
    B = x.shape[0]
    pt, ph, pw = patch_size
    F_val = grid_sizes[0, 0]
    H_val = grid_sizes[0, 1]
    W_val = grid_sizes[0, 2]
    
    # x: [B, F, H*W, out_dim*pt*ph*pw] -> reshape to [B, F, H, W, pt, ph, pw, out_dim]
    seq_per_frame = H_val * W_val
    x = x[:, :, :seq_per_frame]  # trim padding if any
    x = x.reshape(B, F_val, H_val, W_val, pt, ph, pw, out_dim)
    
    # Rearrange: [B, F, H, W, pt, ph, pw, C] -> [B, C, F*pt, H*ph, W*pw]
    # einsum equivalent of 'b f h w p q r c -> b c (f p) (h q) (w r)'
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B, C, F, pt, H, ph, W, pw]
    x = x.reshape(B, out_dim, F_val * pt, H_val * ph, W_val * pw)
    
    return x


# =============================================================================
# TRT-Safe Full Model
# =============================================================================

class TRTCausalWanModel(nn.Module):
    """
    TRT-exportable CausalWanModel.
    
    This model has a flat tensor-only forward signature suitable for ONNX export.
    It wraps the same architecture as CausalWanModel but replaces all
    TRT-incompatible patterns.
    
    The model handles only the _forward_inference path (streaming with KV cache).
    Training path is not needed for TRT.
    """
    
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6
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
        self.head_dim = dim // num_heads
        
        # Embeddings (same as original)
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TRTAttentionBlock(
                dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        
        # Head
        self.head = TRTCausalHead(dim, out_dim, patch_size, eps)
        
        # Precomputed RoPE frequencies (real-valued)
        rope_freqs = precompute_rope_freqs_real(1024, self.head_dim)
        self.register_buffer('rope_cos_t', rope_freqs[0])
        self.register_buffer('rope_sin_t', rope_freqs[1])
        self.register_buffer('rope_cos_h', rope_freqs[2])
        self.register_buffer('rope_sin_h', rope_freqs[3])
        self.register_buffer('rope_cos_w', rope_freqs[4])
        self.register_buffer('rope_sin_w', rope_freqs[5])
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        current_start: torch.Tensor,
        current_end: torch.Tensor,
        # Flattened KV cache: [B, num_layers, max_cache_len, num_heads, head_dim]
        all_kv_k: torch.Tensor,
        all_kv_v: torch.Tensor,
        all_kv_seq_lens: torch.Tensor,
        all_local_start_indices: torch.Tensor,
        # Flattened cross-attn cache: [B, num_layers, ctx_len, num_heads, head_dim]
        all_crossattn_k: torch.Tensor,
        all_crossattn_v: torch.Tensor,
    ):
        """
        TRT-exportable forward pass.
        
        Args:
            x: [B, C_in, F, H, W] — input latent (already BCHW format)
            timestep: [B, F] — diffusion timestep per frame
            context: [B, text_len, C_text] — ALREADY ENCODED text embeddings
                     (text_embedding is applied here)
            current_start: [B] — KV cache start position (Int64)
            current_end: [B] — KV cache end position (Int64)
            all_kv_k: [B, num_layers, max_cache_len, num_heads, head_dim]
            all_kv_v: [B, num_layers, max_cache_len, num_heads, head_dim]
            all_kv_seq_lens: [B, num_layers] — valid lengths per layer
            all_local_start_indices: [B, num_layers] — write positions per layer
            all_crossattn_k: [B, num_layers, ctx_len, num_heads, head_dim]
            all_crossattn_v: [B, num_layers, ctx_len, num_heads, head_dim]
            
        Returns:
            output: [B, C_out, F_out, H_out, W_out] — predicted flow
            all_kv_k: updated KV cache keys
            all_kv_v: updated KV cache values
            all_kv_seq_lens: updated cache lengths
        """
        B = x.shape[0]
        
        # 1. Patch embedding
        x = self.patch_embedding(x)  # [B, dim, F_p, H_p, W_p]
        grid_sizes = torch.tensor(
            [[x.shape[2], x.shape[3], x.shape[4]]],
            dtype=torch.long, device=x.device
        ).expand(B, -1)  # [B, 3]
        
        x = x.flatten(2).transpose(1, 2)  # [B, F_p*H_p*W_p, dim]
        S = x.shape[1]
        
        # 2. Time embedding
        t_flat = timestep.flatten()  # [B*F]
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_flat).to(x.dtype)
        )
        e0 = self.time_projection(e).reshape(
            B, -1, 6, self.dim)  # [B, F, 6, dim]
        
        # 3. Text embedding (applies the MLP)
        context = self.text_embedding(context)  # [B, text_len, dim]
        
        # 4. Compute frame start for RoPE
        F_grid = grid_sizes[0, 0]
        HW = grid_sizes[0, 1] * grid_sizes[0, 2]
        start_frame = current_start[0] // HW
        
        # 5. Run transformer blocks
        for i, block in enumerate(self.blocks):
            # Extract per-layer cache slices
            layer_kv_k = all_kv_k[:, i]           # [B, max_cache, N, D]
            layer_kv_v = all_kv_v[:, i]
            layer_seq_len = all_kv_seq_lens[:, i]  # [B]
            layer_start = all_local_start_indices[:, i]  # [B]
            layer_cross_k = all_crossattn_k[:, i]  # [B, ctx_len, N, D]
            layer_cross_v = all_crossattn_v[:, i]
            
            x, updated_k, updated_v, updated_seq = block(
                x, e0, grid_sizes,
                self.rope_cos_t, self.rope_sin_t,
                self.rope_cos_h, self.rope_sin_h,
                self.rope_cos_w, self.rope_sin_w,
                layer_kv_k, layer_kv_v, layer_seq_len, layer_start,
                layer_cross_k, layer_cross_v,
                start_frame=start_frame,
            )
            
            # Write back updated cache
            all_kv_k = all_kv_k.clone()
            all_kv_v = all_kv_v.clone()
            all_kv_seq_lens = all_kv_seq_lens.clone()
            all_kv_k[:, i] = updated_k
            all_kv_v[:, i] = updated_v
            all_kv_seq_lens[:, i] = updated_seq
        
        # 6. Head
        num_frames = e0.shape[1]
        e_head = e.reshape(B, num_frames, 1, self.dim)  # [B, F, 1, dim]
        x = self.head(x, e_head)  # [B, F, frame_seqlen, out_dim*prod(ps)]
        
        # 7. Unpatchify
        output = trt_unpatchify(x, grid_sizes, self.patch_size, self.out_dim)
        
        return output, all_kv_k, all_kv_v, all_kv_seq_lens
