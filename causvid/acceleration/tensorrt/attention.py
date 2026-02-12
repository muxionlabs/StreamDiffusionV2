# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT-compatible attention implementations.

Replaces FlexAttention with standard scaled_dot_product_attention for ONNX export.
Handles KV cache as explicit tensor inputs/outputs for TensorRT binding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


# Local RMSNorm implementation to avoid external dependencies
class WanRMSNorm(nn.Module):
    """RMSNorm implementation for TensorRT compatibility."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight



def create_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def create_block_causal_mask(
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a block-wise causal mask for video diffusion.
    
    Each frame block can only attend to itself and previous blocks.
    This replaces FlexAttention's dynamic block mask with a static tensor.
    """
    total_length = num_frames * frame_seqlen
    mask = torch.zeros(total_length, total_length, device=device, dtype=dtype)
    
    # Block-wise causal: each block attends to all previous blocks
    block_size = frame_seqlen * num_frame_per_block
    for i in range(0, total_length, block_size):
        block_end = min(i + block_size, total_length)
        # This block can attend to everything up to block_end
        mask[i:block_end, block_end:] = float('-inf')
    
    return mask


def trt_rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings - TensorRT compatible version.
    
    Uses real-number operations (sin/cos rotation) instead of complex multiplication
    for ONNX compatibility.
    
    Args:
        x: [B, L, num_heads, head_dim]
        grid_sizes: [B, 3] containing (F, H, W)
        freqs: [max_seq, head_dim // 2] - the cos/sin frequencies
    
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    b, seq_len, n, d = x.shape
    half_d = d // 2
    
    # Split freqs for F, H, W dimensions  
    freq_splits = [half_d - 2 * (half_d // 3), half_d // 3, half_d // 3]
    freqs_split = freqs.split(freq_splits, dim=1)
    
    output = []
    for i in range(b):
        # Use torch.unbind to preserve symbolic graph (avoid .tolist())
        # grid_sizes is [B, 3], so grid_sizes[i] is [3]
        f, h, w = torch.unbind(grid_sizes[i], dim=0)
        
        # Int64 for Shapes (ONNX Requirement) - Force cast to be sure
        f, h, w = f.long(), h.long(), w.long()
        
        # Int32 for Math (TRT Overflow Prevention)
        f_int, h_int, w_int = f.int(), h.int(), w.int()
        actual_seq_len = f_int * h_int * w_int
        
        # Shape arg must be Int64 to match other dims in reshape
        actual_seq_len_64 = actual_seq_len.long()
        
        # Get the relevant portion
        # Slicing works with Int32 or Int64
        x_i = x[i, :actual_seq_len].float()  # [seq, n, d]
        
        # Build frequency tensor for this sample
        # Define shape constants as Long Tensors (for view)
        one_t = torch.tensor(1, device=x.device, dtype=torch.long)
        
        # Calculate split sizes explicitly
        d_split_0 = half_d - 2 * (half_d // 3)
        d_split_1 = half_d // 3
        d_split_2 = half_d // 3
        
        d0_t = torch.tensor(d_split_0, device=x.device, dtype=torch.long)
        d1_t = torch.tensor(d_split_1, device=x.device, dtype=torch.long)
        d2_t = torch.tensor(d_split_2, device=x.device, dtype=torch.long)
        half_d_t = torch.tensor(half_d, device=x.device, dtype=torch.long)
        
        # Use Int64 (f, h, w) for view/expand shapes
        # Use explicit split dims
        freqs_f = freqs_split[0][:f].view(f, one_t, one_t, d0_t).expand(f, h, w, d0_t)
        freqs_h = freqs_split[1][:h].view(one_t, h, one_t, d1_t).expand(f, h, w, d1_t)
        freqs_w = freqs_split[2][:w].view(one_t, one_t, w, d2_t).expand(f, h, w, d2_t)
        
        # Reshape uses Int64 dims: [Actual(64), 1(64), HalfD(64)]
        freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(actual_seq_len_64, one_t, half_d_t)  # [seq, 1, half_d]
        
        # Real-number rotary embedding (avoid complex numbers for ONNX)
        x_real = x_i[..., 0::2]
        x_imag = x_i[..., 1::2]
        
        # freqs_i is [seq, 1, half_d] -> expand to [seq, n, half_d]
        cos_freqs = freqs_i.cos().expand(-1, n, -1)
        sin_freqs = freqs_i.sin().expand(-1, n, -1)
        
        # Rotate
        # (a + ib) * (cos + isin) = (acos - bsin) + i(asin + bcos)
        out_real = x_real * cos_freqs - x_imag * sin_freqs
        out_imag = x_real * sin_freqs + x_imag * cos_freqs
        
        # Interleave back: stack on last dim then flatten
        x_rotated = torch.cat([out_real.unsqueeze(-1), out_imag.unsqueeze(-1)], dim=-1)
        d_t = torch.tensor(d, device=x.device, dtype=torch.long)
        n_t = torch.tensor(n, device=x.device, dtype=torch.long)
        # Using -1 for inferred dims is safer 
        # Reshape uses Int64 dims: [Actual(64), N(64), D(64)]
        x_rotated = x_rotated.reshape(actual_seq_len_64, n_t, d_t)
        
        # Handle padding
        if actual_seq_len < seq_len:
            x_rotated = torch.cat([x_rotated, x[i, actual_seq_len:].float()], dim=0)
        
        output.append(x_rotated)
    
    return torch.stack(output).type_as(x)


def trt_causal_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary position embeddings for causal streaming inference.
    
    Uses real-number operations (sin/cos rotation) instead of complex multiplication
    for ONNX compatibility.
    
    Args:
        x: [B, L, num_heads, head_dim]
        grid_sizes: [B, 3] containing (F, H, W)  
        freqs: [max_seq, head_dim // 2]
        start_frame: [B] frame index offset for each batch item
    
    Returns:
        Tensor with RoPE applied
    """
    b, seq_len, n, d = x.shape
    half_d = d // 2
    
    freq_splits = [half_d - 2 * (half_d // 3), half_d // 3, half_d // 3]
    freqs_split = freqs.split(freq_splits, dim=1)
    
    output = []
    for i in range(b):
        f, h, w = torch.unbind(grid_sizes[i], dim=0)
        # Cast to Int32 for shape operations consistent with TRT 10
        f, h, w = f.int(), h.int(), w.int()
        actual_seq_len = f * h * w
        
        # start_frame should be tensor
        sf = start_frame[i].int()
    for i in range(b):
        f, h, w = torch.unbind(grid_sizes[i], dim=0)
        
        # Int64 for Shapes (ONNX Requirement) - Force cast
        f, h, w = f.long(), h.long(), w.long()
        
        # Int32 for Math (TRT Overflow Prevention 64-bit)
        f_int, h_int, w_int = f.int(), h.int(), w.int()
        actual_seq_len = f_int * h_int * w_int
        
        # Cast back to Long for Reshape args
        actual_seq_len_64 = actual_seq_len.long()
        
        # start_frame indices need Int32 for calculation
        sf = start_frame[i].int()
        
        # Slicing works with Int32 or Int64
        x_i = x[i, :actual_seq_len].float()  # [seq, n, d]
        
        # Build frequency tensor with offset
        # Use explicit dynamic indexing to prevent constant folding of 'sf'
        idx_f = torch.arange(f_int, device=x.device) + sf
        freq_max = int(freqs_split[0].shape[0]) - 1
        idx_f = idx_f.int().clamp(max=freq_max) 
        
        # Define shape constants as Long Tensors (for view)
        one_t = torch.tensor(1, device=x.device, dtype=torch.long)
        
        # Calculate split sizes explicitly
        # freq_splits = [half_d - 2 * (half_d // 3), half_d // 3, half_d // 3]
        d_split_0 = half_d - 2 * (half_d // 3)
        d_split_1 = half_d // 3
        d_split_2 = half_d // 3
        
        d0_t = torch.tensor(d_split_0, device=x.device, dtype=torch.long)
        d1_t = torch.tensor(d_split_1, device=x.device, dtype=torch.long)
        d2_t = torch.tensor(d_split_2, device=x.device, dtype=torch.long)
        half_d_t = torch.tensor(half_d, device=x.device, dtype=torch.long)
        
        # Use Int64 (f, h, w) for view/expand shapes
        # Use explicit split dims
        freqs_f = freqs_split[0][idx_f].view(f, one_t, one_t, d0_t).expand(f, h, w, d0_t)
        freqs_h = freqs_split[1][:h].view(one_t, h, one_t, d1_t).expand(f, h, w, d1_t)
        freqs_w = freqs_split[2][:w].view(one_t, one_t, w, d2_t).expand(f, h, w, d2_t)
        
        # Reshape uses Int64 dims: [Actual(64), 1(64), HalfD(64)]
        freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(actual_seq_len_64, one_t, half_d_t)
        
        # Real-number rotary embedding
        x_real = x_i[..., 0::2]
        x_imag = x_i[..., 1::2]
        
        cos_freqs = freqs_i.cos().expand(-1, n, -1)
        sin_freqs = freqs_i.sin().expand(-1, n, -1)
        
        out_real = x_real * cos_freqs - x_imag * sin_freqs
        out_imag = x_real * sin_freqs + x_imag * cos_freqs
        
        x_rotated = torch.cat([out_real.unsqueeze(-1), out_imag.unsqueeze(-1)], dim=-1)
        d_t = torch.tensor(d, device=x.device, dtype=torch.long)
        n_t = torch.tensor(n, device=x.device, dtype=torch.long)
        # Using -1 for inferred dims is safer 
        # Reshape uses Int64 dims: [Actual(64), N(64), D(64)]
        x_rotated = x_rotated.reshape(actual_seq_len_64, n_t, d_t)
        
        if actual_seq_len < seq_len:
            x_rotated = torch.cat([x_rotated, x[i, actual_seq_len:].float()], dim=0)
            
        output.append(x_rotated)
    
    return torch.stack(output).type_as(x)


class TRTCausalSelfAttention(nn.Module):
    """
    TensorRT-compatible causal self-attention.
    
    Replaces FlexAttention with scaled_dot_product_attention for ONNX export.
    KV cache is handled as explicit input/output tensors.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.scale = self.head_dim ** -0.5
        
        # Layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        kv_cache_k: Optional[torch.Tensor] = None,
        kv_cache_v: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        current_start: Optional[torch.Tensor] = None,
        current_end: Optional[torch.Tensor] = None,
        start_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with optional KV cache.
        
        Args:
            x: Input tensor [B, L, C]
            seq_lens: Sequence lengths [B]
            grid_sizes: Grid sizes [B, 3]
            freqs: RoPE frequencies
            causal_mask: Pre-computed causal mask [L, L] or None
            kv_cache_k: Key cache [B, cache_len, num_heads, head_dim]
            kv_cache_v: Value cache [B, cache_len, num_heads, head_dim]
            cache_seqlens: Current cache lengths [B]
            current_start: Start indices for cache update [B]
            current_end: End indices for cache update [B]
        
        Returns:
            output: Attention output [B, L, C]
            new_kv_cache_k: Updated key cache
            new_kv_cache_v: Updated value cache
        """
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        
        # Compute Q, K, V
        # Compute Q, K, V
        # Compute Q, K, V
        # Compute Q, K, V
        # shape_64 uses python ints b, n, d directly
        # No intermediate Int32 tensors needed for this part
        # Explicit Shape Construction
        # Use Int64 for Reshape (Standard ONNX/TRT requirement)
        # Avoid implicit 'view' to control types precisely
        # Use explicit 's' instead of -1 to avoid TRT Shape Overflow
        shape_64 = torch.tensor([b, s, n, d], device=x.device, dtype=torch.long)
        resh_dims = torch.unbind(shape_64)
        
        q = self.norm_q(self.q(x)).reshape(*resh_dims)
        k = self.norm_k(self.k(x)).reshape(*resh_dims)
        v = self.v(x).reshape(*resh_dims)
        
        # Apply RoPE
        if kv_cache_k is None:
            # Non-streaming: full sequence RoPE
            q = trt_rope_apply(q, grid_sizes, freqs)
            k = trt_rope_apply(k, grid_sizes, freqs)
        else:
            # Streaming: offset RoPE based on current position
            # Ensure frame_seqlen is tensor
            grid_i = grid_sizes[0] # Assume B=1 or same grid
            
            # Use Int32 for arithmetic safety
            f_g, h_g, w_g = grid_i[0].int(), grid_i[1].int(), grid_i[2].int()
            frame_seqlen = f_g * h_g * w_g
            
            # Symbolic division
            if start_frame is None and current_start is not None:
                # Floor division of Int32 tensors
                start_frame = torch.div(current_start.int(), frame_seqlen, rounding_mode='floor')
                
            q = trt_causal_rope_apply(q, grid_sizes, freqs, start_frame)
            k = trt_causal_rope_apply(k, grid_sizes, freqs, start_frame)
        
        # Handle KV cache
        if kv_cache_k is not None and kv_cache_v is not None:
            # Update cache with new K, V
            new_kv_cache_k = kv_cache_k.clone()
            new_kv_cache_v = kv_cache_v.clone()
            
            # Vectorized or tensor-friendly update
            # We assume batch size is small (usually 1 for streaming)
            if current_start is not None:
                # Use tensor operations to update cache to preserve graph connections
                indices = torch.arange(s, device=x.device).expand(b, s)
                start_indices = current_start.view(b, 1)
                target_indices = indices + start_indices
                
                # Careful scatter update
                # Since we update a slice [start:start+s], we can generate indices
                # New K/V are [B, S, H, D]
                # Cache is [B, MaxLen, H, D]
                
                # But we can iterate B since it is a loop in graph (or unrolled if small)
                for i in range(b):
                    # Get start as tensor (0-d or 1-d)
                    # Force Int32 for indexing
                    start_t = current_start[i].int()
                    idx = torch.arange(s, device=x.device, dtype=torch.int32) + start_t
                    
                    # Ensure indices are within bounds (clamp or mask)
                    # For streaming, we assume caller manages logic, but let's be safe for tracing
                    max_len = int(new_kv_cache_k.shape[1])
                    mask = idx < max_len
                    valid_idx = idx[mask]
                    
                    if valid_idx.numel() > 0:
                        # Index update: cache[i, valid_idx] = k[i, :num_valid]
                        # We must slice the source 'k' as well if we clamped
                        valid_src = k[i, :valid_idx.numel()]
                        
                        # Use index_put_ or simple indexing which traces to Scatter/IndexPut
                        # Indices must be Long for pytorch indexing, but TRT handles Int32 better for shape calc?
                        # Actually, PyTorch indexing requires Long. 
                        # But if we compute in Int32 and cast to Long at valid_idx usage...
                        new_kv_cache_k[i, valid_idx.long()] = valid_src
                        new_kv_cache_v[i, valid_idx.long()] = v[i, :valid_idx.numel()]
                
                # Define end_idx for slicing the full cache for attention
                # We use max() to handle batching, though usually B=1
                end_idx = (current_start + s).max()
            else:
                # Append logic (non-streaming legacy path)
                for i in range(b):
                    start_idx = int(cache_seqlens[i].item()) if cache_seqlens is not None else 0
                    end_idx = start_idx + s
                    new_kv_cache_k[i, start_idx:end_idx] = k[i]
                    new_kv_cache_v[i, start_idx:end_idx] = v[i]
            
            # Use full cache for attention
            k_full = new_kv_cache_k[:, :end_idx]
            v_full = new_kv_cache_v[:, :end_idx]
        else:
            k_full = k
            v_full = v
            new_kv_cache_k = None
            new_kv_cache_v = None
        
        # Reshape for attention: [B, num_heads, L, head_dim]
        # Reshape for attention: [B, num_heads, L, head_dim]
        # STRICT INT32 STRATEGY
        q = q.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)
        
        # Compute attention using SDPA (ONNX compatible)
        # Compute attention using SDPA (ONNX compatible)
        # Dynamic Masking for Ring Buffer / Infinite Streaming
        # We cannot rely on the static 'causal_mask' input because:
        # 1. In Linear phase (Log >> 0), slicing mask[:s] uses rows 0..s instead of Log..Log+s
        # 2. In Ring phase, Physical ordering != Logical ordering
        
        # Strategy: 
        # - Default: Attend to everything (History is valid)
        # - Constraint: Within the current chunk (Physical: current_start...current_start+s), enforces causality.
        
        # CHUNKED ATTENTION STRATEGY
        # The attention mask [B, 1, s, total_k] can exceed Int32 limit (2.14B)
        # e.g., s=100k, total_k=24k -> 2.4 Billion -> TRT Shape Overflow
        # We split 's' into chunks to keep mask size safe.
        
        CHUNK_SIZE = 4096
        
        if s > CHUNK_SIZE and kv_cache_k is not None:
            # Chunked execution
            out_chunks = []
            total_k = k_full.shape[2]
            
            # Common constants for mask generation
            
            # Use Int32 for indices
            col_indices = torch.arange(total_k, device=q.device, dtype=torch.int32).reshape(1, 1, 1, total_k)
            
            if current_start is not None:
                c_start = current_start[0]
                c_end = c_start + s
                
                # Expand dims for broadcasting
                c_start_exp = c_start.reshape(1, 1, 1, 1)
                c_end_exp = c_end.reshape(1, 1, 1, 1)
                
                # Valid End Logic
                f_g, h_g, w_g = torch.unbind(grid_sizes[0], dim=0)
                frame_len = (f_g.int() * h_g.int() * w_g.int())
                sf_val = start_frame[0].int()
                logical_valid = (sf_val * frame_len) + s
                total_k_t = torch.tensor([total_k], device=q.device, dtype=torch.int32)
                valid_end = torch.min(logical_valid, total_k_t)
                valid_end_exp = valid_end.reshape(1, 1, 1, 1)
            
            # Loop over query chunks
            for i in range(0, s, CHUNK_SIZE):
                q_chunk = q[:, :, i:i+CHUNK_SIZE, :]
                chunk_len = q_chunk.shape[2]
                
                attn_mask_chunk = None
                
                if current_start is not None:
                    # Generate mask for this chunk
                    # row_indices must be offset by 'i'
                    row_indices = torch.arange(chunk_len, device=q.device, dtype=torch.int32).reshape(1, 1, chunk_len, 1) + i
                    
                    # --- Mask Logic Reused ---
                    # 1. Intra-Chunk Future Block
                    is_in_current_block = (col_indices >= c_start_exp) & (col_indices < c_end_exp)
                    is_future_in_block = col_indices > (c_start_exp + row_indices)
                    mask_intra_future = is_in_current_block & is_future_in_block
                    
                    # 2. Uninitialized Memory
                    mask_uninitialized = col_indices >= valid_end_exp
                    
                    # 3. Combine
                    mask_block = mask_intra_future | mask_uninitialized
                    # Convert to float mask for better TRT stability
                    attn_mask_chunk = torch.zeros(mask_block.shape, device=q.device, dtype=q.dtype)
                    attn_mask_chunk.masked_fill_(mask_block, float('-inf'))
                    
                elif causal_mask is not None:
                    # Static fallback (sliced for chunk)
                    attn_mask_chunk = causal_mask[i:i+chunk_len, :total_k]
                
                # Run SDPA for chunk
                # Ensure mask is matching dtype
                if attn_mask_chunk is not None and attn_mask_chunk.dtype != q.dtype:
                    attn_mask_chunk = attn_mask_chunk.to(q.dtype)
                    
                o_chunk = F.scaled_dot_product_attention(
                    q_chunk, k_full, v_full,
                    attn_mask=attn_mask_chunk,
                    dropout_p=0.0,
                    is_causal=False
                )
                out_chunks.append(o_chunk)
            
            out = torch.cat(out_chunks, dim=2)
            
        else:
            # Standard single-pass execution (retains original logic logic for small s)
            attn_mask = None
            if s > 1 and kv_cache_k is not None and current_start is not None:
                # Construct dynamic mask
                # Shape: [B, 1, s, Total_K] -> Broadcast over heads
                total_k = k_full.shape[2]
                
                # APPLY ORIGINAL MASK LOGIC
                c_start = current_start[0] # Tensor scalar-like
                c_end = c_start + s
                
                col_indices = torch.arange(total_k, device=q.device, dtype=torch.int32).reshape(1, 1, 1, total_k)
                row_indices = torch.arange(s, device=q.device, dtype=torch.int32).reshape(1, 1, s, 1)
                
                # Valid End
                f_g, h_g, w_g = torch.unbind(grid_sizes[0], dim=0)
                frame_len = (f_g.int() * h_g.int() * w_g.int()) 
                sf_val = start_frame[0].int() 
                logical_valid = (sf_val * frame_len) + s 
                total_k_t = torch.tensor([total_k], device=q.device, dtype=torch.int32)
                valid_end = torch.min(logical_valid, total_k_t)
                
                # Intra-Chunk Causal + Uninitialized
                c_start_exp = c_start.reshape(1, 1, 1, 1)
                c_end_exp = c_end.reshape(1, 1, 1, 1)
                valid_end_exp = valid_end.reshape(1, 1, 1, 1)
                
                is_in_current_block = (col_indices >= c_start_exp) & (col_indices < c_end_exp)
                is_future_in_block = col_indices > (c_start_exp + row_indices)
                mask_intra_future = is_in_current_block & is_future_in_block
                
                mask_uninitialized = col_indices >= valid_end_exp
                
                mask_block = mask_intra_future | mask_uninitialized
                
                # Use float mask
                attn_mask = torch.zeros(mask_block.shape, device=q.device, dtype=q.dtype)
                attn_mask.masked_fill_(mask_block, float('-inf'))
                
            elif causal_mask is not None:
                 attn_mask = causal_mask[:s, :k_full.shape[2]]
    
            out = F.scaled_dot_product_attention(
                q, k_full, v_full,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False
            )
        
        # Reshape back: [B, L, C]
        # Reshape back: [B, L, C]
        # Use explicit self.dim for sequence length
        out = out.transpose(1, 2).reshape(b, s, self.dim)
        out = self.o(out)
        
        return out, new_kv_cache_k, new_kv_cache_v


class TRTCrossAttention(nn.Module):
    """
    TensorRT-compatible cross-attention for text conditioning.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
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
        crossattn_cache_k: Optional[torch.Tensor] = None,
        crossattn_cache_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cross-attention with optional caching.
        
        Args:
            x: Query input [B, L, C]
            context: Key/Value context [B, ctx_len, C]
            context_lens: Context lengths [B]
            crossattn_cache_k: Cached keys [B, ctx_len, num_heads, head_dim]
            crossattn_cache_v: Cached values [B, ctx_len, num_heads, head_dim]
        
        Returns:
            output, cached_k, cached_v
        """
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        
        q = self.norm_q(self.q(x)).reshape(b, s, n, d)
        
        # Use cache if available, otherwise compute K, V
        if crossattn_cache_k is not None and crossattn_cache_v is not None:
            k = crossattn_cache_k
            v = crossattn_cache_v
        else:
            # Explicit views to avoid TRT overflow
            # Use Int64 for shape construction
            ctx_len = context.shape[1]
            shape_64 = torch.tensor([b, ctx_len, n, d], device=x.device, dtype=torch.long)
            resh_dims = torch.unbind(shape_64)
            
            k = self.norm_k(self.k(context)).reshape(*resh_dims)
            v = self.v(context).reshape(*resh_dims)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [B, n, s, d]
        k = k.transpose(1, 2)  # [B, n, ctx_len, d]
        v = v.transpose(1, 2)  # [B, n, ctx_len, d]
        
        # CHUNKED ATTENTION FOR CROSS-ATTN
        # Logits size: S * Ctx * Heads = 99840 * 4096 * 12 = 4.9 Billion -> Overflow Int32
        CHUNK_SIZE = 4096
        
        if s > CHUNK_SIZE:
             out_chunks = []
             for i in range(0, s, CHUNK_SIZE):
                 q_chunk = q[:, :, i:i+CHUNK_SIZE, :]
                 # SDPA
                 o_chunk = F.scaled_dot_product_attention(q_chunk, k, v, dropout_p=0.0)
                 out_chunks.append(o_chunk)
             
             out = torch.cat(out_chunks, dim=2)
        else:
             # SDPA
             out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        
        # Reshape back to [B, s, dim] explicitly
        out = out.transpose(1, 2).reshape(b, s, self.dim)
        out = self.o(out)
        
        # Return with cache (transposed back)
        return out, k.transpose(1, 2), v.transpose(1, 2)
