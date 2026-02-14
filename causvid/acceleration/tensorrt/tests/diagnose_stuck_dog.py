#!/usr/bin/env python3
"""
Diagnose 'stuck dog' issue by comparing PyTorch vs TRT RoPE and model outputs.

Tests:
1. Does trt_rope_apply produce different outputs for different start_frame?
2. Does trt_rope_apply match causal_rope_apply numerically?
3. Does TRTCausalWanModel produce different outputs for different current_start?
"""
import sys
import os
import torch
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from causvid.models.wan.causal_model import causal_rope_apply
from causvid.models.wan.wan_base.modules.model import rope_params
from causvid.acceleration.tensorrt.trt_model import (
    trt_rope_apply, precompute_rope_freqs_real
)


def test_rope_sensitivity():
    """Test: does trt_rope_apply produce DIFFERENT output for different start_frame?"""
    print("=" * 70)
    print("TEST 1: RoPE sensitivity to start_frame")
    print("=" * 70)
    
    head_dim = 128
    num_heads = 12
    H, W = 30, 52
    seq_len = H * W  # 1560
    
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.float32)
    grid_sizes = torch.tensor([[1, H, W]], dtype=torch.long)
    
    rope_freqs = precompute_rope_freqs_real(1024, head_dim)
    cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = rope_freqs
    
    # Test with start_frame = 0
    out_0 = trt_rope_apply(x, grid_sizes, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
                           start_frame=0)
    # Test with start_frame = 1
    out_1 = trt_rope_apply(x, grid_sizes, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
                           start_frame=1)
    # Test with start_frame = 5
    out_5 = trt_rope_apply(x, grid_sizes, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
                           start_frame=5)
    
    diff_01 = (out_0 - out_1).abs().max().item()
    diff_05 = (out_0 - out_5).abs().max().item()
    
    print(f"  max|out(sf=0) - out(sf=1)| = {diff_01:.8f}")
    print(f"  max|out(sf=0) - out(sf=5)| = {diff_05:.8f}")
    
    if diff_01 < 1e-6:
        print("  ❌ FAIL: start_frame has NO effect on output!")
        print("  This means RoPE temporal offset is broken.")
    else:
        print("  ✅ PASS: start_frame changes the output.")
    
    # Also test with start_frame as tensor vs int
    out_tensor = trt_rope_apply(x, grid_sizes, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
                                start_frame=torch.tensor(1))
    diff_tensor = (out_1 - out_tensor).abs().max().item()
    print(f"  max|out(sf=1_int) - out(sf=1_tensor)| = {diff_tensor:.8f}")
    if diff_tensor > 1e-6:
        print("  ❌ WARN: tensor vs int start_frame gives different results!")
    
    return diff_01 > 1e-6


def test_rope_parity_with_original():
    """Test: does trt_rope_apply match causal_rope_apply?"""
    print("\n" + "=" * 70)
    print("TEST 2: trt_rope_apply vs causal_rope_apply parity")
    print("=" * 70)
    
    head_dim = 128
    num_heads = 12
    H, W = 30, 52
    seq_len = H * W
    
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.float32)
    grid_sizes = torch.tensor([[1, H, W]], dtype=torch.long)
    
    # Original freqs (complex)
    freqs_orig = rope_params(1024, head_dim)  # [1024, 64] complex
    
    # TRT precomputed
    rope_freqs = precompute_rope_freqs_real(1024, head_dim)
    cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = rope_freqs
    
    for sf in [0, 1, 5, 10]:
        # Original causal_rope_apply
        out_orig = causal_rope_apply(x, grid_sizes, freqs_orig, start_frame=sf)
        
        # TRT trt_rope_apply
        out_trt = trt_rope_apply(x, grid_sizes, cos_t, sin_t, cos_h, sin_h,
                                  cos_w, sin_w, start_frame=sf)
        
        diff = (out_orig.float() - out_trt.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  start_frame={sf}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}", end="")
        
        if max_diff < 0.01:
            print(" ✅")
        elif max_diff < 0.1:
            print(" ⚠️ (marginal)")
        else:
            print(" ❌ DIVERGENT")
            # Find which dimension diverges most
            dim_diffs = diff[0].mean(dim=(0, 1))  # [head_dim]
            half_d = head_dim // 2
            c_t = half_d - 2 * (half_d // 3)  # 22
            c_h = half_d // 3  # 21
            c_w = half_d // 3  # 21
            
            # In interleaved format, temporal covers dims 0-43
            temporal_diff = dim_diffs[:c_t*2].mean().item()
            height_diff = dim_diffs[c_t*2:(c_t+c_h)*2].mean().item()
            width_diff = dim_diffs[(c_t+c_h)*2:].mean().item()
            print(f"    temporal_mean_err={temporal_diff:.6f}, "
                  f"height_mean_err={height_diff:.6f}, "
                  f"width_mean_err={width_diff:.6f}")


def test_rope_cos_sin_values():
    """Check the actual cos/sin values at different frame positions."""
    print("\n" + "=" * 70)
    print("TEST 3: RoPE cos/sin values at different positions")
    print("=" * 70)
    
    head_dim = 128
    half_d = 64
    c_t = half_d - 2 * (half_d // 3)  # 22
    
    # Original
    freqs_orig = rope_params(1024, head_dim)
    
    # TRT
    rope_freqs = precompute_rope_freqs_real(1024, head_dim)
    cos_t, sin_t = rope_freqs[0], rope_freqs[1]
    
    print("  Frame 0:")
    print(f"    orig temporal cos[:5] = {freqs_orig[0, :c_t].real.float()[:5].tolist()}")
    print(f"    trt  temporal cos[:5] = {cos_t[0, :5].tolist()}")
    print(f"    orig temporal sin[:5] = {freqs_orig[0, :c_t].imag.float()[:5].tolist()}")
    print(f"    trt  temporal sin[:5] = {sin_t[0, :5].tolist()}")
    
    print("  Frame 1:")
    print(f"    orig temporal cos[:5] = {freqs_orig[1, :c_t].real.float()[:5].tolist()}")
    print(f"    trt  temporal cos[:5] = {cos_t[1, :5].tolist()}")
    
    print("  Frame 5:")
    print(f"    orig temporal cos[:5] = {freqs_orig[5, :c_t].real.float()[:5].tolist()}")
    print(f"    trt  temporal cos[:5] = {cos_t[5, :5].tolist()}")
    
    # Check if cos/sin at frame 0 vs frame 1 are different
    diff = (cos_t[0] - cos_t[1]).abs().max().item()
    print(f"\n  cos_t[0] vs cos_t[1] max diff: {diff:.8f}")
    if diff < 1e-8:
        print("  ❌ cos_t doesn't change between frames!")
    else:
        print("  ✅ cos_t changes between frames")


def test_onnx_tracing():
    """Test: does torch.jit.trace preserve the dynamic start_frame behavior?"""
    print("\n" + "=" * 70)
    print("TEST 4: ONNX tracing — does current_start remain dynamic?")
    print("=" * 70)
    
    head_dim = 128
    num_heads = 12
    H, W = 30, 52
    seq_len = H * W
    
    # Create a simple module that wraps trt_rope_apply
    class RoPEModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            rope_freqs = precompute_rope_freqs_real(1024, head_dim)
            self.register_buffer('cos_t', rope_freqs[0])
            self.register_buffer('sin_t', rope_freqs[1])
            self.register_buffer('cos_h', rope_freqs[2])
            self.register_buffer('sin_h', rope_freqs[3])
            self.register_buffer('cos_w', rope_freqs[4])
            self.register_buffer('sin_w', rope_freqs[5])
            
        def forward(self, x, current_start):
            grid_sizes = torch.tensor([[1, H, W]], dtype=torch.long, device=x.device)
            start_frame = current_start[0]  # same as trt_model.py
            return trt_rope_apply(x, grid_sizes,
                                  self.cos_t, self.sin_t,
                                  self.cos_h, self.sin_h,
                                  self.cos_w, self.sin_w,
                                  start_frame=start_frame)

    module = RoPEModule()
    
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, num_heads, head_dim)
    cs0 = torch.tensor([0], dtype=torch.long)
    
    # Direct call
    out_direct_0 = module(x, torch.tensor([0], dtype=torch.long))
    out_direct_1 = module(x, torch.tensor([1], dtype=torch.long))
    direct_diff = (out_direct_0 - out_direct_1).abs().max().item()
    print(f"  Direct call: max|out(cs=0) - out(cs=1)| = {direct_diff:.8f}")
    
    # Traced module
    traced = torch.jit.trace(module, (x, cs0))
    out_traced_0 = traced(x, torch.tensor([0], dtype=torch.long))
    out_traced_1 = traced(x, torch.tensor([1], dtype=torch.long))
    traced_diff = (out_traced_0 - out_traced_1).abs().max().item()
    print(f"  Traced call:  max|out(cs=0) - out(cs=1)| = {traced_diff:.8f}")
    
    if traced_diff < 1e-6:
        print("  ❌ FAIL: torch.jit.trace CONSTANT-FOLDED the start_frame!")
        print("  The TRT engine ignores current_start — THIS IS THE ROOT CAUSE!")
    else:
        print("  ✅ PASS: torch.jit.trace preserves dynamic behavior")
    
    # Also compare traced output with direct output
    trace_vs_direct_0 = (out_traced_0 - out_direct_0).abs().max().item()
    trace_vs_direct_1 = (out_traced_1 - out_direct_1).abs().max().item()
    print(f"  Traced vs Direct (cs=0): {trace_vs_direct_0:.8f}")
    print(f"  Traced vs Direct (cs=1): {trace_vs_direct_1:.8f}")
    
    return traced_diff > 1e-6


if __name__ == '__main__':
    print("Stuck Dog Diagnostic — RoPE & Model Parity Tests")
    print("=" * 70)
    
    results = {}
    results['sensitivity'] = test_rope_sensitivity()
    test_rope_parity_with_original()
    test_rope_cos_sin_values()
    results['tracing'] = test_onnx_tracing()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  RoPE sensitivity to start_frame: {'✅' if results['sensitivity'] else '❌'}")
    print(f"  Tracing preserves dynamic:       {'✅' if results['tracing'] else '❌'}")
    
    if not results['tracing']:
        print("\n  >>> ROOT CAUSE IDENTIFIED: torch.jit.trace constant-folds start_frame")
        print("  >>> FIX: Make start_frame an explicit input tensor, not computed from grid_sizes")
