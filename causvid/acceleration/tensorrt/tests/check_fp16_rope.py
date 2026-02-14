#!/usr/bin/env python3
"""Quick check: does FP16 storage of RoPE cos/sin values cause significant precision loss?"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from causvid.acceleration.tensorrt.trt_model import precompute_rope_freqs_real

cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = precompute_rope_freqs_real(1024, 128)

print("=== FP16 storage precision loss for RoPE cos/sin ===\n")
for name, tensor in [("cos_t", cos_t), ("sin_t", sin_t),
                      ("cos_h", cos_h), ("sin_h", sin_h),
                      ("cos_w", cos_w), ("sin_w", sin_w)]:
    fp16 = tensor.half().float()
    diff = (tensor - fp16).abs()
    print(f"{name}: max_fp16_loss={diff.max().item():.8f}, "
          f"mean_fp16_loss={diff.mean().item():.8f}, "
          f"shape={list(tensor.shape)}")

print("\n=== Spatial freq values at key positions ===")
# Height: positions 0-29, bands 22-42
# First check if any cos/sin values are numerically zero in FP16
for pos in [0, 1, 15, 29]:
    cos_h_fp32 = cos_h[pos]  # [21]
    cos_h_fp16 = cos_h[pos].half()
    sin_h_fp32 = sin_h[pos]
    sin_h_fp16 = sin_h[pos].half()
    zero_cos = (cos_h_fp16 == 1.0).sum().item()
    zero_sin = (sin_h_fp16 == 0.0).sum().item()
    print(f"  h={pos}: fp16 cos==1.0 count: {zero_cos}/21, "
          f"fp16 sin==0.0 count: {zero_sin}/21")

print("\n=== Full-dim RoPE end-to-end: FP16 vs FP32 ===")
from causvid.models.wan.causal_model import causal_rope_apply
from causvid.models.wan.wan_base.modules.model import rope_params
from causvid.acceleration.tensorrt.trt_model import trt_rope_apply

torch.manual_seed(42)
H, W = 30, 52
x = torch.randn(1, H*W, 12, 128)
grid_sizes = torch.tensor([[1, H, W]], dtype=torch.long)
freqs_orig = rope_params(1024, 128)

# Original (float64 complex)
out_orig = causal_rope_apply(x, grid_sizes, freqs_orig, start_frame=0)

# TRT with FP32 buffers
out_trt_fp32 = trt_rope_apply(x, grid_sizes, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, start_frame=0)

# TRT with FP16 buffers (simulating what happens in TRT engine)
out_trt_fp16 = trt_rope_apply(x, grid_sizes,
                               cos_t.half(), sin_t.half(),
                               cos_h.half(), sin_h.half(),
                               cos_w.half(), sin_w.half(),
                               start_frame=0)

diff_fp32 = (out_orig.float() - out_trt_fp32.float()).abs()
diff_fp16 = (out_orig.float() - out_trt_fp16.float()).abs()

print(f"  FP32 buffers vs original: max={diff_fp32.max():.8f}, mean={diff_fp32.mean():.8f}")
print(f"  FP16 buffers vs original: max={diff_fp16.max():.8f}, mean={diff_fp16.mean():.8f}")

# Breakdown by dimension
half_d = 64
c_t = half_d - 2*(half_d//3)
c_h = half_d // 3
# In interleaved format: temporal=0..2*c_t-1, height=2*c_t..2*(c_t+c_h)-1
t_diff = diff_fp16[0, :, :, :2*c_t].mean().item()
h_diff = diff_fp16[0, :, :, 2*c_t:2*(c_t+c_h)].mean().item()
w_diff = diff_fp16[0, :, :, 2*(c_t+c_h):].mean().item()
print(f"  FP16 err by axis: temporal={t_diff:.8f}, height={h_diff:.8f}, width={w_diff:.8f}")
