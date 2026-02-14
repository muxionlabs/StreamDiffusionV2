#!/usr/bin/env python3
"""
Diagnose green grass: test if full-dim spatial + FP16 ITSELF causes 
the issue (Python level), or if it only happens in the TRT engine.

Tests:
1. Python TRT model in bfloat16 with full-dim → should work (we proved this)
2. Python TRT model in FP16 with full-dim → does this produce green grass?
3. If FP16 Python works but TRT engine doesn't → issue is TRT compilation
4. If FP16 Python also fails → issue is FP16 precision, need different fix

Also checks: are the ONNX RoPE constants float32 as intended?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
import numpy as np

# ============================================
# Step 1: Check ONNX RoPE buffer dtypes
# ============================================
print("=== Step 1: Checking ONNX RoPE buffer dtypes ===")
try:
    import onnx
    model_onnx = onnx.load("engines/wan_causal_dit.onnx")
    
    # Find RoPE constants in the ONNX graph
    rope_names = ['rope_cos_t', 'rope_sin_t', 'rope_cos_h', 'rope_sin_h', 
                  'rope_cos_w', 'rope_sin_w']
    
    # Check initializers (constant weights)
    for init in model_onnx.graph.initializer:
        for rname in rope_names:
            if rname in init.name:
                # onnx.TensorProto data types: 1=FLOAT, 10=FLOAT16, 7=INT64
                dtype_map = {1: 'float32', 10: 'float16', 7: 'int64', 6: 'int32'}
                dtype_str = dtype_map.get(init.data_type, f'unknown({init.data_type})')
                print(f"  {init.name}: dtype={dtype_str}, shape={list(init.dims)}")
                break
    
    # Count total float32 vs float16 initializers
    dtype_counts = {}
    for init in model_onnx.graph.initializer:
        dt = {1: 'fp32', 10: 'fp16', 7: 'i64', 6: 'i32'}.get(init.data_type, 'other')
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    print(f"  Total initializer dtypes: {dtype_counts}")
    
    del model_onnx  # free memory
except Exception as e:
    print(f"  Could not load ONNX: {e}")


# ============================================
# Step 2: Python TRT model comparison: bf16 vs fp16
# ============================================
print("\n=== Step 2: Python TRT model — bf16 vs fp16 with full-dim spatial ===")

from causvid.acceleration.tensorrt.trt_model import TRTCausalWanModel
from causvid.acceleration.tensorrt.export_onnx import T2V_1_3B_CONFIG

model_config = T2V_1_3B_CONFIG.copy()

# Create and load in bfloat16
model_bf16 = TRTCausalWanModel(**model_config).to('cuda', dtype=torch.bfloat16)
from causvid.acceleration.tensorrt.weight_converter import load_checkpoint_for_trt
load_checkpoint_for_trt("ckpts/wan_causal_dmd_v2v/model.pt", model_bf16, strict=False)
model_bf16.eval()

# Create and load in float16
model_fp16 = TRTCausalWanModel(**model_config).to('cuda', dtype=torch.float16)
load_checkpoint_for_trt("ckpts/wan_causal_dmd_v2v/model.pt", model_fp16, strict=False)
model_fp16.eval()

# Check RoPE buffer dtypes
print(f"  BF16 model rope_sin_h dtype: {model_bf16.rope_sin_h.dtype}")
print(f"  FP16 model rope_sin_h dtype: {model_fp16.rope_sin_h.dtype}")

# Verify spatial freq values (full-dim should have small values for high bands)
sin_h = model_fp16.rope_sin_h[1]  # position h=1
print(f"  FP16 sin_h[1] range: [{sin_h.min():.6f}, {sin_h.max():.6f}]")
print(f"  FP16 sin_h[1] near-zero count: {(sin_h.abs() < 0.001).sum()}/{len(sin_h)}")

sin_h_bf = model_bf16.rope_sin_h[1]
print(f"  BF16 sin_h[1] range: [{sin_h_bf.min():.6f}, {sin_h_bf.max():.6f}]")
print(f"  BF16 sin_h[1] near-zero count: {(sin_h_bf.abs() < 0.001).sum()}/{len(sin_h_bf)}")

# Create inputs
num_layers = model_config.get('num_layers', 30)
num_heads = model_config.get('num_heads', 12)
head_dim = model_config.get('head_dim', 128)
text_len = model_config.get('text_len', 512)
frame_seq_len = 1560
H_lat, W_lat = 60, 104
cache_size = frame_seq_len

torch.manual_seed(42)

# Common inputs (generate in float32, convert to each dtype)
x_raw = torch.randn(1, 16, 1, H_lat, W_lat, device='cuda')
ts = torch.tensor([[500]], device='cuda', dtype=torch.int64)
ctx = torch.randn(1, text_len, 4096, device='cuda')
cs = torch.tensor([0], device='cuda', dtype=torch.int64)
ce = torch.tensor([frame_seq_len], device='cuda', dtype=torch.int64)
kv_k = torch.zeros(1, num_layers, cache_size, num_heads, head_dim, device='cuda')
kv_v = torch.zeros(1, num_layers, cache_size, num_heads, head_dim, device='cuda')
kv_seq = torch.zeros(1, num_layers, device='cuda', dtype=torch.int64)
kv_start = torch.zeros(1, num_layers, device='cuda', dtype=torch.int64)
cross_k = torch.randn(1, num_layers, text_len, num_heads, head_dim, device='cuda') * 0.1
cross_v = torch.randn(1, num_layers, text_len, num_heads, head_dim, device='cuda') * 0.1

# Run BF16 model
with torch.no_grad():
    out_bf16 = model_bf16(
        x_raw.to(torch.bfloat16), ts, ctx.to(torch.bfloat16), cs, ce,
        kv_k.to(torch.bfloat16), kv_v.to(torch.bfloat16), kv_seq, kv_start,
        cross_k.to(torch.bfloat16), cross_v.to(torch.bfloat16),
    )[0]

# Run FP16 model
with torch.no_grad():
    out_fp16 = model_fp16(
        x_raw.to(torch.float16), ts, ctx.to(torch.float16), cs, ce,
        kv_k.to(torch.float16), kv_v.to(torch.float16), kv_seq, kv_start,
        cross_k.to(torch.float16), cross_v.to(torch.float16),
    )[0]

diff = (out_bf16.float() - out_fp16.float()).abs()
print(f"\n  BF16 output: norm={out_bf16.float().norm():.2f}, range=[{out_bf16.min():.4f}, {out_bf16.max():.4f}]")
print(f"  FP16 output: norm={out_fp16.float().norm():.2f}, range=[{out_fp16.min():.4f}, {out_fp16.max():.4f}]")
print(f"  |BF16 - FP16|: max={diff.max():.4f}, mean={diff.mean():.6f}")

# Check if FP16 output looks "green grass" (very uniform, low variance)
fp16_std = out_fp16.float().std()
bf16_std = out_bf16.float().std()
print(f"  BF16 output std: {bf16_std:.4f}")
print(f"  FP16 output std: {fp16_std:.4f}")

if fp16_std < bf16_std * 0.1:
    print("  ❌ FP16 output is MUCH smoother than BF16 — likely degenerate (green grass)")
elif diff.max() > 10.0:
    print("  ❌ HUGE bf16/fp16 diff — FP16 precision issue in model computation")
else:
    print("  ✅ FP16 and BF16 outputs are reasonably close")


# ============================================  
# Step 3: Check for NaN/Inf in FP16
# ============================================
print(f"\n  FP16 output has NaN: {out_fp16.isnan().any()}")
print(f"  FP16 output has Inf: {out_fp16.isinf().any()}")


# ============================================
# Step 4: Test if the issue is in the engine only
# ============================================
print("\n=== Step 3: Engine output comparison ===")
try:
    from causvid.acceleration.tensorrt.engine_wrapper import TRTEngineWrapper
    engine = TRTEngineWrapper("engines/wan_causal_dit.engine", device=torch.device("cuda"))
    
    engine_inputs = {
        'x': x_raw.to(torch.float16),
        'timestep': ts,
        'current_start': cs,
        'all_kv_k': kv_k.to(torch.float16),
        'all_kv_v': kv_v.to(torch.float16),
        'all_kv_seq_lens': kv_seq,
        'all_local_start_indices': kv_start,
        'all_crossattn_k': cross_k.to(torch.float16),
        'all_crossattn_v': cross_v.to(torch.float16),
    }
    
    with torch.no_grad():
        engine_out = engine.infer(engine_inputs)['output']
    
    # Compare TRT engine vs Python FP16
    diff_engine = (engine_out.float() - out_fp16.float()).abs()
    print(f"  Engine output: norm={engine_out.float().norm():.2f}, range=[{engine_out.min():.4f}, {engine_out.max():.4f}]")
    print(f"  Engine std: {engine_out.float().std():.4f}")
    print(f"  |Engine - Python FP16|: max={diff_engine.max():.4f}, mean={diff_engine.mean():.6f}")
    
    if diff_engine.max() < 1.0:
        print("  ✅ Engine matches Python FP16 closely")  
        print("  → If green grass, the issue is FP16 itself, not TRT")
    else:
        print("  ❌ Engine diverges from Python FP16")
        print("  → TRT compilation introduces errors beyond FP16 precision")
except Exception as e:
    print(f"  Engine test failed: {e}")

print("\n=== Done ===")
