#!/usr/bin/env python3
"""
Test if the TRT engine is sensitive to current_start changes.

If the engine constant-folded current_start during ONNX tracing,
then changing current_start would have NO effect on the output.
This would be the root cause of the "stuck dog" — the model
generates every frame as if it were frame 0.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
from causvid.acceleration.tensorrt.engine_wrapper import TRTEngineWrapper

ENGINE_PATH = "engines/wan_causal_dit.engine"

print("=== Loading TRT engine ===")
engine = TRTEngineWrapper(ENGINE_PATH, device=torch.device("cuda"))
meta = engine.metadata

# Print actual engine I/O names — this is crucial for diagnosing missing inputs
print(f"\n  Engine INPUT names:  {engine.input_names}")
print(f"  Engine OUTPUT names: {engine.output_names}")
print(f"  Engine I/O dtypes:   {engine.io_dtypes}")

# Dimensions
num_layers = meta.get('num_layers', 30)
num_heads = meta.get('num_heads', 12)
head_dim = meta.get('head_dim', 128)
frame_seq_len = meta.get('frame_seq_len', 1560)
text_len = meta.get('text_len', 512)
text_dim = 4096
H_lat, W_lat = 60, 104
min_cache = meta.get('min_cache_len', frame_seq_len)

torch.manual_seed(42)


def make_inputs(current_start_val=0, seed=42, cache_size=None):
    """Create engine inputs with specified current_start.
    
    Dynamically builds the dict based on what the engine actually expects.
    """
    if cache_size is None:
        cache_size = min_cache
        
    torch.manual_seed(seed)
    
    # Start with inputs we know the engine needs
    inputs = {}
    
    all_possible = {
        'x': lambda: torch.randn(1, 16, 1, H_lat, W_lat, device='cuda', dtype=torch.float16),
        'timestep': lambda: torch.tensor([[500]], device='cuda', dtype=torch.int64),
        'context': lambda: torch.randn(1, text_len, text_dim, device='cuda', dtype=torch.float16),
        'current_start': lambda: torch.tensor([current_start_val], device='cuda', dtype=torch.int64),
        'current_end': lambda: torch.tensor([frame_seq_len], device='cuda', dtype=torch.int64),
        'all_kv_k': lambda: torch.randn(1, num_layers, cache_size, num_heads, head_dim,
                                          device='cuda', dtype=torch.float16) * 0.01,
        'all_kv_v': lambda: torch.randn(1, num_layers, cache_size, num_heads, head_dim,
                                          device='cuda', dtype=torch.float16) * 0.01,
        'all_kv_seq_lens': lambda: torch.zeros(1, num_layers, device='cuda', dtype=torch.int64),
        'all_local_start_indices': lambda: torch.zeros(1, num_layers, device='cuda', dtype=torch.int64),
        'all_crossattn_k': lambda: torch.randn(1, num_layers, text_len, num_heads, head_dim,
                                                 device='cuda', dtype=torch.float16) * 0.1,
        'all_crossattn_v': lambda: torch.randn(1, num_layers, text_len, num_heads, head_dim,
                                                 device='cuda', dtype=torch.float16) * 0.1,
    }
    
    for name in engine.input_names:
        if name in all_possible:
            inputs[name] = all_possible[name]()
        else:
            print(f"  WARNING: Engine expects unknown input '{name}'")
    
    return inputs


# ============================================
# Test 1: Does the engine respond to current_start changes?
# ============================================
print("\n=== TEST 1: current_start sensitivity ===")

if 'current_start' not in engine.input_names:
    print("  ❌ CRITICAL: 'current_start' is NOT an engine input!")
    print("  → TRT optimized it away completely!")
    print("  → This IS the root cause of the stuck dog!")
else:
    print("  'current_start' IS an engine input — testing sensitivity...")
    
    inputs_cs0 = make_inputs(current_start_val=0, seed=42)
    inputs_cs1 = make_inputs(current_start_val=1, seed=42)
    inputs_cs5 = make_inputs(current_start_val=5, seed=42)
    
    with torch.no_grad():
        out_cs0 = engine.infer(inputs_cs0)['output'].clone()
        out_cs1 = engine.infer(inputs_cs1)['output'].clone()
        out_cs5 = engine.infer(inputs_cs5)['output'].clone()
    
    diff_01 = (out_cs0.float() - out_cs1.float()).abs()
    diff_05 = (out_cs0.float() - out_cs5.float()).abs()
    
    print(f"  cs=0 output norm: {out_cs0.float().norm():.4f}")
    print(f"  cs=1 output norm: {out_cs1.float().norm():.4f}")
    print(f"  cs=5 output norm: {out_cs5.float().norm():.4f}")
    print(f"  |out(cs=0) - out(cs=1)| max={diff_01.max():.6f}, mean={diff_01.mean():.8f}")
    print(f"  |out(cs=0) - out(cs=5)| max={diff_05.max():.6f}, mean={diff_05.mean():.8f}")
    
    if diff_01.max() < 1e-6:
        print("  ❌ FAIL: Engine IGNORES current_start! Output is identical!")
        print("  → current_start IS in the graph but has no effect")
        print("  → Something in the model doesn't use it properly")
    elif diff_01.max() < 0.001:
        print("  ⚠️  Engine barely responds to current_start (very small diff)")
    else:
        print("  ✅ PASS: Engine output changes with current_start")


# ============================================
# Test 2: Does the engine respond to input x changes?
# ============================================
print("\n=== TEST 2: Input x sensitivity ===")

inputs_a = make_inputs(current_start_val=0, seed=42)
inputs_b = make_inputs(current_start_val=0, seed=99)

with torch.no_grad():
    out_a = engine.infer(inputs_a)['output'].clone() 
    out_b = engine.infer(inputs_b)['output'].clone()

diff_ab = (out_a.float() - out_b.float()).abs()
print(f"  seed42 norm: {out_a.float().norm():.4f}")
print(f"  seed99 norm: {out_b.float().norm():.4f}")
print(f"  max diff: {diff_ab.max():.6f}, mean: {diff_ab.mean():.8f}")

if diff_ab.max() < 1e-6:
    print("  ❌ FAIL: Engine ignores input data!")
else:
    print("  ✅ PASS: Engine responds to input data changes")


# ============================================
# Test 3: context sensitivity
# ============================================
if 'context' in engine.input_names:
    print("\n=== TEST 3: context sensitivity ===")
    inputs_ctx_a = make_inputs(current_start_val=0, seed=42)
    inputs_ctx_b = make_inputs(current_start_val=0, seed=42)
    # Change ONLY context
    inputs_ctx_b['context'] = torch.randn_like(inputs_ctx_b['context'])
    
    with torch.no_grad():
        out_ctx_a = engine.infer(inputs_ctx_a)['output'].clone()
        out_ctx_b = engine.infer(inputs_ctx_b)['output'].clone()
    
    diff_ctx = (out_ctx_a.float() - out_ctx_b.float()).abs()
    print(f"  max diff: {diff_ctx.max():.6f}, mean: {diff_ctx.mean():.8f}")
    if diff_ctx.max() < 1e-6:
        print("  ❌ FAIL: Engine ignores context!")
    else:
        print("  ✅ PASS: Engine responds to context changes")
else:
    print("\n=== TEST 3: context IS NOT an engine input ===")
    print("  This is expected if cross-attn KV is pre-computed outside the engine")


# ============================================
# Test 4: KV cache content sensitivity (CORRECTED)
# ============================================
# With local_start=0, new tokens OVERWRITE positions 0..1559 and mask
# everything beyond → old cache content can't matter. 
# FIX: Set local_start=1560 so new tokens go to 1560..3119 while OLD
# entries at 0..1559 are PRESERVED and ATTENDED to.
print("\n=== TEST 4: KV cache content sensitivity (local_start=1560) ===")

cache_size = frame_seq_len * 3  # 4680 — enough room for 3 frames

def make_kv_test_inputs(kv_seed, data_seed=42):
    """Create inputs where local_start=1560 and KV cache has one frame of data."""
    torch.manual_seed(data_seed)
    inputs = {}
    for name in engine.input_names:
        if name == 'x':
            inputs[name] = torch.randn(1, 16, 1, H_lat, W_lat, device='cuda', dtype=torch.float16)
        elif name == 'timestep':
            inputs[name] = torch.tensor([[500]], device='cuda', dtype=torch.int64)
        elif name == 'current_start':
            inputs[name] = torch.tensor([1], device='cuda', dtype=torch.int64)  # frame 1
        elif name == 'all_kv_k':
            torch.manual_seed(kv_seed)  # different seed for KV content
            inputs[name] = torch.randn(1, num_layers, cache_size, num_heads, head_dim,
                                        device='cuda', dtype=torch.float16)
        elif name == 'all_kv_v':
            # Continue from kv_seed (torch state carries over)
            inputs[name] = torch.randn(1, num_layers, cache_size, num_heads, head_dim,
                                        device='cuda', dtype=torch.float16)
        elif name == 'all_kv_seq_lens':
            # NOT USED by model (overwritten by write_end) but pass anyway
            inputs[name] = torch.full((1, num_layers), frame_seq_len, 
                                       device='cuda', dtype=torch.int64)
        elif name == 'all_local_start_indices':
            # KEY: write NEW tokens at position 1560, preserving 0..1559
            inputs[name] = torch.full((1, num_layers), frame_seq_len,
                                       device='cuda', dtype=torch.int64)
        elif name == 'all_crossattn_k':
            torch.manual_seed(data_seed + 1000)
            inputs[name] = torch.randn(1, num_layers, text_len, num_heads, head_dim,
                                        device='cuda', dtype=torch.float16) * 0.1
        elif name == 'all_crossattn_v':
            inputs[name] = torch.randn(1, num_layers, text_len, num_heads, head_dim,
                                        device='cuda', dtype=torch.float16) * 0.1
    return inputs

inputs_kv_a = make_kv_test_inputs(kv_seed=100, data_seed=42)
inputs_kv_b = make_kv_test_inputs(kv_seed=200, data_seed=42)

# Verify: same x, same current_start, different KV content
assert torch.equal(inputs_kv_a['x'], inputs_kv_b['x']), "x should be identical"
assert torch.equal(inputs_kv_a['current_start'], inputs_kv_b['current_start'])
assert not torch.equal(inputs_kv_a['all_kv_k'], inputs_kv_b['all_kv_k']), "KV should differ"
print(f"  Cache size: {cache_size}, local_start: {frame_seq_len}")
print(f"  → New tokens at positions {frame_seq_len}..{frame_seq_len*2-1}")
print(f"  → Old cache at positions 0..{frame_seq_len-1} should affect attention")

with torch.no_grad():
    out_kv_a = engine.infer(inputs_kv_a)['output'].clone()
    out_kv_b = engine.infer(inputs_kv_b)['output'].clone()

diff_kv = (out_kv_a.float() - out_kv_b.float()).abs()
print(f"  out_kv_a norm: {out_kv_a.float().norm():.4f}")
print(f"  out_kv_b norm: {out_kv_b.float().norm():.4f}")
print(f"  max diff: {diff_kv.max():.6f}, mean: {diff_kv.mean():.8f}")
if diff_kv.max() < 1e-6:
    print("  ❌ FAIL: Engine IGNORES KV cache content even with local_start=1560!")
    print("  → The attention mechanism cannot see previous frame context")
    print("  → This is the ROOT CAUSE of the stuck dog!")
elif diff_kv.max() < 0.01:
    print("  ⚠️  Very small KV sensitivity — attention barely uses cached context")
else:
    print("  ✅ PASS: Engine output changes when KV cache content changes")


# ============================================
# Summary
# ============================================
print("\n=== Missing/extra inputs check ===")
wrapper_inputs = {'x', 'timestep', 'current_start', 
                   'all_kv_k', 'all_kv_v', 'all_kv_seq_lens',
                   'all_local_start_indices', 'all_crossattn_k', 'all_crossattn_v'}
engine_inputs = set(engine.input_names)

missing = engine_inputs - wrapper_inputs
extra = wrapper_inputs - engine_inputs
if missing:
    print(f"  ⚠️  Engine expects these inputs that WRAPPER DOES NOT SEND: {missing}")
if extra:
    print(f"  ⚠️  Wrapper sends these inputs that ENGINE DOES NOT EXPECT: {extra}")
if not missing and not extra:
    print("  ✅ Wrapper input names match engine input names exactly")

print("\n=== Done ===")
