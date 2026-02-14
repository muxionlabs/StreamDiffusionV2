#!/usr/bin/env python3
"""
Compare PyTorch CausalWanModel vs TRTCausalWanModel at the model level.

Loads both models with the same weights, feeds identical inputs, and compares
outputs. This bypasses the TRT engine entirely to check if the TRT-compatible
PyTorch model produces the same outputs as the original.

This isolates whether the issue is in:
  (a) TRT-safe model logic (Python-level) → mismatch here
  (b) TRT engine compilation/tracing → match here, mismatch with engine
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("configs/wan_causal_dmd_v2v.yaml")

# ============================================
# 1. Load original PyTorch CausalWanModel
# ============================================
print("=== Loading original PyTorch CausalWanModel ===")
from causvid.models.wan.causal_model import CausalWanModel

original_model = CausalWanModel(
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
    eps=1e-6,
)
original_model = original_model.to(device='cuda', dtype=torch.bfloat16)

# Load weights
ckpt = torch.load("ckpts/wan_causal_dmd_v2v/model.pt", map_location="cpu")
if 'generator' in ckpt:
    state_dict = ckpt['generator']
elif 'generator_ema' in ckpt:
    state_dict = ckpt['generator_ema']
else:
    state_dict = ckpt

# Extract model-level state dict (strip 'model.' prefix if present)
model_sd = {}
for k, v in state_dict.items():
    if k.startswith('model.'):
        model_sd[k[6:]] = v
    else:
        model_sd[k] = v

original_model.load_state_dict(model_sd, strict=True)
original_model.eval()
print(f"  Loaded {sum(p.numel() for p in original_model.parameters())/1e6:.1f}M params")

# ============================================
# 2. Load TRT-compatible model 
# ============================================
print("\n=== Loading TRT-compatible model ===")
from causvid.acceleration.tensorrt.trt_model import TRTCausalWanModel
from causvid.acceleration.tensorrt.weight_converter import load_checkpoint_for_trt

trt_model = TRTCausalWanModel(
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
    eps=1e-6,
)
trt_model = trt_model.to(device='cuda', dtype=torch.bfloat16)
load_checkpoint_for_trt("ckpts/wan_causal_dmd_v2v/model.pt", trt_model, strict=False)
trt_model.eval()
print(f"  Loaded {sum(p.numel() for p in trt_model.parameters())/1e6:.1f}M params")

# ============================================
# 3. Create test inputs
# ============================================
print("\n=== Creating test inputs ===")
torch.manual_seed(42)

B = 1
H, W = 480, 832
H_lat, W_lat = H // 8, W // 8  # 60, 104
num_frames = 1
text_len = 512
dim = 1536
num_heads = 12
head_dim = 128
num_layers = 30

# Latent space input
x = torch.randn(B, 16, num_frames, H_lat, W_lat, device='cuda', dtype=torch.bfloat16)
timestep = torch.tensor([[500]], device='cuda', dtype=torch.int64)

# Text context (random but same for both)
context = torch.randn(B, text_len, 4096, device='cuda', dtype=torch.bfloat16)

# KV cache (empty at start)
H_p = H_lat // 2  # 30
W_p = W_lat // 2  # 52
frame_seq_len = H_p * W_p  # 1560
max_cache = frame_seq_len * 10  # 15600

current_start = 0
current_end = frame_seq_len

# ============================================
# 4. Run original model (single frame, no cache)
# ============================================
print("\n=== Running original model (frame 0, fresh cache) ===")

# Initialize KV cache for original model
orig_kv_cache = []
for i in range(num_layers):
    orig_kv_cache.append({
        'k': torch.zeros(B, max_cache, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
        'v': torch.zeros(B, max_cache, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
        'global_end_index': torch.tensor([0], device='cuda', dtype=torch.long),
        'local_end_index': torch.tensor([0], device='cuda', dtype=torch.long),
    })
orig_crossattn = []
for i in range(num_layers):
    orig_crossattn.append({
        'k': torch.zeros(B, text_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
        'v': torch.zeros(B, text_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
        'is_init': False,
    })

cs = torch.tensor([current_start], device='cuda', dtype=torch.long)
ce = torch.tensor([current_end], device='cuda', dtype=torch.long)

with torch.no_grad():
    orig_output = original_model(
        x, context, timestep,
        kv_cache=orig_kv_cache,
        crossattn_cache=orig_crossattn,
        current_start=cs,
        current_end=ce,
    )

print(f"  Original output shape: {orig_output.shape}")
print(f"  Output range: [{orig_output.min():.4f}, {orig_output.max():.4f}]")
print(f"  Output mean: {orig_output.mean():.4f}, std: {orig_output.std():.4f}")

# Check KV cache was written
print(f"  KV cache[0] local_end: {orig_kv_cache[0]['local_end_index']}")
print(f"  KV cache[0] global_end: {orig_kv_cache[0]['global_end_index']}")
orig_kv_k0_norm = orig_kv_cache[0]['k'][0, :frame_seq_len].float().norm().item()
print(f"  KV cache[0] k[:1560] norm: {orig_kv_k0_norm:.4f}")

# ============================================
# 5. Run TRT model (same inputs)
# ============================================
print("\n=== Running TRT model (frame 0, fresh cache) ===")

# Initialize KV cache for TRT model
trt_kv_cache = []
for i in range(num_layers):
    trt_kv_cache.append({
        'k': torch.zeros(B, max_cache, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
        'v': torch.zeros(B, max_cache, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
        'global_end_index': torch.tensor([0], device='cuda', dtype=torch.long),
        'local_end_index': torch.tensor([0], device='cuda', dtype=torch.long),
    })
trt_crossattn = []
for i in range(num_layers):
    trt_crossattn.append({
        'k': torch.zeros(B, text_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
        'v': torch.zeros(B, text_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
        'is_init': False,
    })

with torch.no_grad():
    trt_output = trt_model(
        x, context, timestep,
        kv_cache=trt_kv_cache,
        crossattn_cache=trt_crossattn,
        current_start=cs,
        current_end=ce,
    )

print(f"  TRT model output shape: {trt_output.shape}")
print(f"  Output range: [{trt_output.min():.4f}, {trt_output.max():.4f}]")
print(f"  Output mean: {trt_output.mean():.4f}, std: {trt_output.std():.4f}")

# Check KV cache was written
print(f"  KV cache[0] local_end: {trt_kv_cache[0]['local_end_index']}")
trt_kv_k0_norm = trt_kv_cache[0]['k'][0, :frame_seq_len].float().norm().item()
print(f"  KV cache[0] k[:1560] norm: {trt_kv_k0_norm:.4f}")

# ============================================
# 6. Compare outputs
# ============================================
print("\n=== Comparing outputs ===")
diff = (orig_output.float() - trt_output.float()).abs()
print(f"  Max diff: {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")
print(f"  Relative max: {(diff / (orig_output.float().abs() + 1e-8)).max():.6f}")

# Compare KV caches
for layer_idx in [0, 15, 29]:
    kv_diff_k = (orig_kv_cache[layer_idx]['k'].float() - trt_kv_cache[layer_idx]['k'].float()).abs()
    print(f"  Layer {layer_idx} KV-K max diff: {kv_diff_k[:, :frame_seq_len].max():.6f}")

# ============================================
# 7. Run frame 1 with existing cache
# ============================================
print("\n=== Running frame 1 with cache from frame 0 ===")
torch.manual_seed(123)
x1 = torch.randn(B, 16, 1, H_lat, W_lat, device='cuda', dtype=torch.bfloat16)
ts1 = torch.tensor([[547]], device='cuda', dtype=torch.int64)
cs1 = torch.tensor([frame_seq_len], device='cuda', dtype=torch.long)  # token index = 1560
ce1 = torch.tensor([frame_seq_len * 2], device='cuda', dtype=torch.long)

with torch.no_grad():
    orig_out_1 = original_model(
        x1, context, ts1,
        kv_cache=orig_kv_cache, crossattn_cache=orig_crossattn,
        current_start=cs1, current_end=ce1,
    )
    trt_out_1 = trt_model(
        x1, context, ts1,
        kv_cache=trt_kv_cache, crossattn_cache=trt_crossattn,
        current_start=cs1, current_end=ce1,
    )

diff1 = (orig_out_1.float() - trt_out_1.float()).abs()
print(f"  Frame 1 max diff: {diff1.max():.6f}")
print(f"  Frame 1 mean diff: {diff1.mean():.6f}")

# Check if each model's output changes between frames
orig_frame_diff = (orig_output.float() - orig_out_1.float()).abs().max()
trt_frame_diff = (trt_output.float() - trt_out_1.float()).abs().max()
print(f"\n  Original: frame0 vs frame1 max diff: {orig_frame_diff:.6f}")
print(f"  TRT:      frame0 vs frame1 max diff: {trt_frame_diff:.6f}")

if trt_frame_diff < 0.01:
    print("  ⚠️  TRT model outputs are nearly IDENTICAL between frames!")
    print("  This suggests temporal encoding (RoPE or cache) is broken.")
elif trt_frame_diff > 0.01 and diff1.max() > 1.0:
    print("  ⚠️  TRT model changes between frames but diverges from original!")
    print("  This suggests numeric differences accumulate across frames.")
else:
    print("  ✅  Both models change between frames and match.")

print("\n=== Done ===")
