#!/usr/bin/env python3
"""
Compare PyTorch CausalWanModel vs TRTCausalWanModel at the model level.

Calls each model with its OWN correct forward signature and compares final outputs.
This bypasses the TRT engine to isolate model logic bugs from TRT compilation bugs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
import torch.nn.functional as F

print("=== Loading models ===")

# Model config
model_kwargs = dict(
    model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16,
    dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
    num_heads=12, num_layers=30, qk_norm=True, cross_attn_norm=True, eps=1e-6,
)

# --- Original model ---
from causvid.models.wan.causal_model import CausalWanModel
original_model = CausalWanModel(**model_kwargs).to(device='cuda', dtype=torch.bfloat16)

ckpt = torch.load("ckpts/wan_causal_dmd_v2v/model.pt", map_location="cpu")
state_dict = ckpt.get('generator', ckpt.get('generator_ema', ckpt))
model_sd = {k.removeprefix('model.'): v for k, v in state_dict.items()}
original_model.load_state_dict(model_sd, strict=True)
original_model.eval()

# --- TRT model ---
from causvid.acceleration.tensorrt.trt_model import TRTCausalWanModel
from causvid.acceleration.tensorrt.weight_converter import load_checkpoint_for_trt
trt_model = TRTCausalWanModel(**model_kwargs).to(device='cuda', dtype=torch.bfloat16)
load_checkpoint_for_trt("ckpts/wan_causal_dmd_v2v/model.pt", trt_model, strict=False)
trt_model.eval()
print(f"  Both models loaded: {sum(p.numel() for p in original_model.parameters())/1e6:.1f}M params")

# === Create test inputs ===
print("\n=== Creating test inputs ===")
torch.manual_seed(42)

B, H, W = 1, 480, 832
H_lat, W_lat = H // 8, W // 8  # 60, 104
H_p, W_p = H_lat // 2, W_lat // 2  # 30, 52
frame_seq_len = H_p * W_p  # 1560
num_layers, num_heads, head_dim = 30, 12, 128
max_cache = frame_seq_len * 10

# Raw latent input: [B, C_in, F, H_lat, W_lat]
x_raw = torch.randn(B, 16, 1, H_lat, W_lat, device='cuda', dtype=torch.bfloat16)
t_val = 500
timestep_BF = torch.tensor([[t_val]], device='cuda', dtype=torch.int64)  # [B, F]
t_flat = torch.tensor([t_val], device='cuda', dtype=torch.int64)          # [B*F]

# Text context
raw_context = torch.randn(B, 512, 4096, device='cuda', dtype=torch.bfloat16)

current_start_val = 0
current_end_val = frame_seq_len

cs = torch.tensor([current_start_val], device='cuda', dtype=torch.long)
ce = torch.tensor([current_end_val], device='cuda', dtype=torch.long)

# === Initialize KV caches ===
def make_kv_cache():
    kv = []
    for _ in range(num_layers):
        kv.append({
            'k': torch.zeros(B, max_cache, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
            'v': torch.zeros(B, max_cache, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
            'global_end_index': torch.tensor([0], device='cuda', dtype=torch.long),
            'local_end_index': torch.tensor([0], device='cuda', dtype=torch.long),
        })
    return kv

def make_crossattn_cache():
    ca = []
    for _ in range(num_layers):
        ca.append({
            'k': torch.zeros(B, 512, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
            'v': torch.zeros(B, 512, num_heads, head_dim, device='cuda', dtype=torch.bfloat16),
            'is_init': False,
        })
    return ca

# ============================================
# Run ORIGINAL model (frame 0)
# ============================================
print("\n=== Running original model (frame 0) ===")
orig_kv = make_kv_cache()
orig_ca = make_crossattn_cache()

# Original model expects: x=list of [C,F,H,W], t=[B*F], context=list of [L,C_text], seq_len=int
x_list = [x_raw[0]]            # list of [C_in, F, H_lat, W_lat] — single batch item
ctx_list = [raw_context[0]]     # list of [512, 4096] — single batch item

with torch.no_grad():
    orig_output = original_model(
        x_list, t_flat, ctx_list,
        seq_len=max_cache,
        kv_cache=orig_kv,
        crossattn_cache=orig_ca,
        current_start=cs,
        current_end=ce,
    )

print(f"  Output shape: {orig_output.shape}")
print(f"  Range: [{orig_output.min():.4f}, {orig_output.max():.4f}]")
print(f"  KV[0] local_end: {orig_kv[0]['local_end_index']}, global_end: {orig_kv[0]['global_end_index']}")
orig_kv_norm = orig_kv[0]['k'][0, :frame_seq_len].float().norm().item()
print(f"  KV[0] k[:1560] norm: {orig_kv_norm:.4f}")

# ============================================
# Run TRT model (frame 0) — same inputs
# ============================================
print("\n=== Running TRT model (frame 0) ===")
trt_kv = make_kv_cache()
trt_ca = make_crossattn_cache()

# TRT model needs flat KV tensors
all_kv_k = torch.stack([c['k'] for c in trt_kv], dim=1)  # [B, L, cache, N, D]
all_kv_v = torch.stack([c['v'] for c in trt_kv], dim=1)
all_kv_seq = torch.stack([c['global_end_index'] for c in trt_kv], dim=1)  # [B, L]
all_kv_start = torch.stack([c['local_end_index'] for c in trt_kv], dim=1)

# Pre-compute cross-attn KV using original model's text_embedding + cross-attn projections
# (This is what trt_wrapper._precompute_crossattn_kv does)
with torch.no_grad():
    ctx_embedded = original_model.text_embedding(raw_context)

all_cross_k_list, all_cross_v_list = [], []
with torch.no_grad():
    for i, block in enumerate(original_model.blocks):
        k = block.cross_attn.norm_k(block.cross_attn.k(ctx_embedded)).view(B, -1, num_heads, head_dim)
        v = block.cross_attn.v(ctx_embedded).view(B, -1, num_heads, head_dim)
        all_cross_k_list.append(k)
        all_cross_v_list.append(v)
        # Also write to trt_ca for consistency 
        trt_ca[i]['k'] = k
        trt_ca[i]['v'] = v
        trt_ca[i]['is_init'] = True

all_cross_k = torch.stack(all_cross_k_list, dim=1)  # [B, L, 512, N, D]
all_cross_v = torch.stack(all_cross_v_list, dim=1)

with torch.no_grad():
    trt_output, out_kv_k, out_kv_v, out_kv_seq = trt_model(
        x_raw,                # [B, C, F, H, W]
        timestep_BF,          # [B, F]
        raw_context,          # [B, 512, 4096] — TRT model applies text_embedding internally
        cs,                   # [B] current_start (frame index = 0)
        ce,                   # [B] current_end
        all_kv_k, all_kv_v, all_kv_seq, all_kv_start,
        all_cross_k, all_cross_v,
    )

print(f"  Output shape: {trt_output.shape}")
print(f"  Range: [{trt_output.min():.4f}, {trt_output.max():.4f}]")
trt_kv_norm = out_kv_k[0, 0, :frame_seq_len].float().norm().item()
print(f"  KV[0] k[:1560] norm: {trt_kv_norm:.4f}")

# ============================================
# Compare outputs
# ============================================
print("\n=== Comparing frame 0 outputs ===")
diff = (orig_output.float() - trt_output.float()).abs()
print(f"  Max diff: {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")

if diff.max() > 1.0:
    print("  ❌ LARGE DIVERGENCE — bug is in TRT-compatible model logic")
    # Find which dim diverges most
    for d in range(orig_output.shape[1]):
        d_diff = diff[0, d].max().item()
        if d_diff > 0.1:
            print(f"      Channel {d}: max_diff={d_diff:.4f}")
elif diff.max() > 0.01:
    print("  ⚠️  Moderate difference — likely FP precision or minor logic diff") 
else:
    print("  ✅ Models match! Bug is in TRT engine compilation, not model logic")

# ============================================
# Compare KV cache content
# ============================================
print("\n=== Comparing KV cache after frame 0 ===")
for layer_idx in [0, 15, 29]:
    orig_k = orig_kv[layer_idx]['k'][0, :frame_seq_len].float()
    trt_k = out_kv_k[0, layer_idx, :frame_seq_len].float()
    kv_diff = (orig_k - trt_k).abs()
    print(f"  Layer {layer_idx}: KV-K max_diff={kv_diff.max():.6f}, mean={kv_diff.mean():.8f}")

# ============================================
# Run frame 1 with cached context
# ============================================
print("\n=== Running frame 1 with cache from frame 0 ===")
torch.manual_seed(123)
x1_raw = torch.randn(B, 16, 1, H_lat, W_lat, device='cuda', dtype=torch.bfloat16)
t1_val = 547
ts1_BF = torch.tensor([[t1_val]], device='cuda', dtype=torch.int64)
t1_flat = torch.tensor([t1_val], device='cuda', dtype=torch.int64)
cs1 = torch.tensor([frame_seq_len], device='cuda', dtype=torch.long)
ce1 = torch.tensor([frame_seq_len * 2], device='cuda', dtype=torch.long)

# Original model frame 1
with torch.no_grad():
    orig_out1 = original_model(
        [x1_raw[0]], t1_flat, ctx_list,
        seq_len=max_cache,
        kv_cache=orig_kv, crossattn_cache=orig_ca,
        current_start=cs1, current_end=ce1,
    )

# TRT model frame 1 — use updated KV from frame 0
all_kv_seq1 = out_kv_seq  # Updated after frame 0
all_kv_start1 = out_kv_seq.clone()  # local_start = local_end from prev
with torch.no_grad():
    trt_out1, _, _, _ = trt_model(
        x1_raw, ts1_BF, raw_context,
        cs1, ce1,
        out_kv_k, out_kv_v, all_kv_seq1, all_kv_start1,
        all_cross_k, all_cross_v,
    )

diff1 = (orig_out1.float() - trt_out1.float()).abs()
print(f"  Frame 1 max diff: {diff1.max():.6f}")
print(f"  Frame 1 mean diff: {diff1.mean():.6f}")

# Cross-frame variation check
orig_frame_change = (orig_output.float() - orig_out1.float()).abs().max()
trt_frame_change = (trt_output.float() - trt_out1.float()).abs().max()
print(f"\n  Original frame0→frame1 change: {orig_frame_change:.6f}")
print(f"  TRT      frame0→frame1 change: {trt_frame_change:.6f}")

if trt_frame_change < 0.001:
    print("  ❌ TRT model stuck — output doesn't change between frames!")
elif diff1.max() > 1.0:
    print("  ⚠️  Both change but diverge — error accumulates across frames")
else:
    print("  ✅  Both models change and match across frames")

print("\n=== Done ===")
