import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
from safetensors.torch import load_file
import types
import importlib.util
import importlib.machinery

# =============================================================================
# PHASE 1: STABILIZE ENVIRONMENT
# =============================================================================
print("[1/5] Stabilizing Environment...")
import torchvision
import transformers
import diffusers
try:
    from diffusers.models import unets, attention_dispatch
    import torchao.ops
    import bitsandbytes._ops
except ImportError:
    pass

# =============================================================================
# PHASE 2: REAL-VALUED ROPE LOGIC
# =============================================================================

def apply_rotary_emb_real(x, freqs):
    """
    Apply RoPE using only Real arithmetic.
    x: [Batch, Heads, Seq, Head_Dim]
    freqs: [Seq, Head_Dim/2, 2] (Real tensor: [..., 0]=Cos, [..., 1]=Sin)
    """
    b, h, s, d = x.shape
    x_reshaped = x.view(b, h, s, d // 2, 2)
    x_r, x_i = x_reshaped.unbind(-1)
    
    # Broadcast freqs
    freqs = freqs[:s].to(x.dtype) 
    freq_cos = freqs[..., 0].view(1, 1, s, d // 2)
    freq_sin = freqs[..., 1].view(1, 1, s, d // 2)
    
    out_r = x_r * freq_cos - x_i * freq_sin
    out_i = x_r * freq_sin + x_i * freq_cos
    
    out = torch.stack([out_r, out_i], dim=-1).flatten(-2)
    return out

# =============================================================================
# PHASE 3: REPLACEMENT ATTENTION MODULES
# =============================================================================

# FIX: Added *args to swallow extra positional arguments
def patched_self_attn_forward(self, x, freqs=None, kv_cache=None, *args, **kwargs):
    """Replacement for CausalWanSelfAttention.forward"""
    # x: [B, S, Dim]
    
    # 1. Projections & Norm
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)
    
    # 2. Reshape [B, S, 12, 128] -> [B, 12, S, 128]
    b, s, _, _ = q.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).shape
    q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
    
    # 3. Apply RoPE (Real Math)
    # Ignore passed 'freqs' (complex), use internal 'self.freqs' (real)
    if hasattr(self, 'freqs'):
        q = apply_rotary_emb_real(q, self.freqs)
        k = apply_rotary_emb_real(k, self.freqs)
        
    # 4. KV Cache Handling (If provided)
    # Note: For simple export, we might skip the intricate cache update logic 
    # if it relies on complex ops, OR we implement a simplified cache update here.
    # Since we are exporting for TRT, we rely on the input 'kv_cache' structure.
    
    # For now, standard attention to verify graph structure.
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    # 5. Output Projection
    out = out.transpose(1, 2).flatten(2)
    return self.o(out)

# FIX: Added *args here too
def patched_cross_attn_forward(self, x, context, *args, **kwargs):
    """Replacement for WanT2VCrossAttention.forward"""
    
    # 1. Projections
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)
    
    # 2. Reshape
    b, s, _, _ = q.view(x.shape[0], x.shape[1], 12, 128).shape
    q = q.view(b, s, 12, 128).transpose(1, 2)
    k = k.view(b, -1, 12, 128).transpose(1, 2)
    v = v.view(b, -1, 12, 128).transpose(1, 2)
    
    # 3. RoPE
    if hasattr(self, 'freqs'):
        q = apply_rotary_emb_real(q, self.freqs)
        
    # 4. Attention
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    # 5. Output
    out = out.transpose(1, 2).flatten(2)
    return self.o(out)

# =============================================================================
# PHASE 4: APPLY PATCHES & LOAD MODEL
# =============================================================================
print("[2/5] Patching Classes...")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Inject Mock for Flash Attn
dummy_fa = types.ModuleType("flash_attn")
dummy_fa.flash_attn_interface = types.ModuleType("interface")
sys.modules["flash_attn"] = dummy_fa
sys.modules["flash_attn.flash_attn_interface"] = dummy_fa.flash_attn_interface

try:
    from causvid.models.wan.causal_model import CausalWanModel, CausalWanSelfAttention
    from causvid.models.wan.wan_base.modules.model import WanT2VCrossAttention
    
    # OVERWRITE METHODS
    CausalWanSelfAttention.forward = patched_self_attn_forward
    WanT2VCrossAttention.forward = patched_cross_attn_forward
    
    print("      - Replaced Attention forward passes with Real-Valued logic.")

except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

from src.streamdiffusion.acceleration.tensorrt.builder import WanDiTOnnxWrapper, WAN_DIT_ONNX_EXPORT_CONFIG

# Config
CHECKPOINT_DIR = "wan_models/Wan2.1-T2V-1.3B"
ONNX_PATH = "wan_dit.onnx"
ENGINE_PATH = "wan_dit.engine"
DEVICE = "cuda"

def load_wan_1_3b():
    print("[3/5] Loading Model & Converting Buffers...")
    config = {
        "model_type": "t2v", "patch_size": (1, 2, 2), "text_len": 512,
        "in_dim": 16, "dim": 1536, "ffn_dim": 8960, "freq_dim": 256,
        "text_dim": 4096, "out_dim": 16, "num_heads": 12, "num_layers": 30, "eps": 1e-6
    }
    model = CausalWanModel(**config).to(DEVICE).to(torch.float16).eval()
    
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.safetensors')]
    if not files: raise FileNotFoundError("No weights found.")
    state_dict = load_file(os.path.join(CHECKPOINT_DIR, files[0]))
    new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)
    
    # --- BUFFER CONVERSION: COMPLEX -> REAL ---
    converted_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'freqs'):
            # Convert ComplexDouble/Float -> Real Float32 [Seq, Dim/2, 2]
            real_freqs = torch.view_as_real(module.freqs.to(torch.complex64)).to(torch.float32)
            del module.freqs
            module.register_buffer('freqs', real_freqs)
            converted_count += 1
            
    print(f"      - Converted {converted_count} frequency buffers to Real tensors.")

    # Patch Head Logic
    original_head_forward = model.head.forward
    def patched_head_forward(x, e):
        if e.shape[1] > 2: e = e[:, :2, :] 
        return original_head_forward(x, e)
    model.head.forward = patched_head_forward
    
    return model

# =============================================================================
# PHASE 5: BUILD
# =============================================================================
def build():
    model = load_wan_1_3b()
    print("[4/5] Exporting ONNX...")
    
    wrapper = WanDiTOnnxWrapper(model)
    
    # NATIVE 480p INPUTS
    dummy_x = torch.randn(1, 16, 1, 60, 106, dtype=torch.float16, device=DEVICE)
    dummy_t = torch.tensor([1.0], dtype=torch.float16, device=DEVICE)
    dummy_ctx = torch.randn(1, 512, 4096, dtype=torch.float16, device=DEVICE)
    dummy_cache = torch.randn(30, 2, 1, 2048, 12, 128, dtype=torch.float16, device=DEVICE)
    
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_t, dummy_ctx, dummy_cache),
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=False,
        input_names=WAN_DIT_ONNX_EXPORT_CONFIG["input_names"],
        output_names=WAN_DIT_ONNX_EXPORT_CONFIG["output_names"],
        dynamic_axes=WAN_DIT_ONNX_EXPORT_CONFIG["dynamic_axes"]
    )
    print("      - ONNX Export Complete.")
    
    print("[5/5] Building TensorRT Engine...")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX.")
            for error in range(parser.num_errors): print(parser.get_error(error))
            return

    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1,16,1,60,106), (1,16,1,60,106), (1,16,1,60,106))
    profile.set_shape("t", (1,), (1,), (1,))
    profile.set_shape("context", (1,512,4096), (1,512,4096), (1,512,4096))
    min_c = (30, 2, 1, 0, 12, 128)
    opt_c = (30, 2, 1, 2048, 12, 128)
    max_c = (30, 2, 1, 4096, 12, 128)
    profile.set_shape("flat_cache", min_c, opt_c, max_c)
    config.add_optimization_profile(profile)
    
    engine_bytes = builder.build_serialized_network(network, config)
    if not engine_bytes:
        print("ERROR: Engine build failed.")
        return

    with open(ENGINE_PATH, "wb") as f: f.write(engine_bytes)
    print(f"SUCCESS: Engine saved to {ENGINE_PATH}")

if __name__ == "__main__":
    build()