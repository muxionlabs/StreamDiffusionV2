import sys
import types
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib.machinery import ModuleSpec
import argparse
import gc
from omegaconf import OmegaConf

# =========================
# 1. SAFETY SETTINGS
# =========================
torch.set_grad_enabled(False)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# =========================
# 2. MOCK FLASH ATTENTION (FIXED)
# =========================
def manual_attention_forward(q, k, v):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    scale = q_t.size(-1) ** -0.5
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_t)
    return output.transpose(1, 2)

def universal_mock(q, *args, **kwargs):
    k = kwargs.get("k") or kwargs.get("k_cache") or args[0]
    v = kwargs.get("v") or kwargs.get("v_cache") or args[1]

    if q.dim() == 3:
        return manual_attention_forward(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)

    return manual_attention_forward(q, k, v)

# Create mock modules WITH __spec__
mock_flash = types.ModuleType("flash_attn")
mock_interface = types.ModuleType("flash_attn.flash_attn_interface")

mock_flash.__spec__ = ModuleSpec(name="flash_attn", loader=None)
mock_interface.__spec__ = ModuleSpec(name="flash_attn.flash_attn_interface", loader=None)

mock_flash.flash_attn_func = universal_mock
mock_flash.flash_attn_varlen_func = universal_mock
mock_flash.flash_attn_with_kvcache = universal_mock

mock_interface.flash_attn_func = universal_mock
mock_interface.flash_attn_varlen_func = universal_mock
mock_interface.flash_attn_with_kvcache = universal_mock

mock_flash.flash_attn_interface = mock_interface

sys.modules["flash_attn"] = mock_flash
sys.modules["flash_attn.flash_attn_interface"] = mock_interface

warnings.filterwarnings("ignore")

# =========================
# 3. IMPORT WAN
# =========================
from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
import causvid.models.wan.wan_base.modules.model as wan_model_module
import causvid.models.wan.wan_base.modules.attention as wan_attention_module

# =========================
# 4. TRT-SAFE ROPE (NO-OP)
# =========================
def causal_rope_apply_trt(x, grid_sizes, freqs, start_frame=0):
    if freqs.is_complex():
        freqs = torch.view_as_real(freqs).float()
    return x  # safe no-op for export

def force_patch_model(model):
    print(">>>Patching model for TRT...")

    for _, module in model.named_modules():
        if hasattr(module, "freqs") and module.freqs.is_complex():
            real_freqs = torch.view_as_real(module.freqs).float().contiguous()
            del module.freqs
            module.register_buffer("freqs", real_freqs)

    wan_attention_module.flash_attention = universal_mock
    wan_model_module.flash_attention = universal_mock

    for mod in sys.modules.values():
        if hasattr(mod, "causal_rope_apply"):
            mod.causal_rope_apply = causal_rope_apply_trt

# =========================
# 5. PIPELINE WRAPPER
# =========================
class SingleGPUInferencePipeline:
    def __init__(self, config):
        self.pipeline = CausalStreamInferencePipeline(config, device="cpu")

    def load_model(self, checkpoint_folder):
        ckpt = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")
        state_dict = ckpt["generator"] if "generator" in ckpt else ckpt
        self.pipeline.generator.load_state_dict(state_dict, strict=False)

# =========================
# 6. STATELESS TRT WRAPPER
# =========================
class WanTRTWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x, t, context):
        return self.model(
            x,
            t=t,
            context=context,
            seq_len=32760
        )

# =========================
# 7. MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_folder", required=True)
    parser.add_argument("--output_onnx", default="wan_stream_stateless.onnx")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # REQUIRED KEYS
    config.model_type = "T2V-1.3B"
    config.model_name = "wan"
    config.height = 480
    config.width = 832
    config.num_frames = 81
    config.num_kv_cache = 4
    config.num_sink_tokens = 0
    config.adapt_sink_threshold = 0.0

    print(">>> Loading Pipeline...")
    pipeline = SingleGPUInferencePipeline(config)
    pipeline.load_model(args.checkpoint_folder)

    model = pipeline.pipeline.generator.model.to(dtype=torch.float32)
    force_patch_model(model)

    wrapper = WanTRTWrapper(model).eval()

    del pipeline, config
    gc.collect()

    mini_x = torch.randn(1, 16, 1, 60, 104)
    mini_t = torch.tensor([[500, 500]])
    mini_ctx = torch.randn(1, 512, 4096)

    print(">>> Exporting ONNX...")

    torch.onnx.export(
        wrapper,
        (mini_x, mini_t, mini_ctx),
        args.output_onnx,
        opset_version=14,
        input_names=["x", "t", "context"],
        output_names=["flow_output"],
        save_as_external_data=True,
    )

    print(f">>>ONNX Export Complete: {args.output_onnx}")

if __name__ == "__main__":
    main()
