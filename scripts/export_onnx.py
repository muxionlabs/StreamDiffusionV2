"""
Phase 1: The Wrapper & Export (Baked 480p Strategy)
Function: Exports Wan2.1 to ONNX with FIXED 480p resolution.
          1. Sets input size to exactly 832x480 (Latent: 104x60).
          2. Avoids dynamic shape bugs by baking the target resolution.
          3. Uses CPU to handle the large trace without VRAM OOM.
"""

import sys
import types
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib.machinery import ModuleSpec

# ==============================================================================
#  1. MANUAL ATTENTION (CPU SAFE)
# ==============================================================================
def manual_attention_forward(q, k, v, **kwargs):
    # q, k, v: [Batch, Seq, Heads, Dim] -> Transpose to [Batch, Heads, Seq, Dim]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    
    scale = q_t.size(-1) ** -0.5
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_t)
    
    return output.transpose(1, 2)

# ==============================================================================
#  2. MOCK FLASH ATTENTION LIB
# ==============================================================================
mock_flash = types.ModuleType("flash_attn")
mock_interface = types.ModuleType("flash_attn.flash_attn_interface")

mock_flash.__spec__ = ModuleSpec(name="flash_attn", loader=None)
mock_flash.__file__ = "mock_flash_attn.py"
mock_flash.__path__ = []
mock_interface.__spec__ = ModuleSpec(name="flash_attn.flash_attn_interface", loader=None)
mock_interface.__file__ = "mock_flash_interface.py"

def universal_mock(q, *args, **kwargs):
    k = kwargs.get('k') or kwargs.get('k_cache')
    v = kwargs.get('v') or kwargs.get('v_cache')
    if k is None and len(args) > 0: k = args[0]
    if v is None and len(args) > 1: v = args[1]
    if q.dim() == 3: return manual_attention_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
    return manual_attention_forward(q, k, v)

mock_flash.flash_attn_func = universal_mock
mock_flash.flash_attn_varlen_func = universal_mock
mock_flash.flash_attn_with_kvcache = universal_mock
mock_interface.flash_attn_func = universal_mock
mock_interface.flash_attn_varlen_func = universal_mock
mock_interface.flash_attn_with_kvcache = universal_mock
mock_flash.flash_attn_interface = mock_interface

sys.modules["flash_attn"] = mock_flash
sys.modules["flash_attn.flash_attn_interface"] = mock_interface

print(">>>Flash Attention Lib intercepted.")

# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
import argparse
import gc
from omegaconf import OmegaConf

os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCH_ONNX_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
import causvid.models.wan.causal_model as causal_module 
import causvid.models.wan.wan_base.modules.model as wan_model_module
import causvid.models.wan.wan_base.modules.attention as wan_attention_module

# ==============================================================================
#  3. PATCH THE MODEL
# ==============================================================================
def causal_rope_apply_trt(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2 
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    if freqs.is_complex(): freqs = torch.view_as_real(freqs).float()
    freqs_t, freqs_h, freqs_w = freqs.split(split_sizes, dim=1)
    output = []
    if isinstance(grid_sizes, torch.Tensor): grid_list = grid_sizes.tolist()
    else: grid_list = grid_sizes
    for i, (f, h, w) in enumerate(grid_list):
        seq_len = f * h * w
        sf = start_frame[i].item() if isinstance(start_frame, torch.Tensor) and start_frame.numel() > 1 else (start_frame.item() if isinstance(start_frame, torch.Tensor) else start_frame)
        ft = freqs_t[sf : sf + f].view(f, 1, 1, -1, 2).expand(f, h, w, -1, 2)
        fh = freqs_h[:h].view(1, h, 1, -1, 2).expand(f, h, w, -1, 2)
        fw = freqs_w[:w].view(1, 1, w, -1, 2).expand(f, h, w, -1, 2)
        freqs_i = torch.cat([ft, fh, fw], dim=-2).reshape(seq_len, 1, -1, 2)
        current_seq = x.shape[1] 
        limit = min(seq_len, current_seq)
        x_i_slice = x[i, :limit].float()
        x_i_reshaped = x_i_slice.reshape(limit, n, -1, 2)
        re_x, im_x = x_i_reshaped.unbind(-1)
        cos_f, sin_f = freqs_i[:limit].unbind(-1)
        re_out = re_x * cos_f - im_x * sin_f
        im_out = re_x * sin_f + im_x * cos_f
        x_out = torch.stack([re_out, im_out], dim=-1).flatten(2)
        if current_seq > limit: x_out = torch.cat([x_out, x[i, limit:]], dim=0)
        output.append(x_out)
    return torch.stack(output).type_as(x)

def force_patch_model(model):
    print(">>>APPLYING PATCHES...")
    for name, module in model.named_modules():
        if hasattr(module, 'freqs') and module.freqs.is_complex():
            real_freqs = torch.view_as_real(module.freqs).float().contiguous()
            del module.freqs
            module.register_buffer('freqs', real_freqs)
    causal_module.causal_rope_apply = causal_rope_apply_trt
    wan_attention_module.flash_attention = universal_mock
    wan_model_module.flash_attention = universal_mock
    print("Patches applied.")

# ==============================================================================
#  WRAPPER
# ==============================================================================
class SingleGPUInferencePipeline:
    def __init__(self, config):
        self.config = config
        self.pipeline = CausalStreamInferencePipeline(config, device="cpu") 
    def load_model(self, checkpoint_folder: str):
        ckpt = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")
        state_dict = ckpt['generator'] if 'generator' in ckpt else ckpt
        self.pipeline.generator.load_state_dict(state_dict, strict=False)

class WanTRTWrapper(nn.Module):
    def __init__(self, original_model, num_layers=30):
        super().__init__()
        self.model = original_model
        self.num_layers = num_layers
    def forward(self, x, t, context, kv_flat, cross_flat):
        dummy_idx = torch.tensor([0], device=x.device, dtype=torch.long)
        kv_cache = []
        for i in range(self.num_layers):
            k, v = torch.split(kv_flat[:, i], 128, dim=-1)
            kv_cache.append({"k": k.clone(), "v": v.clone(), "global_end_index": dummy_idx, "local_end_index": dummy_idx})
        crossattn_cache = []
        for i in range(self.num_layers):
            k, v = torch.split(cross_flat[:, i], 128, dim=-1)
            crossattn_cache.append({"k": k.clone(), "v": v.clone(), "is_init": True})
        flow_pred = self.model(x, t=t, context=context, seq_len=32760, kv_cache=kv_cache, crossattn_cache=crossattn_cache, current_start=dummy_idx, current_end=dummy_idx)
        new_kv = []
        for i in range(self.num_layers):
            new_kv.append(torch.cat([kv_cache[i]['k'], kv_cache[i]['v']], dim=-1).unsqueeze(1))
        return flow_pred, torch.cat(new_kv, dim=1)

# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_onnx", type=str, default="wan_stream.onnx")
    parser.add_argument("--output_folder", type=str, default="debug_out") 
    parser.add_argument("--prompt_file_path", type=str, default="prompt.txt")
    parser.add_argument("--video_path", type=str, required=False, default=None)
    parser.add_argument("--image_path", type=str, required=False, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.7)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=2) 
    parser.add_argument("--model_type", type=str, default="T2V-1.3B")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--fixed_noise_scale", action="store_true", default=False)
    parser.add_argument("--img2img", action="store_true", default=False)

    args = parser.parse_args()
    device = torch.device("cpu")
    
    print(">>> Loading Model (CPU Mode)...")
    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    
    pipeline_manager = SingleGPUInferencePipeline(config)
    pipeline_manager.load_model(args.checkpoint_folder)
    dit_model = pipeline_manager.pipeline.generator.model.to(dtype=torch.float32)
    force_patch_model(dit_model)
    
    print(">>> Wrapping Model...")
    trt_wrapper = WanTRTWrapper(dit_model).to(device)
    trt_wrapper.eval()

    # --- KEY CHANGE: TARGET 480p SHAPES ---
    # 832 / 8 = 104
    # 480 / 8 = 60
    # Latent Area = 6240
    print(">>> Generating 480p Dummy Tensors (104x60 latents)...")
    # x: [Batch, Channel, Time, Height, Width]
    mini_x = torch.randn(1, 16, 1, 60, 104, device=device, dtype=torch.float32)
    mini_t = torch.tensor([[500, 500]], device=device, dtype=torch.long)
    mini_ctx = torch.randn(1, 512, 4096, device=device, dtype=torch.float32)
    # kv_cache: [Batch, Layers, SeqLen, Heads, HeadDim]
    # SeqLen = Time * H * W = 1 * 60 * 104 = 6240
    mini_kv = torch.randn(1, 30, 6240, 12, 256, device=device, dtype=torch.float32) 
    mini_cross = torch.randn(1, 30, 512, 12, 256, device=device, dtype=torch.float32)

    print(">>> Cleaning RAM...")
    del pipeline_manager
    gc.collect()

    # Only define DYNAMIC BATCH, NOT dynamic spatial dims
    dynamic_axes = {
        "x": {0: "batch", 2: "time"}, # Fixed H/W
        "t": {0: "batch"},
        "context": {0: "batch"},
        "kv_cache": {0: "batch"}, # Fixed SeqLen
        "cross_cache": {0: "batch"},
        "flow_output": {0: "batch", 2: "time"},
        "kv_cache_updated": {0: "batch"}
    }

    print(f">>> Exporting to {args.output_onnx} (CPU / 480p FIXED)...")
    
    torch.onnx.export(
        trt_wrapper,
        (mini_x, mini_t, mini_ctx, mini_kv, mini_cross),
        args.output_onnx,
        export_params=True,
        opset_version=14, 
        do_constant_folding=False, 
        input_names=["x", "t", "context", "kv_cache", "cross_cache"],
        output_names=["flow_output", "kv_cache_updated"],
        dynamic_axes=dynamic_axes, 
        save_as_external_data=True,
        keep_initializers_as_inputs=False,
        verbose=False
    )
    
    print(f">>>ONNX Export Complete! Check for {args.output_onnx}")

if __name__ == "__main__":
    main()