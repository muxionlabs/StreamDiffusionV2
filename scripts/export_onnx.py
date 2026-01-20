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

# 1. MEMORY & THREAD SAFETY
torch.set_grad_enabled(False)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- Mock Flash Attention ---
def manual_attention_forward(q, k, v, **kwargs):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    scale = q_t.size(-1) ** -0.5
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_t)
    return output.transpose(1, 2)

mock_flash = types.ModuleType("flash_attn")
mock_interface = types.ModuleType("flash_attn.flash_attn_interface")
mock_flash.__spec__ = ModuleSpec(name="flash_attn", loader=None)
mock_interface.__spec__ = ModuleSpec(name="flash_attn.flash_attn_interface", loader=None)

def universal_mock(q, *args, **kwargs):
    k = kwargs.get('k') or kwargs.get('k_cache')
    v = kwargs.get('v') or kwargs.get('v_cache')
    if k is None and len(args) > 0: k = args[0]
    if v is None and len(args) > 1: v = args[1]
    if q.dim() == 3:
        return manual_attention_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
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

warnings.filterwarnings("ignore")

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
import causvid.models.wan.wan_base.modules.model as wan_model_module
import causvid.models.wan.wan_base.modules.attention as wan_attention_module

# --- TRT Compatible RoPE Function ---
def causal_rope_apply_trt(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2 
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    if freqs.is_complex(): 
        freqs = torch.view_as_real(freqs).float()
    
    if freqs.dim() == 5 and freqs.shape[-1] == 2:
         freqs_t, freqs_h, freqs_w = freqs.split(split_sizes, dim=3)
    else:
         freqs_t, freqs_h, freqs_w = freqs.split(split_sizes, dim=1)

    output = []
    if isinstance(grid_sizes, torch.Tensor): grid_list = grid_sizes.tolist()
    else: grid_list = grid_sizes
    
    for i, (f, h, w) in enumerate(grid_list):
        seq_len = f * h * w
        sf = start_frame[i].item() if isinstance(start_frame, torch.Tensor) and start_frame.numel() > 1 else (start_frame.item() if isinstance(start_frame, torch.Tensor) else start_frame)
        
        ft = freqs_t[sf : sf + f].reshape(f, 1, 1, -1, 2).expand(f, h, w, -1, 2)
        fh = freqs_h[:h].reshape(1, h, 1, -1, 2).expand(f, h, w, -1, 2)
        fw = freqs_w[:w].reshape(1, 1, w, -1, 2).expand(f, h, w, -1, 2)
        
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
        if current_seq > limit: 
            x_out = torch.cat([x_out, x[i, limit:]], dim=0)
        output.append(x_out)
        
    return torch.stack(output).type_as(x)

def force_patch_model(model):
    print(">>>Patching Complex Ops and RoPE for TRT...")
    for name, module in model.named_modules():
        if hasattr(module, 'freqs') and module.freqs.is_complex():
            real_freqs = torch.view_as_real(module.freqs).float().contiguous()
            del module.freqs
            module.register_buffer('freqs', real_freqs)
            
    wan_attention_module.flash_attention = universal_mock
    wan_model_module.flash_attention = universal_mock

    patched_count = 0
    for mod_name, module in sys.modules.items():
        if hasattr(module, 'causal_rope_apply'):
            print(f"    -> Patching causal_rope_apply in {mod_name}")
            module.causal_rope_apply = causal_rope_apply_trt
            patched_count += 1
    if patched_count == 0:
        print("WARNING: No modules found with 'causal_rope_apply'. Patch might fail!")

class SingleGPUInferencePipeline:
    def __init__(self, config):
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

    def forward(self, x, t, context, kv_flat, cross_flat, cross_init, k_starts, k_ends):
        kv_cache, crossattn_cache = [], []

        for i in range(self.num_layers):
            k, v = torch.split(kv_flat[:, i], 128, dim=-1)
            kv_cache.append({
                "k": k.clone(),
                "v": v.clone(),
                "global_end_index": k_ends,
                "local_end_index": k_ends
            })

        for i in range(self.num_layers):
            k, v = torch.split(cross_flat[:, i], 128, dim=-1)
            crossattn_cache.append({
                "k": k.clone(),
                "v": v.clone(),
                "is_init": cross_init[:, i]
            })

        flow_pred = self.model(
            x, t=t, context=context, seq_len=32760,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=k_starts,
            current_end=k_ends
        )

        new_kv, new_cross, new_init = [], [], []

        for i in range(self.num_layers):
            new_kv.append(torch.cat([kv_cache[i]["k"], kv_cache[i]["v"]], dim=-1).unsqueeze(1))
            new_cross.append(torch.cat([crossattn_cache[i]["k"], crossattn_cache[i]["v"]], dim=-1).unsqueeze(1))
            
            # Boolean <-> Tensor Fix
            val = crossattn_cache[i]["is_init"]
            if isinstance(val, bool):
                val_tensor = torch.tensor([val], device=x.device, dtype=torch.bool)
                new_init.append(val_tensor.unsqueeze(1))
            else:
                new_init.append(val.unsqueeze(1))

        return (
            flow_pred,
            torch.cat(new_kv, dim=1),
            torch.cat(new_cross, dim=1),
            torch.cat(new_init, dim=1),
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_folder", required=True)
    parser.add_argument("--output_onnx", default="wan_stream.onnx")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    
    print(">>> Loading Pipeline...")
    pipeline_manager = SingleGPUInferencePipeline(config)
    pipeline_manager.load_model(args.checkpoint_folder)

    # 1. EXTRACT GENERATOR & PATCH
    model = pipeline_manager.pipeline.generator.model.to(dtype=torch.float32)
    force_patch_model(model)
    wrapper = WanTRTWrapper(model).eval()

    # 2. CLEANUP
    if hasattr(pipeline_manager.pipeline, 'text_encoder'): del pipeline_manager.pipeline.text_encoder
    if hasattr(pipeline_manager.pipeline, 'vae'): del pipeline_manager.pipeline.vae
    if hasattr(pipeline_manager.pipeline, 'tokenizer'): del pipeline_manager.pipeline.tokenizer
    pipeline_manager.pipeline.generator.model = None
    del pipeline_manager
    del config
    gc.collect()
    torch.cuda.empty_cache()

    # 3. DUMMY INPUTS (Corrected Dtypes)
    token_count = 1560 
    
    print(f">>> Creating Dummy Inputs for {token_count} tokens...")
    mini_x = torch.randn(1, 16, 1, 60, 104)
    mini_t = torch.tensor([[500, 500]])
    mini_ctx = torch.randn(1, 512, 4096)
    
    mini_kv = torch.randn(1, 30, token_count, 12, 256)
    mini_cross = torch.randn(1, 30, 512, 12, 256)
    mini_cross_init = torch.zeros(1, 30, dtype=torch.bool)
    
    # FIX: Int32 for Indices (Critical for TRT Shape Tensors)
    mini_start = torch.tensor([0], dtype=torch.int32)
    mini_end = torch.tensor([token_count], dtype=torch.int32)

    print(">>> Exporting ONNX (Lite Mode)...")
    torch.onnx.export(
        wrapper,
        (mini_x, mini_t, mini_ctx, mini_kv, mini_cross, mini_cross_init, mini_start, mini_end),
        args.output_onnx,
        opset_version=14,
        input_names=["x", "t", "context", "kv_cache", "cross_cache", "cross_init", "k_starts", "k_ends"],
        output_names=["flow_output", "kv_cache_updated", "cross_cache_updated", "cross_init_updated"],
        dynamic_axes={
            "x": {0: "batch"},
            "kv_cache": {0: "batch", 2: "seq_len"},
            "cross_cache": {0: "batch", 2: "seq_len"},
            "kv_cache_updated": {0: "batch", 2: "seq_len"},
            "cross_cache_updated": {0: "batch", 2: "seq_len"}
        },
        save_as_external_data=True
    )

    print(f">>>ONNX Export Complete: {args.output_onnx}")

if __name__ == "__main__":
    main()