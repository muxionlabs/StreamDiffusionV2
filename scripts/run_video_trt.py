"""
Function: Runs real video-to-video inference.
          - Explicitly moves VAE to GPU to fix RuntimeErrors.
          - Uses 'stream_encode' for Wan compatibility.
          - Runs TensorRT 12 FPS engine.
"""

import tensorrt as trt
import torch
import cv2
import numpy as np
import argparse
import os
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

# Import your existing wrappers
from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class TRTModel:
    def __init__(self, engine_path, device="cuda"):
        print(f"Loading TensorRT Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()
        self.inputs = {}
        self.outputs = {}
        self.allocations = []
        
        # Universal Binding Setup
        num_io = self.engine.num_io_tensors if hasattr(self.engine, "num_io_tensors") else self.engine.num_bindings
        for i in range(num_io):
            if hasattr(self.engine, "get_tensor_name"):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            else:
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = self.engine.get_binding_dtype(i)
                is_input = self.engine.binding_is_input(i)

            torch_dtype = torch.float32
            if dtype == trt.float16: torch_dtype = torch.float16
            elif dtype == trt.int32: torch_dtype = torch.int32
            elif dtype == trt.int64: torch_dtype = torch.int64

            tensor = torch.empty(tuple(shape), dtype=torch_dtype, device=device)
            self.allocations.append(tensor) 
            self.context.set_tensor_address(name, tensor.data_ptr())
            
            if is_input: self.inputs[name] = tensor
            else: self.outputs[name] = tensor

    def run(self, feed_dict):
        for name, tensor in feed_dict.items():
            if name in self.inputs:
                if self.inputs[name].dtype != tensor.dtype:
                    self.inputs[name].copy_(tensor.to(self.inputs[name].dtype))
                else:
                    self.inputs[name].copy_(tensor)
        
        self.context.execute_async_v3(self.stream.cuda_stream)
        return self.outputs

def get_text_embeddings(pipeline, prompt, device):
    """
    Robustly encode text, handling Dict/Tuple/Object return types.
    """
    encoder = pipeline.text_encoder
    encoder.to(device)
    encoder.eval()
    
    try:
        result = encoder([prompt])
    except:
        tokenizer = getattr(pipeline, "tokenizer", None)
        if not tokenizer and hasattr(encoder, "tokenizer"): tokenizer = encoder.tokenizer
        
        if tokenizer:
            tokens = tokenizer([prompt], return_tensors="pt", padding="max_length", max_length=512, truncation=True)
            input_ids = tokens.input_ids.to(device)
            result = encoder(input_ids)
        else:
            raise RuntimeError("Could not run text encoder.")

    if isinstance(result, dict):
        if "last_hidden_state" in result: return result["last_hidden_state"]
        elif "pooler_output" in result: return result["pooler_output"]
        return list(result.values())[0]

    if isinstance(result, (tuple, list)): return result[0]
    if hasattr(result, "last_hidden_state"): return result.last_hidden_state
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wan_causal_dmd_v2v.yaml")
    parser.add_argument("--ckpt", default="ckpts/wan_causal_dmd_v2v")
    parser.add_argument("--engine", default="wan_stream.engine")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="trt_final.mp4")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    
    # Dummy args
    parser.add_argument("--noise_scale", type=float, default=0.7)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fixed_noise_scale", action="store_true")
    parser.add_argument("--img2img", action="store_true")
    
    args = parser.parse_args()

    device = "cuda"
    VAE_SCALING_FACTOR = 0.13025 
    
    print("Loading PyTorch components...")
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    config.model_type = "T2V-1.3B" 
    
    pt_pipe = CausalStreamInferencePipeline(config, device=device)
    ckpt = torch.load(os.path.join(args.ckpt, "model.pt"), map_location="cpu")
    pt_pipe.generator.load_state_dict(ckpt, strict=False)
    
    # --- FIXED: Explicitly move VAE to GPU ---
    print("Moving VAE to CUDA...")
    pt_pipe.vae.to(device)
    pt_pipe.vae.eval()
    
    # Free up VRAM (Delete PyTorch DiT)
    pt_pipe.generator.model = None 
    torch.cuda.empty_cache()
    
    # Load TensorRT
    trt_engine = TRTModel(args.engine, device=device)

    # Setup Inputs
    prompt = "A dog walks on the grass, realistic"
    print(f"Encoding prompt: '{prompt}'")
    
    context = get_text_embeddings(pt_pipe, prompt, device)
    context = context.to(dtype=torch.float32)
    
    # Cache
    kv_cache = torch.zeros(1, 30, 6240, 12, 256, device=device, dtype=torch.float32)
    cross_cache = torch.zeros(1, 30, 512, 12, 256, device=device, dtype=torch.float32)
    t = torch.tensor([[500, 500]], device=device, dtype=torch.int64)

    # Video Loop
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (args.width, args.height))
    
    print("Streaming Inference...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((args.width, args.height))
        pixel_values = transforms.ToTensor()(img).unsqueeze(0).to(device)
        pixel_values = pixel_values * 2.0 - 1.0
        
        with torch.no_grad():
            # [B, C, T, H, W]
            inp_video = pixel_values.unsqueeze(2)
            
            # --- FIXED: VAE is now on CUDA, so this call will work ---
            latents = pt_pipe.vae.stream_encode(inp_video)
            
            # Helper: handle variable return shapes
            if latents.shape[1] != 16 and latents.shape[2] == 16:
                 latents = latents.transpose(1, 2)
                 
            # Apply Scaling & Cast
            latents = (latents * VAE_SCALING_FACTOR).to(dtype=torch.float32)

        # TRT Step
        outputs = trt_engine.run({
            "x": latents, "t": t, "context": context,
            "kv_cache": kv_cache, "cross_cache": cross_cache
        })
        
        denoised = outputs["flow_output"]
        kv_cache.copy_(outputs["kv_cache_updated"])
        
        # Decode
        with torch.no_grad():
            decoded = pt_pipe.vae.stream_decode_to_pixel(denoised / VAE_SCALING_FACTOR)
            
        decoded_np = decoded.squeeze().permute(1, 2, 0).cpu().numpy()
        decoded_np = (decoded_np * 0.5 + 0.5).clip(0, 1) * 255
        out.write(cv2.cvtColor(decoded_np.astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(".", end="", flush=True)

    cap.release()
    out.release()
    print(f"Video saved to {args.output}")

if __name__ == "__main__":
    main()