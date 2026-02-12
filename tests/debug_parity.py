
import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
from omegaconf import OmegaConf

# Import paths
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from causvid.acceleration.tensorrt.engines.simple_engines import DiTEngineStreaming

def run_parity_check(args):
    device = torch.device("cuda")
    
    # 1. Load PyTorch Pipeline (Reference) - Only needed if we rebuild OR verify
    # If we already have the engine and skip rebuild, we can load it later, but let's just be consistent.
    
    print("Loading PyTorch Pipeline...")
    config = OmegaConf.load(args.config)
    
    max_seq_len = 24000
    
    # Inject missing args that are usually CLI args
    config.model_type = "T2V-1.3B"
    config.height = 480
    config.width = 832
    # Ensure other keys if needed
    if not hasattr(config, "model_name"):
         config.model_name = "Wan2.1-T2V-1.3B"
         
    # Force float16 for fair comparison
    # We only load the FULL model if we need to build exporting ONNX.
    
    # Check if engine exists first to save time/memory
    debug_engine_path = "trt_engines_debug/dit_streaming.engine"
    if os.path.exists(debug_engine_path) and not args.force_rebuild:
        print(f"Engine {debug_engine_path} exists. Skipping rebuild.")
        engine_path = debug_engine_path
        # We still need a PT pipe for verification
        pipe_pt = CausalStreamInferencePipeline(config, device=device)
        ckpt_path = os.path.join(args.checkpoint, "model.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            pipe_pt.generator.load_state_dict(state_dict, strict=False)
        pipe_pt.generator.to(device, dtype=torch.float16)
        pipe_pt.text_encoder.to(device, dtype=torch.float16)
    else:
        # Full rebuild path
        pipe_pt = CausalStreamInferencePipeline(config, device=device)
        ckpt_path = os.path.join(args.checkpoint, "model.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            pipe_pt.generator.load_state_dict(state_dict, strict=False)
        pipe_pt.generator.to(device, dtype=torch.float16)
        pipe_pt.text_encoder.to(device, dtype=torch.float16)
        
        # 2. Rebuild TRT Engine on the fly to ensure profile match
        print("Rebuilding TRT Engine (debug)...")
        from causvid.acceleration.tensorrt.builder import EngineBuilder
        builder = EngineBuilder(
            engine_dir="trt_engines_debug",
            model_path=args.checkpoint,
            model_type="T2V-1.3B",
            fp16=True,
            device=str(device),
            force_rebuild=True
        )
        # Build with EXACT constraints of this script
        max_seq_len = 24000
        engine_dict = builder.build_all(
            pipe_pt,
            batch_size=1,
            height=480,
            width=832,
            num_frames=1, # Streaming chunk size
            streaming=True,
            max_seq_len=max_seq_len,
            skip_onnx_optimize=True, # Faster debug build
            skip_t5=True,
            skip_vae=True
        )
        engine_path = engine_dict["dit"]
        
        if args.build_only:
            print("Engine built. Exiting --build_only mode.")
            return

        # CRITICAL: build_all destroys the PyTorch model to save RAM. We must reload it.
        print("Reloading PyTorch Pipeline after engine build...")
        del builder
        if 'pipe_pt' in locals():
            del pipe_pt
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Re-merge config for safety
        config = OmegaConf.load(args.config)
        config.model_type = "T2V-1.3B"
        config.height = 480
        config.width = 832
        if not hasattr(config, "model_name"):
             config.model_name = "Wan2.1-T2V-1.3B"
        
        pipe_pt = CausalStreamInferencePipeline(config, device=device)
        
        # Reload weights logic from above
        ckpt_path = os.path.join(args.checkpoint, "model.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            pipe_pt.generator.load_state_dict(state_dict, strict=False)

        pipe_pt.generator.to(device, dtype=torch.float16)
        pipe_pt.text_encoder.to(device, dtype=torch.float16)
    
    # 2.5 Prepare PyTorch internal state (again)
    # Manually init state for PT
    prompt = "A cinematic video of a futuristic city"
    pipe_pt.conditional_dict = pipe_pt.text_encoder(text_prompts=[prompt])
    # Init KV cache
    batch_size = 1
    pipe_pt._initialize_kv_cache(batch_size=batch_size, dtype=torch.float16, device=device)
    pipe_pt._initialize_crossattn_cache(batch_size=batch_size, dtype=torch.float16, device=device)
    
    # Init hidden states
    pipe_pt.hidden_states = torch.zeros(
        (batch_size, 16, 480//8, 832//8), 
        device=device, 
        dtype=torch.float16
    ).unsqueeze(1) # [B, F, C, H, W]
    
    pipe_pt.kv_cache_starts = torch.zeros((batch_size,), device=device, dtype=torch.long)
    pipe_pt.kv_cache_ends = torch.zeros((batch_size,), device=device, dtype=torch.long)
    pipe_pt.timestep = torch.zeros((batch_size,), device=device, dtype=torch.long)

    print(f"Loading TRT Engine from {engine_path}...")
    engine = DiTEngineStreaming(str(engine_path), device=str(device), use_cuda_graph=False)
    
    # 3. Alloc Memory
    cache_shape = (1, 2, 1, max_seq_len, 12, 128) # 1 layer/chunk
    trt_kv_cache = [torch.zeros(cache_shape, device=device, dtype=torch.float16) for _ in range(30)] # 30 chunks
    
    # 4. Synthesize Inputs
    # 1 Frame input
    bs = 1
    c = 16
    h, w = 60, 104 # 480x832 latents
    noisy_latents = torch.randn(bs, 1, c, h, w, device=device, dtype=torch.float16)
    timestep = torch.tensor([[600]], device=device, dtype=torch.long)
    
    # Fake embeddings (zeros) or run text encoder
    # Let's run text encoder to be sure
    prompt = "A cinematic shot"
    cond_dict = pipe_pt.text_encoder(text_prompts=[prompt])
    pipe_pt.conditional_dict = cond_dict
    context = cond_dict["prompt_embeds"].to(dtype=torch.float16) # [1, 512, 4096]
    
    # 5. Run PyTorch Inference (1 step)
    print("\n--- Running PyTorch (Step 0) ---")
    # PyTorch expects [B, C, T, H, W] for inputs usually? 
    # WanCausalBlock expects x=[B, C, T, H, W]
    # Re-check inference_stream signature
    # def inference_stream(self, noise, current_start, current_end, current_step)
    
    # Manually call generator to inspect internal shapes if needed, 
    # but let's use the public API first.
    
    # We need to hack the PT cache to be empty corresponding to Frame 0
    pipe_pt.kv_cache1 = [
        {'k': torch.zeros(bs, max_seq_len, 12, 128, device=device, dtype=torch.float16), 
         'v': torch.zeros(bs, max_seq_len, 12, 128, device=device, dtype=torch.float16),
         'global_end_index': torch.tensor([0], device=device, dtype=torch.long),
         'local_end_index': torch.tensor([0], device=device, dtype=torch.long)} 
        for _ in range(30)
    ]
    # Initialize crossattn_cache (num_blocks, 512, heads, head_dim)
    pipe_pt.crossattn_cache = []
    for _ in range(pipe_pt.num_transformer_blocks):
        pipe_pt.crossattn_cache.append({
            "k": torch.zeros(bs, 512, pipe_pt.num_heads, 128, dtype=torch.float16, device=device),
            "v": torch.zeros(bs, 512, pipe_pt.num_heads, 128, dtype=torch.float16, device=device),
            "is_init": False,
        })
    
    # Initialize pipeline hidden states and cache pointers (as if prepare was called)
    pipe_pt.hidden_states = torch.zeros(
        (bs, 1, c, h, w), dtype=torch.float16, device=device
    )
    pipe_pt.kv_cache_starts = torch.zeros(bs, dtype=torch.long, device=device)
    pipe_pt.kv_cache_ends = torch.zeros(bs, dtype=torch.long, device=device)
    pipe_pt.timestep = torch.tensor([600], dtype=torch.long, device=device)

    # RUN PT
    with torch.no_grad():
        out_pt = pipe_pt.inference_stream(noisy_latents, current_start=0, current_end=0, current_step=timestep[0,0].item())
    

    # 6. Run TRT Inference (Step 0)
    print("\n--- Running TRT (Step 0) ---")
    current_start_t = torch.tensor([0], device=device, dtype=torch.long)
    start_frame_t = torch.tensor([0], device=device, dtype=torch.long) # Logical frame 0
    trt_context = context
    
    # TRT Input x: [B, F, C, H, W]
    trt_x = noisy_latents
    trt_timestep = timestep
    
    out_trt, _ = engine(
        trt_x,
        trt_timestep,
        trt_context,
        trt_kv_cache,
        current_start_t,
        start_frame_t
    )
    
    
    # TRT Output is [B, C, T, H, W] (1, 16, 1, 60, 104)
    # PT Output is [B, T, C, H, W] (1, 1, 16, 60, 104)
    # Align TRT to PT
    if out_trt.ndim == 5 and out_trt.shape[1] == 16: # B, C, T, H, W
        out_trt = out_trt.permute(0, 2, 1, 3, 4) # -> B, T, C, H, W
            
    if out_trt.shape != out_pt.shape:
        print(f"Shape Mismatch! TRT {out_trt.shape} vs PT {out_pt.shape}")
            
    # Compare
    diff = (out_pt - out_trt).abs()
    print(f"\n[Comparison Step 0]")
    print(f"Max Diff: {diff.max().item():.5f}")
    print(f"Mean Diff: {diff.mean().item():.5f}")
    
    if diff.max() > 1e-1:
        print("!!! HIGH DIVERGENCE !!!")
    else:
        print(">>> MATCH <<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/wan_causal_dmd_v2v.yaml")
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="ckpts/wan_causal_dmd_v2v")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild TRT engine")
    parser.add_argument("--build_only", action="store_true", help="Only build TRT engine and exit")
    args = parser.parse_args()

    run_parity_check(args)
