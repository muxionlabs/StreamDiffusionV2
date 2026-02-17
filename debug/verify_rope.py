import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from causvid.models.wan.wan_base.modules.model import rope_params as rope_params_orig
from causvid.acceleration.tensorrt.trt_wrapper import TRTWanDiffusionWrapper

def test_rope_parity():
    print("--- Testing RoPE Parity ---")
    device = "cuda"
    max_seq_len = 24000 # Typical for video
    head_dim = 64
    
    # 1. Reference (Original)
    # The original model calculates 'freqs' using rope_params
    full_freqs_ref = rope_params_orig(max_seq_len, head_dim).to(device)
    
    # 2. Target (TRT Wrapper Logic)
    # We want to see how TRT Wrapper generates these.
    # We can inspect the code or instantiate a dummy wrapper context
    
    # Test with correct dimensions found in logs
    # Log: [DEBUG_ROPE] freqs_t shape: torch.Size([1024, 22]) -> implies c_t=22
    # If c_t=22, then half_dim must be around 64.
    # half_dim=64 -> head_dim=128.
    
    test_dims = [64, 128]
    
    for head_dim in test_dims:
        print(f"\n--- Testing head_dim={head_dim} ---")
        half_dim = head_dim // 2
        print(f"half_dim: {half_dim}")
        
        # Reference Splits
        c = half_dim
        split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
        print(f"Reference Split Sizes: {split_sizes}")
        
        # TRT Logic
        trt_c_h = half_dim // 3
        trt_c_w = half_dim // 3
        trt_c_t = half_dim - 2 * (half_dim // 3)
        trt_splits = [trt_c_t, trt_c_h, trt_c_w]
        print(f"TRT Split Sizes: {trt_splits}")
        
        if split_sizes == trt_splits:
            print(">> MATCH")
        else:
            print(">> MISMATCH")
            
        # Verify Values
        # trt_params = rope_params_orig(max_seq_len, head_dim)
        # We can implement the 'manual' calculation here to see if it matches PyTorch's rope_params
        
        if head_dim == 128:
             # Check if [22, 21, 21] logic holds
             if trt_splits == [22, 21, 21]:
                 print(">> Confirmed: head_dim=128 produces the [22, 21, 21] split seen in logs.")


if __name__ == "__main__":
    test_rope_parity()
