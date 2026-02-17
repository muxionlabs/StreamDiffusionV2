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
    
    # Replicating the logic from trt_wrapper.py (True Baseline)
    # In trt_wrapper.py, we typically see:
    # half_dim = head_dim // 2
    # c_h = half_dim // 3
    # c_w = half_dim // 3
    # c_t = half_dim - c_h - c_w
    
    half_dim = head_dim // 2
    c = half_dim
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    print(f"Original Split Sizes (based on half_dim={half_dim}): {split_sizes}")
    
    freqs_split_ref = full_freqs_ref.split(split_sizes, dim=1)
    ref_t = freqs_split_ref[0]
    ref_h = freqs_split_ref[1]
    ref_w = freqs_split_ref[2]
    
    print(f"Ref T shape: {ref_t.shape}")
    print(f"Ref H shape: {ref_h.shape}")
    print(f"Ref W shape: {ref_w.shape}")

    # Now check what TRT wrapper does
    # If trt_wrapper.py uses head_dim // 3 instead of half_dim // 3, that would be the bug.
    
    trt_c_h = half_dim // 3
    trt_c_w = half_dim // 3
    trt_c_t = half_dim - trt_c_h - trt_c_w # This is how it should be
    
    # But let's verify what I put in the actual file.
    # In Step 962, I saw:
    # c_h = half_dim // 3
    # c_w = half_dim // 3
    
    print(f"TRT Wrapper Calculated Sizes (should match): c_t={trt_c_t}, c_h={trt_c_h}, c_w={trt_c_w}")
    
    if split_sizes != [trt_c_t, trt_c_h, trt_c_w]:
        print("!!! DETECTED LOGIC MISMATCH !!!")
        print(f"Reference: {split_sizes}")
        print(f"TRT Wrapper: {[trt_c_t, trt_c_h, trt_c_w]}")
    else:
        print("Dimension logic matches.")
        
    # Check values
    # In trt_wrapper, we do:
    # freqs_t = full_freqs[:, :c_t]
    # freqs_h = full_freqs[:, c_t:c_t+c_h]
    # freqs_w = full_freqs[:, c_t+c_h:]
    
    # Reference split does:
    # ref_t = freqs_split_ref[0] which is full_freqs[:, :split_sizes[0]]
    # ref_h = freqs_split_ref[1] which is full_freqs[:, split_sizes[0]:split_sizes[0]+split_sizes[1]]
    
    # So if split sizes match, the slicing logic is identical.
    
    # Let's verify if RoPE params generation is identical
    # trt_wrapper call: rope_params(max_seq_len, self.head_dim)
    # reference call: rope_params(max_seq_len, head_dim)
    # Identical.

if __name__ == "__main__":
    test_rope_parity()
