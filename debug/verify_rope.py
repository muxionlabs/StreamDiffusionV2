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
    c_t = head_dim // 3
    c_h = head_dim // 3
    c_w = head_dim // 3 # roughly
    
    # In trt_wrapper.py:
    # full_freqs = rope_params(max_seq_len, self.head_dim).to(device)
    # freqs_t = full_freqs[:, :c_t]
    # freqs_h = full_freqs[:, c_t:c_t+c_h]
    # freqs_w = full_freqs[:, c_t+c_h:]
    
    # Let's verify the slice indices match exactly what the model expects
    # In wan/wan_base/modules/model.py:
    # freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    c = head_dim
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    print(f"Original Split Sizes: {split_sizes}")
    
    freqs_split_ref = full_freqs_ref.split(split_sizes, dim=1)
    ref_t = freqs_split_ref[0]
    ref_h = freqs_split_ref[1]
    ref_w = freqs_split_ref[2]
    
    print(f"Ref T shape: {ref_t.shape}")
    print(f"Ref H shape: {ref_h.shape}")
    print(f"Ref W shape: {ref_w.shape}")

    # Now check what TRT wrapper does (based on my recent edit)
    # c_t = head_dim // 3
    # c_h = head_dim // 3
    # c_w = head_dim // 3 NOTE: This might be the bug! 
    # If head_dim=64, 64//3 = 21. 
    # 21 + 21 + 21 = 63. Missing 1 dimension!
    
    trt_c_t = head_dim // 3
    trt_c_h = head_dim // 3
    trt_c_w = head_dim // 3
    
    print(f"TRT Wrapper Calculated Sizes: c_t={trt_c_t}, c_h={trt_c_h}, c_w={trt_c_w}")
    print(f"Sum: {trt_c_t + trt_c_h + trt_c_w} vs Expected {head_dim}")
    
    if (trt_c_t + trt_c_h + trt_c_w) != head_dim:
        print("!!! DETECTED DIMENSION MISMATCH IN TRT WRAPPER LOGIC !!!")
        print("Model split logic: [c - 2*(c//3), c//3, c//3]")
        print(f"Model T size: {split_sizes[0]}")
        
    else:
        print("Dimension logic matches (unexpected based on my manual calculation)")

if __name__ == "__main__":
    test_rope_parity()
