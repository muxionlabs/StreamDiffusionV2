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
        
        # GENERATE REFERENCE VALUES
        full_freqs_ref = rope_params_orig(max_seq_len, head_dim).to(device)
        freqs_split_ref = full_freqs_ref.split(split_sizes, dim=1)
        ref_t = freqs_split_ref[0]
        ref_h = freqs_split_ref[1]
        ref_w = freqs_split_ref[2]
        
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
                 
             # Value Inspection for Numerical Stability
             print("\n--- Inspecting Values ---")
             
             # Reference Frequencies
             freqs_t, freqs_h, freqs_w = ref_t, ref_h, ref_w
             
             # Calculate magnitudes (should be 1.0 for polar form? No, these are freqs * pos)
             # Wait, rope_params returns complex numbers: exp(i * theta)
             # theta = pos * freq
             # absolute value is always 1.0. This is rotational.
             # So freqs themselves ARE NOT the issue for overflow/underflow in values, but their rotations are.
             # Wait, in the TRT wrapper we pass:
             # 'rope_cos_t': freqs_t.real.to(dtype=torch.float32)
             # 'rope_sin_t': freqs_t.imag.to(dtype=torch.float32)
             
             # Let's check min/max/mean of the real/imag parts.
             # They should be in [-1, 1].
             
             print(f"Freqs T (Real): Min={freqs_t.real.min():.4f}, Max={freqs_t.real.max():.4f}, Mean={freqs_t.real.mean():.4f}")
             print(f"Freqs H (Real): Min={freqs_h.real.min():.4f}, Max={freqs_h.real.max():.4f}, Mean={freqs_h.real.mean():.4f}")
             print(f"Freqs W (Real): Min={freqs_w.real.min():.4f}, Max={freqs_w.real.max():.4f}, Mean={freqs_w.real.mean():.4f}")
             
             # But wait! rope_params returns the PRECOMPUTED COS/SIN factors (i.e., rotation matrices).
             # It computes:
             # freqs = torch.outer(pos, 1.0 / theta^(2i/d))
             # then returns polar(1, freqs).
             # So the values ARE ALWAYS ON THE UNIT CIRCLE.
             # Their magnitude IS 1.0.
             
             mag_t = freqs_t.abs()
             mag_h = freqs_h.abs()
             mag_w = freqs_w.abs()
             
             print(f"Freqs T Magnitude: Min={mag_t.min():.6f}, Max={mag_t.max():.6f}")
             print(f"Freqs H Magnitude: Min={mag_h.min():.6f}, Max={mag_h.max():.6f}")
             print(f"Freqs W Magnitude: Min={mag_w.min():.6f}, Max={mag_w.max():.6f}")
             
             # If magnitude != 1.0, then something is wrong with rope_params.
             # If magnitude == 1.0, then INPUTS to TRT are fine (range [-1, 1]).
             # The instability comes from the Attention operation ITSELF using these rotations.
             
             # Alternatively, maybe the frequencies (theta values) are causing aliasing?
             # Let's inspect the RAW THETA values if possible? 
             # No, we only have the complex outputs.
             
             pass


if __name__ == "__main__":
    test_rope_parity()
