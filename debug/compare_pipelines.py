import torch
import os
import argparse
import glob
import numpy as np

def compare_tensors(t1, t2, name):
    t1 = t1.float().cpu()
    t2 = t2.float().cpu()
    
    if t1.shape != t2.shape:
        print(f"[{name}] SHAPE MISMATCH: {t1.shape} vs {t2.shape}")
        return
    
    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Cosine Similarity
    t1_flat = t1.view(-1)
    t2_flat = t2.view(-1)
    if t1_flat.norm() == 0 or t2_flat.norm() == 0:
        cos_sim = 0.0
    else:
        cos_sim = torch.nn.functional.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)).item()
        
    print(f"[{name}] Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f} | Cos Sim: {cos_sim:.6f}")
    
    if max_diff > 0.1:
        print(f"    !!! SIGNIFICANT DIVERGENCE DETECTED !!!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str, required=True, help="Reference directory (PyTorch)")
    parser.add_argument("--dir2", type=str, required=True, help="Target directory (TRT)")
    args = parser.parse_args()
    
    # 1. Compare VAE Latents
    f1 = os.path.join(args.dir1, "vae_latents_input.pt")
    f2 = os.path.join(args.dir2, "vae_latents_input.pt")
    if os.path.exists(f1) and os.path.exists(f2):
        print("\n--- Comparing VAE Latents ---")
        compare_tensors(torch.load(f1), torch.load(f2), "VAE_Latents")
    else:
        print("VAE Latents missing in one or both directories.")

    # 2. Compare Steps
    steps1 = sorted(glob.glob(os.path.join(args.dir1, "denoised_step_*.pt")))
    steps2 = sorted(glob.glob(os.path.join(args.dir2, "denoised_step_*.pt")))
    
    print(f"\n--- Comparing Denoising Steps ({len(steps1)} vs {len(steps2)}) ---")
    
    for s1, s2 in zip(steps1, steps2):
        name = os.path.basename(s1)
        if name != os.path.basename(s2):
            print(f"Filename mismatch: {s1} vs {s2}")
            break
            
        t1 = torch.load(s1)
        t2 = torch.load(s2)
        compare_tensors(t1, t2, name)

    # 3. Compare Output
    f1 = os.path.join(args.dir1, "vae_decoded_output.pt")
    f2 = os.path.join(args.dir2, "vae_decoded_output.pt")
    if os.path.exists(f1) and os.path.exists(f2):
        print("\n--- Comparing VAE Output ---")
        compare_tensors(torch.load(f1), torch.load(f2), "VAE_Output")

if __name__ == "__main__":
    main()
