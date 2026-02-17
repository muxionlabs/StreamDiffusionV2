
import torch
import sys

def inspect_checkpoint(ckpt_path):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    print(f"Top-level keys: {list(ckpt.keys())}")
    
    if isinstance(ckpt, dict) and 'generator' in ckpt:
        sd = ckpt['generator']
        print("Using 'generator' key.")
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        print("Using 'state_dict' key.")
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        sd = ckpt['model']
        print("Using 'model' key.")
    else:
        sd = ckpt
        print("Using root dictionary as state_dict.")
        
    print(f"Total keys in state_dict: {len(sd)}")
    print("First 20 keys:")
    for i, k in enumerate(list(sd.keys())[:20]):
        print(f"  {k}")

    print("\n--- Searching for 'text_embedding' ---")
    for k in sd.keys():
        if "text_embedding" in k:
            print(f"{k}: {sd[k].shape}")
            
    print("\n--- Searching for 'cross_attn' (Block 0) ---")
    for k in sd.keys():
        if "blocks.0" in k and "cross_attn" in k:
            print(f"{k}: {sd[k].shape}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug/inspect_ckpt.py <path_to_ckpt>")
        sys.exit(1)
    inspect_checkpoint(sys.argv[1])
