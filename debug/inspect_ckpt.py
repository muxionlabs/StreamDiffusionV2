
import torch
import sys

def inspect_checkpoint(ckpt_path):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if "model." in list(ckpt.keys())[0]:
        print("Prefix 'model.' detected.")
        
    print("\n--- Text Embedding Keys ---")
    for k in ckpt.keys():
        if "text_embedding" in k:
            print(f"{k}: {ckpt[k].shape}")
            
    print("\n--- Cross Attention Keys (Block 0) ---")
    for k in ckpt.keys():
        if "blocks.0.cross_attn" in k:
            print(f"{k}: {ckpt[k].shape}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug/inspect_ckpt.py <path_to_ckpt>")
        sys.exit(1)
    inspect_checkpoint(sys.argv[1])
