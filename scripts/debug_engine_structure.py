import tensorrt as trt
import torch
import cv2
import numpy as np
import os
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def test_structure_safe(engine_path, video_path):
    print(f"Safe Structure Test: {video_path} -> {engine_path}")
    
    if not os.path.exists(engine_path):
        print("Engine not found.")
        return

    # 1. Load Engine
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    # 2. Get Input Frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read video frame.")
        return

    # 3. Setup Inputs (Logic copied from working Raw script)
    stream = torch.cuda.current_stream()
    inputs = []
    outputs = []
    
    # These matched your successful run logs
    shapes = {
        "x": (1, 16, 1, 60, 104),
        "t": (1, 2),
        "context": (1, 128, 1024),
        "kv_cache": (1, 30, 6240, 12, 256),
        "cross_cache": (1, 30, 512, 12, 256)
    }

    # Resize frame for injection
    latent_h, latent_w = 60, 104
    small_frame = cv2.resize(frame, (latent_w, latent_h))

    print("--- Binding Tensors ---")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        dtype = engine.get_tensor_dtype(name)
        
        # Get shape
        base_shape = engine.get_tensor_shape(name)
        # Resolve dynamic dims using our known good shapes
        final_shape = tuple([s if s != -1 else shapes[name][k] for k, s in enumerate(base_shape)])
        
        print(f"Binding {name}: {final_shape}")

        # Create Tensor
        torch_dtype = torch.float16 if dtype == trt.float16 else torch.float32
        
        if is_input:
            if name == "x":
                # === INJECTION ===
                tensor = torch.zeros(final_shape, device="cuda", dtype=torch_dtype)
                # Normalize 0-255 -> -1 to 1
                rgb = torch.from_numpy(small_frame).cuda().float().permute(2, 0, 1)
                rgb = (rgb / 127.5) - 1.0
                # Inject into first 3 channels
                tensor[0, 0:3, 0, :, :] = rgb.to(dtype=torch_dtype)
            elif name == "t":
                tensor = torch.tensor([[500, 500]], device="cuda", dtype=torch.int64)
            else:
                tensor = torch.randn(final_shape, device="cuda", dtype=torch_dtype)
            
            context.set_input_shape(name, final_shape)
        else:
            tensor = torch.empty(final_shape, device="cuda", dtype=torch_dtype)
            outputs.append((name, tensor))

        context.set_tensor_address(name, tensor.data_ptr())

    # 4. Run
    print("Running Engine (Safe Mode)...")
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # 5. Visualize
    print("Generating Output...")
    
    # Find output tensor
    target_tensor = outputs[0][1]
    for n, t in outputs:
        if "flow" in n or "output" in n: target_tensor = t; break
            
    data = target_tensor.squeeze().float().cpu().numpy() # [16, 60, 104]
    
    # Map first 3 channels to RGB for structure check
    ch0 = data[0]
    ch1 = data[1]
    ch2 = data[2]
    
    # Normalize for display
    def norm(x): return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    heatmap = cv2.merge([norm(ch0), norm(ch1), norm(ch2)])
    heatmap = cv2.resize(heatmap, (512, 300), interpolation=cv2.INTER_NEAREST)
    
    # Input ref
    input_ref = cv2.resize(frame, (512, 300))
    
    # Stitch
    combined = np.hstack([input_ref, heatmap])
    cv2.putText(combined, "Input", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(combined, "Engine Output (Latent Structure)", (522,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    cv2.imwrite("debug_structure_safe.png", combined)
    print(f"Saved: debug_structure_safe.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original.mp4")
    parser.add_argument("--engine", default="wan_stream.engine")
    args = parser.parse_args()
    
    test_structure_safe(args.engine, args.input)