import tensorrt as trt
import torch
import numpy as np
import time

# Config
ENGINE_PATH = "wan_dit.engine"
DEVICE = "cuda"

class TRTWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        try:
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()
            with trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        except FileNotFoundError:
            print(f"CRITICAL ERROR: {engine_path} not found.")
            exit(1)
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        
        self.bindings = [None] * self.engine.num_io_tensors
        self.inputs = {}
        self.outputs = {}
        self.name_map = {} 

        print("\n[Engine Inspection] Scanning binding names...")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            
            # Map TRT dtype to Torch dtype
            torch_dtype = torch.float16 if dtype == trt.DataType.HALF else torch.float32
            if dtype == trt.DataType.INT32: torch_dtype = torch.int32
            
            print(f"  - Binding {i}: Name='{name}', Mode={mode}, Shape={shape}, Dtype={dtype}")

            dims = len(shape)
            
            # 1. Output
            if mode == trt.TensorIOMode.OUTPUT:
                self.name_map["output"] = name
                # Fix for dynamic output shape
                alloc_shape = tuple(s if s > 0 else 1 for s in shape)
                if alloc_shape == (1, 1, 1, 1, 1): alloc_shape = (1, 16, 1, 60, 106)
                
                tensor = torch.zeros(alloc_shape, dtype=torch_dtype, device=DEVICE)
                self.outputs[name] = tensor
                
            # 2. Inputs
            else:
                alloc_shape = list(shape)
                tensor = torch.zeros(tuple(alloc_shape), dtype=torch_dtype, device=DEVICE)
                self.inputs[name] = tensor
                
                # Identify role by dimensions
                if dims == 6: self.name_map["cache"] = name
                elif dims == 5: self.name_map["x"] = name
                elif dims == 3: self.name_map["context"] = name
                elif dims == 1: self.name_map["t"] = name
            
            self.context.set_tensor_address(name, tensor.data_ptr())
            self.bindings[i] = tensor.data_ptr()

    def infer(self, x, t, context, cache):
        nm = self.name_map
        
        # Robust Copy: Only copy if the engine actually asks for it
        if "x" in nm: self.inputs[nm["x"]].copy_(x)
        if "t" in nm: self.inputs[nm["t"]].copy_(t)
        if "context" in nm: self.inputs[nm["context"]].copy_(context)
        if "cache" in nm: self.inputs[nm["cache"]].copy_(cache)
        
        # Robust Shape Setting
        if "x" in nm: self.context.set_input_shape(nm["x"], x.shape)
        if "t" in nm: self.context.set_input_shape(nm["t"], t.shape)
        if "context" in nm: self.context.set_input_shape(nm["context"], context.shape)
        if "cache" in nm: self.context.set_input_shape(nm["cache"], cache.shape)
        
        self.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()
        
        return self.outputs[nm["output"]]

if __name__ == "__main__":
    print(f"--- Testing TensorRT Engine: {ENGINE_PATH} ---")
    model = TRTWrapper(ENGINE_PATH)
    print("Engine loaded and mapped successfully.")
    
    print("Generating inputs...")
    x = torch.randn(1, 16, 1, 60, 106, dtype=torch.float16, device=DEVICE)
    t = torch.tensor([1.0], dtype=torch.float16, device=DEVICE)
    ctx = torch.randn(1, 512, 4096, dtype=torch.float16, device=DEVICE)
    cache = torch.randn(30, 2, 1, 2048, 12, 128, dtype=torch.float16, device=DEVICE)
    
    print("Warming up...")
    for _ in range(3):
        model.infer(x, t, ctx, cache)
        
    print("Measuring latency...")
    start_time = time.time()
    iters = 50
    for _ in range(iters):
        output = model.infer(x, t, ctx, cache)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / iters * 1000
    print(f"Average Inference Time: {avg_latency:.2f} ms")
    print(f"FPS: {1000/avg_latency:.2f}")
    
    print(f"Output Mean: {output.mean().item():.4f}")
    if output.mean().item() == 0.0:
        print("WARNING: Output is all zeros! Something is wrong.")
    elif torch.isnan(output).any():
        print("CRITICAL: Output contains NaNs!")
    else:
        print("SUCCESS: Output looks valid.")