"""
Phase 4: The Benchmark (Version Agnostic)
Function: Loads the TensorRT Engine and measures raw inference speed.
          - Detects TRT 10.x vs TRT 8.x/9.x automatically.
          - Allocates buffers correctly for your version.
"""

import tensorrt as trt
import torch
import time
import argparse
import sys

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTWrapper:
    def __init__(self, engine_path):
        print(f"I: Loading engine from {engine_path}...")
        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"E: Failed to load engine. {e}")
            sys.exit(1)
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()
        
        self.inputs = {}
        self.outputs = {}
        self.allocations = []

        print("I: Allocating buffers...")
        
        # --- PATH A: TRT 10.x (New API) ---
        if hasattr(self.engine, "num_io_tensors"):
            print("I: Detected TensorRT 10.x+ API")
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.engine.get_tensor_shape(name)
                mode = self.engine.get_tensor_mode(name)
                is_input = (mode == trt.TensorIOMode.INPUT)
                
                self._allocate_tensor(name, dtype, shape, is_input)

        # --- PATH B: TRT 8.x/9.x (Legacy API) ---
        elif hasattr(self.engine, "num_bindings"):
            print("I: Detected TensorRT 8.x/9.x API")
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)
                
                self._allocate_tensor(name, dtype, shape, is_input)
        
        else:
            print("E: Unknown TensorRT Version. Could not find tensor count.")
            sys.exit(1)

    def _allocate_tensor(self, name, dtype, shape, is_input):
        # Map TRT dtype to PyTorch dtype
        if dtype == trt.float16:
            torch_dtype = torch.float16
        elif dtype == trt.float32:
            torch_dtype = torch.float32
        elif dtype == trt.int32:
            torch_dtype = torch.int32
        elif dtype == trt.int64:
            torch_dtype = torch.int64
        elif dtype == trt.int8:
            torch_dtype = torch.int8
        else:
            torch_dtype = torch.float32

        # Create Tensor
        tensor = torch.empty(tuple(shape), dtype=torch_dtype, device="cuda")
        self.allocations.append(tensor) # Keep alive
        
        # Bind memory
        self.context.set_tensor_address(name, tensor.data_ptr())

        if is_input:
            self.inputs[name] = tensor
        else:
            self.outputs[name] = tensor
            
        type_str = str(dtype).split('.')[-1]
        io_str = "Input " if is_input else "Output"
        print(f"   [{io_str}] {name}: {shape} [{type_str}]")

    def infer(self, feed_dict):
        # 1. Copy Inputs
        for name, tensor in feed_dict.items():
            if name in self.inputs:
                if self.inputs[name].dtype != tensor.dtype:
                    self.inputs[name].copy_(tensor.to(self.inputs[name].dtype))
                else:
                    self.inputs[name].copy_(tensor)
        
        # 2. Run Inference
        self.context.execute_async_v3(self.stream.cuda_stream)
        
        # 3. Return
        return self.outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="wan_stream.engine")
    args = parser.parse_args()

    # 1. Load Engine
    trt_model = TRTWrapper(args.engine)

    # 2. Create Dummy Data (480p Baked)
    print("\nI: Generating dummy inputs (480p)...")
    device = "cuda"
    
    dummy_inputs = {
        "x": torch.randn(1, 16, 1, 60, 104, dtype=torch.float32, device=device),
        "t": torch.tensor([[500, 500]], dtype=torch.int64, device=device),
        "context": torch.randn(1, 512, 4096, dtype=torch.float32, device=device),
        "kv_cache": torch.randn(1, 30, 6240, 12, 256, dtype=torch.float32, device=device),
        "cross_cache": torch.randn(1, 30, 512, 12, 256, dtype=torch.float32, device=device)
    }

    # 3. Warmup
    print("I: Warming up GPU...")
    for _ in range(5):
        trt_model.infer(dummy_inputs)
    torch.cuda.synchronize()

    # 4. Benchmark
    print("I: Benchmarking (100 iterations)...")
    start_time = time.time()
    iterations = 100
    
    for _ in range(iterations):
        trt_model.context.execute_async_v3(trt_model.stream.cuda_stream)
    
    torch.cuda.synchronize()
    end_time = time.time()

    # 5. Stats
    total_time = end_time - start_time
    avg_time = total_time / iterations
    fps = 1.0 / avg_time
    
    print("\n" + "="*40)
    print(f" RESULTS: Wan 2.1 (480p) TensorRT")
    print(f" Avg Latency: {avg_time*1000:.2f} ms")
    print(f" Throughput:  {fps:.2f} FPS (Steps per Sec)")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()