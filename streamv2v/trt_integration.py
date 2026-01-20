import torch
import torch.nn as nn
import tensorrt as trt
import os
import sys

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class DummyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.sink_size = 0
        self.adapt_sink_thr = 0

class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttn()

class TRTEngineWrapper:
    def __init__(self, engine_path, device="cuda"):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")
            
        print(f"[TRT] Loading Engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream(device=device)
        self.io_tensors = {}
        self.tensor_names = set()
        
        print("[TRT] Allocating Buffers...")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.tensor_names.add(name)
            
            # Aliasing (Memory Optimization)
            if "updated" in name:
                base = name.replace("_updated", "")
                if base in self.io_tensors:
                    self.context.set_tensor_address(name, self.io_tensors[base].data_ptr())
                    continue
            
            # Dtype Handling
            dtype_trt = self.engine.get_tensor_dtype(name)
            if dtype_trt == trt.float32: dtype = torch.float32
            elif dtype_trt == trt.float16: dtype = torch.float16
            elif dtype_trt == trt.int32: dtype = torch.int32
            elif dtype_trt == trt.int64: dtype = torch.int64
            else: dtype = torch.float32
            
            # Shape Handling
            shape = self.engine.get_tensor_shape(name)
            dims = list(shape)
            if -1 in dims: 
                # Fallback defaults just in case
                if "kv" in name: dims = [1, 30, 6240, 12, 256]
                elif "cross" in name: dims = [1, 30, 512, 12, 256]
                elif "t" in name: dims = [1, 2]
                elif "context" in name: dims = [1, 512, 4096]
                elif "start" in name or "end" in name: dims = [1]
                else: dims = [1, 16, 1, 60, 104] 
            
            t = torch.zeros(tuple(dims), device=device, dtype=dtype).contiguous()
            self.io_tensors[name] = t
            self.context.set_tensor_address(name, t.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, tuple(dims))

        # Verify Critical Inputs only (Relaxed check)
        # We don't check for 'cross_cache' or 'k_ends' as TRT often optimizes them out.
        required = ["context", "k_starts"]
        missing = [req for req in required if req not in self.tensor_names]
        if missing:
            print(f"\n[TRT ERROR] Engine is still missing: {missing}")
            sys.exit(1)
        print(f"[TRT] Ready. Active Bindings: {list(self.tensor_names)}")

    def infer(self, x, t, context, k_start, k_end):
        # 1. Inputs (x, t)
        target_x = self.io_tensors["x"]
        if x.dtype != target_x.dtype: x = x.to(target_x.dtype)
        target_x.copy_(x)

        target_t = self.io_tensors["t"]
        if t.numel() != target_t.numel():
             target_t.copy_(t.view(1, -1).expand(target_t.shape))
        else:
             target_t.copy_(t)

        # 2. Context (Text)
        # Safe copy: engine might have optimized away 'context' (unlikely now) 
        # or kept it. We only copy if it exists.
        if "context" in self.io_tensors and context is not None:
            target_ctx = self.io_tensors["context"]
            target_ctx.zero_()
            src = context
            if src.shape[0] > target_ctx.shape[0]: src = src[:target_ctx.shape[0]]
            if src.shape[0] < target_ctx.shape[0]: src = src.expand(target_ctx.shape[0], -1, -1)
            target_ctx[:, :src.shape[1], :].copy_(src.to(target_ctx.dtype))

        # 3. Indices
        # Only copy if the binding exists (Adaptive)
        if "k_starts" in self.io_tensors:
            self.io_tensors["k_starts"].copy_(k_start)
        
        if "k_ends" in self.io_tensors:
            self.io_tensors["k_ends"].copy_(k_end)

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.io_tensors["flow_output"]

class TRTCausalWanModel(nn.Module):
    def __init__(self, engine_path, original_model):
        super().__init__()
        self.model_type = original_model.model_type
        self.patch_size = original_model.patch_size
        self.in_dim = original_model.in_dim
        self.dim = original_model.dim
        self.num_frame_per_block = 1 
        self.num_heads = original_model.num_heads
        self.blocks = nn.ModuleList([DummyBlock() for _ in range(original_model.num_layers)])
        self.engine = TRTEngineWrapper(engine_path)
        
        self.frame_seq_length = (480//16) * (832//16) 
        self.current_start = 0
        self.current_end = 0

    def forward(self, x, t, **kwargs):
        if isinstance(x, list): x = x[0]
        batch_size = x.shape[0]
        input_frames = x.shape[2]
        outputs_batch = []
        
        for b in range(batch_size):
            outputs_time = []
            
            # Extract Context
            ctx = kwargs.get('context', kwargs.get('conditional_dict', {}).get('prompt_embeds'))
            if ctx is not None and ctx.shape[0] == batch_size: ctx = ctx[b:b+1]

            for i in range(input_frames):
                x_single = x[b:b+1, :, i:i+1, :, :]
                t_single = t[b] if t.dim()>0 else t
                
                # Advance Time
                self.current_start = self.current_end
                self.current_end = self.current_end + self.frame_seq_length
                
                k_start = torch.tensor([self.current_start], device=x.device, dtype=torch.long)
                k_end = torch.tensor([self.current_end], device=x.device, dtype=torch.long)
                
                # Infer (Adaptive)
                flow = self.engine.infer(x_single, t_single, ctx, k_start, k_end)
                
                # Pass Through (Pipeline handles math)
                outputs_time.append(flow.to(x_single.dtype))
            
            outputs_batch.append(torch.cat(outputs_time, dim=2))
        return torch.cat(outputs_batch, dim=0)

def replace_model_with_trt(pipeline, engine_path):
    print(f"[TRT] Swapping PyTorch Model with {engine_path}...")
    original = pipeline.generator.model
    trt_model = TRTCausalWanModel(engine_path, original)
    pipeline.generator.model = trt_model
    return pipeline