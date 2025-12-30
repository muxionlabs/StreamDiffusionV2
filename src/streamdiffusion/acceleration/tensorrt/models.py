import torch
import tensorrt as trt

class WanDiTTensorRTEngineWrapper:
    def __init__(self, engine_path, device="cuda"):
        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        
        # Persistent Flat Cache
        # [Layers=30, KV=2, B=1, Heads=12, Len=1024, Dim=128]
        self.cache_buffer = torch.zeros(30, 2, 1, 12, 1024, 128, dtype=torch.float16, device=device)
        self.output_buffer = torch.zeros(1, 16, 1, 60, 106, dtype=torch.float16, device=device)

    def forward(self, x, t, context, **kwargs):
        # 1. Roll Cache (Shift Left)
        # Shift on Dim 4 (Length)
        self.cache_buffer = torch.roll(self.cache_buffer, shifts=-1, dims=4)
        
        # 2. Bindings
        self.context.set_tensor_address("x", x.data_ptr())
        self.context.set_tensor_address("t", t.data_ptr())
        self.context.set_tensor_address("context", context.data_ptr())
        self.context.set_tensor_address("flat_cache", self.cache_buffer.data_ptr())
        self.context.set_tensor_address("flow_pred", self.output_buffer.data_ptr())
        
        # 3. Infer
        self.context.execute_async_v3(self.stream)
        
        return self.output_buffer