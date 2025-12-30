import tensorrt as trt
import torch
import torch.nn as nn

class DummyAttr:
    """A dummy object that swallows attribute assignments (sink_size, etc.)"""
    def __init__(self):
        self.sink_size = 0
        self.freqs = None 

class DummyBlock:
    """Mock for a Transformer Block"""
    def __init__(self):
        self.self_attn = DummyAttr()
        self.cross_attn = DummyAttr()

class DummyModel:
    """Mock for the inner .model attribute"""
    def __init__(self):
        # The pipeline loops over 30 blocks to set cache params.
        # We create 30 dummy blocks to satisfy this loop.
        self.blocks = [DummyBlock() for _ in range(30)]
        self.num_frame_per_block = 1

class WanTRTBackend(nn.Module):
    def __init__(self, engine_path, device="cuda"):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.ERROR)
        try:
            with open(engine_path, "rb") as f:
                self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        self.device = device
        
        # Mock Structure
        self.model = DummyModel()
        
        # Binding Indices
        self.idx_x = self.engine.get_tensor_name(0) # 'x'
        self.idx_t = self.engine.get_tensor_name(1) # 't'
        self.idx_ctx = self.engine.get_tensor_name(2) # 'context'
        self.idx_out = self.engine.get_tensor_name(3) # 'flow_pred'
        
        # Pre-allocate output buffer (Native 480p: 60x106 latents)
        self.output_tensor = torch.empty((1, 16, 1, 60, 106), dtype=torch.float16, device=device)
        self.context.set_tensor_address(self.idx_out, self.output_tensor.data_ptr())

    def forward(self, noisy_image_or_video, timestep, conditional_dict, **kwargs):
        """
        Drop-in replacement for WanDiffusion.forward.
        Ignores 'kv_cache', 'crossattn_cache', etc.
        """
        # --- INPUT UNPACKING LOGIC ---
        # 1. Handle Conditional Dict (Text Embeddings)
        context = conditional_dict
        if isinstance(context, list):
            context = context[0]
        
        if isinstance(context, dict):
            # Wan/StreamDiffusion usually stores the main embedding in 'context' or 'hidden_states'
            # We look for the tensor that matches our expected dimensions [B, 512, 4096]
            if "context" in context:
                context = context["context"]
            else:
                # Fallback: grab the first value that is a Tensor
                for v in context.values():
                    if isinstance(v, torch.Tensor):
                        context = v
                        break
        
        # 2. Ensure Tensor layout
        # TensorRT pointers require contiguous memory
        if not context.is_contiguous():
            context = context.contiguous()
        if not noisy_image_or_video.is_contiguous():
            noisy_image_or_video = noisy_image_or_video.contiguous()
        if not timestep.is_contiguous():
            timestep = timestep.contiguous()

        # 3. Bind Inputs
        self.context.set_tensor_address(self.idx_x, noisy_image_or_video.data_ptr())
        self.context.set_tensor_address(self.idx_t, timestep.data_ptr())
        self.context.set_tensor_address(self.idx_ctx, context.data_ptr())
        
        # 4. Enqueue
        self.context.execute_async_v3(self.stream)
        
        # 5. Return result
        return self.output_tensor.clone()
    
    def get_scheduler(self):
        return None