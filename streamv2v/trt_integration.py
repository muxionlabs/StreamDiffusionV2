import torch
import torch.nn as nn
import tensorrt as trt
import torch.nn.functional as F

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class TRTEngineWrapper:
    def __init__(self, engine_path, device="cuda"):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream(device=device)
        self.device = device
        self.io = {}

        print("[TRT] Allocating IO tensors...")

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            dtype = torch.float16 if self.engine.get_tensor_dtype(name) == trt.float16 else torch.float32

            tensor = torch.zeros(shape, device=device, dtype=dtype).contiguous()
            self.io[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

            print(f"  -> {name}: {shape} {dtype}")

        print("[TRT] Engine ready (stateless).")

    def infer(self, x, t, context):
        self.io["x"].copy_(x)
        self.io["t"].copy_(t)
        self.io["context"].copy_(context)

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return self.io["flow_output"]


class TRTCausalWanModel(nn.Module):
    def __init__(self, engine_path, device="cuda"):
        super().__init__()
        self.engine = TRTEngineWrapper(engine_path, device=device)

    def forward(self, x, *args, **kwargs):
        # ---- timestep ----
        if "t" in kwargs:
            t = kwargs["t"]
        elif len(args) > 0:
            t = args[0]
        else:
            raise RuntimeError("Missing timestep")

        # ---- context ----
        if "context" in kwargs:
            context = kwargs["context"]
        elif "conditional_dict" in kwargs:
            context = kwargs["conditional_dict"]["prompt_embeds"]
        else:
            raise RuntimeError("Missing context")

        # Normalize timestep
        if t.dim() == 2:
            t = t[:, 0]
        t = t.unsqueeze(1)  # [B, 1]

        if isinstance(x, list):
            x = x[0]

        B, C, T, H, W = x.shape

        outputs = []

        for b in range(B):
            frames = []
            for i in range(T):
                x_single = x[b:b+1, :, i:i+1]
                t_single = t[b:b+1]
                ctx = context[b:b+1]

                out = self.engine.infer(x_single, t_single, ctx)
                out = torch.nan_to_num(out, nan=0.0).clamp(-100, 100)

                frames.append(out.to(x.dtype))

            outputs.append(torch.cat(frames, dim=2))

        out = torch.cat(outputs, dim=0)  # [B, 2, 1560, 64]

        # ---- Token → Spatial ----
        B, C, N, D = out.shape

        Ht, Wt = 30, 52   # Patch grid from TRT
        assert N == Ht * Wt, f"Unexpected token count: {N}"

        # [B, 2, 1560, 64] → [B, 2, 64, 30, 52]
        out = out.reshape(B, C, Ht, Wt, D).permute(0, 1, 4, 2, 3).contiguous()

        # ---- Reduce patch dim 64 → 2 ----
        # Keep 2-channel flow
        out = out.mean(dim=2)  # [B, 2, 30, 52]

        # Expand 2 → 16 channels for WAN compatibility
        out = out.repeat(1, 8, 1, 1)  # [B, 16, 30, 52]


        out = F.interpolate(out, size=(60, 104), mode="bilinear", align_corners=False)
        # Expand temporal
        out = out.unsqueeze(2).repeat(1, 1, T, 1, 1)  # [B, 16, T, 60, 104]

        # MATCH VAE DTYPE
        return out.to(torch.bfloat16)


def replace_model_with_trt(pipeline, engine_path, device="cuda"):
    print(f"[TRT] Replacing PyTorch model with TRT engine: {engine_path}")
    pipeline.generator.model = TRTCausalWanModel(engine_path, device=device)
    return pipeline
