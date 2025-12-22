import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

class TRTInferenceWrapper:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        # Collect input/output binding info
        self.input_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        self.output_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]

    def infer(self, inputs_dict):
        # Prepare device buffers
        bindings = [None] * self.engine.num_bindings
        device_buffers = {}
        for name in self.input_names:
            idx = self.engine.get_binding_index(name)
            shape = self.context.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            inp = inputs_dict[name]
            if not inp.flags['C_CONTIGUOUS']:
                inp = np.ascontiguousarray(inp)
            device_buffers[name] = cuda.mem_alloc(inp.nbytes)
            cuda.memcpy_htod_async(device_buffers[name], inp, self.stream)
            bindings[idx] = int(device_buffers[name])
        output_host = {}
        for name in self.output_names:
            idx = self.engine.get_binding_index(name)
            shape = self.context.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            out = np.empty(shape, dtype=dtype)
            device_buffers[name] = cuda.mem_alloc(out.nbytes)
            bindings[idx] = int(device_buffers[name])
            output_host[name] = out
        # Run inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=int(self.stream.handle))
        # Copy outputs back
        for name in self.output_names:
            cuda.memcpy_dtoh_async(output_host[name], device_buffers[name], self.stream)
        self.stream.synchronize()
        return [output_host[name] for name in self.output_names]
