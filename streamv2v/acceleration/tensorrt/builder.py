import os
import torch
from .utilities import export_onnx, optimize_onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class EngineBuilder:
	def __init__(self, model, engine_dir, device='cuda'):
		self.model = model
		self.engine_dir = engine_dir
		self.device = device

	def build(self, dummy_inputs, name_prefix="model", opset=17, force_export=False, fp16=True, max_batch_size=1):
		"""
		Export ONNX and build TensorRT engine for the model.
		"""
		os.makedirs(self.engine_dir, exist_ok=True)
		onnx_path = os.path.join(self.engine_dir, f"{name_prefix}.onnx")
		onnx_opt_path = os.path.join(self.engine_dir, f"{name_prefix}.opt.onnx")
		engine_path = os.path.join(self.engine_dir, f"{name_prefix}.engine")

		# Export ONNX
		if force_export or not os.path.exists(onnx_path):
			export_onnx(self.model, dummy_inputs, onnx_path, opset=opset)
		else:
			print(f"Found cached ONNX: {onnx_path}")
		# Optimize ONNX
		if force_export or not os.path.exists(onnx_opt_path):
			optimize_onnx(onnx_path, onnx_opt_path)
		else:
			print(f"Found cached optimized ONNX: {onnx_opt_path}")

		# Build TensorRT engine
		if force_export or not os.path.exists(engine_path):
			print(f"Building TensorRT engine from {onnx_opt_path} ...")
			TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
			builder = trt.Builder(TRT_LOGGER)
			network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
			network = builder.create_network(network_flags)
			parser = trt.OnnxParser(network, TRT_LOGGER)
			with open(onnx_opt_path, "rb") as f:
				if not parser.parse(f.read()):
					for i in range(parser.num_errors):
						print(parser.get_error(i))
					raise RuntimeError("Failed to parse ONNX for TensorRT.")
			config = builder.create_builder_config()
			config.max_workspace_size = 1 << 30  # 1GB
			if fp16:
				config.set_flag(trt.BuilderFlag.FP16)
			profile = builder.create_optimization_profile()
			for i in range(network.num_inputs):
				input_name = network.get_input(i).name
				shape = network.get_input(i).shape
				# Set min/opt/max shape for dynamic batch (if needed)
				if -1 in shape:
					min_shape = tuple(s if s != -1 else 1 for s in shape)
					opt_shape = tuple(s if s != -1 else max_batch_size for s in shape)
					max_shape = tuple(s if s != -1 else max_batch_size for s in shape)
					profile.set_shape(input_name, min_shape, opt_shape, max_shape)
			config.add_optimization_profile(profile)
			engine = builder.build_engine(network, config)
			with open(engine_path, "wb") as f:
				f.write(engine.serialize())
			print(f"TensorRT engine saved to {engine_path}")
		else:
			print(f"Found cached engine: {engine_path}")
		return onnx_path, onnx_opt_path, engine_path
