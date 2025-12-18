
import os
import torch
from .utilities import export_onnx, optimize_onnx

class EngineBuilder:
	def __init__(self, model, engine_dir, device='cuda'):
		self.model = model
		self.engine_dir = engine_dir
		self.device = device

	def build(self, dummy_inputs, name_prefix="model", opset=17, force_export=False):
		"""
		Export ONNX and (optionally) build TensorRT engine for the model.
		"""
		os.makedirs(self.engine_dir, exist_ok=True)
		onnx_path = os.path.join(self.engine_dir, f"{name_prefix}.onnx")
		onnx_opt_path = os.path.join(self.engine_dir, f"{name_prefix}.opt.onnx")
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
		# TensorRT engine build placeholder (to be implemented)
		engine_path = os.path.join(self.engine_dir, f"{name_prefix}.engine")
		print(f"[TODO] Build TensorRT engine from {onnx_opt_path} and save to {engine_path}")
		return onnx_path, onnx_opt_path, engine_path
