
import os
import torch
import onnx
import onnxruntime as ort
import numpy as np

def export_onnx(model, dummy_inputs, onnx_path, opset=17, dynamic_axes=None, input_names=None, output_names=None):
	"""
	Export a PyTorch model to ONNX format.
	"""
	model.eval()
	torch.onnx.export(
		model,
		dummy_inputs,
		onnx_path,
		export_params=True,
		opset_version=opset,
		do_constant_folding=True,
		input_names=input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
	)
	print(f"Exported ONNX model to {onnx_path}")

def check_onnx(onnx_path):
	"""
	Check ONNX model for validity.
	"""
	model = onnx.load(onnx_path)
	onnx.checker.check_model(model)
	print(f"ONNX model at {onnx_path} is valid.")

def run_onnx_inference(onnx_path, inputs_dict):
	"""
	Run inference on an ONNX model using ONNX Runtime.
	"""
	sess = ort.InferenceSession(onnx_path)
	outputs = sess.run(None, inputs_dict)
	return outputs

def optimize_onnx(onnx_path, optimized_path=None):
	"""
	Placeholder for ONNX optimization (can use onnxoptimizer, onnxsim, etc.).
	"""
	# For now, just copy the file
	if optimized_path is None:
		optimized_path = onnx_path.replace('.onnx', '.opt.onnx')
	import shutil
	shutil.copy(onnx_path, optimized_path)
	print(f"Optimized ONNX model saved to {optimized_path}")
	return optimized_path
