
import os

class EngineManager:
	def __init__(self, engine_dir):
		self.engine_dir = engine_dir
		os.makedirs(self.engine_dir, exist_ok=True)
		self.engines = {}

	def get_engine_path(self, name_prefix):
		return os.path.join(self.engine_dir, f"{name_prefix}.engine")

	def has_engine(self, name_prefix):
		return os.path.exists(self.get_engine_path(name_prefix))

	def load_engine(self, name_prefix):
		engine_path = self.get_engine_path(name_prefix)
		if not os.path.exists(engine_path):
			raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
		# Placeholder: actual TensorRT engine loading logic goes here
		print(f"[TODO] Load TensorRT engine from {engine_path}")
		self.engines[name_prefix] = engine_path
		return engine_path

	def get_or_build_engine(self, builder, dummy_inputs, name_prefix="model", opset=17, force_export=False):
		if not self.has_engine(name_prefix):
			print(f"Engine not found for {name_prefix}, building...")
			_, _, engine_path = builder.build(dummy_inputs, name_prefix, opset, force_export)
		else:
			engine_path = self.get_engine_path(name_prefix)
			print(f"Found cached engine: {engine_path}")
		return self.load_engine(name_prefix)
