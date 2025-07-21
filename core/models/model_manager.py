import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import yaml
import os
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pickle

class ModelManager:
    """Manages ML model lifecycle: training, conversion, deployment, versioning"""
    
    def __init__(self, config_path='settings.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_dir = 'models'
        self.exports_dir = f'{self.model_dir}/exports'
        self.checkpoints_dir = f'{self.model_dir}/checkpoints'
        
        os.makedirs(self.exports_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        self.model_registry = {}
        self.load_model_registry()
    
    def convert_pytorch_to_tflite(self, pytorch_model, input_shape: Tuple[int, ...], 
                                 model_name: str, optimization_level: str = 'full') -> str:
        """Convert PyTorch model to optimized TFLite"""
        
        logging.info(f"Converting {model_name} to TFLite...")
        
        # 1. Convert PyTorch to ONNX
        dummy_input = torch.randn(1, *input_shape)
        onnx_path = f'{self.exports_dir}/{model_name}.onnx'
        
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # 2. Convert ONNX to TensorFlow SavedModel
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        saved_model_path = f'{self.exports_dir}/{model_name}_saved_model'
        tf_rep.export_graph(saved_model_path)
        
        # 3. Convert to TFLite with optimization
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        if optimization_level == 'full':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif optimization_level == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Add representative dataset for quantization
            converter.representative_dataset = self.get_representative_dataset
        
        tflite_model = converter.convert()
        
        # 4. Save TFLite model
        tflite_path = f'{self.exports_dir}/{model_name}.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # 5. Validate conversion
        self.validate_tflite_model(tflite_path, input_shape)
        
        # 6. Register model
        model_info = {
            'name': model_name,
            'path': tflite_path,
            'input_shape': input_shape,
            'created_at': datetime.utcnow().isoformat(),
            'optimization': optimization_level,
            'size_bytes': os.path.getsize(tflite_path),
            'fingerprint': self.calculate_model_fingerprint(tflite_path)
        }
        
        self.register_model(model_info)
        
        logging.info(f"✅ TFLite model saved: {tflite_path} ({model_info['size_bytes']} bytes)")
        return tflite_path
    
    def get_representative_dataset(self):
        """Generate representative dataset for quantization"""
        # Load recent feature data for calibration
        try:
            with open('data/features/recent_features.pkl', 'rb') as f:
                features_df = pickle.load(f)
            
            # Convert to numpy array and yield samples
            features_array = features_df.values.astype(np.float32)
            
            for i in range(min(100, len(features_array))):
                yield [features_array[i:i+1]]
                
        except FileNotFoundError:
            # Generate synthetic data if no real data available
            input_shape = (1, 11)  # Adjust based on your feature count
            for _ in range(100):
                yield [np.random.random(input_shape).astype(np.float32)]
    
    def validate_tflite_model(self, tflite_path: str, input_shape: Tuple[int, ...]):
        """Validate TFLite model can run inference"""
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test with random input
        test_input = np.random.random((1, *input_shape)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        logging.info(f"✅ TFLite validation passed - Output shape: {output_data.shape}")
        return True
    
    def calculate_model_fingerprint(self, model_path: str) -> str:
        """Calculate SHA256 fingerprint of model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def register_model(self, model_info: Dict):
        """Register model in model registry"""
        self.model_registry[model_info['name']] = model_info
        self.save_model_registry()
    
    def load_model_registry(self):
        """Load model registry from disk"""
        registry_path = f'{self.model_dir}/model_registry.json'
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                self.model_registry = json.load(f)
    
    def save_model_registry(self):
        """Save model registry to disk"""
        registry_path = f'{self.model_dir}/model_registry.json'
        with open(registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def get_latest_model(self, model_name: str) -> Optional[Dict]:
        """Get latest version of a model"""
        return self.model_registry.get(model_name)
    
    def benchmark_model_performance(self, model_path: str, input_shape: Tuple[int, ...], 
                                  num_runs: int = 1000) -> Dict:
        """Benchmark TFLite model performance"""
        import time
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        
        # Warmup
        test_input = np.random.random((1, *input_shape)).astype(np.float32)
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'avg_inference_time_ms': np.mean(times) * 1000,
            'p95_inference_time_ms': np.percentile(times, 95) * 1000,
            'p99_inference_time_ms': np.percentile(times, 99) * 1000,
            'throughput_per_second': 1.0 / np.mean(times)
        }

class TFLiteInferenceEngine:
    """Optimized TFLite inference engine for production"""
    
    def __init__(self, model_path: str, num_threads: int = 4):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logging.info(f"TFLite engine initialized: {model_path}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Fast inference with TFLite model"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        features = features.astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output.squeeze()
    
    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Batch inference for multiple samples"""
        results = []
        for features in features_batch:
            result = self.predict(features)
            results.append(result)
        return np.array(results)
