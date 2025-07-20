#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time

def validate_transformer_model():
    """Validate that transformer model works correctly"""
    print("🔍 Validating Transformer model...")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path='models/transformers/renaissance_model.tflite')
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        
        # Test inference
        test_input = np.random.random((1, 120, 45)).astype(np.float32)
        
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        inference_time = time.time() - start_time
        
        print(f"✅ Inference successful")
        print(f"   Prediction: {prediction[0][0]:.4f}")
        print(f"   Inference time: {inference_time*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

if __name__ == "__main__":
    validate_transformer_model()
