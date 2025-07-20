#!/usr/bin/env python3
import sys
sys.path.append('../models')
from renaissance_transformer import *
import tensorflow as tf

def main():
    print("🚀 Training Renaissance Transformer Model...")
    
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    model = train_renaissance_transformer()
    
    # Save model
    model.save('models/transformers/renaissance_model.h5')
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('models/transformers/renaissance_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ Renaissance Transformer model saved!")

if __name__ == "__main__":
    main()
