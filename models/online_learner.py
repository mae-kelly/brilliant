import numpy as np
import tensorflow as tf
import asyncio
from collections import deque

class OnlineLearner:
    def __init__(self):
        self.model = None
        self.performance_history = deque(maxlen=1000)
        self.retrain_threshold = 100
        
    async def load_models(self):
        try:
            self.interpreter = tf.lite.Interpreter('models/model_weights.tflite')
            self.interpreter.allocate_tensors()
            print("✅ Online learner loaded")
        except:
            print("⚠️ Online learner using fallback")
    
    async def predict(self, features):
        if self.interpreter:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            self.interpreter.set_tensor(input_details[0]['index'], features.reshape(1, -1).astype(np.float32))
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(output_details[0]['index'])[0][0]
            confidence = abs(prediction - 0.5) * 2
            return prediction, confidence
        return 0.5, 0.5
    
    async def update_on_trade_result(self, features, prediction, outcome, pnl, confidence):
        self.performance_history.append({'prediction': prediction, 'outcome': outcome, 'pnl': pnl})

online_learner = OnlineLearner()
