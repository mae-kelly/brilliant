import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {
        "confidence_threshold": 0.75, "momentum_threshold": 0.65, "volatility_threshold": 0.10,
        "liquidity_threshold": 50000, "min_liquidity_threshold": 10000, "max_risk_score": 0.4,
        "max_slippage": get_dynamic_config().get("max_slippage", 0.03), "stop_loss_threshold": get_dynamic_config().get("stop_loss_threshold", 0.05), "take_profit_threshold": get_dynamic_config().get("take_profit_threshold", 0.12),
        "max_hold_time": get_dynamic_config().get("max_hold_time", 300), "min_price_change": 5, "max_price_change": 15,
        "max_position_size": 10.0, "starting_capital": 10.0
    }
    def update_performance(*args): pass

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
