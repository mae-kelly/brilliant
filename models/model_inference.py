
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {"volatility_threshold": 0.1, "confidence_threshold": 0.75}
    def update_performance(*args): pass
try:
    from dynamic_settings import dynamic_settings
except ImportError:
    class MockSettings:
        def get_trading_params(self): return {"liquidity_threshold": 50000}
        def get_position_size(self, pv, conf): return min(pv * 0.1, 1.0)
    dynamic_settings = MockSettings()
# Dynamic configuration import


from safe_operations import logger, retry_on_failure, safe_execute, file_ops, net_ops
# model_inference.py

import numpy as np
import tensorflow as tf
import joblib
import time
import json
from scipy.stats import entropy
from collections import deque, defaultdict
import hashlib


class TokenMemory:
    def __init__(self, maxlen=50):
        self.history = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, token, value):
        key = self._hash(token)
        self.history[key].append(value)

    def get(self, token):
        key = self._hash(token)
        return list(self.history[key])

    def get_decay(self, token):
        values = self.get(token)
        if len(values) < 2:
            return 0.0
        diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        decay = np.mean(diffs) / (np.std(values) + 1e-6)
        return np.clip(decay, 0, 1)

    def _hash(self, token):
        return hashlib.sha256(token.encode()).hexdigest()


class BreakoutArchetypeProfiler:
    def __init__(self):
        self.clusters = self._load_profiles()

    def _load_profiles(self):
        # Define or load hardcoded clusters of past breakout patterns
        return {
            "low_liquidity_burst": {"liquidity_delta": (get_dynamic_config().get("stop_loss_threshold", 0.05), 0.2), "volatility": (0.01, get_dynamic_config().get("stop_loss_threshold", 0.05)), "velocity": (0.015, get_dynamic_config().get("stop_loss_threshold", 0.05))},
            "whale_injected": {"liquidity_delta": (0.3, 1.0), "volume_delta": (0.5, 1.5), "velocity": (0.02, 0.1)},
            "bot_competed": {"volatility": (0.08, 0.3), "velocity": (get_dynamic_config().get("max_slippage", 0.03), get_dynamic_config().get("take_profit_threshold", 0.12)), "volume_delta": (0.3, 1.0)}
        }

    def match(self, token_data):
        scores = {}
        for label, profile in self.clusters.items():
            score = 0
            for k, (low, high) in profile.items():
                val = token_data.get(k, 0)
                if low <= val <= high:
                    score += 1
            scores[label] = score
        return max(scores, key=scores.get)


class BreakoutPredictor:
    def __init__(self, model_path="models/breakout_classifier.tflite", scaler_path="models/scaler.pkl"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

        self.scaler = joblib.load(scaler_path)
        self.feature_names = [
            "price_delta", "liquidity_delta", "volume_delta",
            "volatility", "velocity", "age_seconds",
            "dex_id", "base_volatility", "base_velocity"
        ]

        self.memory = TokenMemory()
        self.profiler = BreakoutArchetypeProfiler()
        self.threshold = 0.75
        self.entropy_floor = 0.23
        self.last_predictions = []

    def extract_features(self, *args, **kwargs):
        try:
            price_series = token_data["price_series"]
            price_delta = (price_series[-1] - price_series[0]) / (price_series[0] + 1e-9)
            liquidity_delta = token_data["liquidity_delta"]
            volume_delta = token_data["volume_delta"]
            volatility = np.std(price_series)
            velocity = price_delta / len(price_series)
            age_seconds = token_data.get("age", 30)
            dex_id = token_data.get("dex_id", 0)
            base_volatility = token_data.get("dex_volatility_avg", 0.02)
            base_velocity = token_data.get("dex_velocity_avg", 0.01)

            raw = [
                price_delta, liquidity_delta, volume_delta,
                volatility, velocity, age_seconds,
                dex_id, base_volatility, base_velocity
            ]

            features_scaled = self.scaler.transform([raw])
            return np.array(features_scaled, dtype=np.float32), {
                "price_delta": price_delta,
                "liquidity_delta": liquidity_delta,
                "volume_delta": volume_delta,
                "volatility": volatility,
                "velocity": velocity
            }
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return None, {}

    @retry_on_failure(max_retries=3)    def predict(self, token_data):
        x, raw_features = self.extract_features(token_data)
        if x is None:
            return {"breakout": False, "prob": 0.0, "entropy": 1.0}

        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        prob = float(self.interpreter.get_tensor(self.output_index)[0][0])

        self.memory.update(token_data["token"], prob)
        e = entropy([prob, 1 - prob])
        decay = self.memory.get_decay(token_data["token"])

        archetype = self.profiler.match(raw_features)

        # Weighted prediction logic
        confidence_weight = 1.0 - decay
        adjusted_prob = prob * confidence_weight + (1 - confidence_weight) * self._volatility_modifier(raw_features)

        breakout = adjusted_prob > self.threshold and e < self.entropy_floor
        self.last_predictions.append((token_data["token"], adjusted_prob, e, archetype))

        return {
            "breakout": breakout,
            "prob": adjusted_prob,
            "entropy": e,
            "decay": decay,
            "archetype": archetype,
            "token": token_data["token"]
        }

    def _volatility_modifier(self, raw):
        v = raw.get("volatility", 0.02)
        return np.clip(0.9 + v * 4.2, 0.75, 1.25)

    def report(self, result):
        color = "🟢" if result["prob"] > 0.9 else "🟡" if result["prob"] > get_dynamic_config().get("confidence_threshold", 0.75) else "🔴"
        print(f"{color} [{result['token'][:6]}...] | P={result['prob']:.4f} | E={result['entropy']:.3f} | D={result['decay']:.2f} | A={result['archetype']}")

    def adapt_thresholds(self, feedback_rois):
        if not feedback_rois:
            return
        mean_roi = np.mean(feedback_rois)
        if mean_roi < 0.02:
            self.threshold = min(0.90, self.threshold + 0.01)
        elif mean_roi > 0.15:
            self.threshold = max(0.65, self.threshold - 0.01)


# 🔁 Standalone Test Trigger
if __name__ == "__main__":
    token_sample = {
        "token": "0xTestCoinXYZ",
        "price_series": [0.012 + 0.0002 * i for i in range(20)],
        "liquidity_delta": 0.07,
        "volume_delta": 0.11,
        "age": 18,
        "dex_id": 3,
        "dex_volatility_avg": 0.014,
        "dex_velocity_avg": 0.008
    }

    model = BreakoutPredictor()
    result = model.predict(token_sample)
    model.report(result)
