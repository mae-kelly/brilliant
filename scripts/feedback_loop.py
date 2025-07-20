
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

import os
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from collections import deque
import tensorflow as tf

from model_inference import BreakoutPredictor
from utils import convert_model_to_tflite

class CognitiveFeedbackLoop:
    def __init__(self, log_dir="logs", retrain_threshold=150, update_interval=20):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.trade_log_path = os.path.join(log_dir, "executed_trades.jsonl")
        self.feedback_log_path = os.path.join(log_dir, "feedback_summary.jsonl")
        self.model_history_path = os.path.join(log_dir, "model_registry.jsonl")
        self.episodic_memory = []
        self.retrain_data = deque(maxlen=2000)
        self.episode = 0
        self.predictor = BreakoutPredictor()
        self.performance_window = deque(maxlen=50)
        self.volatility_state = "neutral"

    def load_trades(self):
        trades = []
        if not os.path.exists(self.trade_log_path):
            return trades
        with open(self.trade_log_path, "r") as f:
            for line in f:
                try:
                    trade = json.loads(line.strip())
                    if "roi" in trade and "features" in trade:
                        trades.append(trade)
except Exception as e:
    logger.error(f"Error in operation: {e}")
    continue
        return trades

    def classify_regime(self, roi_list):
        std = np.std(roi_list)
        if std > 0.4:
            return "volatile"
        elif std < 0.1:
            return "choppy"
        return "neutral"

    def compute_fitness_hash(self, model_weights):
        flat = np.concatenate([w.flatten() for w in model_weights])
        return hashlib.sha256(flat.tobytes()).hexdigest()

    def evaluate_model(self, model, X_val, y_val):
        eval_loss, eval_acc = model.evaluate(X_val, y_val, verbose=0)
        return {"loss": float(eval_loss), "accuracy": float(eval_acc)}

    def record_model_metadata(self, model, model_path, tflite_path, fitness_score):
        metadata = {
            "episode": self.episode,
            "timestamp": datetime.utcnow().isoformat(),
            "model_path": model_path,
            "tflite_path": tflite_path,
            "fitness_score": fitness_score,
            "hash": self.compute_fitness_hash(model.get_weights())
        }
        with open(self.model_history_path, "a") as f:
            f.write(json.dumps(metadata) + "\n")

    def retrain_model(self, trades):
        X = np.array([t["features"] for t in trades])
        y = np.array([1 if t["roi"] > 0 else 0 for t in trades])

        val_split = int(0.8 * len(X))
        X_train, X_val = X[:val_split], X[val_split:]
        y_train, y_val = y[:val_split], y[val_split:]

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=8, batch_size=24, verbose=0)

        evaluation = self.evaluate_model(model, X_val, y_val)
        model_path = os.path.join(self.log_dir, f"model_ep{self.episode}.h5")
        tflite_path = os.path.join(self.log_dir, f"model_ep{self.episode}.tflite")
        model.save(model_path)

        convert_model_to_tflite(model_path, tflite_path)
        self.predictor.reload_model(tflite_path)

        self.record_model_metadata(model, model_path, tflite_path, evaluation["accuracy"])

    def summarize_and_log_feedback(self, trades):
        rois = [t["roi"] for t in trades]
        volatility = np.std(rois)
        mean_roi = np.mean(rois)
        win_rate = sum(r > 0 for r in rois) / len(rois)
        self.performance_window.append(mean_roi)

        feedback = {
            "episode": self.episode,
            "timestamp": datetime.utcnow().isoformat(),
            "volatility": volatility,
            "mean_roi": mean_roi,
            "win_rate": win_rate,
            "strategy_regime": self.classify_regime(rois),
            "threshold": self.predictor.threshold,
            "model_id": f"ep{self.episode}"
        }

        with open(self.feedback_log_path, "a") as f:
            f.write(json.dumps(feedback) + "\n")

        return mean_roi, win_rate

    def adapt_thresholds(self, win_rate, mean_roi):
        if win_rate < 0.4:
            self.predictor.threshold = min(0.98, self.predictor.threshold + 0.02)
        elif mean_roi > 0.15:
            self.predictor.threshold = max(0.6, self.predictor.threshold - 0.01)

    def archive_trades(self):
        archive_path = self.trade_log_path + f".ep{self.episode}"
        if os.path.exists(self.trade_log_path):
            os.rename(self.trade_log_path, archive_path)

    def run(self):
        trades = self.load_trades()
        if len(trades) < self.update_interval:
            print(f"[WAIT] Awaiting trades. Current: {len(trades)}")
            return

        mean_roi, win_rate = self.summarize_and_log_feedback(trades)
        self.adapt_thresholds(win_rate, mean_roi)
        self.retrain_data.extend(trades)

        if len(self.retrain_data) >= self.retrain_threshold:
            print(f"[RETRAIN] Triggered at Episode {self.episode}...")
            self.retrain_model(list(self.retrain_data))

        self.archive_trades()
        self.episode += 1

if __name__ == "__main__":
    loop = CognitiveFeedbackLoop()
    while True:
        loop.run()
        time.sleep(60)
