import os
import json
import numpy as np
import pandas as pd
import time
from collections import deque
from datetime import datetime, timedelta
from utils.feedback_utils import load_trade_logs, compute_roi_deltas
from utils.token_profiler import compute_entropy_score
from scipy.signal import savgol_filter


class RegimeClassifier:
    def __init__(self, window=50):
        self.window = window
        self.returns = deque(maxlen=window)
        self.entropies = deque(maxlen=window)
        self.last_regime = None

    def classify(self, roi, entropy):
        self.returns.append(roi)
        self.entropies.append(entropy)

        if len(self.returns) < self.window:
            return "loading"

        vol = np.std(self.returns)
        mean_entropy = np.mean(self.entropies)

        if vol < 0.05 and mean_entropy < 0.4:
            self.last_regime = "stable"
        elif vol > 0.15 and mean_entropy > 0.6:
            self.last_regime = "chaotic"
        elif mean_entropy > 0.75:
            self.last_regime = "speculative"
        else:
            self.last_regime = "unknown"

        return self.last_regime


class KalmanFilter:
    def __init__(self, R=0.01, Q=1e-4):
        self.x = 0
        self.P = 1.0
        self.R = R
        self.Q = Q

    def update(self, z):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return self.x


class AdaptiveOptimizer:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.param_bounds = {
            "slippage_tolerance": (0.001, 0.05),
            "hold_duration": (10, 600),
            "entry_confidence": (0.6, 0.99),
            "exit_momentum_drop": (0.001, 0.02)
        }

        self.state = {
            "slippage_tolerance": self.config.get("default_slippage", 0.01),
            "hold_duration": self.config.get("default_hold_seconds", 180),
            "entry_confidence": self.config.get("entry_threshold", 0.85),
            "exit_momentum_drop": self.config.get("exit_threshold", 0.005)
        }

        self.performance_history = deque(maxlen=300)
        self.entropy_history = deque(maxlen=300)
        self.kalman = KalmanFilter()
        self.regime = RegimeClassifier()
        self.last_meta_learning = time.time()

        self.episode_rewards = []
        self.episode_window = deque(maxlen=5)

        self.fallback_state = {
            "slippage_tolerance": 0.02,
            "hold_duration": 90,
            "entry_confidence": 0.9,
            "exit_momentum_drop": 0.01
        }

    def clip(self, key, value):
        low, high = self.param_bounds[key]
        return float(np.clip(value, low, high))

    def reward_from_roi(self, roi):
        if roi > 1.5: return 3
        elif roi > 1.2: return 2
        elif roi > 1.05: return 1
        elif roi > 0.95: return -1
        elif roi > 0.85: return -2
        else: return -3

    def update(self, token_meta, trade_outcome):
        roi = trade_outcome.get("roi", 1.0)
        entropy = compute_entropy_score(token_meta)
        smoothed_roi = self.kalman.update(roi)

        self.performance_history.append(smoothed_roi)
        self.entropy_history.append(entropy)

        regime = self.regime.classify(roi, entropy)
        reward = self.reward_from_roi(roi)
        self.episode_window.append(reward)

        # === Recursive Parameter Adjustments ===

        # Confidence tuning
        if regime == "stable" and reward > 0:
            self.state["entry_confidence"] = self.clip("entry_confidence", self.state["entry_confidence"] - 0.005)
        elif regime == "chaotic":
            self.state["entry_confidence"] = self.clip("entry_confidence", self.state["entry_confidence"] + 0.01)

        # Exit tuning
        if reward >= 2:
            self.state["exit_momentum_drop"] = self.clip("exit_momentum_drop", self.state["exit_momentum_drop"] - 0.001)
        elif reward <= -2:
            self.state["exit_momentum_drop"] = self.clip("exit_momentum_drop", self.state["exit_momentum_drop"] + 0.002)

        # Slippage tuning
        if reward < 0:
            self.state["slippage_tolerance"] = self.clip("slippage_tolerance", self.state["slippage_tolerance"] + 0.0015)
        elif reward > 0:
            self.state["slippage_tolerance"] = self.clip("slippage_tolerance", self.state["slippage_tolerance"] - 0.001)

        # Hold duration tuning
        if np.mean(self.episode_window) > 0:
            self.state["hold_duration"] = self.clip("hold_duration", self.state["hold_duration"] + 10)
        elif np.mean(self.episode_window) < -1:
            self.state["hold_duration"] = self.clip("hold_duration", self.state["hold_duration"] - 10)

        self.meta_learning_check()

    def meta_learning_check(self):
        if time.time() - self.last_meta_learning < 240:
            return

        logs = load_trade_logs("logs/trade_log.csv")
        if logs.empty: return

        roi_series = compute_roi_deltas(logs)
        entropy_array = np.array(self.entropy_history)

        if len(roi_series) > 50:
            smoothed = savgol_filter(roi_series.values, 21, 3)
            spike_count = np.sum(np.diff(np.sign(np.diff(smoothed))) != 0)

            if spike_count > 15 or np.mean(entropy_array[-30:]) > 0.85:
                # Reset config to safe
                self.state.update(self.fallback_state)

        self.last_meta_learning = time.time()

    def get_state(self):
        return self.state

    def save_state(self, path="config/optimizer_state.json"):
        with open(path, "w") as f:
            json.dump(self.state, f, indent=2)

    def load_state(self, path="config/optimizer_state.json"):
        if os.path.exists(path):
            with open(path) as f:
                self.state = json.load(f)
