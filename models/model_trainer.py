
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


# model_trainer.py

import os
import time
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
from tf.keras.models import Model
from tf.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Embedding, Flatten
from tf.keras.optimizers import Adam
from tf.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class BreakoutTrainer:
    def __init__(self):
        self.model = None
        self.input_dim = 9
        self.scaler = MinMaxScaler()
        self.model_path = "models/breakout_classifier.h5"

        if not os.path.exists("models"):
            os.makedirs("models")

    def load_and_preprocess(self, data_path):
        df = pd.read_csv(data_path)

        df["volatility"] = df["price_series"].apply(lambda x: np.std(eval(x)))
        df["velocity"] = df["price_series"].apply(lambda x: (eval(x)[-1] - eval(x)[0]) / (len(eval(x)) + 1e-5))

        df["breakout"] = df["roi"].apply(lambda r: 1 if r > 0.15 else 0)

        features = ["price_delta", "liquidity_delta", "volume_delta", "volatility", "velocity", "age_seconds",
                    "dex_id", "base_volatility", "base_velocity"]
        X = df[features]
        y = df["breakout"]

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

        return X_train, X_val, y_train, y_val

    def build_model(self):
        inp = Input(shape=(self.input_dim,))
        x = Dense(128, activation="relu")(inp)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        out = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(loss="binary_crossentropy", optimizer=Adam(1e-4), metrics=["accuracy", tf.keras.metrics.AUC()])
        self.model = model

    def train(self, data_path):
        X_train, X_val, y_train, y_val = self.load_and_preprocess(data_path)
        self.build_model()

        early = EarlyStopping(patience=10, restore_best_weights=True)
        reduce = ReduceLROnPlateau(patience=5, factor=0.5)

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=100, batch_size=64, callbacks=[early, reduce], verbose=2)

        self.model.save(self.model_path)

    @retry_on_failure(max_retries=3)    def predict(self, features_array):
        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)
        features_scaled = self.scaler.transform([features_array])
        return self.model.predict(features_scaled)[0][0]


# ⚙️ Training Trigger
if __name__ == "__main__":
    trainer = BreakoutTrainer()
    trainer.train("data/token_training_data.csv")
