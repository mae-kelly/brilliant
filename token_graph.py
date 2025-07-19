
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
import os
import uuid
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from scipy.stats import entropy

sns.set(style="whitegrid")

class TokenGraph:
    def __init__(self, output_dir="charts", interactive=False):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.interactive = interactive

    def _normalize(self, series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    def _compute_entropy(self, series, bins=15):
        hist, _ = np.histogram(series, bins=bins, density=True)
        return entropy(hist + 1e-9)

    def _compute_momentum(self, price_series):
        return price_series.diff().fillna(0)

    def _compute_volatility(self, price_series, window=5):
        return price_series.rolling(window).std().fillna(0)

    def _detect_breakout_signals(self, price_series, momentum_series):
        local_max = argrelextrema(price_series.values, np.greater, order=3)[0]
        local_min = argrelextrema(price_series.values, np.less, order=3)[0]
        breakouts = []
        for i in range(3, len(momentum_series)):
            if momentum_series.iloc[i] > momentum_series.iloc[i - 1] * 2 and price_series.iloc[i] > price_series.iloc[i - 1] * 1.1:
                breakouts.append(i)
        return local_max, local_min, breakouts

    def _classify_regimes(self, volatility_series):
        regimes = []
        threshold_high = volatility_series.mean() + volatility_series.std()
        threshold_low = volatility_series.mean() - volatility_series.std()
        for v in volatility_series:
            if v >= threshold_high:
                regimes.append('HIGH')
            elif v <= threshold_low:
                regimes.append('LOW')
            else:
                regimes.append('MEDIUM')
        return regimes

    def _predictive_divergence(self, price, volume):
        norm_price = self._normalize(price)
        norm_volume = self._normalize(volume)
        divergence = norm_volume - norm_price
        return divergence

    def plot_static_graph(self, token_data, token_symbol=None):
        df = pd.DataFrame(token_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')

        price = df['price']
        volume = df['volume']
        liquidity = df['liquidity']
        momentum = self._compute_momentum(price)
        volatility = self._compute_volatility(price)
        entropy_series = df['price'].rolling(window=5).apply(self._compute_entropy)
        divergence = self._predictive_divergence(price, volume)

        local_max, local_min, breakouts = self._detect_breakout_signals(price, momentum)
        regimes = self._classify_regimes(volatility)

        fig, axs = plt.subplots(6, 1, figsize=(18, 24), sharex=True)

        axs[0].plot(price, label='Price', color='black')
        axs[0].scatter(df.index[local_max], price.iloc[local_max], color='green', label='Peaks')
        axs[0].scatter(df.index[local_min], price.iloc[local_min], color='red', label='Troughs')
        axs[0].scatter(df.index[breakouts], price.iloc[breakouts], color='blue', label='Breakouts', marker='x')
        axs[0].set_title(f"{token_symbol or 'Token'} Price")

        for i, regime in enumerate(regimes):
            if regime == 'HIGH':
                axs[0].axvspan(df.index[i], df.index[i], color='red', alpha=0.05)
            elif regime == 'LOW':
                axs[0].axvspan(df.index[i], df.index[i], color='green', alpha=0.05)

        axs[1].plot(volume, label='Volume', color='orange')
        axs[1].set_title("Volume")

        axs[2].plot(liquidity, label='Liquidity', color='purple')
        axs[2].set_title("Liquidity")

        axs[3].plot(momentum, label='Momentum', color='blue')
        axs[3].set_title("Momentum")

        axs[4].plot(entropy_series, label='Entropy', color='brown')
        axs[4].axhline(entropy_series.mean() + entropy_series.std(), linestyle='--', color='red', alpha=0.4, label='Entropy Spike Threshold')
        axs[4].set_title("Entropy (Volatility Complexity)")

        axs[5].plot(divergence, label='Vol-Px Divergence', color='magenta')
        axs[5].axhline(0, linestyle='--', color='gray', alpha=0.3)
        axs[5].set_title("Volume vs Price Divergence (Predictive Anomaly)")

        for ax in axs:
            ax.grid(True)
            ax.legend()

        fig.autofmt_xdate()
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"{token_symbol or uuid.uuid4()}.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def visualize(self, token_data, token_symbol=None):
        return self.plot_static_graph(token_data, token_symbol)
