"""
=============================================================================
UTILITY: Financial Data Loading and Preprocessing
=============================================================================

Handles data acquisition from multiple sources, feature engineering for
RL state representations, and normalization/scaling for neural networks.
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MarketData:
    """Container for processed market data."""
    prices: np.ndarray
    returns: np.ndarray
    features: np.ndarray
    feature_names: List[str]
    dates: Optional[np.ndarray] = None
    tickers: Optional[List[str]] = None


class FinancialDataLoader:
    """
    Loads and preprocesses financial data for RL environments.

    Supports:
      - Yahoo Finance via yfinance
      - Synthetic data generation (for testing/research)
      - CSV file loading
    """

    @staticmethod
    def load_from_yfinance(
        tickers: List[str],
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
    ) -> MarketData:
        """Load real market data from Yahoo Finance."""
        try:
            import yfinance as yf
            data = yf.download(tickers, start=start_date, end=end_date)

            if len(tickers) == 1:
                prices = data['Close'].values
            else:
                prices = data['Close'].values

            returns = np.diff(prices, axis=0) / prices[:-1]
            features = FeatureEngineering.compute_all_features(prices)
            feature_names = FeatureEngineering.get_feature_names()

            return MarketData(
                prices=prices,
                returns=returns,
                features=features,
                feature_names=feature_names,
                dates=data.index.values[1:],  # align with returns
                tickers=tickers,
            )
        except ImportError:
            print("yfinance not installed. Using synthetic data.")
            return FinancialDataLoader.generate_synthetic_data(len_data=1000)

    @staticmethod
    def generate_synthetic_data(
        len_data: int = 2000,
        n_assets: int = 1,
        seed: int = 42,
    ) -> MarketData:
        """
        Generate synthetic market data using Geometric Brownian Motion
        with regime switching.

        GBM: dS = mu*S*dt + sigma*S*dW

        With regime switching:
          - Bull regime: high mu, low sigma
          - Bear regime: negative mu, high sigma
          - Transition between regimes follows a Markov chain
        """
        np.random.seed(seed)
        dt = 1.0 / 252

        # Regime parameters
        regimes = {
            'bull': {'mu': 0.15, 'sigma': 0.12},
            'bear': {'mu': -0.10, 'sigma': 0.25},
            'sideways': {'mu': 0.02, 'sigma': 0.08},
        }
        regime_transition = np.array([
            [0.98, 0.01, 0.01],  # bull -> bull/bear/sideways
            [0.02, 0.96, 0.02],  # bear stays sticky
            [0.02, 0.02, 0.96],  # sideways is stable
        ])

        all_prices = []
        for asset in range(n_assets):
            prices = np.zeros(len_data)
            prices[0] = 100.0 * (1 + 0.1 * asset)
            current_regime = 0  # start in bull

            for t in range(1, len_data):
                current_regime = np.random.choice(3, p=regime_transition[current_regime])
                regime_name = ['bull', 'bear', 'sideways'][current_regime]
                params = regimes[regime_name]

                drift = (params['mu'] - 0.5 * params['sigma'] ** 2) * dt
                diffusion = params['sigma'] * np.sqrt(dt) * np.random.randn()
                prices[t] = prices[t - 1] * np.exp(drift + diffusion)

            all_prices.append(prices)

        prices = np.column_stack(all_prices) if n_assets > 1 else all_prices[0]
        returns = np.diff(prices, axis=0) / prices[:-1] if n_assets == 1 else np.diff(prices, axis=0) / prices[:-1]

        features = FeatureEngineering.compute_all_features(prices)
        feature_names = FeatureEngineering.get_feature_names()

        return MarketData(
            prices=prices,
            returns=returns,
            features=features,
            feature_names=feature_names,
        )

    @staticmethod
    def generate_correlated_assets(
        n_assets: int = 5,
        n_days: int = 2000,
        correlation: float = 0.5,
        seed: int = 42,
    ) -> MarketData:
        """
        Generate correlated asset prices for portfolio optimization.
        Uses a Cholesky decomposition to induce correlation structure.
        """
        np.random.seed(seed)

        mus = np.random.uniform(0.05, 0.20, n_assets)
        sigmas = np.random.uniform(0.10, 0.30, n_assets)

        corr_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(corr_matrix, 1.0)

        L = np.linalg.cholesky(corr_matrix)

        dt = 1.0 / 252
        prices = np.zeros((n_days, n_assets))
        prices[0] = 100.0

        for t in range(1, n_days):
            z = np.random.randn(n_assets)
            corr_z = L @ z

            for i in range(n_assets):
                drift = (mus[i] - 0.5 * sigmas[i] ** 2) * dt
                diffusion = sigmas[i] * np.sqrt(dt) * corr_z[i]
                prices[t, i] = prices[t - 1, i] * np.exp(drift + diffusion)

        returns = np.diff(prices, axis=0) / prices[:-1]
        feature_list = []
        for i in range(n_assets):
            feat = FeatureEngineering.compute_all_features(prices[:, i])
            feature_list.append(feat)

        features = np.concatenate(feature_list, axis=1)
        feature_names = [
            f"{name}_asset{i}"
            for i in range(n_assets)
            for name in FeatureEngineering.get_feature_names()
        ]

        tickers = [f"ASSET_{i}" for i in range(n_assets)]

        return MarketData(
            prices=prices,
            returns=returns,
            features=features,
            feature_names=feature_names,
            tickers=tickers,
        )


class FeatureEngineering:
    """
    Computes technical features from price data for RL state representation.

    STATE DESIGN PHILOSOPHY:
    The state must encode enough history to approximate the Markov property.
    Raw prices alone are insufficient. We compute features that capture:
      - Trend (moving averages, momentum)
      - Volatility (rolling std, ATR proxy)
      - Mean reversion (RSI, price-to-MA ratio)
      - Volume profile (if available)
    """

    @staticmethod
    def compute_all_features(prices: np.ndarray) -> np.ndarray:
        """Compute full feature set from price series."""
        if prices.ndim == 1:
            prices_1d = prices
        else:
            prices_1d = prices[:, 0]

        n = len(prices_1d)
        returns = np.zeros(n)
        returns[1:] = np.diff(prices_1d) / prices_1d[:-1]

        features = {
            'returns': returns,
            'log_returns': np.log1p(returns),
            'ma_5_ratio': FeatureEngineering._ma_ratio(prices_1d, 5),
            'ma_20_ratio': FeatureEngineering._ma_ratio(prices_1d, 20),
            'ma_50_ratio': FeatureEngineering._ma_ratio(prices_1d, 50),
            'volatility_10': FeatureEngineering._rolling_volatility(returns, 10),
            'volatility_30': FeatureEngineering._rolling_volatility(returns, 30),
            'rsi_14': FeatureEngineering._rsi(prices_1d, 14),
            'momentum_10': FeatureEngineering._momentum(prices_1d, 10),
            'momentum_30': FeatureEngineering._momentum(prices_1d, 30),
            'macd': FeatureEngineering._macd(prices_1d),
            'bollinger_pos': FeatureEngineering._bollinger_position(prices_1d, 20),
        }

        result = np.column_stack(list(features.values()))
        result = np.nan_to_num(result, 0.0)
        return result[1:]  # drop first row (no return)

    @staticmethod
    def get_feature_names() -> List[str]:
        return [
            'returns', 'log_returns', 'ma_5_ratio', 'ma_20_ratio',
            'ma_50_ratio', 'volatility_10', 'volatility_30', 'rsi_14',
            'momentum_10', 'momentum_30', 'macd', 'bollinger_pos',
        ]

    @staticmethod
    def _ma_ratio(prices: np.ndarray, window: int) -> np.ndarray:
        """Price / Moving Average - captures relative position to trend."""
        ma = np.convolve(prices, np.ones(window) / window, mode='full')[:len(prices)]
        ma[:window] = prices[:window]
        return prices / ma - 1.0

    @staticmethod
    def _rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
        """Annualized rolling standard deviation of returns."""
        vol = np.zeros_like(returns)
        for i in range(window, len(returns)):
            vol[i] = np.std(returns[i - window:i]) * np.sqrt(252)
        vol[:window] = vol[window] if window < len(vol) else 0.0
        return vol

    @staticmethod
    def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Relative Strength Index (0-100).
        RSI > 70: overbought (potential reversal down)
        RSI < 30: oversold (potential reversal up)
        """
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros_like(prices, dtype=float)
        avg_loss = np.zeros_like(prices, dtype=float)

        avg_gain[period] = np.mean(gains[1:period + 1])
        avg_loss[period] = np.mean(losses[1:period + 1])

        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi / 100.0  # normalize to [0, 1]

    @staticmethod
    def _momentum(prices: np.ndarray, lookback: int) -> np.ndarray:
        """Price momentum: current price / price n periods ago - 1."""
        mom = np.zeros_like(prices)
        for i in range(lookback, len(prices)):
            mom[i] = prices[i] / prices[i - lookback] - 1.0
        return mom

    @staticmethod
    def _macd(prices: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
        """MACD: difference between fast and slow exponential moving averages."""
        ema_fast = FeatureEngineering._ema(prices, fast)
        ema_slow = FeatureEngineering._ema(prices, slow)
        return (ema_fast - ema_slow) / prices

    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _bollinger_position(prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Position within Bollinger Bands (-1 to 1)."""
        ma = np.convolve(prices, np.ones(window) / window, mode='full')[:len(prices)]
        std = np.zeros_like(prices)
        for i in range(window, len(prices)):
            std[i] = np.std(prices[i - window:i])
        std[:window] = std[window] if window < len(std) else 1.0

        upper = ma + 2 * std
        lower = ma - 2 * std
        band_width = upper - lower
        band_width = np.where(band_width > 0, band_width, 1.0)

        return (prices - lower) / band_width * 2 - 1  # map to [-1, 1]
