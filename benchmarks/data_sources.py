"""
===========================================================================
Data Sources: Synthetic, Yahoo Finance, CSV
===========================================================================

Unified data interface so every benchmark uses the same data regardless
of source.  Switch from synthetic to real data with one flag:

    python benchmarks/run_benchmarks.py --data synthetic
    python benchmarks/run_benchmarks.py --data yahoo --tickers AAPL MSFT
    python benchmarks/run_benchmarks.py --data csv --path data.csv
===========================================================================
"""

import sys, os
import numpy as np
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import FinancialDataLoader, FeatureEngineering
from environments.stock_trading_env import StockTradingEnv


class DataSource:
    """
    Unified data provider that creates train + test environments.

    Supports three backends:
      1. Synthetic (always available, reproducible)
      2. Yahoo Finance via yfinance (needs internet)
      3. CSV file (local data)
    """

    @staticmethod
    def from_synthetic(
        n_days: int = 3000,
        seed: int = 42,
        train_ratio: float = 0.67,
        reward_type: str = "sharpe",
    ) -> Tuple[StockTradingEnv, StockTradingEnv, dict]:
        """Generate synthetic GBM data with regime switching."""
        data = FinancialDataLoader.generate_synthetic_data(
            len_data=n_days, seed=seed,
        )
        return DataSource._split_and_build(
            data.prices, data.features, train_ratio, reward_type,
            meta={"source": "synthetic", "n_days": n_days, "seed": seed},
        )

    @staticmethod
    def from_yahoo(
        ticker: str = "SPY",
        start: str = "2018-01-01",
        end: str = "2024-01-01",
        train_ratio: float = 0.67,
        reward_type: str = "sharpe",
    ) -> Tuple[StockTradingEnv, StockTradingEnv, dict]:
        """Load real market data from Yahoo Finance."""
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end, progress=False)
            prices = df["Close"].values.flatten()
            features = FeatureEngineering.compute_all_features(prices)
            meta = {"source": "yahoo", "ticker": ticker, "start": start,
                    "end": end, "n_days": len(prices)}
            return DataSource._split_and_build(
                prices, features, train_ratio, reward_type, meta=meta,
            )
        except ImportError:
            print("  yfinance not installed. Falling back to synthetic data.")
            print("  Install with: pip install yfinance")
            return DataSource.from_synthetic(reward_type=reward_type)
        except Exception as e:
            print(f"  Yahoo download failed ({e}). Falling back to synthetic.")
            return DataSource.from_synthetic(reward_type=reward_type)

    @staticmethod
    def from_csv(
        path: str,
        price_column: str = "Close",
        train_ratio: float = 0.67,
        reward_type: str = "sharpe",
    ) -> Tuple[StockTradingEnv, StockTradingEnv, dict]:
        """Load data from a local CSV file."""
        import pandas as pd
        df = pd.read_csv(path)
        prices = df[price_column].values.astype(float)
        features = FeatureEngineering.compute_all_features(prices)
        meta = {"source": "csv", "path": path, "n_days": len(prices)}
        return DataSource._split_and_build(
            prices, features, train_ratio, reward_type, meta=meta,
        )

    @staticmethod
    def _split_and_build(
        prices, features, train_ratio, reward_type, meta,
    ) -> Tuple[StockTradingEnv, StockTradingEnv, dict]:
        split = int(len(prices) * train_ratio)
        features_aligned = features[:len(prices) - 1]

        train_prices = prices[:split]
        test_prices = prices[split:]
        train_feat = features_aligned[:split - 1] if len(features_aligned) >= split else features_aligned
        test_feat = features_aligned[split - 1:] if len(features_aligned) >= split else features_aligned

        train_env = StockTradingEnv(
            prices=train_prices,
            features=FeatureEngineering.compute_all_features(train_prices),
            reward_type=reward_type,
            discrete_actions=True,
        )
        test_env = StockTradingEnv(
            prices=test_prices,
            features=FeatureEngineering.compute_all_features(test_prices),
            reward_type=reward_type,
            discrete_actions=True,
        )
        meta["train_days"] = len(train_prices)
        meta["test_days"] = len(test_prices)
        return train_env, test_env, meta
