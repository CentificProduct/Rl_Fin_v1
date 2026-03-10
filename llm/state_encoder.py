"""
===========================================================================
LLM as State Encoder / Sentiment Feature Generator
===========================================================================

Two components:

1. LLMStateEncoder
   - Generates synthetic financial news from market conditions (no API needed)
   - Asks the LLM to score sentiment/risk/confidence
   - Produces a fixed-dim feature vector for the RL agent's state

2. LLMAugmentedTradingEnv
   - Gymnasium wrapper around StockTradingEnv
   - Appends LLM-derived sentiment features to the observation
   - Original 16-dim state becomes 19-dim: [..., sentiment, risk, confidence]
===========================================================================
"""

import json
import sys
import os
import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Tuple, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.providers import LLMProvider, get_provider, _parse_json_response


# Headline templates based on market conditions
_HEADLINE_TEMPLATES = {
    "strong_bull": [
        "Markets surge as strong momentum continues, indices hit new highs",
        "Investor confidence soars amid robust economic indicators",
        "Bull run extends: analysts raise price targets across the board",
    ],
    "mild_bull": [
        "Markets edge higher on steady economic data",
        "Gradual gains continue as earnings beat expectations",
        "Positive sentiment supports modest market advance",
    ],
    "flat": [
        "Markets trade sideways as investors await key data releases",
        "Mixed signals keep indices range-bound in quiet session",
        "Low volatility persists as market participants stay cautious",
    ],
    "mild_bear": [
        "Markets slip on profit-taking after recent gains",
        "Mild selling pressure as economic concerns weigh on sentiment",
        "Indices retreat modestly amid mixed corporate guidance",
    ],
    "strong_bear": [
        "Sharp sell-off as recession fears grip markets",
        "Volatility spikes to multi-month highs on heavy selling",
        "Bear market territory: indices plunge on systemic risk concerns",
    ],
    "high_vol": [
        "Wild swings as volatility explodes, VIX hits elevated levels",
        "Chaotic trading session with extreme intraday reversals",
        "Risk-off mode: investors flee to safe havens amid turbulence",
    ],
}

SENTIMENT_PROMPT = """You are a financial sentiment analyst. Score this market headline.

Headline: "{headline}"

Provide scores as JSON:
- sentiment: float from -1.0 (very bearish) to +1.0 (very bullish)
- risk: float from 0.0 (low risk) to 1.0 (extreme risk)
- confidence: float from 0.0 (uncertain) to 1.0 (very confident)

Respond ONLY with JSON: {{"sentiment": <float>, "risk": <float>, "confidence": <float>}}"""


class LLMStateEncoder:
    """
    Generates sentiment features from market context using an LLM.

    For demo purposes, it creates synthetic headlines from market
    indicators (so no external news API is needed), then asks the LLM
    to score them.  In production, replace _generate_headline() with
    a real news feed.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        update_every_n: int = 5,
        cache_size: int = 128,
    ):
        self.provider = provider or get_provider("auto")
        self.update_every_n = update_every_n

        self._step = 0
        self._cached_features = np.array([0.0, 0.3, 0.5], dtype=np.float32)
        self._history: List[np.ndarray] = []
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """
        Return a 3-dim sentiment vector: [sentiment, risk, confidence].
        Cached and updated every N steps to avoid excessive LLM calls.
        """
        self._step += 1

        if self._step % self.update_every_n != 1 and self._step > 1:
            return self._cached_features

        headline = self._generate_headline(obs)

        cache_key = headline[:50]
        if cache_key in self._cache:
            self._cached_features = self._cache[cache_key]
            return self._cached_features

        prompt = SENTIMENT_PROMPT.format(headline=headline)

        try:
            result = self.provider.chat_json([{"role": "user", "content": prompt}])
            features = np.array([
                float(result.get("sentiment", 0.0)),
                float(result.get("risk", 0.3)),
                float(result.get("confidence", 0.5)),
            ], dtype=np.float32)
            features[0] = np.clip(features[0], -1.0, 1.0)
            features[1] = np.clip(features[1], 0.0, 1.0)
            features[2] = np.clip(features[2], 0.0, 1.0)
        except Exception:
            features = self._cached_features

        self._cached_features = features
        self._cache[cache_key] = features
        if len(self._cache) > self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._history.append(features.copy())
        return features

    def reset(self):
        self._step = 0
        self._cached_features = np.array([0.0, 0.3, 0.5], dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return 3

    @property
    def feature_names(self) -> List[str]:
        return ["llm_sentiment", "llm_risk", "llm_confidence"]

    def _generate_headline(self, obs: np.ndarray) -> str:
        """Create a synthetic headline from market indicators."""
        momentum = float(obs[8]) if len(obs) > 8 else 0.0
        vol = float(obs[5]) if len(obs) > 5 else 0.15
        rsi = float(obs[7]) if len(obs) > 7 else 0.5

        if vol > 0.35:
            regime = "high_vol"
        elif momentum > 0.03:
            regime = "strong_bull"
        elif momentum > 0.005:
            regime = "mild_bull"
        elif momentum < -0.03:
            regime = "strong_bear"
        elif momentum < -0.005:
            regime = "mild_bear"
        else:
            regime = "flat"

        templates = _HEADLINE_TEMPLATES[regime]
        idx = self._step % len(templates)
        return templates[idx]


class LLMAugmentedTradingEnv(gym.Wrapper):
    """
    Gymnasium wrapper that augments the observation with LLM sentiment features.

    Original observation: [market_features(12) + portfolio_features(4)] = 16-dim
    Augmented observation: [market_features(12) + portfolio_features(4) + llm_features(3)] = 19-dim

    Drop-in replacement for StockTradingEnv:
        env = LLMAugmentedTradingEnv(StockTradingEnv(...), provider=my_llm)
    """

    def __init__(self, env: gym.Env, provider: Optional[LLMProvider] = None, **encoder_kwargs):
        super().__init__(env)
        self.encoder = LLMStateEncoder(provider=provider, **encoder_kwargs)

        orig_shape = env.observation_space.shape[0]
        new_dim = orig_shape + self.encoder.feature_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(new_dim,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.encoder.reset()
        augmented = self._augment(obs)
        return augmented, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        augmented = self._augment(obs)
        return augmented, reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        sentiment_features = self.encoder.encode(obs)
        return np.concatenate([obs, sentiment_features]).astype(np.float32)
