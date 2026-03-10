"""
===========================================================================
LLM as Trading Policy Agent
===========================================================================

Two variants:

1. LLMDirectPolicy  -- Pure LLM: reads market state, outputs trade action.
   Registered as a benchmark strategy for head-to-head comparison.

2. LLMGuidedRL      -- LLM acts as a teacher that biases the RL agent's
   exploration.  The LLM suggestion has weight `guidance_strength` in
   the action selection, decaying over training as the agent learns.

Both convert the raw observation vector into readable market text so
the LLM can reason about it.
===========================================================================
"""

import json
import sys
import os
import numpy as np
from typing import Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.providers import LLMProvider, get_provider, _parse_json_response
from benchmarks.registry import BenchmarkRegistry, BaseStrategy


ACTION_NAMES = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
ACTION_MAP = {name: i for i, name in enumerate(ACTION_NAMES)}

POLICY_PROMPT = """You are a quantitative trading agent managing a stock portfolio.

CURRENT MARKET STATE:
- 5-day trend: {trend_5d}
- 20-day trend: {trend_20d}
- 10-day volatility (annualised): {vol:.1%}
- RSI (0-1, below 0.3 = oversold, above 0.7 = overbought): {rsi:.2f}
- 10-day momentum: {momentum:+.4f}
- MACD signal: {macd:+.4f}
- Bollinger position (-1=lower band, +1=upper band): {bollinger:+.2f}

PORTFOLIO STATE:
- Current position: {position:.0%} invested
- Unrealised P&L: {pnl:+.2%}
- Drawdown from peak: {drawdown:.2%}

Choose ONE action from: strong_sell, sell, hold, buy, strong_buy

Consider:
- Oversold RSI with positive momentum -> potential buy
- Overbought RSI with negative momentum -> potential sell
- High volatility -> reduce position size (sell/hold)
- Large drawdown -> be conservative (hold/sell)

Respond ONLY with JSON: {{"action": "<action>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}"""


def _obs_to_market_text(obs: np.ndarray, info: Dict = None) -> Dict:
    """Convert numerical observation to readable context for the LLM."""
    ma5 = float(obs[2]) if len(obs) > 2 else 0
    ma20 = float(obs[3]) if len(obs) > 3 else 0

    def trend_word(ratio):
        if ratio > 0.02:
            return f"strongly bullish ({ratio:+.3f})"
        elif ratio > 0.005:
            return f"mildly bullish ({ratio:+.3f})"
        elif ratio < -0.02:
            return f"strongly bearish ({ratio:+.3f})"
        elif ratio < -0.005:
            return f"mildly bearish ({ratio:+.3f})"
        return f"flat ({ratio:+.3f})"

    return {
        "trend_5d": trend_word(ma5),
        "trend_20d": trend_word(ma20),
        "vol": float(obs[5]) if len(obs) > 5 else 0.15,
        "rsi": float(obs[7]) if len(obs) > 7 else 0.5,
        "momentum": float(obs[8]) if len(obs) > 8 else 0.0,
        "macd": float(obs[10]) if len(obs) > 10 else 0.0,
        "bollinger": float(obs[11]) if len(obs) > 11 else 0.0,
        "position": float(obs[-4]) if len(obs) > 3 else 0.0,
        "pnl": float(info.get("total_return", 0)) if info else 0.0,
        "drawdown": float(info.get("max_drawdown", 0)) if info else 0.0,
    }


def _parse_action(result: Dict) -> int:
    """Convert LLM JSON response to an action index (0-4)."""
    action_str = str(result.get("action", "hold")).lower().strip()
    return ACTION_MAP.get(action_str, 2)  # default to hold


# =====================================================================
# Variant 1: LLM as direct trading policy
# =====================================================================

class LLMDirectPolicy(BaseStrategy):
    """
    The LLM directly decides every trade action.
    Registered as a benchmark strategy for fair comparison.
    """

    def __init__(self, provider: Optional[LLMProvider] = None, model_name: str = ""):
        self.provider = provider or get_provider("auto")
        self._model_label = model_name or self.provider.model_name
        self._last_info: Dict = {}
        self._call_count = 0
        self._action_counts = np.zeros(5, dtype=int)

    def train(self, env, config: Dict):
        pass

    def predict(self, obs: np.ndarray, info: Dict = None) -> int:
        self._last_info = info or {}
        self._call_count += 1

        context = _obs_to_market_text(obs, info)
        prompt = POLICY_PROMPT.format(**context)

        try:
            result = self.provider.chat_json([{"role": "user", "content": prompt}])
            action = _parse_action(result)
        except Exception:
            action = 2  # hold on failure

        self._action_counts[action] += 1
        return action

    def reset(self):
        self._last_info = {}


# =====================================================================
# Variant 2: LLM-guided exploration for RL agents
# =====================================================================

class LLMGuidedRL:
    """
    LLM acts as a teacher that biases the RL agent's action selection.

    During training, the agent's epsilon-greedy or softmax exploration
    is blended with the LLM's suggestion:

        if random() < guidance_strength:
            action = llm_action
        else:
            action = agent_action

    guidance_strength decays from initial value to 0 over training,
    letting the agent eventually act independently.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        guidance_strength: float = 0.5,
        decay_rate: float = 0.995,
        min_strength: float = 0.0,
        query_every_n: int = 3,
    ):
        self.provider = provider or get_provider("auto")
        self.guidance_strength = guidance_strength
        self.decay_rate = decay_rate
        self.min_strength = min_strength
        self.query_every_n = query_every_n

        self._step = 0
        self._cached_action = 2
        self._current_strength = guidance_strength

    def suggest_action(self, obs: np.ndarray, info: Dict = None) -> Optional[int]:
        """
        Return the LLM's suggested action, or None if skipping this step.
        Call this alongside the RL agent's own action selection.
        """
        self._step += 1

        if np.random.random() > self._current_strength:
            return None

        if self._step % self.query_every_n != 0:
            return self._cached_action

        context = _obs_to_market_text(obs, info)
        prompt = POLICY_PROMPT.format(**context)

        try:
            result = self.provider.chat_json([{"role": "user", "content": prompt}])
            self._cached_action = _parse_action(result)
        except Exception:
            pass

        return self._cached_action

    def select_action(
        self,
        agent_action: int,
        obs: np.ndarray,
        info: Dict = None,
    ) -> int:
        """Blend agent's action with LLM suggestion."""
        llm_action = self.suggest_action(obs, info)
        if llm_action is not None:
            return llm_action
        return agent_action

    def decay(self):
        """Call at the end of each episode to decay guidance strength."""
        self._current_strength = max(
            self.min_strength,
            self._current_strength * self.decay_rate,
        )

    def reset(self):
        self._step = 0
        self._cached_action = 2
