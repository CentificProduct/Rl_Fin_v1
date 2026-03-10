"""
===========================================================================
LLM as Reward Model for RL Agents
===========================================================================

The LLM evaluates trade decisions and produces a quality score that gets
blended into the RL reward signal.  This implements the pattern from
FinRL-DeepSeek and SAPPO where LLM judgment augments numerical rewards.

Flow:
    Agent takes action -> env returns base reward
    -> LLM scores the trade context -> blended reward sent to agent

The LLM is NOT called on every step (too slow).  Instead it is called:
    - Every N steps (configurable)
    - Only when the agent actually trades (position changes)
    - Results are cached and interpolated between calls
===========================================================================
"""

import json
import numpy as np
from typing import Dict, Optional, List
from collections import OrderedDict

from .providers import LLMProvider, get_provider, _parse_json_response


ACTION_NAMES = ["strong_sell", "sell", "hold", "buy", "strong_buy"]

TRADE_EVAL_PROMPT = """You are a quantitative trading analyst evaluating a trade decision.

MARKET STATE:
- Price trend (5d MA ratio): {ma5_ratio:+.4f}
- Price trend (20d MA ratio): {ma20_ratio:+.4f}
- Volatility (10d annualised): {vol_10:.1%}
- RSI (14-period, 0-1 scale): {rsi:.2f}
- Momentum (10d): {momentum:+.4f}
- Current drawdown from peak: {drawdown:.2%}

PORTFOLIO STATE:
- Current position: {position:.1%} of portfolio
- Unrealised P&L: {unrealized_pnl:+.2%}
- Cash ratio: {cash_ratio:.1%}

TRADE DECISION: {action_name}

Evaluate this trade decision on a scale from -1.0 (terrible) to +1.0 (excellent).
Consider: Is the action appropriate given the trend, volatility, RSI, and current position?

Respond ONLY with JSON: {{"score": <float -1 to 1>, "reasoning": "<one sentence>"}}"""


class LLMRewardModel:
    """
    Uses an LLM to score trade quality as a reward component.

    The score is blended with the base numerical reward:
        final_reward = (1 - llm_weight) * base_reward + llm_weight * llm_score
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        llm_weight: float = 0.3,
        eval_every_n_steps: int = 5,
        only_on_trades: bool = True,
        cache_size: int = 256,
    ):
        self.provider = provider or get_provider("auto")
        self.llm_weight = llm_weight
        self.eval_every_n_steps = eval_every_n_steps
        self.only_on_trades = only_on_trades

        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._step_count = 0
        self._last_score = 0.0
        self._scores_history: List[float] = []

    def score_trade(
        self,
        action: int,
        obs: np.ndarray,
        info: Dict,
    ) -> float:
        """
        Ask the LLM to evaluate this trade decision.
        Returns a score in [-1.0, +1.0].
        """
        self._step_count += 1

        if self.only_on_trades and action == 2:
            return self._last_score * 0.95  # decay toward zero for holds

        if self._step_count % self.eval_every_n_steps != 0:
            return self._last_score

        cache_key = self._make_cache_key(action, obs)
        if cache_key in self._cache:
            return self._cache[cache_key]

        context = self._build_context(action, obs, info)
        prompt = TRADE_EVAL_PROMPT.format(**context)

        try:
            messages = [{"role": "user", "content": prompt}]
            result = self.provider.chat_json(messages)
            score = float(result.get("score", 0.0))
            score = max(-1.0, min(1.0, score))
        except Exception as e:
            score = 0.0

        self._cache[cache_key] = score
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        self._last_score = score
        self._scores_history.append(score)
        return score

    def blend_reward(
        self,
        base_reward: float,
        action: int,
        obs: np.ndarray,
        info: Dict,
    ) -> float:
        """Compute the blended reward: base + LLM component."""
        llm_score = self.score_trade(action, obs, info)
        blended = (1 - self.llm_weight) * base_reward + self.llm_weight * llm_score
        return blended

    def reset(self):
        """Reset per-episode state."""
        self._step_count = 0
        self._last_score = 0.0

    @property
    def avg_score(self) -> float:
        if not self._scores_history:
            return 0.0
        return float(np.mean(self._scores_history[-50:]))

    def _build_context(self, action: int, obs: np.ndarray, info: Dict) -> Dict:
        return {
            "ma5_ratio": float(obs[2]) if len(obs) > 2 else 0.0,
            "ma20_ratio": float(obs[3]) if len(obs) > 3 else 0.0,
            "vol_10": float(obs[5]) if len(obs) > 5 else 0.15,
            "rsi": float(obs[7]) if len(obs) > 7 else 0.5,
            "momentum": float(obs[8]) if len(obs) > 8 else 0.0,
            "drawdown": float(info.get("max_drawdown", 0)),
            "position": float(obs[-4]) if len(obs) > 3 else 0.0,
            "unrealized_pnl": float(info.get("total_return", 0)),
            "cash_ratio": float(obs[-3]) if len(obs) > 3 else 1.0,
            "action_name": ACTION_NAMES[action] if 0 <= action < 5 else "hold",
        }

    def _make_cache_key(self, action: int, obs: np.ndarray) -> tuple:
        quantized = tuple(np.round(obs[:6], 2))
        return (action, quantized)
