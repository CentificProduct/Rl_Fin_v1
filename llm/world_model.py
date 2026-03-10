"""
===========================================================================
LLM as World Model / Scenario Generator
===========================================================================

The LLM predicts what happens next in the market given the current state,
producing structured scenarios that an RL agent can use for planning.

This is analogous to Model Predictive Control (MPC) in advanced/model_based.py
but uses an LLM instead of a neural network ensemble.

Usage:
    wm = LLMWorldModel(provider)
    scenarios = wm.generate_scenarios(obs, n_scenarios=5)
    best_action = wm.plan_action(obs, planning_horizon=3)
===========================================================================
"""

import json
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from .providers import LLMProvider, get_provider, _parse_json_response


@dataclass
class MarketScenario:
    """A single predicted future scenario."""
    direction: str   # "up", "down", "flat"
    magnitude: float  # expected return magnitude
    volatility_change: float  # change in volatility
    reasoning: str
    probability: float = 0.5


SCENARIO_PROMPT = """You are a market forecasting model. Given the current market state, predict the most likely next-period outcome.

CURRENT STATE:
- 5-day price trend: {trend_5d:+.4f} (positive = up)
- 20-day price trend: {trend_20d:+.4f}
- 10-day annualised volatility: {vol:.1%}
- RSI (0-1 scale): {rsi:.2f}
- 10-day momentum: {momentum:+.4f}
- Current position: {position:.0%}
- Drawdown from peak: {drawdown:.2%}

Consider mean-reversion (extreme RSI tends to reverse), momentum continuation, and volatility clustering.

Predict the next period outcome as JSON:
{{
  "direction": "<up|down|flat>",
  "magnitude": <float, expected return e.g. 0.01 for +1%>,
  "volatility_change": <float, change in vol e.g. 0.02 for +2% increase>,
  "probability": <float 0-1, confidence in this scenario>,
  "reasoning": "<one sentence>"
}}

Respond ONLY with the JSON object."""

MULTI_SCENARIO_PROMPT = """You are a market scenario generator. Given the current state, generate {n} distinct scenarios for the next period.

CURRENT STATE:
- 5-day price trend: {trend_5d:+.4f}
- 20-day price trend: {trend_20d:+.4f}
- Volatility: {vol:.1%}
- RSI: {rsi:.2f}
- Momentum: {momentum:+.4f}

Generate {n} scenarios as a JSON array. Each scenario has: direction, magnitude, volatility_change, probability, reasoning.
Probabilities should sum to approximately 1.0.

Respond ONLY with a JSON array: [{{"direction": "...", "magnitude": ..., "volatility_change": ..., "probability": ..., "reasoning": "..."}}]"""


class LLMWorldModel:
    """
    Uses an LLM to generate forward-looking market scenarios.

    Can be used for:
    1. Single-step forecasting
    2. Multi-scenario Monte Carlo planning
    3. Action evaluation via simulated outcomes
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_provider("auto")
        self._scenario_cache: Dict[tuple, List[MarketScenario]] = {}

    def predict_next(self, obs: np.ndarray, info: Dict = None) -> MarketScenario:
        """Predict the single most likely next-period outcome."""
        context = self._build_context(obs, info)
        prompt = SCENARIO_PROMPT.format(**context)

        try:
            result = self.provider.chat_json([{"role": "user", "content": prompt}])
            return MarketScenario(
                direction=str(result.get("direction", "flat")),
                magnitude=float(result.get("magnitude", 0.0)),
                volatility_change=float(result.get("volatility_change", 0.0)),
                reasoning=str(result.get("reasoning", "")),
                probability=float(result.get("probability", 0.5)),
            )
        except Exception:
            return MarketScenario("flat", 0.0, 0.0, "prediction failed", 0.5)

    def generate_scenarios(
        self, obs: np.ndarray, n_scenarios: int = 3, info: Dict = None,
    ) -> List[MarketScenario]:
        """Generate multiple distinct scenarios for planning."""
        cache_key = tuple(np.round(obs[:6], 2))
        if cache_key in self._scenario_cache:
            return self._scenario_cache[cache_key]

        context = self._build_context(obs, info)
        context["n"] = n_scenarios
        prompt = MULTI_SCENARIO_PROMPT.format(**context)

        try:
            raw = self.provider.chat(
                [{"role": "user", "content": prompt}], json_mode=True
            )
            parsed = json.loads(raw) if isinstance(raw, str) else raw

            if isinstance(parsed, dict) and "scenarios" in parsed:
                parsed = parsed["scenarios"]
            if not isinstance(parsed, list):
                parsed = [parsed]

            scenarios = []
            for s in parsed[:n_scenarios]:
                scenarios.append(MarketScenario(
                    direction=str(s.get("direction", "flat")),
                    magnitude=float(s.get("magnitude", 0.0)),
                    volatility_change=float(s.get("volatility_change", 0.0)),
                    reasoning=str(s.get("reasoning", "")),
                    probability=float(s.get("probability", 1.0 / n_scenarios)),
                ))
        except Exception:
            scenarios = [
                MarketScenario("up", 0.005, 0.0, "base case up", 0.4),
                MarketScenario("flat", 0.0, 0.0, "base case flat", 0.3),
                MarketScenario("down", -0.005, 0.01, "base case down", 0.3),
            ][:n_scenarios]

        self._scenario_cache[cache_key] = scenarios
        return scenarios

    def plan_action(
        self,
        obs: np.ndarray,
        info: Dict = None,
        n_scenarios: int = 3,
    ) -> Tuple[int, Dict]:
        """
        Use scenario-based planning to choose the best action.

        For each scenario, evaluate what action would be best:
        - Market going up -> buy/strong_buy
        - Market going down -> sell/strong_sell
        - High volatility increase -> hold/reduce position

        Weight by scenario probability and return the best action.
        """
        scenarios = self.generate_scenarios(obs, n_scenarios, info)

        action_scores = np.zeros(5)  # [strong_sell, sell, hold, buy, strong_buy]

        for scenario in scenarios:
            p = scenario.probability
            mag = scenario.magnitude
            vol_change = scenario.volatility_change

            if scenario.direction == "up":
                action_scores[4] += p * (1.0 + abs(mag) * 10)  # strong_buy
                action_scores[3] += p * 0.7  # buy
            elif scenario.direction == "down":
                action_scores[0] += p * (1.0 + abs(mag) * 10)  # strong_sell
                action_scores[1] += p * 0.7  # sell
            else:
                action_scores[2] += p * 1.0  # hold

            if vol_change > 0.05:
                action_scores[2] += p * 0.5  # high vol -> more hold
                action_scores[0] += p * 0.3  # reduce exposure
                action_scores[4] -= p * 0.3  # less aggressive buy

        best_action = int(np.argmax(action_scores))
        return best_action, {
            "action_scores": action_scores.tolist(),
            "n_scenarios": len(scenarios),
            "scenarios": [
                {"dir": s.direction, "mag": s.magnitude, "prob": s.probability}
                for s in scenarios
            ],
        }

    def clear_cache(self):
        self._scenario_cache.clear()

    def _build_context(self, obs: np.ndarray, info: Dict = None) -> Dict:
        return {
            "trend_5d": float(obs[2]) if len(obs) > 2 else 0.0,
            "trend_20d": float(obs[3]) if len(obs) > 3 else 0.0,
            "vol": float(obs[5]) if len(obs) > 5 else 0.15,
            "rsi": float(obs[7]) if len(obs) > 7 else 0.5,
            "momentum": float(obs[8]) if len(obs) > 8 else 0.0,
            "position": float(obs[-4]) if len(obs) > 3 else 0.0,
            "drawdown": float(info.get("max_drawdown", 0)) if info else 0.0,
        }
