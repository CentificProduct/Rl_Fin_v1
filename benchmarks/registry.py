"""
===========================================================================
Extensible Benchmark Registry
===========================================================================

Provides a plug-in architecture so any new strategy (custom RL agent,
open-source library, classical quant model) can be registered with one
decorator and automatically included in every benchmark run.

HOW TO ADD YOUR OWN STRATEGY:
    from benchmarks.registry import BenchmarkRegistry

    @BenchmarkRegistry.register("My Strategy", category="rl")
    class MyStrategy:
        def train(self, env, config):
            ...
        def predict(self, obs):
            return action

That's it -- the runner picks it up automatically.
===========================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Type


@dataclass
class StrategyResult:
    """Standardised result container for every benchmark entry."""
    name: str
    category: str  # 'baseline', 'classical', 'rl', 'open_source'
    portfolio_values: np.ndarray = field(default_factory=lambda: np.array([]))
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    annual_volatility: float = 0.0
    calmar_ratio: float = 0.0
    trade_count: int = 0
    total_cost: float = 0.0
    train_time: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy:
    """
    Interface every benchmark strategy must follow.

    Subclass this and implement train() + predict().
    Or use the functional register_fn() shortcut for simple strategies.
    """
    name: str = "Unnamed"
    category: str = "custom"

    def train(self, env, config: Dict) -> None:
        """Train the strategy (no-op for rule-based)."""
        pass

    def predict(self, obs: np.ndarray, info: Dict = None) -> Any:
        """Return an action given the current observation."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state between evaluation episodes."""
        pass


class BenchmarkRegistry:
    """
    Global registry of benchmark strategies.

    Strategies are stored by name and tagged with a category so the
    reporting layer can group them (baselines vs RL vs open-source).
    """
    _strategies: Dict[str, Dict] = {}

    @classmethod
    def register(cls, name: str, category: str = "custom"):
        """Decorator to register a strategy class."""
        def wrapper(klass):
            cls._strategies[name] = {
                "cls": klass,
                "category": category,
                "name": name,
            }
            klass.name = name
            klass.category = category
            return klass
        return wrapper

    @classmethod
    def register_fn(
        cls,
        name: str,
        predict_fn: Callable,
        category: str = "custom",
        train_fn: Optional[Callable] = None,
        reset_fn: Optional[Callable] = None,
    ):
        """Register a simple function-based strategy without writing a class."""
        class _FnStrategy(BaseStrategy):
            pass
        _FnStrategy.name = name
        _FnStrategy.category = category
        _FnStrategy.predict = lambda self, obs, info=None: predict_fn(obs, info)
        if train_fn:
            _FnStrategy.train = lambda self, env, cfg: train_fn(env, cfg)
        if reset_fn:
            _FnStrategy.reset = lambda self: reset_fn()
        cls._strategies[name] = {
            "cls": _FnStrategy,
            "category": category,
            "name": name,
        }

    @classmethod
    def get(cls, name: str) -> BaseStrategy:
        """Instantiate and return a registered strategy."""
        entry = cls._strategies[name]
        instance = entry["cls"]()
        instance.name = entry["name"]
        instance.category = entry["category"]
        return instance

    @classmethod
    def list_all(cls) -> List[Dict]:
        """List every registered strategy."""
        return [
            {"name": v["name"], "category": v["category"]}
            for v in cls._strategies.values()
        ]

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        return [v["name"] for v in cls._strategies.values() if v["category"] == category]

    @classmethod
    def clear(cls):
        cls._strategies.clear()
