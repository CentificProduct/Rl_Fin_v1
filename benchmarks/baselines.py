"""
===========================================================================
Open-Source & Classical Baselines
===========================================================================

Provides a library of ready-to-use baselines that are registered into
the BenchmarkRegistry automatically.  Includes:

 BASELINES (no training):
   - Random agent
   - Buy & Hold
   - Always-cash (do nothing)

 CLASSICAL QUANT (rule-based):
   - SMA crossover (trend following)
   - RSI mean-reversion
   - Bollinger Band breakout
   - Momentum (12-1 month)
   - MACD signal

 OPEN-SOURCE RL (via Stable-Baselines3, optional):
   - SB3 PPO
   - SB3 A2C
   - SB3 DQN

 CUSTOM RL (this project):
   - Our DQN (Double+Dueling)
   - Our PPO
   - Our Q-Learning

Every strategy follows the BaseStrategy interface so the runner can
train / evaluate / compare them uniformly.
===========================================================================
"""

import sys, os, time
import numpy as np
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.registry import BenchmarkRegistry, BaseStrategy


# =====================================================================
#  CATEGORY 1 : BASELINES (trivial / no learning)
# =====================================================================

@BenchmarkRegistry.register("Random", category="baseline")
class RandomStrategy(BaseStrategy):
    def predict(self, obs, info=None):
        return np.random.randint(5)


@BenchmarkRegistry.register("Buy & Hold", category="baseline")
class BuyHoldStrategy(BaseStrategy):
    """Always take the 'strong buy' action (action 4)."""
    def predict(self, obs, info=None):
        return 4


@BenchmarkRegistry.register("Always Cash", category="baseline")
class AlwaysCashStrategy(BaseStrategy):
    """Never enter a position (action 2 = hold)."""
    def predict(self, obs, info=None):
        return 2


# =====================================================================
#  CATEGORY 2 : CLASSICAL QUANT (rule-based, no NN training)
# =====================================================================

@BenchmarkRegistry.register("SMA Crossover", category="classical")
class SMACrossover(BaseStrategy):
    """
    Classic dual moving-average crossover.
    Buy when fast SMA > slow SMA, sell otherwise.
    Uses the ma_5_ratio (idx 2) and ma_20_ratio (idx 3) features.
    """
    def predict(self, obs, info=None):
        ma5 = obs[2] if len(obs) > 2 else 0
        ma20 = obs[3] if len(obs) > 3 else 0
        if ma5 > 0.005 and ma5 > ma20:
            return 4  # strong buy
        elif ma5 < -0.005 and ma5 < ma20:
            return 0  # strong sell
        return 2  # hold


@BenchmarkRegistry.register("RSI Mean-Reversion", category="classical")
class RSIMeanReversion(BaseStrategy):
    """
    Buy when RSI is oversold (<0.3), sell when overbought (>0.7).
    RSI feature is at index 7.
    """
    def predict(self, obs, info=None):
        rsi = obs[7] if len(obs) > 7 else 0.5
        if rsi < 0.25:
            return 4  # strong buy (oversold)
        elif rsi < 0.35:
            return 3  # buy
        elif rsi > 0.75:
            return 0  # strong sell (overbought)
        elif rsi > 0.65:
            return 1  # sell
        return 2  # hold


@BenchmarkRegistry.register("Bollinger Breakout", category="classical")
class BollingerBreakout(BaseStrategy):
    """
    Trade Bollinger Band breakouts.
    Bollinger position feature at index 11 (range ~[-1, 1]).
    """
    def predict(self, obs, info=None):
        boll = obs[11] if len(obs) > 11 else 0
        if boll > 0.8:
            return 4  # breakout up -> buy
        elif boll < -0.8:
            return 0  # breakout down -> sell
        elif boll > 0.3:
            return 3
        elif boll < -0.3:
            return 1
        return 2


@BenchmarkRegistry.register("Momentum", category="classical")
class MomentumStrategy(BaseStrategy):
    """
    10-day momentum (feature idx 8). Go long when positive, short when negative.
    """
    def predict(self, obs, info=None):
        mom = obs[8] if len(obs) > 8 else 0
        if mom > 0.03:
            return 4
        elif mom > 0.01:
            return 3
        elif mom < -0.03:
            return 0
        elif mom < -0.01:
            return 1
        return 2


@BenchmarkRegistry.register("MACD Signal", category="classical")
class MACDSignalStrategy(BaseStrategy):
    """
    Trade based on MACD (feature idx 10).
    Positive MACD -> bullish, negative -> bearish.
    """
    def predict(self, obs, info=None):
        macd = obs[10] if len(obs) > 10 else 0
        if macd > 0.005:
            return 4
        elif macd > 0.001:
            return 3
        elif macd < -0.005:
            return 0
        elif macd < -0.001:
            return 1
        return 2


@BenchmarkRegistry.register("Volatility Regime", category="classical")
class VolatilityRegimeStrategy(BaseStrategy):
    """
    Reduce exposure in high-volatility regimes (risk-off).
    vol_10 at idx 5, vol_30 at idx 6.
    """
    def predict(self, obs, info=None):
        vol = obs[5] if len(obs) > 5 else 0.15
        mom = obs[8] if len(obs) > 8 else 0
        if vol > 0.30:
            return 2  # high vol -> cash
        if mom > 0.01:
            return 4 if vol < 0.15 else 3
        elif mom < -0.01:
            return 0 if vol < 0.15 else 1
        return 2


# =====================================================================
#  CATEGORY 3 : OPEN-SOURCE RL (Stable-Baselines3)
# =====================================================================

@BenchmarkRegistry.register("SB3-PPO", category="open_source")
class SB3PPOStrategy(BaseStrategy):
    """
    Stable-Baselines3 PPO agent.

    Falls back gracefully if stable-baselines3 is not installed.
    Install with:  pip install stable-baselines3
    """
    def __init__(self):
        self.model = None

    def train(self, env, config: Dict):
        try:
            from stable_baselines3 import PPO
            total_steps = config.get("total_timesteps", 10000)
            self.model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=5,
                gamma=0.99,
                verbose=0,
            )
            self.model.learn(total_timesteps=total_steps)
        except ImportError:
            print("  [SB3-PPO] stable-baselines3 not installed -- skipping training")
            self.model = None

    def predict(self, obs, info=None):
        if self.model is None:
            return 2  # hold fallback
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


@BenchmarkRegistry.register("SB3-A2C", category="open_source")
class SB3A2CStrategy(BaseStrategy):
    """Stable-Baselines3 A2C agent."""
    def __init__(self):
        self.model = None

    def train(self, env, config: Dict):
        try:
            from stable_baselines3 import A2C
            total_steps = config.get("total_timesteps", 10000)
            self.model = A2C(
                "MlpPolicy", env,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                verbose=0,
            )
            self.model.learn(total_timesteps=total_steps)
        except ImportError:
            print("  [SB3-A2C] stable-baselines3 not installed -- skipping training")
            self.model = None

    def predict(self, obs, info=None):
        if self.model is None:
            return 2
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


@BenchmarkRegistry.register("SB3-DQN", category="open_source")
class SB3DQNStrategy(BaseStrategy):
    """Stable-Baselines3 DQN agent."""
    def __init__(self):
        self.model = None

    def train(self, env, config: Dict):
        try:
            from stable_baselines3 import DQN
            total_steps = config.get("total_timesteps", 10000)
            self.model = DQN(
                "MlpPolicy", env,
                learning_rate=1e-4,
                buffer_size=50000,
                batch_size=64,
                gamma=0.99,
                verbose=0,
            )
            self.model.learn(total_timesteps=total_steps)
        except ImportError:
            print("  [SB3-DQN] stable-baselines3 not installed -- skipping training")
            self.model = None

    def predict(self, obs, info=None):
        if self.model is None:
            return 2
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


# =====================================================================
#  CATEGORY 4 : CUSTOM RL (this project's agents)
# =====================================================================

@BenchmarkRegistry.register("Custom DQN", category="rl")
class CustomDQNStrategy(BaseStrategy):
    """Our Double-Dueling DQN implementation."""
    def __init__(self):
        self.agent = None

    def train(self, env, config: Dict):
        from agents.dqn_agent import DQNAgent
        obs, _ = env.reset()
        episodes = config.get("episodes", 15)
        self.agent = DQNAgent(
            state_dim=len(obs), action_dim=5,
            hidden_dims=[64, 32],
            double_dqn=True, dueling=True,
            learning_rate=5e-4,
            epsilon_decay_steps=episodes * 400,
            buffer_size=20000, batch_size=32,
        )
        for _ in range(episodes):
            self.agent.train_episode(env)

    def predict(self, obs, info=None):
        if self.agent is None:
            return 2
        return self.agent.select_action(obs, training=False)


@BenchmarkRegistry.register("Custom PPO", category="rl")
class CustomPPOStrategy(BaseStrategy):
    """Our PPO implementation."""
    def __init__(self):
        self.agent = None

    def train(self, env, config: Dict):
        from agents.ppo_agent import PPOTrader
        import torch
        obs, _ = env.reset()
        iters = config.get("iterations", 10)
        self.agent = PPOTrader(
            state_dim=len(obs), action_dim=5,
            hidden_dim=64, continuous=False,
            rollout_length=256, n_epochs=3,
        )
        for _ in range(iters):
            rollout = self.agent.collect_rollout(env)
            self.agent.update(rollout)

    def predict(self, obs, info=None):
        if self.agent is None:
            return 2
        import torch
        st = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            a, _, _, _ = self.agent.network.get_action_and_value(st)
        return a.item()


@BenchmarkRegistry.register("Custom Q-Learning", category="rl")
class CustomQLearningStrategy(BaseStrategy):
    """Our tabular Q-Learning implementation."""
    def __init__(self):
        self.agent = None

    def train(self, env, config: Dict):
        from agents.q_learning import QLearningTrader
        episodes = config.get("episodes", 50)
        self.agent = QLearningTrader(
            alpha=0.1, gamma=0.99, epsilon_decay=0.97,
        )
        for _ in range(episodes):
            self.agent.train_episode(env)

    def predict(self, obs, info=None):
        if self.agent is None:
            return 2
        feats = np.array([
            obs[9] if len(obs) > 9 else 0,
            obs[5] if len(obs) > 5 else 0.15,
            obs[7] if len(obs) > 7 else 0.5,
            obs[-4] if len(obs) > 3 else 0,
        ])
        state = self.agent.discretize_state(feats)
        return self.agent.select_action(state, training=False)


# =====================================================================
#  Auto-register everything
# =====================================================================

def register_all_baselines():
    """No-op: registration happens at import via decorators above."""
    pass
