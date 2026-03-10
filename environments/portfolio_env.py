"""
=============================================================================
MODULE 5: Portfolio Allocation Environment
=============================================================================

THEORY:
-------
Portfolio allocation extends single-asset trading to multi-asset optimization.
The agent must decide how to distribute capital across N assets at each step.

MARKOWITZ MEETS RL:
  Classical mean-variance optimization:
    max_w  w'mu - (lambda/2) * w'Sigma*w
    s.t.   sum(w) = 1, w >= 0

  RL replaces the static optimization with dynamic, adaptive allocation:
  - The agent learns time-varying optimal weights
  - It naturally handles non-stationary correlations
  - Transaction costs are part of the decision process
  - No need to estimate mu and Sigma explicitly

ACTION SPACE:
  Continuous: w in R^N, sum(w) = 1 (portfolio weights via softmax)

REWARD OPTIONS:
  - Portfolio return: r_t = w_t' * r_{assets,t}
  - Risk-adjusted: Sharpe of the portfolio return stream
  - Utility: CRRA utility U(r) = r^{1-gamma} / (1-gamma)
=============================================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import FinancialDataLoader


class PortfolioAllocationEnv(gym.Env):
    """
    Multi-asset portfolio allocation environment.

    The agent outputs portfolio weights at each step.
    The environment simulates returns with transaction costs.
    """
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        prices: Optional[np.ndarray] = None,
        n_assets: int = 5,
        initial_balance: float = 1000000.0,
        transaction_cost: float = 0.001,
        risk_aversion: float = 1.0,
        reward_type: str = "utility",  # 'return', 'sharpe', 'utility'
        rebalance_frequency: int = 1,  # steps between rebalances
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if prices is None:
            data = FinancialDataLoader.generate_correlated_assets(
                n_assets=n_assets, n_days=2000
            )
            self.prices = data.prices
            self.returns = data.returns
            self.n_assets = n_assets
        else:
            self.prices = prices
            self.returns = np.diff(prices, axis=0) / prices[:-1]
            self.n_assets = prices.shape[1]

        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_aversion = risk_aversion
        self.reward_type = reward_type
        self.rebalance_frequency = rebalance_frequency
        self.render_mode = render_mode

        # State: [asset returns (N*lookback), current weights (N), portfolio stats (4)]
        self.lookback = 20
        state_dim = self.n_assets * self.lookback + self.n_assets + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

        # Actions: portfolio weights (softmax applied internally)
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.lookback
        self.portfolio_value = self.initial_balance
        self.weights = np.ones(self.n_assets) / self.n_assets  # equal weight
        self.peak_value = self.initial_balance
        self.returns_history = []
        self.weights_history = [self.weights.copy()]
        self.value_history = [self.initial_balance]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute portfolio rebalancing step.

        1. Normalize action to valid portfolio weights (sum = 1)
        2. Compute turnover and transaction costs
        3. Apply asset returns to get new portfolio value
        4. Compute reward
        """
        # Normalize to valid weights
        new_weights = np.clip(action, 0, None)
        weight_sum = new_weights.sum()
        if weight_sum > 0:
            new_weights /= weight_sum
        else:
            new_weights = np.ones(self.n_assets) / self.n_assets

        # Transaction costs from turnover
        turnover = np.sum(np.abs(new_weights - self.weights))
        cost = turnover * self.transaction_cost * self.portfolio_value

        # Apply market returns
        asset_returns = self.returns[self.current_step]
        portfolio_return = np.dot(new_weights, asset_returns)

        # Update state
        old_value = self.portfolio_value
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - cost
        self.weights = new_weights * (1 + asset_returns)
        self.weights /= self.weights.sum()  # drift-adjusted weights

        realized_return = (self.portfolio_value - old_value) / old_value
        self.returns_history.append(realized_return)
        self.value_history.append(self.portfolio_value)
        self.weights_history.append(self.weights.copy())
        self.peak_value = max(self.peak_value, self.portfolio_value)

        self.current_step += 1

        reward = self._compute_reward(realized_return)

        terminated = self.portfolio_value <= self.initial_balance * 0.5
        truncated = self.current_step >= len(self.returns) - 1

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _compute_reward(self, ret: float) -> float:
        if self.reward_type == "return":
            return ret * 100

        elif self.reward_type == "sharpe":
            if len(self.returns_history) < 2:
                return ret * 100
            arr = np.array(self.returns_history)
            return (arr.mean() / (arr.std() + 1e-8)) * np.sqrt(252)

        elif self.reward_type == "utility":
            # CRRA utility: U(1+r) = (1+r)^(1-gamma) / (1-gamma)
            gamma = self.risk_aversion
            if gamma == 1:
                return np.log(1 + ret)
            else:
                return ((1 + ret) ** (1 - gamma) - 1) / (1 - gamma)

        return ret

    def _get_observation(self) -> np.ndarray:
        start = max(0, self.current_step - self.lookback)
        end = self.current_step
        recent_returns = self.returns[start:end]

        if len(recent_returns) < self.lookback:
            pad = np.zeros((self.lookback - len(recent_returns), self.n_assets))
            recent_returns = np.vstack([pad, recent_returns])

        flat_returns = recent_returns.flatten()

        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        vol = np.std(self.returns_history[-30:]) * np.sqrt(252) if len(self.returns_history) > 1 else 0.0
        sharpe = (np.mean(self.returns_history) / (np.std(self.returns_history) + 1e-8) * np.sqrt(252)) if len(self.returns_history) > 1 else 0.0

        portfolio_stats = np.array([portfolio_return, drawdown, vol, sharpe], dtype=np.float32)

        obs = np.concatenate([flat_returns, self.weights, portfolio_stats]).astype(np.float32)
        return obs

    def _get_info(self) -> Dict:
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        returns_arr = np.array(self.returns_history) if self.returns_history else np.array([0.0])
        sharpe = returns_arr.mean() / (returns_arr.std() + 1e-8) * np.sqrt(252) if len(returns_arr) > 1 else 0.0
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        return {
            "portfolio_value": self.portfolio_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": drawdown,
            "weights": self.weights.copy(),
            "step": self.current_step,
        }


def demonstrate_portfolio_env():
    """Show portfolio environment behavior with equal-weight rebalancing."""
    print("=" * 70)
    print("  CHAPTER 5: PORTFOLIO ALLOCATION ENVIRONMENT")
    print("=" * 70)

    env = PortfolioAllocationEnv(n_assets=5, reward_type="utility")
    obs, info = env.reset(seed=42)

    print(f"\nEnvironment:")
    print(f"  Assets:          {env.n_assets}")
    print(f"  State dim:       {env.observation_space.shape[0]}")
    print(f"  Action dim:      {env.action_space.shape[0]}")

    # Equal-weight baseline
    equal_weights = np.ones(env.n_assets) / env.n_assets
    total_reward = 0
    steps = 0

    while True:
        obs, reward, terminated, truncated, info = env.step(equal_weights)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    print(f"\nEqual-Weight Strategy ({steps} steps):")
    print(f"  Final Value:     ${info['portfolio_value']:,.2f}")
    print(f"  Total Return:    {info['total_return'] * 100:.2f}%")
    print(f"  Sharpe Ratio:    {info['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:    {info['max_drawdown'] * 100:.2f}%")
    print(f"  Total Reward:    {total_reward:.4f}\n")

    return env


if __name__ == "__main__":
    demonstrate_portfolio_env()
