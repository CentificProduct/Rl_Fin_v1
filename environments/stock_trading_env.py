"""
=============================================================================
MODULE 4: Custom Gymnasium Environment for Stock Trading
=============================================================================

THEORY:
-------
A trading environment must translate the financial market into an MDP that
an RL agent can interact with. Key design decisions:

STATE REPRESENTATION:
  The state should encode all information relevant to the trading decision.
  We include: price features, technical indicators, current position, P&L,
  and risk metrics. The state must approximate the Markov property.

ACTION SPACE:
  We support both discrete and continuous actions:
  - Discrete: {Strong Sell, Sell, Hold, Buy, Strong Buy}
  - Continuous: target position in [-1, 1] (short to long)

REWARD FUNCTION:
  The reward is the critical design choice. Options include:
  - Raw returns: r_t = (P_{t+1} - P_t) * position
  - Risk-adjusted: r_t = return / volatility (differential Sharpe ratio)
  - Drawdown-penalized: r_t = return - lambda * max_drawdown
  - Transaction cost-aware: r_t = return - |delta_position| * cost

MARKET IMPACT:
  Real trades move prices. We model linear impact:
    execution_price = price * (1 + impact_coefficient * |order_size|)

CONCEPT - REWARD SHAPING IN FINANCE:
  Naive return-based rewards are sparse and noisy. Shaped rewards can
  accelerate learning by providing denser feedback:
  - Differential Sharpe ratio (Moody & Saffell, 2001)
  - Risk-sensitive rewards using CVaR
  - Curriculum learning: start with simple markets, increase complexity
=============================================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import FinancialDataLoader, FeatureEngineering


class StockTradingEnv(gym.Env):
    """
    A comprehensive stock trading environment following the Gymnasium API.

    Features:
      - Realistic transaction costs and slippage
      - Market impact modeling
      - Multiple reward formulations
      - Position limits and risk constraints
      - Support for both discrete and continuous actions
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        prices: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,  # 10 bps per trade
        slippage: float = 0.0005,  # 5 bps slippage
        max_position: float = 1.0,  # max fraction of capital in position
        reward_type: str = "sharpe",  # 'simple', 'sharpe', 'sortino', 'calmar'
        lookback_window: int = 30,
        discrete_actions: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if prices is None:
            data = FinancialDataLoader.generate_synthetic_data(len_data=2000)
            self.prices = data.prices
            self.features = data.features
        else:
            self.prices = prices
            self.features = features if features is not None else \
                FeatureEngineering.compute_all_features(prices)

        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position = max_position
        self.reward_type = reward_type
        self.lookback_window = lookback_window
        self.discrete_actions = discrete_actions
        self.render_mode = render_mode

        n_features = self.features.shape[1] if self.features.ndim > 1 else 1
        self.state_dim = n_features + 4  # features + [position, cash_ratio, unrealized_pnl, drawdown]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        if discrete_actions:
            self.action_space = spaces.Discrete(5)
            self._action_map = {
                0: -1.0,   # strong sell
                1: -0.5,   # sell
                2: 0.0,    # hold
                3: 0.5,    # buy
                4: 1.0,    # strong buy
            }
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        self._reset_state()

    def _reset_state(self):
        """Initialize/reset all internal state variables."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0  # number of shares
        self.position_value = 0.0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.returns_history = []
        self.portfolio_history = [self.initial_balance]
        self.position_history = [0.0]
        self.trade_count = 0
        self.total_cost = 0.0

        # For differential Sharpe ratio
        self._A = 0.0  # running mean of returns
        self._B = 0.0  # running mean of squared returns
        self._eta = 0.01  # EMA decay for Sharpe estimation

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self._reset_state()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.

        1. Decode action into target position
        2. Execute trade with costs and slippage
        3. Advance market by one step
        4. Compute reward
        5. Return (obs, reward, terminated, truncated, info)
        """
        # Decode action
        if self.discrete_actions:
            target_position_frac = self._action_map[int(action)]
        else:
            target_position_frac = float(np.clip(action, -1.0, 1.0))

        target_position_frac *= self.max_position

        # Current price
        current_price = self.prices[self.current_step]

        # Calculate target number of shares
        target_value = target_position_frac * self.portfolio_value
        target_shares = target_value / current_price
        shares_to_trade = target_shares - self.position

        # Execute trade with transaction costs and slippage
        if abs(shares_to_trade) > 1e-8:
            direction = np.sign(shares_to_trade)
            exec_price = current_price * (1 + direction * self.slippage)
            trade_value = abs(shares_to_trade) * exec_price
            cost = trade_value * self.transaction_cost

            self.balance -= shares_to_trade * exec_price + cost
            self.position = target_shares
            self.trade_count += 1
            self.total_cost += cost

        # Advance market
        self.current_step += 1
        new_price = self.prices[self.current_step]

        # Update portfolio value
        self.position_value = self.position * new_price
        old_portfolio = self.portfolio_value
        self.portfolio_value = self.balance + self.position_value

        # Compute step return
        step_return = (self.portfolio_value - old_portfolio) / old_portfolio
        self.returns_history.append(step_return)
        self.portfolio_history.append(self.portfolio_value)
        self.position_history.append(self.position * new_price / self.portfolio_value)

        # Update peak for drawdown
        self.peak_portfolio = max(self.peak_portfolio, self.portfolio_value)

        # Compute reward
        reward = self._compute_reward(step_return)

        # Check termination
        terminated = False
        if self.portfolio_value <= self.initial_balance * 0.5:  # 50% drawdown limit
            terminated = True
            reward -= 1.0  # penalty for ruin

        truncated = self.current_step >= len(self.prices) - 2

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, step_return: float) -> float:
        """
        Compute reward based on selected reward formulation.

        REWARD ENGINEERING is crucial in financial RL:

        1. SIMPLE RETURN: Direct P&L signal
           + Intuitive, easy to implement
           - Very noisy, doesn't account for risk

        2. DIFFERENTIAL SHARPE RATIO (Moody & Saffell, 2001):
           Approximates the change in Sharpe ratio from one step.
           + Directly optimizes what we care about (risk-adjusted return)
           + Dense signal even when returns are near zero
           - More complex to implement

        3. SORTINO RATIO variant:
           Like Sharpe but only penalizes downside volatility.
           + Better captures asymmetric risk preferences
           - Requires tracking downside returns separately

        4. CALMAR RATIO variant:
           Penalizes based on maximum drawdown.
           + Directly controls worst-case losses
           - Very sparse signal (drawdown changes infrequently)
        """
        if self.reward_type == "simple":
            return step_return * 100  # scale up for better gradient signal

        elif self.reward_type == "sharpe":
            # Differential Sharpe ratio
            delta_A = step_return - self._A
            delta_B = step_return ** 2 - self._B

            self._A += self._eta * delta_A
            self._B += self._eta * delta_B

            denom = (self._B - self._A ** 2)
            if denom > 1e-8:
                dsr = (self._B * delta_A - 0.5 * self._A * delta_B) / (denom ** 1.5)
            else:
                dsr = step_return

            return float(np.clip(dsr, -5, 5))

        elif self.reward_type == "sortino":
            downside_returns = [r for r in self.returns_history if r < 0]
            downside_vol = np.std(downside_returns) if len(downside_returns) > 2 else 0.01
            return step_return / (downside_vol + 1e-8)

        elif self.reward_type == "calmar":
            drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio
            return step_return * 100 - 10 * drawdown

        return step_return

    def _get_observation(self) -> np.ndarray:
        """Construct the state observation vector."""
        idx = min(self.current_step - 1, len(self.features) - 1)
        market_features = self.features[idx]

        position_frac = self.position * self.prices[self.current_step] / self.portfolio_value if self.portfolio_value > 0 else 0
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 1
        unrealized_pnl = (self.portfolio_value - self.initial_balance) / self.initial_balance
        drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio

        portfolio_features = np.array([
            position_frac,
            cash_ratio,
            unrealized_pnl,
            drawdown,
        ], dtype=np.float32)

        obs = np.concatenate([market_features, portfolio_features]).astype(np.float32)
        return obs

    def _get_info(self) -> Dict:
        """Return additional info for logging and analysis."""
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance

        returns_arr = np.array(self.returns_history)
        sharpe = 0.0
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(252)

        drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio

        return {
            "portfolio_value": self.portfolio_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": drawdown,
            "trade_count": self.trade_count,
            "total_cost": self.total_cost,
            "step": self.current_step,
        }


def demonstrate_trading_environment():
    """Run the trading environment with random actions to show its behavior."""
    print("=" * 70)
    print("  CHAPTER 4: STOCK TRADING ENVIRONMENT")
    print("=" * 70)

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)
    obs, info = env.reset(seed=42)

    print(f"\nEnvironment Configuration:")
    print(f"  State dimension:     {env.state_dim}")
    print(f"  Action space:        {env.action_space}")
    print(f"  Initial balance:     ${env.initial_balance:,.0f}")
    print(f"  Transaction cost:    {env.transaction_cost * 100:.1f}%")
    print(f"  Reward type:         {env.reward_type}")

    total_reward = 0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"\nRandom Agent Results ({steps} steps):")
    print(f"  Final Portfolio:     ${info['portfolio_value']:,.2f}")
    print(f"  Total Return:        {info['total_return'] * 100:.2f}%")
    print(f"  Sharpe Ratio:        {info['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:        {info['max_drawdown'] * 100:.2f}%")
    print(f"  Number of Trades:    {info['trade_count']}")
    print(f"  Transaction Costs:   ${info['total_cost']:,.2f}")
    print(f"  Cumulative Reward:   {total_reward:.4f}")

    print("\nNote: Random actions serve as a baseline. RL agents should")
    print("significantly outperform this random strategy.\n")

    return env


if __name__ == "__main__":
    demonstrate_trading_environment()
