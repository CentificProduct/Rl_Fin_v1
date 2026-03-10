"""
=============================================================================
MODULE 15: Financial Reward Shaping
=============================================================================

THEORY:
-------
Reward shaping is the art of designing reward functions that guide RL agents
toward desired behavior. In finance, this is perhaps the most critical
design decision.

REWARD SHAPING THEOREM (Ng et al., 1999):
  Adding a potential-based shaping function F(s, s') = gamma * Phi(s') - Phi(s)
  to the reward does NOT change the optimal policy. This guarantees that
  shaped rewards guide learning without introducing bias.

COMMON PITFALLS:
  1. Sparse rewards (only P&L at episode end): too hard to learn from
  2. Raw returns: very noisy, no risk awareness
  3. Over-shaped rewards: agent hacks the reward instead of trading well
  4. Scale mismatch: different reward components on different scales

BEST PRACTICES FOR FINANCIAL REWARDS:
  1. Use risk-adjusted metrics (Sharpe, Sortino) as primary signal
  2. Add dense intermediate feedback (rolling performance metrics)
  3. Penalize excessive trading (transaction costs are real)
  4. Include drawdown penalties (survival is prerequisite for profit)
  5. Normalize/clip rewards for training stability
=============================================================================
"""

import numpy as np
from typing import Dict, Callable, Optional, List


class FinancialRewardShaper:
    """
    Configurable reward shaping for financial RL environments.

    Combines multiple reward components with configurable weights.
    Each component addresses a specific aspect of trading quality.
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        sharpe_weight: float = 0.5,
        drawdown_penalty: float = 2.0,
        turnover_penalty: float = 0.1,
        position_penalty: float = 0.05,
        survival_bonus: float = 0.01,
        reward_clip: float = 10.0,
        reward_scaling: float = 1.0,
        lookback: int = 30,
    ):
        self.weights = {
            'return': return_weight,
            'sharpe': sharpe_weight,
            'drawdown': drawdown_penalty,
            'turnover': turnover_penalty,
            'position': position_penalty,
            'survival': survival_bonus,
        }
        self.reward_clip = reward_clip
        self.reward_scaling = reward_scaling
        self.lookback = lookback

        self._return_buffer: List[float] = []
        self._peak_value = 1.0
        self._portfolio_value = 1.0

    def compute_reward(
        self,
        step_return: float,
        position_change: float,
        current_position: float,
        portfolio_value: float,
    ) -> float:
        """
        Compute the shaped reward from multiple components.

        Components:
        1. Return component: direct P&L signal (scaled)
        2. Sharpe component: rolling differential Sharpe ratio
        3. Drawdown penalty: penalize approaching drawdown limits
        4. Turnover penalty: discourage excessive trading
        5. Position penalty: penalize extreme positions
        6. Survival bonus: small reward for staying in the game
        """
        self._return_buffer.append(step_return)
        self._portfolio_value = portfolio_value
        self._peak_value = max(self._peak_value, portfolio_value)

        reward = 0.0

        # 1. Return component
        reward += self.weights['return'] * step_return * 100

        # 2. Differential Sharpe ratio
        if len(self._return_buffer) > 5:
            recent = self._return_buffer[-self.lookback:]
            mean_r = np.mean(recent)
            std_r = np.std(recent) + 1e-8
            rolling_sharpe = mean_r / std_r
            reward += self.weights['sharpe'] * rolling_sharpe

        # 3. Drawdown penalty (exponential to penalize large drawdowns more)
        drawdown = (self._peak_value - portfolio_value) / self._peak_value
        reward -= self.weights['drawdown'] * (np.exp(5 * drawdown) - 1)

        # 4. Turnover penalty
        reward -= self.weights['turnover'] * abs(position_change)

        # 5. Position penalty (penalize extreme leverage)
        reward -= self.weights['position'] * max(0, abs(current_position) - 0.8)

        # 6. Survival bonus
        reward += self.weights['survival']

        # Scale and clip
        reward *= self.reward_scaling
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)

        return reward

    def reset(self):
        """Reset internal state for a new episode."""
        self._return_buffer = []
        self._peak_value = 1.0
        self._portfolio_value = 1.0


class CurriculumRewardScheduler:
    """
    Curriculum learning for financial RL via reward scheduling.

    CONCEPT:
    Start with easy objectives and gradually increase difficulty:
    1. Phase 1: Learn to not lose money (survival)
    2. Phase 2: Learn to generate positive returns
    3. Phase 3: Learn risk-adjusted returns (Sharpe)
    4. Phase 4: Full objective with all penalties

    This is analogous to how human traders learn: first don't blow up,
    then learn to make money, then learn to manage risk.
    """

    def __init__(self, total_episodes: int = 1000):
        self.total_episodes = total_episodes
        self.current_episode = 0

        self.phases = [
            {
                'name': 'Survival',
                'start': 0,
                'end': 0.2,
                'config': {'return_weight': 0.1, 'drawdown_penalty': 5.0, 'survival_bonus': 0.1},
            },
            {
                'name': 'BasicProfits',
                'start': 0.2,
                'end': 0.5,
                'config': {'return_weight': 1.0, 'drawdown_penalty': 2.0, 'survival_bonus': 0.05},
            },
            {
                'name': 'RiskAdjusted',
                'start': 0.5,
                'end': 0.8,
                'config': {'return_weight': 0.5, 'sharpe_weight': 1.0, 'drawdown_penalty': 2.0},
            },
            {
                'name': 'FullObjective',
                'start': 0.8,
                'end': 1.0,
                'config': {'return_weight': 1.0, 'sharpe_weight': 0.5, 'drawdown_penalty': 2.0,
                           'turnover_penalty': 0.1, 'position_penalty': 0.05},
            },
        ]

    def get_reward_shaper(self) -> FinancialRewardShaper:
        """Get reward shaper for current curriculum phase."""
        progress = self.current_episode / self.total_episodes

        current_phase = self.phases[-1]
        for phase in self.phases:
            if phase['start'] <= progress < phase['end']:
                current_phase = phase
                break

        shaper = FinancialRewardShaper(**current_phase['config'])
        return shaper

    def advance(self):
        """Advance curriculum by one episode."""
        self.current_episode += 1

    @property
    def current_phase_name(self) -> str:
        progress = self.current_episode / self.total_episodes
        for phase in self.phases:
            if phase['start'] <= progress < phase['end']:
                return phase['name']
        return self.phases[-1]['name']


def demonstrate_reward_shaping():
    """Demonstrate different reward shaping strategies."""
    print("=" * 70)
    print("  CHAPTER 15: FINANCIAL REWARD SHAPING")
    print("=" * 70)

    np.random.seed(42)

    # Simulate a trading episode
    n_steps = 252
    returns = np.random.normal(0.0003, 0.015, n_steps)

    print("\n--- Comparing Reward Shaping Strategies ---")

    configs = {
        "Simple Returns": {"return_weight": 1.0, "sharpe_weight": 0.0, "drawdown_penalty": 0.0},
        "Sharpe-Focused": {"return_weight": 0.5, "sharpe_weight": 1.0, "drawdown_penalty": 0.0},
        "Drawdown-Aware": {"return_weight": 0.5, "sharpe_weight": 0.5, "drawdown_penalty": 3.0},
        "Full Shaped": {"return_weight": 1.0, "sharpe_weight": 0.5, "drawdown_penalty": 2.0,
                        "turnover_penalty": 0.1, "position_penalty": 0.05},
    }

    for name, config in configs.items():
        shaper = FinancialRewardShaper(**config)
        total_reward = 0
        pv = 1.0

        for t in range(n_steps):
            pv *= (1 + returns[t])
            position_change = np.random.uniform(-0.2, 0.2)
            position = np.random.uniform(-1, 1)

            reward = shaper.compute_reward(returns[t], position_change, position, pv)
            total_reward += reward

        print(f"  {name:>20s}: Total Reward = {total_reward:8.2f}")

    # Curriculum learning demo
    print("\n--- Curriculum Learning Schedule ---")
    scheduler = CurriculumRewardScheduler(total_episodes=100)

    for ep in range(100):
        scheduler.advance()
        if (ep + 1) % 25 == 0:
            print(f"  Episode {ep + 1:4d}: Phase = {scheduler.current_phase_name}")

    print("\nKey Insights:")
    print("  1. Reward shaping dramatically affects learned behavior")
    print("  2. Potential-based shaping preserves optimal policy (theory)")
    print("  3. Curriculum learning helps agents learn progressively")
    print("  4. The 'right' reward depends on the investor's objectives\n")


if __name__ == "__main__":
    demonstrate_reward_shaping()
