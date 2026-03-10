"""
=============================================================================
MODULE 7: Q-Learning for Trading
=============================================================================

THEORY:
-------
Q-Learning is the foundational off-policy, model-free RL algorithm.
It learns the optimal Q-function directly without knowing transition dynamics.

ALGORITHM:
  Initialize Q(s, a) arbitrarily
  For each episode:
    For each step:
      Choose a using epsilon-greedy from Q
      Take action a, observe r, s'
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
      s <- s'

KEY PROPERTIES:
  - OFF-POLICY: learns about the optimal policy while following an
    exploratory policy (epsilon-greedy)
  - CONVERGENCE: guaranteed to find Q* given sufficient exploration
    and appropriate learning rate decay (Robbins-Monro conditions)
  - TABULAR: requires discrete state and action spaces

FINANCIAL APPLICATION:
  For tabular Q-learning in finance, we must discretize the state space.
  We bin continuous features (price momentum, volatility, RSI) into
  discrete buckets. This loses information but provides a simple,
  interpretable starting point.

EXPLORATION vs. EXPLOITATION:
  The epsilon-greedy strategy balances:
  - Exploration: try new trades to discover better strategies
  - Exploitation: use what we've learned to maximize returns

  In finance, exploration is risky (real money). This motivates:
  - Sim-to-real: train in simulation, deploy carefully
  - Conservative exploration: upper confidence bounds (UCB)
  - Bayesian approaches: Thompson sampling
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt


class QLearningTrader:
    """
    Tabular Q-Learning agent for discretized trading.

    State discretization:
      - Momentum: 5 bins (very negative to very positive)
      - Volatility: 3 bins (low, medium, high)
      - RSI zone: 3 bins (oversold, neutral, overbought)
      - Position: 3 bins (short, flat, long)
      Total: 5 * 3 * 3 * 3 = 135 states

    Actions: [strong_sell, sell, hold, buy, strong_buy]
    """

    def __init__(
        self,
        n_momentum_bins: int = 5,
        n_vol_bins: int = 3,
        n_rsi_bins: int = 3,
        n_position_bins: int = 3,
        n_actions: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.n_bins = [n_momentum_bins, n_vol_bins, n_rsi_bins, n_position_bins]
        self.total_states = np.prod(self.n_bins)

        # Bin edges for discretization
        self.momentum_bins = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])
        self.vol_bins = np.array([0.10, 0.20])
        self.rsi_bins = np.array([0.30, 0.70])
        self.position_bins = np.array([-0.3, 0.3])

        # Q-table and visit counts
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.visit_count = defaultdict(lambda: np.zeros(n_actions))
        self.training_history: List[Dict] = []

    def discretize_state(self, features: np.ndarray) -> Tuple:
        """
        Convert continuous features to discrete state.

        Takes a feature vector and maps it to a tuple of bin indices.
        This is the key approximation in tabular methods.
        """
        momentum = features[0] if len(features) > 0 else 0
        volatility = features[1] if len(features) > 1 else 0.15
        rsi = features[2] if len(features) > 2 else 0.5
        position = features[3] if len(features) > 3 else 0

        m_bin = np.digitize(momentum, self.momentum_bins)
        v_bin = np.digitize(volatility, self.vol_bins)
        r_bin = np.digitize(rsi, self.rsi_bins)
        p_bin = np.digitize(position, self.position_bins)

        return (m_bin, v_bin, r_bin, p_bin)

    def select_action(self, state: Tuple, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon: explore (random action)
        With probability 1-epsilon: exploit (greedy w.r.t. Q)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def select_action_ucb(self, state: Tuple, c: float = 2.0) -> int:
        """
        Upper Confidence Bound (UCB) action selection.

        UCB(s, a) = Q(s, a) + c * sqrt(ln(N(s)) / N(s, a))

        Balances exploration and exploitation more principally than
        epsilon-greedy. Actions with fewer visits get a bonus.
        """
        total_visits = self.visit_count[state].sum()
        if total_visits == 0:
            return np.random.randint(self.n_actions)

        ucb_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if self.visit_count[state][a] == 0:
                return a  # must try unvisited action
            bonus = c * np.sqrt(np.log(total_visits) / self.visit_count[state][a])
            ucb_values[a] = self.Q[state][a] + bonus

        return int(np.argmax(ucb_values))

    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool,
    ) -> float:
        """
        Q-Learning update rule.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        The max operation is what makes Q-learning OFF-POLICY:
        we update toward the best possible action in s', regardless
        of what action we actually took (or will take).
        """
        current_q = self.Q[state][action]

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - current_q
        self.Q[state][action] += self.alpha * td_error
        self.visit_count[state][action] += 1

        return td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(
        self,
        env,
        max_steps: int = 2000,
    ) -> Dict:
        """Train for one episode and return metrics."""
        obs, info = env.reset()

        relevant_features = np.array([
            obs[9] if len(obs) > 9 else 0,   # momentum
            obs[5] if len(obs) > 5 else 0.15, # volatility
            obs[7] if len(obs) > 7 else 0.5,  # RSI
            obs[-4] if len(obs) > 3 else 0,   # position
        ])

        state = self.discretize_state(relevant_features)
        total_reward = 0
        total_td_error = 0
        steps = 0

        for step in range(max_steps):
            action = self.select_action(state, training=True)

            obs, reward, terminated, truncated, info = env.step(action)

            relevant_features = np.array([
                obs[9] if len(obs) > 9 else 0,
                obs[5] if len(obs) > 5 else 0.15,
                obs[7] if len(obs) > 7 else 0.5,
                obs[-4] if len(obs) > 3 else 0,
            ])
            next_state = self.discretize_state(relevant_features)

            done = terminated or truncated
            td_error = self.update(state, action, reward, next_state, done)

            total_reward += reward
            total_td_error += abs(td_error)
            state = next_state
            steps += 1

            if done:
                break

        self.decay_epsilon()

        metrics = {
            "total_reward": total_reward,
            "avg_td_error": total_td_error / max(steps, 1),
            "epsilon": self.epsilon,
            "portfolio_value": info.get("portfolio_value", 0),
            "total_return": info.get("total_return", 0),
            "sharpe_ratio": info.get("sharpe_ratio", 0),
            "states_visited": len(self.Q),
            "steps": steps,
        }
        self.training_history.append(metrics)
        return metrics

    def get_policy_summary(self) -> Dict:
        """Summarize the learned policy for interpretation."""
        policy = {}
        for state, q_values in self.Q.items():
            action_names = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
            best_action = action_names[np.argmax(q_values)]
            policy[state] = {
                "best_action": best_action,
                "q_values": {name: round(q, 4) for name, q in zip(action_names, q_values)},
                "visits": self.visit_count[state].tolist(),
            }
        return policy


class DoubleQLearningTrader(QLearningTrader):
    """
    Double Q-Learning to address maximization bias.

    PROBLEM WITH Q-LEARNING:
      The max operation max_a' Q(s',a') introduces a positive bias
      because we use the same Q-values to both select and evaluate
      actions. In finance, this overestimates expected returns.

    SOLUTION:
      Maintain two Q-tables. Use one to select actions, the other
      to evaluate them:
        Q1(s,a) <- Q1(s,a) + alpha * [r + gamma * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a)]
      With 50% probability, swap the roles of Q1 and Q2.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Q2 = defaultdict(lambda: np.zeros(self.n_actions))

    def select_action(self, state: Tuple, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        combined_q = self.Q[state] + self.Q2[state]
        return int(np.argmax(combined_q))

    def update(self, state, action, reward, next_state, done) -> float:
        if np.random.random() < 0.5:
            Q_update, Q_eval = self.Q, self.Q2
        else:
            Q_update, Q_eval = self.Q2, self.Q

        current_q = Q_update[state][action]

        if done:
            td_target = reward
        else:
            best_action = np.argmax(Q_update[next_state])
            td_target = reward + self.gamma * Q_eval[next_state][best_action]

        td_error = td_target - current_q
        Q_update[state][action] += self.alpha * td_error
        self.visit_count[state][action] += 1

        return td_error


def demonstrate_q_learning():
    """Train Q-Learning agents on the trading environment."""
    print("=" * 70)
    print("  CHAPTER 7: Q-LEARNING FOR STOCK TRADING")
    print("=" * 70)

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environments.stock_trading_env import StockTradingEnv

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)

    # Standard Q-Learning
    print("\n--- Training Standard Q-Learning (200 episodes) ---")
    q_agent = QLearningTrader(alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.99)

    for ep in range(200):
        metrics = q_agent.train_episode(env)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1:4d}: Return={metrics['total_return'] * 100:7.2f}%  "
                  f"Sharpe={metrics['sharpe_ratio']:6.3f}  "
                  f"Epsilon={metrics['epsilon']:.3f}  "
                  f"States={metrics['states_visited']}")

    # Double Q-Learning
    print("\n--- Training Double Q-Learning (200 episodes) ---")
    dq_agent = DoubleQLearningTrader(alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.99)

    for ep in range(200):
        metrics = dq_agent.train_episode(env)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1:4d}: Return={metrics['total_return'] * 100:7.2f}%  "
                  f"Sharpe={metrics['sharpe_ratio']:6.3f}  "
                  f"Epsilon={metrics['epsilon']:.3f}")

    print("\nKey Insight: Double Q-Learning reduces the positive bias in")
    print("return estimates, leading to more conservative but realistic")
    print("trading strategies.\n")

    return q_agent, dq_agent


if __name__ == "__main__":
    demonstrate_q_learning()
