"""
=============================================================================
MODULE 12: RL-Based Portfolio Optimization
=============================================================================

THEORY:
-------
Portfolio optimization is one of the most natural applications of RL in finance.

CLASSICAL APPROACH (Markowitz, 1952):
  max_w  E[r_p] - (lambda/2) * Var[r_p]
  where r_p = w' * r, w = portfolio weights

  Problems:
  1. Requires accurate estimates of mu (expected returns) and Sigma (covariance)
  2. Static: doesn't adapt to changing market conditions
  3. Estimation error amplification: small errors in mu/Sigma lead to extreme weights
  4. Ignores transaction costs and market impact

RL APPROACH:
  - State: market features + current portfolio weights + risk metrics
  - Action: target portfolio weights (continuous, sum to 1)
  - Reward: risk-adjusted return (Sharpe, Sortino, utility)
  - The agent IMPLICITLY learns mu and Sigma through experience
  - Naturally handles non-stationarity, transaction costs, and constraints

KEY ADVANTAGES OF RL OVER CLASSICAL:
  1. No need to estimate expected returns (the hardest part of finance)
  2. Adaptive: automatically adjusts to regime changes
  3. End-to-end: optimizes the actual objective including all frictions
  4. Can incorporate any constraint via reward shaping or environment design

CONTINUOUS ACTION SPACES:
  Portfolio weights are continuous. We use:
  - PPO with Gaussian policy (output mean and std for each weight)
  - Softmax final layer to ensure weights sum to 1
  - Dirichlet policy for naturally valid weight distributions
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import FinancialDataLoader


class DirichletPolicyNetwork(nn.Module):
    """
    Policy network that outputs Dirichlet distribution parameters.

    WHY DIRICHLET?
    Portfolio weights must be non-negative and sum to 1.
    The Dirichlet distribution naturally satisfies these constraints.
    Its concentration parameters alpha control the "sharpness" of
    the allocation (high alpha = more concentrated portfolio).
    """

    def __init__(self, state_dim: int, n_assets: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
            nn.Softplus(),  # ensure alpha > 0
        )
        # Bias initialization for uniform allocation
        nn.init.constant_(self.network[-2].bias, 1.0)

    def forward(self, state: torch.Tensor) -> Dirichlet:
        alpha = self.network(state) + 1.0  # ensure alpha >= 1 for valid distribution
        return Dirichlet(alpha)


class PortfolioCriticNetwork(nn.Module):
    """Value network for portfolio state evaluation."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RLPortfolioOptimizer:
    """
    RL-based portfolio optimizer using PPO with Dirichlet policy.

    ARCHITECTURE:
      - Policy: Neural net -> Dirichlet parameters -> Sample weights
      - Critic: Neural net -> Scalar value estimate
      - Training: PPO clipped objective with GAE advantages

    REWARD FUNCTION:
      We optimize a utility-based reward:
        r_t = log(1 + w_t' * returns_t) - lambda * turnover_t

      This is equivalent to maximizing the log-growth rate of the portfolio
      (Kelly criterion connection) minus a trading cost penalty.
    """

    def __init__(
        self,
        n_assets: int,
        state_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        cost_penalty: float = 0.001,
    ):
        self.n_assets = n_assets
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.cost_penalty = cost_penalty

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = DirichletPolicyNetwork(state_dim, n_assets, hidden_dim).to(self.device)
        self.critic = PortfolioCriticNetwork(state_dim, hidden_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.training_history: List[Dict] = []

    def select_weights(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Sample portfolio weights from the Dirichlet policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.policy(state_t)
            if training:
                weights = dist.sample()
            else:
                weights = dist.concentration / dist.concentration.sum()

        return weights.cpu().numpy().flatten()

    def compute_gae(self, rewards, values, dones, next_value):
        """Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    def train_on_data(
        self,
        returns_data: np.ndarray,
        features_data: np.ndarray,
        n_episodes: int = 100,
        episode_length: int = 252,
    ) -> List[Dict]:
        """
        Train the portfolio optimizer on historical data.

        Simulates episodes by sampling random starting points in the data.
        """
        n_days = len(returns_data)
        all_metrics = []

        for episode in range(n_episodes):
            start = np.random.randint(0, n_days - episode_length - 1)
            ep_returns = returns_data[start:start + episode_length]
            ep_features = features_data[start:start + episode_length]

            weights = np.ones(self.n_assets) / self.n_assets  # start equal weight

            states, actions, rewards, values, dones, log_probs = [], [], [], [], [], []

            for t in range(episode_length):
                state = np.concatenate([ep_features[t], weights])
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                dist = self.policy(state_t)
                new_weights = dist.sample()
                log_prob = dist.log_prob(new_weights)
                value = self.critic(state_t).squeeze()

                w = new_weights.detach().cpu().numpy().flatten()

                # Portfolio return
                port_return = np.dot(w, ep_returns[t])
                turnover = np.sum(np.abs(w - weights))

                # Reward: log return minus cost
                reward = np.log(1 + port_return) - self.cost_penalty * turnover

                states.append(state)
                actions.append(w)
                rewards.append(reward)
                values.append(value.item())
                dones.append(0.0)
                log_probs.append(log_prob.item())

                # Update weights for next step (drift)
                weights = w * (1 + ep_returns[t])
                weights /= weights.sum()

            dones[-1] = 1.0

            # Compute advantages and update
            final_state = np.concatenate([ep_features[-1], weights])
            final_state_t = torch.FloatTensor(final_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(final_state_t).item()

            advantages, returns_target = self.compute_gae(
                np.array(rewards), np.array(values), np.array(dones), next_value
            )

            self._ppo_update(states, actions, log_probs, advantages, returns_target)

            total_port_return = np.exp(np.sum(rewards)) - 1
            metrics = {
                "episode": episode,
                "total_return": total_port_return,
                "avg_reward": np.mean(rewards),
                "avg_turnover": np.mean([np.sum(np.abs(np.diff(np.array(actions), axis=0)), axis=1).mean()]) if len(actions) > 1 else 0,
            }
            all_metrics.append(metrics)
            self.training_history.append(metrics)

            if (episode + 1) % 20 == 0:
                recent = all_metrics[-20:]
                avg_return = np.mean([m["total_return"] for m in recent])
                print(f"  Episode {episode + 1:4d}: "
                      f"Avg Return={avg_return * 100:.2f}%  "
                      f"AvgReward={metrics['avg_reward']:.6f}")

        return all_metrics

    def _ppo_update(self, states, actions, old_log_probs, advantages, returns):
        """PPO update step."""
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        old_lp_t = torch.FloatTensor(old_log_probs).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(5):
            dist = self.policy(states_t)
            new_lp = dist.log_prob(actions_t)
            entropy = dist.entropy()

            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

            values = self.critic(states_t).squeeze()
            value_loss = nn.MSELoss()(values, ret_t)

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()


def demonstrate_portfolio_optimization():
    """Demonstrate RL portfolio optimization."""
    print("=" * 70)
    print("  CHAPTER 12: RL-BASED PORTFOLIO OPTIMIZATION")
    print("=" * 70)

    data = FinancialDataLoader.generate_correlated_assets(n_assets=5, n_days=2000)

    n_feature_per_asset = 12
    state_dim = n_feature_per_asset * data.prices.shape[1] + data.prices.shape[1]  # features + weights

    optimizer = RLPortfolioOptimizer(
        n_assets=5,
        state_dim=state_dim,
        hidden_dim=128,
    )

    print(f"\nPortfolio Optimization Setup:")
    print(f"  Assets:     {5}")
    print(f"  State dim:  {state_dim}")
    print(f"  Policy:     Dirichlet distribution (natural for portfolio weights)")
    print(f"  Objective:  Log-return minus transaction costs")

    print("\n--- Training RL Portfolio Optimizer (60 episodes) ---")
    metrics = optimizer.train_on_data(
        data.returns, data.features, n_episodes=60, episode_length=200
    )

    # Compare with equal-weight
    equal_returns = np.mean(data.returns, axis=1)
    equal_total = np.exp(np.sum(np.log(1 + equal_returns[:200]))) - 1

    print(f"\n--- Benchmark Comparison ---")
    final_metrics = metrics[-10:]
    rl_return = np.mean([m["total_return"] for m in final_metrics])
    print(f"  RL Optimizer (last 10 eps avg): {rl_return * 100:.2f}%")
    print(f"  Equal Weight (200 days):        {equal_total * 100:.2f}%")

    print("\nKey Insights:")
    print("  1. Dirichlet policy naturally outputs valid portfolio weights")
    print("  2. RL learns to adapt allocation to market conditions")
    print("  3. Cost penalty encourages stable, low-turnover portfolios")
    print("  4. No need to estimate expected returns explicitly\n")

    return optimizer


if __name__ == "__main__":
    demonstrate_portfolio_optimization()
