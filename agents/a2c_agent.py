"""
=============================================================================
MODULE 11: Advantage Actor-Critic (A2C) for Trading
=============================================================================

THEORY:
-------
A2C is a synchronous variant of A3C (Asynchronous Advantage Actor-Critic).
It combines the benefits of:
  - Policy gradient (actor): direct policy optimization
  - Value function (critic): variance reduction via baseline

ARCHITECTURE:
  Actor: pi_theta(a|s)     - outputs action probabilities
  Critic: V_phi(s)         - estimates state value

UPDATE RULES:
  Actor:  theta <- theta + alpha * grad log pi_theta(a|s) * A(s,a)
  Critic: phi <- phi - beta * grad (V_phi(s) - G_t)^2

  where A(s,a) = r + gamma*V(s') - V(s) is the TD advantage

A2C vs PPO:
  - A2C: simpler, direct policy gradient update
  - PPO: adds clipping for more stable updates
  - A2C is faster per iteration but less stable
  - PPO is preferred in practice for its reliability

A2C vs DQN:
  - A2C: on-policy, natural for continuous actions
  - DQN: off-policy (can reuse data), discrete actions only
  - A2C: can be more sample-efficient per update
  - DQN: more sample-efficient overall (replay buffer)
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Optional


class A2CNetwork(nn.Module):
    """Actor-Critic network with separate heads."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if continuous:
            self.actor_mean = nn.Sequential(
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )
            self.actor_log_std = nn.Parameter(-0.5 * torch.ones(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        value = self.critic(features)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            return mean, std, value
        else:
            logits = self.actor(features)
            return logits, value


class A2CTrader:
    """
    Advantage Actor-Critic agent for financial trading.

    KEY CONCEPT - N-STEP RETURNS:
    Instead of using single-step TD error, A2C can use n-step returns:
      G_t:t+n = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n*V(s_{t+n})

    Longer n-steps:
      + Less bias (using more actual rewards)
      - More variance (more randomness in the trajectory)

    For financial applications, n=5 to 20 steps (1-4 trading weeks)
    provides a good balance.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 7e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 5,
        max_grad_norm: float = 0.5,
        continuous: bool = False,
    ):
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = A2CNetwork(state_dim, action_dim, hidden_dim, continuous).to(self.device)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=lr, alpha=0.99, eps=1e-5)

        self.training_history: List[Dict] = []

    def select_action(self, state: np.ndarray, training: bool = True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if self.continuous:
            mean, std, value = self.network(state_t)
            dist = Normal(mean, std)
            if training:
                action = dist.sample()
            else:
                action = mean
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            return action.detach().cpu().numpy().flatten(), log_prob, entropy, value.squeeze()
        else:
            logits, value = self.network(state_t)
            dist = Categorical(logits=logits)
            if training:
                action = dist.sample()
            else:
                action = logits.argmax(-1)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action.item(), log_prob, entropy, value.squeeze()

    def compute_n_step_returns(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        next_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute n-step returns for A2C update.

        G_t = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n*V(s_{t+n})
        """
        returns = []
        R = next_value.detach()

        for t in reversed(range(len(rewards))):
            if dones[t]:
                R = torch.tensor(0.0).to(self.device)
            R = rewards[t] + self.gamma * R
            returns.insert(0, R)

        return torch.stack(returns)

    def train_n_steps(self, env, obs: np.ndarray):
        """Collect n steps and perform A2C update."""
        states, actions, rewards, dones = [], [], [], []
        log_probs, entropies, values = [], [], []

        for _ in range(self.n_steps):
            action, log_prob, entropy, value = self.select_action(obs, training=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)

            obs = next_obs

            if done:
                obs, info = env.reset()

        # Bootstrap value for last state
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if self.continuous:
                _, _, next_value = self.network(state_t)
            else:
                _, next_value = self.network(state_t)
            next_value = next_value.squeeze()

        # Compute returns and advantages
        returns = self.compute_n_step_returns(rewards, values, dones, next_value)
        values_t = torch.stack(values)
        advantages = returns - values_t.detach()

        # Losses
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        policy_loss = -(log_probs_t * advantages.detach()).mean()
        value_loss = (returns - values_t).pow(2).mean()
        entropy_loss = -entropies_t.mean()

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return obs, info, {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "avg_reward": np.mean(rewards),
        }

    def train(self, env, total_steps: int = 50000) -> List[Dict]:
        """Train A2C for a specified number of total steps."""
        obs, info = env.reset()
        all_metrics = []
        step_count = 0
        episode_reward = 0
        episode_count = 0

        while step_count < total_steps:
            obs, info, metrics = self.train_n_steps(env, obs)
            step_count += self.n_steps

            episode_reward += metrics["avg_reward"] * self.n_steps

            if step_count % 5000 < self.n_steps:
                episode_count += 1
                all_metrics.append(metrics)
                self.training_history.append(metrics)

                print(f"  Step {step_count:6d}: "
                      f"AvgReward={metrics['avg_reward']:.4f}  "
                      f"PolicyLoss={metrics['policy_loss']:.4f}  "
                      f"ValueLoss={metrics['value_loss']:.4f}  "
                      f"Entropy={metrics['entropy']:.4f}")

                episode_reward = 0

        return all_metrics


def demonstrate_a2c():
    """Train A2C agent on stock trading."""
    print("=" * 70)
    print("  CHAPTER 11: ADVANTAGE ACTOR-CRITIC (A2C)")
    print("=" * 70)

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environments.stock_trading_env import StockTradingEnv

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)
    obs, _ = env.reset()
    state_dim = len(obs)

    print(f"\nA2C Configuration:")
    print(f"  State dim:    {state_dim}")
    print(f"  N-steps:      5")
    print(f"  Gamma:        0.99")

    agent = A2CTrader(
        state_dim=state_dim,
        action_dim=5,
        n_steps=5,
        continuous=False,
    )

    print("\n--- Training A2C (20,000 steps) ---")
    agent.train(env, total_steps=20000)

    print("\nKey Insights:")
    print("  1. A2C uses n-step returns for better bias-variance tradeoff")
    print("  2. Synchronous training is simpler than async (A3C)")
    print("  3. RMSprop optimizer is traditional for A2C")
    print("  4. Suitable as a simpler alternative to PPO\n")

    return agent


if __name__ == "__main__":
    demonstrate_a2c()
