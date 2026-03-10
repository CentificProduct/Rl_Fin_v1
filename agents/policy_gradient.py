"""
=============================================================================
MODULE 9: Policy Gradient Methods (REINFORCE)
=============================================================================

THEORY:
-------
Policy gradient methods directly optimize the policy without learning Q-values.

THE POLICY GRADIENT THEOREM:
  grad J(theta) = E_pi[ sum_t grad log pi_theta(a_t|s_t) * G_t ]

  where G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k} is the return from time t.

This says: increase the probability of actions that led to high returns,
decrease the probability of actions that led to low returns.

ADVANTAGES OVER DQN:
  - Naturally handles continuous action spaces
  - Can represent stochastic policies (useful for exploration)
  - More stable optimization (smooth policy updates)
  - Better for multi-modal action distributions

DISADVANTAGES:
  - High variance (REINFORCE with baseline helps)
  - Sample inefficient (on-policy, can't reuse old data)
  - Can converge to local optima

BASELINE AND VARIANCE REDUCTION:
  Replace G_t with advantage A_t = G_t - V(s_t):
    grad J = E[ grad log pi(a_t|s_t) * (G_t - V(s_t)) ]

  The baseline V(s_t) doesn't change the expected gradient but
  dramatically reduces variance, accelerating learning.

FINANCIAL APPLICATION:
  Policy gradients are ideal for portfolio allocation where the agent
  must output continuous weights. The policy directly outputs a
  probability distribution over positions.
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Neural network that parameterizes the policy pi_theta(a|s).

    For discrete actions: outputs action probabilities via softmax
    For continuous actions: outputs mean and std of Gaussian policy
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64],
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous

        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh(),
            ])
            prev_dim = h_dim

        self.features = nn.Sequential(*layers)

        if continuous:
            self.mean_head = nn.Linear(prev_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state: torch.Tensor):
        features = self.features(state)

        if self.continuous:
            mean = self.mean_head(features)
            std = self.log_std.exp().expand_as(mean)
            return mean, std
        else:
            logits = self.action_head(features)
            return logits


class ValueNetwork(nn.Module):
    """Baseline network V(s) for variance reduction."""

    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class REINFORCEAgent:
    """
    REINFORCE with baseline (vanilla policy gradient).

    ALGORITHM:
      For each episode:
        1. Generate trajectory tau = (s0,a0,r0, s1,a1,r1, ...) using pi_theta
        2. For each step t:
           G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}  (discounted return)
           A_t = G_t - V_phi(s_t)                    (advantage estimate)
        3. Policy update:
           theta <- theta + alpha_pi * sum_t grad log pi_theta(a_t|s_t) * A_t
        4. Baseline update:
           phi <- phi - alpha_v * sum_t grad (V_phi(s_t) - G_t)^2

    ENTROPY REGULARIZATION:
      Adding an entropy bonus H(pi) to the objective encourages exploration:
        J(theta) = E[sum gamma^t * r_t] + beta * H(pi_theta)

      In finance, this prevents the policy from becoming deterministic too
      quickly, maintaining trading flexibility.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64],
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        continuous: bool = False,
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.continuous = continuous
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims, continuous).to(self.device)
        self.baseline = ValueNetwork(state_dim, hidden_dims).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.baseline.parameters(), lr=value_lr)

        self.training_history: List[Dict] = []

    def select_action(self, state: np.ndarray, training: bool = True):
        """Sample action from the policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad() if not training else torch.enable_grad():
            if self.continuous:
                mean, std = self.policy(state_tensor)
                if training:
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                else:
                    action = mean
                    log_prob = torch.zeros(1)
                return action.cpu().numpy().flatten(), log_prob

            else:
                logits = self.policy(state_tensor)
                dist = Categorical(logits=logits)
                if training:
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                else:
                    action = logits.argmax(dim=-1)
                    log_prob = torch.zeros(1)
                return action.item(), log_prob

    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Compute discounted returns G_t for each timestep.

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
            = r_t + gamma * G_{t+1}

        We compute this efficiently working backwards from the end.
        """
        T = len(rewards)
        returns = np.zeros(T)
        running_return = 0

        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        return returns

    def train_episode(self, env, max_steps: int = 2000) -> Dict:
        """
        Collect one episode and perform policy gradient update.
        """
        obs, info = env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        total_reward = 0

        # Collect trajectory
        for step in range(max_steps):
            action, log_prob = self.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            total_reward += reward

            obs = next_obs
            if done:
                break

        # Compute returns
        returns = self.compute_returns(rewards)

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        log_probs_t = torch.stack([lp if isinstance(lp, torch.Tensor) else torch.tensor(lp) for lp in log_probs]).to(self.device)

        # Normalize returns for stability
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Compute baseline values
        values = self.baseline(states_t).squeeze()
        advantages = returns_t - values.detach()

        # Policy gradient loss
        policy_loss = -(log_probs_t.squeeze() * advantages).mean()

        # Entropy bonus
        with torch.no_grad():
            if self.continuous:
                mean, std = self.policy(states_t)
                entropy = Normal(mean, std).entropy().mean()
            else:
                logits = self.policy(states_t)
                entropy = Categorical(logits=logits).entropy().mean()

        total_policy_loss = policy_loss - self.entropy_coef * entropy

        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Update baseline
        value_loss = nn.MSELoss()(values, returns_t)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        metrics = {
            "total_reward": total_reward,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "portfolio_value": info.get("portfolio_value", 0),
            "total_return": info.get("total_return", 0),
            "sharpe_ratio": info.get("sharpe_ratio", 0),
            "episode_length": len(rewards),
        }
        self.training_history.append(metrics)
        return metrics


def demonstrate_reinforce():
    """Train REINFORCE agent on stock trading."""
    print("=" * 70)
    print("  CHAPTER 9: REINFORCE (POLICY GRADIENT) FOR TRADING")
    print("=" * 70)

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environments.stock_trading_env import StockTradingEnv

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)
    obs, _ = env.reset()
    state_dim = len(obs)

    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=5,
        gamma=0.99,
        entropy_coef=0.01,
        continuous=False,
    )

    print(f"\nREINFORCE Configuration:")
    print(f"  State dim:       {state_dim}")
    print(f"  Action dim:      5 (discrete)")
    print(f"  Gamma:           {agent.gamma}")
    print(f"  Entropy coef:    {agent.entropy_coef}")

    print("\n--- Training REINFORCE (150 episodes) ---")
    for ep in range(150):
        metrics = agent.train_episode(env)

        if (ep + 1) % 30 == 0:
            print(f"  Episode {ep + 1:4d}: "
                  f"Return={metrics['total_return'] * 100:7.2f}%  "
                  f"Sharpe={metrics['sharpe_ratio']:.3f}  "
                  f"PolicyLoss={metrics['policy_loss']:.4f}  "
                  f"Entropy={metrics['entropy']:.4f}")

    print("\nKey Insights:")
    print("  1. REINFORCE directly optimizes the policy (no Q-values needed)")
    print("  2. The baseline V(s) dramatically reduces gradient variance")
    print("  3. Entropy regularization prevents premature convergence")
    print("  4. On-policy: must collect fresh data each episode (less sample efficient)\n")

    return agent


if __name__ == "__main__":
    demonstrate_reinforce()
