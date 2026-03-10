"""
=============================================================================
MODULE 10: Proximal Policy Optimization (PPO) for Trading
=============================================================================

THEORY:
-------
PPO (Schulman et al., 2017) is the most widely used policy gradient algorithm.
It addresses the key weakness of REINFORCE: instability from large policy updates.

THE PROBLEM:
  Vanilla policy gradient can take huge steps that collapse performance.
  Trust region methods (TRPO) constrain the KL divergence between
  old and new policies, but are complex to implement.

PPO's ELEGANT SOLUTION - CLIPPED OBJECTIVE:
  L_CLIP(theta) = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]

  where r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t) (probability ratio)

  The clipping prevents the policy ratio from moving too far from 1,
  ensuring small, safe updates.

WHY PPO IS GREAT FOR FINANCE:
  1. STABILITY: Financial rewards are noisy. PPO's conservative updates
     prevent catastrophic policy changes from unlucky episodes.
  2. SAMPLE EFFICIENCY: PPO reuses each batch for multiple epochs,
     extracting more learning from expensive market data.
  3. CONTINUOUS ACTIONS: Natural for portfolio weight allocation.
  4. SCALABILITY: Works well with parallel environments.

GENERALIZED ADVANTAGE ESTIMATION (GAE):
  A_t^GAE = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
  where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

  GAE interpolates between:
  - lambda=0: TD residual (low variance, high bias)
  - lambda=1: Monte Carlo advantage (high variance, low bias)
  Typically lambda=0.95 works well.
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional


class ActorCriticNetwork(nn.Module):
    """
    Shared backbone with separate actor (policy) and critic (value) heads.

    Sharing features between actor and critic:
    - Reduces parameters and computation
    - Provides implicit regularization
    - Features learned for value estimation help policy learning

    Separation of heads:
    - Actor and critic may need different representations
    - Prevents gradient interference between the two objectives
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Actor head
        if continuous:
            self.actor_mean = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, action_dim),
                nn.Tanh(),  # bound actions to [-1, 1]
            )
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, action_dim),
            )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor):
        shared_features = self.shared(state)
        value = self.critic(shared_features)

        if self.continuous:
            mean = self.actor_mean(shared_features)
            std = self.actor_log_std.exp().expand_as(mean)
            return mean, std, value
        else:
            logits = self.actor(shared_features)
            return logits, value

    def get_action_and_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Get action, log probability, entropy, and value."""
        if self.continuous:
            mean, std, value = self(state)
            dist = Normal(mean, std)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits, value = self(state)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class PPOTrader:
    """
    PPO agent for financial trading.

    TRAINING PROCEDURE:
    1. Collect T timesteps of experience using current policy
    2. Compute GAE advantages
    3. For K epochs:
       a. Sample mini-batches from collected data
       b. Compute clipped policy loss
       c. Compute value loss
       d. Compute entropy bonus
       e. Combined loss = policy_loss + c1*value_loss - c2*entropy
       f. Gradient step
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        rollout_length: int = 2048,
        continuous: bool = False,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.continuous = continuous

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim, continuous
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        self.training_history: List[Dict] = []

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE balances bias and variance in advantage estimation:

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = delta_t + (gamma * lambda) * delta_{t+1} + (gamma*lambda)^2 * delta_{t+2} + ...

        Also returns the TD(lambda) returns for value function targets:
        G_t = A_t + V(s_t)
        """
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def collect_rollout(self, env) -> Dict:
        """Collect rollout_length steps of experience."""
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        obs, info = env.reset()

        for _ in range(self.rollout_length):
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(state_t)

            if self.continuous:
                action_np = action.cpu().numpy().flatten()
            else:
                action_np = action.item()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            states.append(obs)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob.item())
            values.append(value.item())

            obs = next_obs
            if done:
                obs, info = env.reset()

        # Compute next value for GAE
        with torch.no_grad():
            next_state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _, _, _, next_value = self.network.get_action_and_value(next_state_t)
            next_value = next_value.item()

        advantages, returns = self.compute_gae(
            np.array(rewards), np.array(values), np.array(dones), next_value
        )

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "log_probs": np.array(log_probs),
            "values": np.array(values),
            "advantages": advantages,
            "returns": returns,
            "rewards": np.array(rewards),
            "info": info,
        }

    def update(self, rollout: Dict) -> Dict:
        """
        Perform PPO update using collected rollout data.

        CLIPPED OBJECTIVE:
          L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

        The clip prevents large policy changes:
          - If A > 0 (good action): ratio capped at 1+eps (don't get too greedy)
          - If A < 0 (bad action): ratio capped at 1-eps (don't overreact)
        """
        states_t = torch.FloatTensor(rollout["states"]).to(self.device)
        if self.continuous:
            actions_t = torch.FloatTensor(rollout["actions"]).to(self.device)
        else:
            actions_t = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs_t = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages_t = torch.FloatTensor(rollout["advantages"]).to(self.device)
        returns_t = torch.FloatTensor(rollout["returns"]).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(states_t))

            for start in range(0, len(states_t), self.batch_size):
                end = min(start + self.batch_size, len(states_t))
                batch_idx = indices[start:end]

                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                _, new_log_probs, entropy, new_values = \
                    self.network.get_action_and_value(batch_states, batch_actions)

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (also clipped for stability)
                value_loss = nn.MSELoss()(new_values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def train(self, env, n_iterations: int = 50) -> List[Dict]:
        """Full PPO training loop."""
        all_metrics = []

        for iteration in range(n_iterations):
            rollout = self.collect_rollout(env)
            update_info = self.update(rollout)

            avg_reward = rollout["rewards"].mean()
            total_reward = rollout["rewards"].sum()

            metrics = {
                **update_info,
                "avg_reward": avg_reward,
                "total_reward": total_reward,
                "iteration": iteration,
            }
            all_metrics.append(metrics)
            self.training_history.append(metrics)

            if (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration + 1:4d}: "
                      f"AvgReward={avg_reward:.4f}  "
                      f"PolicyLoss={update_info['policy_loss']:.4f}  "
                      f"ValueLoss={update_info['value_loss']:.4f}  "
                      f"Entropy={update_info['entropy']:.4f}")

        return all_metrics


def demonstrate_ppo():
    """Train PPO agent on stock trading."""
    print("=" * 70)
    print("  CHAPTER 10: PROXIMAL POLICY OPTIMIZATION (PPO)")
    print("=" * 70)

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environments.stock_trading_env import StockTradingEnv

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)
    obs, _ = env.reset()
    state_dim = len(obs)

    print(f"\nPPO Configuration:")
    print(f"  State dim:        {state_dim}")
    print(f"  Clip epsilon:     0.2")
    print(f"  GAE lambda:       0.95")
    print(f"  N epochs:         10")
    print(f"  Rollout length:   2048")

    agent = PPOTrader(
        state_dim=state_dim,
        action_dim=5,
        hidden_dim=256,
        continuous=False,
        rollout_length=1024,
        n_epochs=10,
    )

    print("\n--- Training PPO (50 iterations) ---")
    agent.train(env, n_iterations=50)

    print("\nKey Insights:")
    print("  1. PPO clips the policy update to prevent catastrophic changes")
    print("  2. GAE provides low-variance advantage estimates")
    print("  3. Multiple epochs per rollout improve sample efficiency")
    print("  4. Entropy regularization maintains exploration\n")

    return agent


if __name__ == "__main__":
    demonstrate_ppo()
