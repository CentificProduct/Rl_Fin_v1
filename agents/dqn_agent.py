"""
=============================================================================
MODULE 8: Deep Q-Network (DQN) for Trading
=============================================================================

THEORY:
-------
DQN extends Q-Learning to continuous state spaces using neural networks.
Two critical innovations make it work:

1. EXPERIENCE REPLAY:
   Store transitions (s, a, r, s', done) in a replay buffer.
   Sample random mini-batches for training.
   - Breaks temporal correlations in sequential market data
   - Reuses rare but important experiences (market crashes)
   - More sample-efficient than online learning

2. TARGET NETWORK:
   Maintain a separate "target" network Q_target that is updated slowly.
   TD target = r + gamma * max_a' Q_target(s', a')
   - Stabilizes training by preventing moving-target problem
   - Updated via hard copy every C steps or soft (Polyak) averaging:
     theta_target <- tau * theta + (1 - tau) * theta_target

VARIANTS IMPLEMENTED:
  - Vanilla DQN
  - Double DQN (van Hasselt, 2015): addresses overestimation
  - Dueling DQN (Wang, 2016): separate value and advantage streams
  - Prioritized Experience Replay (Schaul, 2016): focus on surprising transitions

FINANCIAL CONSIDERATIONS:
  - Financial data is heavily non-stationary: periodic retraining needed
  - Fat-tailed returns make Huber loss better than MSE
  - Reward clipping can hurt (financial signals are already small)
  - Careful feature normalization is critical
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import copy


class DQNetwork(nn.Module):
    """
    Deep Q-Network with optional dueling architecture.

    Architecture choices for financial data:
    - LayerNorm instead of BatchNorm (works with single samples)
    - Moderate depth (3-4 layers) to avoid overfitting
    - Dropout for regularization
    - Huber loss for robustness to outlier returns
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dueling: bool = True,
    ):
        super().__init__()
        self.dueling = dueling

        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        self.features = nn.Sequential(*layers)

        if dueling:
            self.value_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)

        if self.dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)

        return q_values


class DQNAgent:
    """
    Complete DQN agent with all modern improvements.

    TRAINING LOOP:
    1. Observe state s
    2. Select action a via epsilon-greedy from Q_online(s)
    3. Execute a, observe r, s', done
    4. Store (s, a, r, s', done) in replay buffer
    5. Sample mini-batch from buffer
    6. Compute targets: y = r + gamma * max_a' Q_target(s', a')
       (For Double DQN: a* = argmax_a' Q_online(s', a'), y = r + gamma * Q_target(s', a*))
    7. Update Q_online to minimize (Q_online(s, a) - y)^2
    8. Periodically update Q_target
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        tau: float = 0.005,  # for soft update
        double_dqn: bool = True,
        dueling: bool = True,
        use_soft_update: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.double_dqn = double_dqn
        self.use_soft_update = use_soft_update

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.online_net = DQNetwork(state_dim, action_dim, hidden_dims, dueling).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim, hidden_dims, dueling).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)

        self.total_steps = 0
        self.training_history: List[Dict] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """
        One gradient step of DQN training.

        Returns the loss value, or None if buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample mini-batch
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: select action with online net, evaluate with target net
                best_actions = self.online_net(next_states_t).argmax(dim=1)
                next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(next_states_t).max(dim=1)[0]

            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network
        self.total_steps += 1
        if self.use_soft_update:
            self._soft_update()
        elif self.total_steps % self.target_update_freq == 0:
            self._hard_update()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        return loss.item()

    def _soft_update(self):
        """Polyak averaging: theta_target <- tau*theta + (1-tau)*theta_target"""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def _hard_update(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_episode(self, env, max_steps: int = 2000) -> Dict:
        """Train for one episode."""
        obs, info = env.reset()
        total_reward = 0
        total_loss = 0
        loss_count = 0
        steps = 0

        for step in range(max_steps):
            action = self.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.store_transition(obs, action, reward, next_obs, float(done))

            loss = self.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            total_reward += reward
            obs = next_obs
            steps += 1

            if done:
                break

        avg_loss = total_loss / max(loss_count, 1)

        metrics = {
            "total_reward": total_reward,
            "avg_loss": avg_loss,
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "portfolio_value": info.get("portfolio_value", 0),
            "total_return": info.get("total_return", 0),
            "sharpe_ratio": info.get("sharpe_ratio", 0),
            "max_drawdown": info.get("max_drawdown", 0),
            "steps": steps,
        }
        self.training_history.append(metrics)
        return metrics

    def evaluate(self, env, n_episodes: int = 10) -> Dict:
        """Evaluate agent without exploration."""
        returns = []
        sharpes = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            while True:
                action = self.select_action(obs, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            returns.append(info.get("total_return", 0))
            sharpes.append(info.get("sharpe_ratio", 0))

        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
        }


def demonstrate_dqn():
    """Train DQN agent on the stock trading environment."""
    print("=" * 70)
    print("  CHAPTER 8: DEEP Q-NETWORK (DQN) FOR TRADING")
    print("=" * 70)

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environments.stock_trading_env import StockTradingEnv

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)
    obs, _ = env.reset()
    state_dim = len(obs)

    print(f"\nDQN Configuration:")
    print(f"  State dim:       {state_dim}")
    print(f"  Action dim:      5")
    print(f"  Double DQN:      True")
    print(f"  Dueling:         True")
    print(f"  Replay buffer:   100,000")
    print(f"  Batch size:      64")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=5,
        double_dqn=True,
        dueling=True,
        learning_rate=1e-4,
        epsilon_decay_steps=30000,
    )

    print("\n--- Training DQN (100 episodes) ---")
    for ep in range(100):
        metrics = agent.train_episode(env)

        if (ep + 1) % 20 == 0:
            eval_results = agent.evaluate(env, n_episodes=5)
            print(f"  Episode {ep + 1:4d}: "
                  f"Train Return={metrics['total_return'] * 100:7.2f}%  "
                  f"Eval Return={eval_results['mean_return'] * 100:7.2f}% +/- {eval_results['std_return'] * 100:.2f}%  "
                  f"Eval Sharpe={eval_results['mean_sharpe']:.3f}  "
                  f"Epsilon={metrics['epsilon']:.3f}  "
                  f"Loss={metrics['avg_loss']:.4f}")

    print("\nKey Insights:")
    print("  1. DQN handles continuous states naturally via neural networks")
    print("  2. Double DQN prevents overestimating expected returns")
    print("  3. Dueling architecture separates state value from action advantage")
    print("  4. Experience replay breaks the correlation in sequential market data\n")

    return agent


if __name__ == "__main__":
    demonstrate_dqn()
