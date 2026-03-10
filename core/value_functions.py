"""
=============================================================================
MODULE 2: Value Functions and Temporal Difference Learning
=============================================================================

THEORY:
-------
Value functions are the central objects in RL. They estimate "how good" it is
for an agent to be in a given state (or to take a given action in a state).

STATE VALUE FUNCTION V^pi(s):
  V^pi(s) = E_pi[ sum_{k=0}^{inf} gamma^k * R_{t+k+1} | S_t = s ]

  "Expected cumulative discounted reward starting from state s,
   following policy pi."

ACTION VALUE FUNCTION Q^pi(s, a):
  Q^pi(s, a) = E_pi[ sum_{k=0}^{inf} gamma^k * R_{t+k+1} | S_t = s, A_t = a ]

  "Expected return after taking action a in state s, then following pi."

TEMPORAL DIFFERENCE (TD) LEARNING:
  TD methods learn value functions from experience without a model.
  They bootstrap: update estimates based on other estimates.

  TD(0) update:
    V(s) <- V(s) + alpha * [ R + gamma * V(s') - V(s) ]
                              ^^^^^^^^^^^^^^^^^^^^^^^^^
                              This is the TD error (delta)

  The TD error delta = R + gamma * V(s') - V(s) is the key signal.
  It measures the surprise: how much better (or worse) the outcome was
  compared to what we expected.

FINANCIAL INTERPRETATION:
  - V(s) represents the expected future wealth from the current market state
  - Q(s, a) represents the expected return of a specific trade given conditions
  - The TD error is analogous to "alpha" in finance: excess return over
    the expected return (the "surprise" component)
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from collections import deque
import random


class ValueFunctionApproximator:
    """
    Neural network-based value function approximation.

    WHY APPROXIMATION?
    In finance, the state space is continuous and high-dimensional
    (prices, indicators, positions). We cannot maintain a table for
    every state. Instead, we use a neural network to generalize:

      V_theta(s) ≈ V^pi(s)

    The network learns a mapping from market features to expected value.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        learning_rate: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss: robust to outliers

    def predict(self, state: np.ndarray) -> float:
        """Estimate V(s)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.network(state_tensor).item()

    def update(self, state: np.ndarray, target: float) -> float:
        """
        Update value estimate toward target.

        In TD(0): target = R + gamma * V(s')
        In Monte Carlo: target = G_t (actual return)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        target_tensor = torch.FloatTensor([target]).unsqueeze(0).to(self.device)

        prediction = self.network(state_tensor)
        loss = self.loss_fn(prediction, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def batch_update(self, states: np.ndarray, targets: np.ndarray) -> float:
        """Mini-batch update for efficiency."""
        states_t = torch.FloatTensor(states).to(self.device)
        targets_t = torch.FloatTensor(targets).unsqueeze(1).to(self.device)

        predictions = self.network(states_t)
        loss = self.loss_fn(predictions, targets_t)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()


class QNetworkDueling(nn.Module):
    """
    Dueling DQN Architecture for Q-value estimation.

    CONCEPT - DUELING ARCHITECTURE:
      Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]

    This decomposes the Q-value into:
      - V(s): How good is this state in general?
      - A(s, a): How much better is action a compared to the average?

    FINANCIAL MOTIVATION:
    In many market states, the specific action matters less than being in
    the right state (e.g., during a crash, all actions lose money).
    The dueling architecture naturally captures this by separately learning
    the state value and the action advantage.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Subtract mean advantage for identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class TDLearning:
    """
    Temporal Difference Learning implementations for financial time series.

    TD METHODS HIERARCHY:
      TD(0):    V(s) += alpha * [R + gamma*V(s') - V(s)]
      TD(n):    Uses n-step returns for the target
      TD(lambda): Weighted average of all n-step returns

    Each trades off bias vs. variance:
      - TD(0): Low variance, high bias (bootstraps from a single step)
      - Monte Carlo: Zero bias, high variance (uses actual full returns)
      - TD(n): Interpolates between them
      - TD(lambda): Elegant unification via eligibility traces
    """

    def __init__(
        self,
        state_dim: int,
        gamma: float = 0.99,
        alpha: float = 0.001,
        n_steps: int = 1,
        lambda_trace: float = 0.9,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.n_steps = n_steps
        self.lambda_trace = lambda_trace
        self.value_fn = ValueFunctionApproximator(state_dim)
        self.n_step_buffer: deque = deque(maxlen=n_steps)

    def td_zero_update(
        self, state: np.ndarray, reward: float, next_state: np.ndarray, done: bool
    ) -> float:
        """
        TD(0) update: the simplest TD method.

        Target = R + gamma * V(s')  [or just R if terminal]

        This is the workhorse of many RL algorithms. The TD error
        serves as the learning signal.
        """
        current_value = self.value_fn.predict(state)

        if done:
            td_target = reward
        else:
            next_value = self.value_fn.predict(next_state)
            td_target = reward + self.gamma * next_value

        td_error = td_target - current_value
        loss = self.value_fn.update(state, td_target)

        return td_error

    def n_step_update(
        self,
        state: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        N-step TD update: uses multi-step returns.

        G_t:t+n = R_{t+1} + gamma*R_{t+2} + ... + gamma^{n-1}*R_{t+n}
                  + gamma^n * V(S_{t+n})

        N-step returns reduce bias at the cost of higher variance.
        In finance, this can capture delayed reward signals (e.g., a trade
        that takes several days to realize its profit/loss).
        """
        self.n_step_buffer.append((state, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_steps:
            return None

        n_step_return = 0
        for i, (s, r, ns, d) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * r
            if d:
                break

        if not done:
            _, _, final_next_state, _ = self.n_step_buffer[-1]
            n_step_return += (self.gamma ** self.n_steps) * self.value_fn.predict(final_next_state)

        first_state = self.n_step_buffer[0][0]
        current_value = self.value_fn.predict(first_state)
        td_error = n_step_return - current_value

        self.value_fn.update(first_state, n_step_return)
        return td_error

    def td_lambda_update(
        self,
        episode_transitions: List[Tuple[np.ndarray, float, np.ndarray, bool]],
    ) -> List[float]:
        """
        TD(lambda) with eligibility traces (offline/forward view).

        The lambda-return G_t^lambda is an exponentially weighted average
        of all n-step returns:

          G_t^lambda = (1 - lambda) * sum_{n=1}^{inf} lambda^{n-1} * G_t:t+n

        When lambda=0, this reduces to TD(0).
        When lambda=1, this is Monte Carlo.

        FINANCIAL INSIGHT:
        Lambda controls how far into the future we look for feedback.
        A short-horizon trader might use low lambda (quick feedback),
        while a long-term investor uses high lambda (patient evaluation).
        """
        T = len(episode_transitions)
        td_errors = []

        states = np.array([t[0] for t in episode_transitions])
        rewards = np.array([t[1] for t in episode_transitions])
        next_states = np.array([t[2] for t in episode_transitions])
        dones = np.array([t[3] for t in episode_transitions])

        values = np.array([self.value_fn.predict(s) for s in states])
        next_values = np.array([self.value_fn.predict(s) for s in next_states])

        # Compute lambda-returns (backward view with eligibility traces)
        eligibility = np.zeros_like(values)
        for t in range(T):
            td_error = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]

            eligibility = self.gamma * self.lambda_trace * eligibility
            eligibility[t] = 1.0

            targets = values + self.alpha * td_error * eligibility
            td_errors.append(td_error)

        # Batch update with computed targets
        lambda_returns = self._compute_lambda_returns(rewards, next_values, dones)
        self.value_fn.batch_update(states, lambda_returns)

        return td_errors

    def _compute_lambda_returns(
        self,
        rewards: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute lambda-returns for all timesteps (forward view)."""
        T = len(rewards)
        lambda_returns = np.zeros(T)

        # Work backwards
        next_lambda_return = 0
        for t in reversed(range(T)):
            if dones[t]:
                next_lambda_return = 0

            td_target = rewards[t] + self.gamma * next_values[t] * (1 - dones[t])
            lambda_returns[t] = (
                td_target
                + self.gamma * self.lambda_trace * (1 - dones[t]) * (next_lambda_return - next_values[t])
            )
            next_lambda_return = lambda_returns[t]

        return lambda_returns


def demonstrate_td_learning():
    """
    Demonstrate TD learning on a synthetic financial time series.

    Generates a mean-reverting price process (Ornstein-Uhlenbeck) and
    trains a value function to predict future returns from market states.
    """
    print("=" * 70)
    print("  CHAPTER 2: TEMPORAL DIFFERENCE LEARNING IN FINANCE")
    print("=" * 70)

    np.random.seed(42)

    # Generate Ornstein-Uhlenbeck process (mean-reverting prices)
    n_steps = 1000
    dt = 1.0 / 252  # daily
    mu = 100  # long-term mean
    theta = 0.15  # mean reversion speed
    sigma = 2.0  # volatility

    prices = np.zeros(n_steps)
    prices[0] = mu

    for t in range(1, n_steps):
        dp = theta * (mu - prices[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        prices[t] = prices[t - 1] + dp

    returns = np.diff(prices) / prices[:-1]
    rolling_vol = np.array([
        returns[max(0, i - 20):i].std() if i > 0 else 0.01
        for i in range(len(returns))
    ])
    momentum = np.array([
        returns[max(0, i - 10):i].mean() if i > 0 else 0.0
        for i in range(len(returns))
    ])

    state_dim = 4
    td_learner = TDLearning(state_dim=state_dim, gamma=0.99, alpha=0.001)

    print("\nTraining TD(0) on synthetic market data...")
    td_errors = []

    for t in range(1, len(returns) - 1):
        state = np.array([prices[t] / mu, rolling_vol[t], momentum[t], t / n_steps])
        next_state = np.array([prices[t + 1] / mu, rolling_vol[t + 1], momentum[t + 1], (t + 1) / n_steps])
        reward = returns[t]
        done = (t == len(returns) - 2)

        td_error = td_learner.td_zero_update(state, reward, next_state, done)
        td_errors.append(td_error)

    print(f"  Mean TD Error: {np.mean(td_errors):.6f}")
    print(f"  Std TD Error:  {np.std(td_errors):.6f}")
    print(f"  Final |TD Error|: {abs(td_errors[-1]):.6f}")
    print("\nKey Insight: As learning progresses, TD errors should shrink,")
    print("indicating the value function is becoming more accurate.\n")

    return td_errors


if __name__ == "__main__":
    demonstrate_td_learning()
