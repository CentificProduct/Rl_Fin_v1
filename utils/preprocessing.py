"""
=============================================================================
UTILITY: Feature Preprocessing for RL in Finance
=============================================================================
"""

import numpy as np
from typing import Tuple, Optional


class FeatureNormalizer:
    """
    Online normalization for streaming financial data.

    Uses Welford's algorithm for numerically stable running statistics.
    Critical for RL because:
    1. Neural networks train better with normalized inputs
    2. Financial features span very different scales
    3. Statistics change over time (non-stationarity)
    """

    def __init__(self, shape: Tuple[int, ...]):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self._m2 = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray):
        """Update running statistics with Welford's online algorithm."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2
        self.var = self._m2 / max(self.count - 1, 1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize using running statistics."""
        std = np.sqrt(self.var + 1e-8)
        return (x - self.mean) / std

    def denormalize(self, x_normalized: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        std = np.sqrt(self.var + 1e-8)
        return x_normalized * std + self.mean


class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms.

    Stores (s, a, r, s', done) transitions and samples mini-batches.

    WHY REPLAY BUFFERS MATTER IN FINANCE:
    1. Breaks temporal correlation in market data
    2. Reuses rare but important events (crashes, squeezes)
    3. Stabilizes training of neural network approximators
    """

    def __init__(self, capacity: int = 100000, state_dim: int = 12):
        self.capacity = capacity
        self.state_dim = state_dim
        self.position = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random mini-batch."""
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER).

    Samples transitions with probability proportional to their TD error.
    High-error transitions are surprising and contain more learning signal.

    FINANCIAL MOTIVATION:
    Market crashes and extreme events are rare but crucial. PER ensures
    the agent revisits these high-impact experiences more frequently,
    analogous to how risk managers stress-test against extreme scenarios.
    """

    def __init__(
        self,
        capacity: int = 100000,
        state_dim: int = 12,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        super().__init__(capacity, state_dim)
        self.alpha = alpha  # prioritization exponent
        self.beta_start = beta_start  # importance sampling correction
        self.beta_frames = beta_frames
        self.frame = 0

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        """Anneal beta from beta_start to 1.0 over training."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        self.priorities[self.position] = self.max_priority
        super().push(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        self.frame += 1

        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)

        # Importance sampling weights to correct bias
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        priorities = np.abs(td_errors) + 1e-6
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
