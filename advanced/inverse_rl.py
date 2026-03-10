"""
=============================================================================
MODULE 18: Inverse Reinforcement Learning for Market Analysis
=============================================================================

THEORY:
-------
Inverse RL (IRL) recovers the reward function from observed behavior.
Instead of learning a policy from rewards, we learn the reward function
that explains observed expert behavior.

MOTIVATION IN FINANCE:
  We can observe how successful traders and institutional investors
  make decisions, but we don't know their exact reward function.
  IRL can recover what they're actually optimizing:
  - Are they maximizing returns? Risk-adjusted returns?
  - How do they weigh drawdown vs. returns?
  - What's their effective risk aversion parameter?

ALGORITHMS:
  1. MAXIMUM ENTROPY IRL (Ziebart, 2008):
     Finds a reward function R such that the expert policy is optimal,
     with maximum entropy among all optimal policies.

     max_R  E_expert[sum R(s,a)] - log Z(R)

  2. GENERATIVE ADVERSARIAL IMITATION LEARNING (GAIL):
     Uses a discriminator to distinguish expert from agent behavior.
     No explicit reward recovery, but learns to imitate.

  3. BEHAVIORAL CLONING:
     Simplest approach: supervised learning to match expert actions.
     Suffers from distributional shift (compounding errors).

APPLICATIONS:
  - Recover hedge fund strategies from position data
  - Understand central bank policy functions
  - Learn market maker reward functions
  - Replicate index rebalancing strategies
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional


class RewardNetwork(nn.Module):
    """Neural network that represents the learned reward function."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(-1).float()
        elif action.dtype != torch.float32:
            action = action.float()
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class MaxEntIRL:
    """
    Maximum Entropy Inverse RL.

    The principle of maximum entropy says: among all policies that match
    the expert's feature expectations, choose the one with maximum entropy.

    This gives the reward function:
      R*(s, a) = theta' * phi(s, a)

    where theta is learned to match:
      E_expert[phi(s, a)] = E_pi_R[phi(s, a)]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_actions = n_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_net = RewardNetwork(state_dim, 1, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=lr)

    def compute_expert_feature_expectations(
        self, expert_trajectories: List[List[Tuple[np.ndarray, int]]]
    ) -> torch.Tensor:
        """
        Compute average state-action features across expert trajectories.

        This is what we're trying to match: the feature expectations
        under the expert's policy.
        """
        all_rewards = []

        for trajectory in expert_trajectories:
            for state, action in trajectory:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_t = torch.LongTensor([action]).to(self.device)
                reward = self.reward_net(state_t, action_t)
                all_rewards.append(reward)

        return torch.cat(all_rewards).mean()

    def update(
        self,
        expert_trajectories: List[List[Tuple[np.ndarray, int]]],
        agent_trajectories: List[List[Tuple[np.ndarray, int]]],
    ) -> float:
        """
        Update reward function to match expert behavior.

        The gradient is:
          grad_theta L = E_expert[grad R] - E_agent[grad R]

        This pushes up rewards for expert-like behavior and down for
        agent behavior, until the agent's policy matches the expert's.
        """
        # Expert reward
        expert_reward = self.compute_expert_feature_expectations(expert_trajectories)

        # Agent reward
        agent_reward = self.compute_expert_feature_expectations(agent_trajectories)

        # Loss: maximize expert reward - agent reward
        loss = -(expert_reward - agent_reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_reward(self, state: np.ndarray, action: int) -> float:
        """Get the learned reward for a state-action pair."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.LongTensor([action]).to(self.device)

        with torch.no_grad():
            reward = self.reward_net(state_t, action_t)

        return reward.item()


class BehavioralCloner:
    """
    Behavioral Cloning: supervised learning from expert demonstrations.

    The simplest form of imitation learning:
      min_theta E_{(s,a) ~ D_expert} [ -log pi_theta(a|s) ]

    PROBLEMS:
      - Distributional shift: agent visits states the expert never saw
      - Compounding errors: small mistakes lead to unseen states
      - No recovery: doesn't learn to correct mistakes

    SOLUTIONS:
      - DAgger: Iteratively collect data from agent states with expert labels
      - Noise injection: Train with noisy expert demonstrations
      - Inverse RL: Learn the reward and then optimize (more robust)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128, lr: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_on_demonstrations(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 64,
    ) -> List[float]:
        """Train policy via supervised learning on expert data."""
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        dataset = torch.utils.data.TensorDataset(states_t, actions_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_states, batch_actions in loader:
                logits = self.policy(batch_states)
                loss = self.loss_fn(logits, batch_actions)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        return losses

    def predict(self, state: np.ndarray) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(state_t)
        return logits.argmax(dim=-1).item()


class InverseRLMarketAnalyzer:
    """
    Analyzes market participant behavior using Inverse RL.

    Given observed trading patterns (positions, trades), this module
    recovers the implicit reward function being optimized.
    """

    def __init__(self, state_dim: int, n_actions: int = 5):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.irl = MaxEntIRL(state_dim, 1, n_actions)
        self.cloner = BehavioralCloner(state_dim, n_actions)

    def analyze_expert_strategy(
        self,
        expert_states: np.ndarray,
        expert_actions: np.ndarray,
    ) -> Dict:
        """
        Analyze an expert's trading strategy through IRL.

        Returns:
        - Recovered reward function weights
        - Behavioral clone of the strategy
        - Analysis of what the expert appears to optimize
        """
        print("  Step 1: Behavioral Cloning...")
        bc_losses = self.cloner.train_on_demonstrations(
            expert_states, expert_actions, n_epochs=50
        )

        # Analyze the recovered policy
        print("  Step 2: Analyzing recovered reward function...")
        reward_analysis = {}

        for action in range(self.n_actions):
            action_rewards = []
            for i in range(min(100, len(expert_states))):
                r = self.irl.get_reward(expert_states[i], action)
                action_rewards.append(r)
            reward_analysis[action] = {
                'mean': np.mean(action_rewards),
                'std': np.std(action_rewards),
            }

        # Action distribution analysis
        predicted_actions = [self.cloner.predict(s) for s in expert_states[:200]]
        action_dist = np.bincount(predicted_actions, minlength=self.n_actions)
        action_dist = action_dist / action_dist.sum()

        return {
            'bc_final_loss': bc_losses[-1],
            'reward_by_action': reward_analysis,
            'action_distribution': action_dist,
        }


def demonstrate_inverse_rl():
    """Demonstrate Inverse RL for market analysis."""
    print("=" * 70)
    print("  CHAPTER 18: INVERSE RL FOR MARKET ANALYSIS")
    print("=" * 70)

    np.random.seed(42)

    state_dim = 10
    n_samples = 1000
    n_actions = 5

    # Generate synthetic "expert" data (momentum-following expert)
    expert_states = np.random.randn(n_samples, state_dim)

    expert_actions = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        momentum = expert_states[i, 0]
        volatility = abs(expert_states[i, 1])

        if momentum > 0.5 and volatility < 1.0:
            expert_actions[i] = 4  # strong buy
        elif momentum > 0.1:
            expert_actions[i] = 3  # buy
        elif momentum < -0.5 and volatility < 1.0:
            expert_actions[i] = 0  # strong sell
        elif momentum < -0.1:
            expert_actions[i] = 1  # sell
        else:
            expert_actions[i] = 2  # hold

    print(f"\nSynthetic Expert Data: {n_samples} observations")
    print(f"  Expert rule: momentum-following with volatility filter")
    print(f"  Action distribution: {np.bincount(expert_actions, minlength=5) / n_samples}")

    # Analyze
    analyzer = InverseRLMarketAnalyzer(state_dim, n_actions)
    results = analyzer.analyze_expert_strategy(expert_states, expert_actions)

    action_names = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
    print(f"\n--- Recovered Action Distribution ---")
    for i, name in enumerate(action_names):
        print(f"  {name:>12s}: {results['action_distribution'][i]:.3f}")

    print(f"\n  Behavioral Cloning Final Loss: {results['bc_final_loss']:.4f}")

    print("\nKey Insights:")
    print("  1. IRL recovers WHAT an expert optimizes (not just HOW)")
    print("  2. Behavioral cloning provides a quick imitation baseline")
    print("  3. MaxEnt IRL avoids ambiguity via entropy regularization")
    print("  4. Can reveal hidden objectives of market participants\n")

    return results


if __name__ == "__main__":
    demonstrate_inverse_rl()
