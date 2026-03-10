"""
=============================================================================
MODULE 17: Model-Based RL for Finance
=============================================================================

THEORY:
-------
Model-based RL learns a model of the environment dynamics, then uses it
for planning. In finance, the "model" is a model of market dynamics.

MODEL-FREE vs MODEL-BASED:
  Model-Free:
    + Simple, no modeling assumptions
    + Works with any dynamics
    - Sample inefficient (needs lots of data)
    - Cannot plan or look ahead

  Model-Based:
    + Sample efficient (learns from fewer interactions)
    + Can plan ahead (simulate future scenarios)
    + Can incorporate domain knowledge (financial models)
    - Model errors compound (model exploitation)
    - More complex to implement

ARCHITECTURE:
  1. WORLD MODEL: Learns s_{t+1} = f(s_t, a_t) + noise
  2. REWARD MODEL: Learns r_t = g(s_t, a_t)
  3. PLANNER: Uses the model to search for optimal actions
     (e.g., Model Predictive Control, Monte Carlo Tree Search)

FINANCIAL WORLD MODELS:
  1. Neural Network Dynamics: s' = NN(s, a)
  2. Gaussian Process: Provides uncertainty estimates
  3. Ensemble: Multiple models for epistemic uncertainty
  4. Physics-Informed: Embed financial models (GBM, Heston) into NN

DYNA ARCHITECTURE:
  Combines real experience with simulated experience:
  1. Take real action, observe real transition
  2. Update model with real transition
  3. Generate N simulated transitions from model
  4. Update policy with real + simulated data
  This multiplies sample efficiency by N+1.
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional


class FinancialWorldModel(nn.Module):
    """
    Neural network world model for financial dynamics.

    Learns to predict:
    1. Next state: s_{t+1} = f(s_t, a_t)
    2. Reward: r_t = g(s_t, a_t)

    Uses an ensemble of models for uncertainty estimation.
    High uncertainty -> the model is unreliable -> be conservative.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        ensemble_size: int = 5,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.state_dim = state_dim

        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, state_dim + 1),  # next_state + reward
            )
            for _ in range(ensemble_size)
        ])

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, model_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward using a specific ensemble member."""
        if action.dim() == 1:
            action = action.unsqueeze(-1).float()
        elif action.dtype != torch.float32:
            action = action.float()

        x = torch.cat([state, action], dim=-1)
        output = self.models[model_idx](x)

        # Predict state delta (residual learning) and reward
        next_state_delta = output[..., :self.state_dim]
        reward = output[..., -1]

        next_state = state + next_state_delta

        return next_state, reward

    def predict_with_uncertainty(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict using all ensemble members and estimate uncertainty.

        Epistemic uncertainty (model disagreement) is estimated as
        the variance across ensemble predictions.
        """
        all_next_states = []
        all_rewards = []

        for i in range(self.ensemble_size):
            ns, r = self.forward(state, action, model_idx=i)
            all_next_states.append(ns)
            all_rewards.append(r)

        next_states = torch.stack(all_next_states)
        rewards = torch.stack(all_rewards)

        mean_state = next_states.mean(dim=0)
        std_state = next_states.std(dim=0)
        mean_reward = rewards.mean(dim=0)
        std_reward = rewards.std(dim=0)

        return mean_state, std_state, mean_reward, std_reward


class ModelBasedFinancialRL:
    """
    Model-based RL system with Dyna-style planning.

    DYNA ALGORITHM:
    1. Real step:
       - Observe state s, take action a
       - Observe next state s', reward r
       - Store (s, a, r, s') in real buffer
       - Update world model with this transition
       - Update policy with this transition

    2. Planning steps (repeated K times):
       - Sample s from real buffer
       - Choose a using current policy
       - Simulate s', r from world model
       - Update policy with simulated transition

    MODEL PREDICTIVE CONTROL (MPC):
    Instead of learning a policy, use the model to plan online:
    1. Generate N random action sequences of horizon H
    2. Simulate each sequence through the world model
    3. Evaluate cumulative reward for each sequence
    4. Execute the first action of the best sequence
    5. Re-plan at each step (receding horizon)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        model_lr: float = 1e-3,
        planning_horizon: int = 10,
        n_simulations: int = 100,
        dyna_steps: int = 5,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.n_simulations = n_simulations
        self.dyna_steps = dyna_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.world_model = FinancialWorldModel(
            state_dim, 1, hidden_dim
        ).to(self.device)

        self.model_optimizer = optim.Adam(
            self.world_model.parameters(), lr=model_lr
        )

        self.real_buffer: List[Tuple] = []
        self.training_history: List[Dict] = []

    def update_model(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> float:
        """Update world model with a real transition."""
        self.real_buffer.append((state, action, reward, next_state))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.LongTensor([action]).to(self.device)
        reward_t = torch.FloatTensor([reward]).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        total_loss = 0
        for i in range(self.world_model.ensemble_size):
            pred_next, pred_reward = self.world_model(state_t, action_t, model_idx=i)

            state_loss = nn.MSELoss()(pred_next, next_state_t)
            reward_loss = nn.MSELoss()(pred_reward, reward_t)
            loss = state_loss + reward_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.world_model.ensemble_size

    def mpc_action(self, state: np.ndarray) -> int:
        """
        Model Predictive Control: plan ahead using the world model.

        Evaluates many random action sequences and picks the best one.
        Uses ensemble uncertainty to be pessimistic about uncertain outcomes.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_t = state_t.expand(self.n_simulations, -1)

        # Generate random action sequences
        action_sequences = torch.randint(
            0, self.action_dim,
            (self.n_simulations, self.planning_horizon)
        ).to(self.device)

        total_rewards = torch.zeros(self.n_simulations).to(self.device)
        current_states = state_t.clone()

        with torch.no_grad():
            for t in range(self.planning_horizon):
                actions = action_sequences[:, t]

                mean_next, std_next, mean_reward, std_reward = \
                    self.world_model.predict_with_uncertainty(current_states, actions)

                # Pessimistic evaluation: mean - k * std (risk-averse planning)
                pessimistic_reward = mean_reward - 0.5 * std_reward
                total_rewards += pessimistic_reward * (0.99 ** t)

                current_states = mean_next

        best_idx = total_rewards.argmax()
        best_first_action = action_sequences[best_idx, 0].item()

        return best_first_action

    def generate_simulated_data(
        self, batch_size: int = 32
    ) -> List[Tuple[np.ndarray, int, float, np.ndarray]]:
        """
        Generate simulated transitions from the world model.

        Used in the Dyna planning step to augment real experience.
        """
        if len(self.real_buffer) < batch_size:
            return []

        indices = np.random.choice(len(self.real_buffer), batch_size)
        simulated_data = []

        for idx in indices:
            state, _, _, _ = self.real_buffer[idx]
            action = np.random.randint(self.action_dim)

            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_t = torch.LongTensor([action]).to(self.device)

            with torch.no_grad():
                mean_next, std_next, mean_reward, std_reward = \
                    self.world_model.predict_with_uncertainty(state_t, action_t)

                # Only trust predictions with low uncertainty
                uncertainty = std_next.mean().item()
                if uncertainty < 0.5:
                    next_state = mean_next.squeeze().cpu().numpy()
                    reward = mean_reward.item()
                    simulated_data.append((state, action, reward, next_state))

        return simulated_data


def demonstrate_model_based():
    """Demonstrate model-based RL for financial trading."""
    print("=" * 70)
    print("  CHAPTER 17: MODEL-BASED RL IN FINANCE")
    print("=" * 70)

    state_dim = 16
    action_dim = 5

    agent = ModelBasedFinancialRL(
        state_dim=state_dim,
        action_dim=action_dim,
        planning_horizon=10,
        n_simulations=50,
    )

    print(f"\nModel-Based RL Configuration:")
    print(f"  World model:     Ensemble of {agent.world_model.ensemble_size} neural networks")
    print(f"  Planning:        MPC with horizon {agent.planning_horizon}")
    print(f"  Simulations:     {agent.n_simulations} per step")
    print(f"  Dyna steps:      {agent.dyna_steps}")

    # Synthetic training with random transitions
    print("\n--- Training World Model (100 transitions) ---")
    np.random.seed(42)

    for i in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randint(action_dim)
        next_state = state + np.random.randn(state_dim) * 0.01
        reward = np.random.randn() * 0.01

        loss = agent.update_model(state, action, reward, next_state)

        if (i + 1) % 25 == 0:
            print(f"  Step {i + 1}: Model Loss = {loss:.6f}")

    # Test MPC planning
    print("\n--- MPC Planning Test ---")
    test_state = np.random.randn(state_dim)
    action = agent.mpc_action(test_state)
    print(f"  MPC selected action: {action}")

    # Test uncertainty estimation
    print("\n--- Uncertainty Estimation ---")
    state_t = torch.FloatTensor(test_state).unsqueeze(0)
    for a in range(action_dim):
        action_t = torch.LongTensor([a])
        _, std_s, mean_r, std_r = agent.world_model.predict_with_uncertainty(state_t, action_t)
        print(f"  Action {a}: Mean Reward={mean_r.item():.4f}  "
              f"Reward Std={std_r.item():.4f}  State Std={std_s.mean().item():.4f}")

    # Generate simulated data
    sim_data = agent.generate_simulated_data(batch_size=10)
    print(f"\n  Generated {len(sim_data)} trusted simulated transitions")

    print("\nKey Insights:")
    print("  1. World models multiply sample efficiency via Dyna planning")
    print("  2. Ensemble uncertainty prevents model exploitation")
    print("  3. MPC provides online planning without explicit policy")
    print("  4. Pessimistic evaluation handles model error conservatively\n")

    return agent


if __name__ == "__main__":
    demonstrate_model_based()
