"""
=============================================================================
MODULE 3: Bellman Equations - The Recursive Heart of RL
=============================================================================

THEORY:
-------
The Bellman equations express a recursive relationship: the value of a state
equals the immediate reward plus the discounted value of successor states.

BELLMAN EXPECTATION EQUATION (for a fixed policy pi):
  V^pi(s) = sum_a pi(a|s) * [ R(s,a) + gamma * sum_{s'} P(s'|s,a) V^pi(s') ]

BELLMAN OPTIMALITY EQUATION (for the optimal policy):
  V*(s) = max_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) V*(s') ]
  Q*(s,a) = R(s,a) + gamma * sum_{s'} P(s'|s,a) * max_{a'} Q*(s',a')

FINANCIAL INTERPRETATION:
  The Bellman equation for portfolio optimization says:
  "The value of a portfolio state equals the best immediate trading
   reward plus the discounted expected value of where that trade takes us."

  This is deeply connected to the Hamilton-Jacobi-Bellman (HJB) equation
  in continuous-time finance, which governs optimal control of portfolios
  under stochastic dynamics (Merton's problem).
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class BellmanSolver:
    """
    Solves Bellman equations for financial decision problems.

    Provides implementations of both the expectation and optimality
    equations, with visualization of the convergence process.
    """

    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma

    def solve_bellman_expectation(
        self,
        n_states: int,
        transition_matrix: np.ndarray,
        reward_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Solve V^pi = R^pi + gamma * P^pi * V^pi analytically.

        Rearranging: V^pi = (I - gamma * P^pi)^{-1} * R^pi

        This is the matrix form of the Bellman expectation equation.
        For small state spaces, we can solve it exactly with linear algebra.

        FINANCIAL EXAMPLE:
        If states represent discrete market conditions and R^pi is the
        expected return under a fixed trading strategy, this gives us the
        exact expected cumulative return from each starting condition.
        """
        I = np.eye(n_states)
        V = np.linalg.solve(I - self.gamma * transition_matrix, reward_vector)
        return V

    def iterative_bellman_optimality(
        self,
        n_states: int,
        n_actions: int,
        transitions: np.ndarray,  # shape: (n_states, n_actions, n_states)
        rewards: np.ndarray,  # shape: (n_states, n_actions)
        theta: float = 1e-10,
        max_iters: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Solve Bellman optimality equation iteratively.

        V_{k+1}(s) = max_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) * V_k(s') ]

        Returns optimal V*, optimal policy, and convergence history.
        """
        V = np.zeros(n_states)
        convergence_history = []

        for k in range(max_iters):
            V_new = np.zeros(n_states)

            for s in range(n_states):
                q_values = np.zeros(n_actions)
                for a in range(n_actions):
                    q_values[a] = rewards[s, a] + self.gamma * np.dot(transitions[s, a], V)
                V_new[s] = np.max(q_values)

            delta = np.max(np.abs(V_new - V))
            convergence_history.append(delta)
            V = V_new

            if delta < theta:
                break

        # Extract optimal policy
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = rewards[s, a] + self.gamma * np.dot(transitions[s, a], V)
            policy[s] = np.argmax(q_values)

        return V, policy, convergence_history

    def q_value_iteration(
        self,
        n_states: int,
        n_actions: int,
        transitions: np.ndarray,
        rewards: np.ndarray,
        theta: float = 1e-10,
        max_iters: int = 10000,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Q-value iteration: directly compute optimal Q*.

        Q_{k+1}(s,a) = R(s,a) + gamma * sum_{s'} P(s'|s,a) * max_{a'} Q_k(s',a')

        Q-values are more useful in practice because they directly tell us
        the value of each action, enabling action selection without
        knowing the transition model.
        """
        Q = np.zeros((n_states, n_actions))
        convergence = []

        for k in range(max_iters):
            Q_new = np.zeros_like(Q)

            for s in range(n_states):
                for a in range(n_actions):
                    Q_new[s, a] = rewards[s, a] + self.gamma * np.dot(
                        transitions[s, a], np.max(Q, axis=1)
                    )

            delta = np.max(np.abs(Q_new - Q))
            convergence.append(delta)
            Q = Q_new

            if delta < theta:
                break

        return Q, convergence


def demonstrate_bellman_equations():
    """
    Demonstrate Bellman equations with a concrete financial example.

    SCENARIO: An investor faces 3 market states and 3 actions.
    We solve for the optimal strategy using both analytical and
    iterative methods, then compare.
    """
    print("=" * 70)
    print("  CHAPTER 3: BELLMAN EQUATIONS IN FINANCIAL DECISION-MAKING")
    print("=" * 70)

    n_states = 3  # [expansion, recession, recovery]
    n_actions = 3  # [aggressive, moderate, conservative]
    state_names = ["Expansion", "Recession", "Recovery"]
    action_names = ["Aggressive", "Moderate", "Conservative"]

    # Transition probabilities P[s, a, s']
    P = np.zeros((n_states, n_actions, n_states))

    # Expansion state transitions
    P[0, 0] = [0.6, 0.2, 0.2]  # aggressive in expansion
    P[0, 1] = [0.5, 0.2, 0.3]  # moderate in expansion
    P[0, 2] = [0.4, 0.3, 0.3]  # conservative in expansion

    # Recession state transitions
    P[1, 0] = [0.1, 0.5, 0.4]  # aggressive in recession
    P[1, 1] = [0.2, 0.4, 0.4]  # moderate in recession
    P[1, 2] = [0.2, 0.3, 0.5]  # conservative in recession

    # Recovery state transitions
    P[2, 0] = [0.4, 0.1, 0.5]  # aggressive in recovery
    P[2, 1] = [0.3, 0.2, 0.5]  # moderate in recovery
    P[2, 2] = [0.3, 0.2, 0.5]  # conservative in recovery

    # Reward matrix R[s, a] (expected single-period returns, in %)
    R = np.array([
        [12.0, 7.0, 3.0],   # expansion: aggressive earns most
        [-8.0, -2.0, 1.0],  # recession: conservative protects
        [6.0, 4.0, 2.5],    # recovery: aggressive captures upside
    ]) / 100.0

    solver = BellmanSolver(gamma=0.95)

    # Method 1: Iterative Bellman Optimality
    print("\n--- Iterative Bellman Optimality Solution ---")
    V_star, opt_policy, history = solver.iterative_bellman_optimality(
        n_states, n_actions, P, R
    )

    print(f"  Converged in {len(history)} iterations")
    print(f"\n  Optimal Value Function:")
    for i, (name, v) in enumerate(zip(state_names, V_star)):
        action = action_names[opt_policy[i]]
        print(f"    {name:>12s}: V* = {v:8.4f}  |  Optimal Action = {action}")

    # Method 2: Q-Value Iteration
    print("\n--- Q-Value Analysis ---")
    Q_star, q_history = solver.q_value_iteration(n_states, n_actions, P, R)

    print(f"  Converged in {len(q_history)} iterations")
    print(f"\n  Q*(s, a) Table:")
    header = "              " + "  ".join(f"{a:>14s}" for a in action_names)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, name in enumerate(state_names):
        values = "  ".join(f"{Q_star[i, j]:14.6f}" for j in range(n_actions))
        best = " <-- BEST" if True else ""
        print(f"    {name:>10s}: {values}")

    # Method 3: Analytical solution for a fixed policy
    print("\n--- Analytical Solution (Bellman Expectation, Moderate Policy) ---")
    moderate_P = P[:, 1, :]  # transition matrix under moderate policy
    moderate_R = R[:, 1]  # rewards under moderate policy
    V_moderate = solver.solve_bellman_expectation(n_states, moderate_P, moderate_R)

    for name, v in zip(state_names, V_moderate):
        print(f"    {name:>12s}: V^moderate = {v:.6f}")

    # Compare: optimal vs moderate policy
    print("\n--- Value of Optimal Policy vs. Moderate-Only Policy ---")
    for name, v_opt, v_mod in zip(state_names, V_star, V_moderate):
        improvement = v_opt - v_mod
        pct = (improvement / abs(v_mod)) * 100 if v_mod != 0 else 0
        print(f"    {name:>12s}: Improvement = {improvement:.4f} ({pct:.1f}%)")

    print("\nKey Insight: The Bellman optimality equation reveals that the best")
    print("strategy is state-dependent. Being aggressive in expansion/recovery")
    print("and conservative in recession captures significantly more value than")
    print("any single fixed strategy.\n")

    return V_star, Q_star, opt_policy


if __name__ == "__main__":
    demonstrate_bellman_equations()
