"""
=============================================================================
MODULE 1: Markov Decision Process (MDP) Foundations for Finance
=============================================================================

THEORY:
-------
A Markov Decision Process is the mathematical framework underlying all of RL.
An MDP is defined by the tuple (S, A, P, R, gamma) where:

  S     = State space (e.g., portfolio holdings, market indicators)
  A     = Action space (e.g., buy/sell/hold decisions)
  P     = Transition probability P(s'|s, a) - how the market evolves
  R     = Reward function R(s, a, s') - profit/loss signal
  gamma = Discount factor in [0, 1] - time preference of money

FINANCIAL INTERPRETATION:
-------------------------
In finance, the MDP framework maps naturally:
  - States encode market conditions: prices, volumes, technical indicators,
    portfolio positions, and risk metrics.
  - Actions represent trading decisions: position sizing, asset allocation
    weights, or discrete buy/sell/hold.
  - Transitions capture market dynamics: price movements, regime changes,
    and the impact of the agent's own trades (market impact).
  - Rewards reflect financial objectives: returns, risk-adjusted returns
    (Sharpe ratio), or drawdown-penalized P&L.
  - The discount factor models the time value of money and the agent's
    planning horizon.

KEY PROPERTY - MARKOV ASSUMPTION:
  P(s_{t+1} | s_t, a_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t, a_t)

  In finance, this is an approximation. Markets have long memory (momentum,
  mean reversion). We handle this by engineering states that encode sufficient
  history (e.g., moving averages, rolling volatility).
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt


class MarketRegime(Enum):
    """Discrete market regimes for simplified MDP modeling."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


class TradingAction(Enum):
    """Discrete trading actions."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class FinancialState:
    """
    Encodes the current state of the financial environment.

    The state must satisfy the Markov property: it should contain all
    information needed to make optimal decisions going forward.
    """
    price: float
    position: float  # current holdings (-1 to 1, normalized)
    cash: float
    volatility: float  # rolling volatility
    momentum: float  # price momentum indicator
    regime: MarketRegime
    rsi: float  # relative strength index (0-100)
    spread: float  # bid-ask spread
    time_step: int

    def to_vector(self) -> np.ndarray:
        """Convert state to numerical vector for function approximation."""
        regime_encoding = {
            MarketRegime.BULL: [1, 0, 0, 0],
            MarketRegime.BEAR: [0, 1, 0, 0],
            MarketRegime.SIDEWAYS: [0, 0, 1, 0],
            MarketRegime.HIGH_VOLATILITY: [0, 0, 0, 1],
        }
        return np.array([
            self.price,
            self.position,
            self.cash,
            self.volatility,
            self.momentum,
            self.rsi / 100.0,
            self.spread,
            self.time_step,
            *regime_encoding[self.regime]
        ], dtype=np.float32)


@dataclass
class Transition:
    """A single SARS' transition tuple."""
    state: FinancialState
    action: TradingAction
    reward: float
    next_state: FinancialState
    done: bool


class MarkovDecisionProcess:
    """
    Generic MDP solver using dynamic programming.

    This demonstrates the theoretical foundations before we move to
    model-free RL methods that don't require known transition dynamics.

    CONCEPT - VALUE ITERATION:
      V_{k+1}(s) = max_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) * V_k(s') ]

    CONCEPT - POLICY ITERATION:
      1. Policy Evaluation:  V^pi(s) = R(s, pi(s)) + gamma * sum P * V^pi(s')
      2. Policy Improvement: pi'(s) = argmax_a [ R(s,a) + gamma * sum P * V^pi(s') ]
    """

    def __init__(
        self,
        states: List[str],
        actions: List[str],
        transition_probs: Dict[str, Dict[str, Dict[str, float]]],
        rewards: Dict[str, Dict[str, float]],
        gamma: float = 0.99,
    ):
        self.states = states
        self.actions = actions
        self.P = transition_probs  # P[s][a][s'] = probability
        self.R = rewards  # R[s][a] = immediate reward
        self.gamma = gamma
        self.n_states = len(states)
        self.n_actions = len(actions)

    def value_iteration(self, theta: float = 1e-8, max_iters: int = 10000) -> Tuple[Dict, Dict]:
        """
        Value Iteration Algorithm.

        Iteratively applies the Bellman optimality equation until convergence.
        Returns the optimal value function and policy.

        Convergence is guaranteed because the Bellman operator is a
        contraction mapping under the sup-norm.
        """
        V = {s: 0.0 for s in self.states}
        history = []

        for iteration in range(max_iters):
            delta = 0
            V_new = {}

            for s in self.states:
                action_values = []
                for a in self.actions:
                    q_sa = self.R[s][a]
                    for s_prime in self.states:
                        q_sa += self.gamma * self.P[s][a].get(s_prime, 0) * V[s_prime]
                    action_values.append(q_sa)

                V_new[s] = max(action_values)
                delta = max(delta, abs(V_new[s] - V[s]))

            V = V_new
            history.append(delta)

            if delta < theta:
                print(f"Value iteration converged in {iteration + 1} iterations")
                break

        policy = {}
        for s in self.states:
            best_action = None
            best_value = float('-inf')
            for a in self.actions:
                q_sa = self.R[s][a]
                for s_prime in self.states:
                    q_sa += self.gamma * self.P[s][a].get(s_prime, 0) * V[s_prime]
                if q_sa > best_value:
                    best_value = q_sa
                    best_action = a
            policy[s] = best_action

        return V, policy

    def policy_iteration(self, max_iters: int = 1000) -> Tuple[Dict, Dict]:
        """
        Policy Iteration Algorithm.

        Alternates between policy evaluation (computing V^pi) and
        policy improvement (greedy w.r.t. V^pi). Converges in fewer
        iterations than value iteration but each iteration is more expensive.
        """
        policy = {s: self.actions[0] for s in self.states}
        V = {s: 0.0 for s in self.states}

        for iteration in range(max_iters):
            # Policy Evaluation
            for _ in range(1000):
                delta = 0
                for s in self.states:
                    a = policy[s]
                    v_new = self.R[s][a]
                    for s_prime in self.states:
                        v_new += self.gamma * self.P[s][a].get(s_prime, 0) * V[s_prime]
                    delta = max(delta, abs(v_new - V[s]))
                    V[s] = v_new
                if delta < 1e-8:
                    break

            # Policy Improvement
            stable = True
            for s in self.states:
                old_action = policy[s]
                best_value = float('-inf')
                for a in self.actions:
                    q_sa = self.R[s][a]
                    for s_prime in self.states:
                        q_sa += self.gamma * self.P[s][a].get(s_prime, 0) * V[s_prime]
                    if q_sa > best_value:
                        best_value = q_sa
                        policy[s] = a

                if policy[s] != old_action:
                    stable = False

            if stable:
                print(f"Policy iteration converged in {iteration + 1} iterations")
                break

        return V, policy


class FinancialMDP(MarkovDecisionProcess):
    """
    A concrete financial MDP modeling a simplified market with regimes.

    States: Market regimes (Bull, Bear, Sideways)
    Actions: Buy, Hold, Sell
    Transitions: Regime switching probabilities (calibrated to market data)
    Rewards: Expected returns conditional on regime and action

    This demonstrates how abstract MDP concepts map to financial reality.
    """

    @staticmethod
    def create_market_regime_mdp(gamma: float = 0.95) -> 'FinancialMDP':
        """
        Factory method creating a market regime MDP with realistic parameters.

        Transition probabilities are calibrated to typical regime persistence:
        - Bull markets tend to persist (70% self-transition)
        - Bear markets are shorter-lived (50% self-transition)
        - Sideways markets are intermediate (60% self-transition)
        """
        states = ["bull", "bear", "sideways"]
        actions = ["buy", "hold", "sell"]

        P = {
            "bull": {
                "buy":  {"bull": 0.70, "bear": 0.10, "sideways": 0.20},
                "hold": {"bull": 0.65, "bear": 0.15, "sideways": 0.20},
                "sell": {"bull": 0.60, "bear": 0.20, "sideways": 0.20},
            },
            "bear": {
                "buy":  {"bull": 0.20, "bear": 0.50, "sideways": 0.30},
                "hold": {"bull": 0.15, "bear": 0.55, "sideways": 0.30},
                "sell": {"bull": 0.10, "bear": 0.60, "sideways": 0.30},
            },
            "sideways": {
                "buy":  {"bull": 0.30, "bear": 0.15, "sideways": 0.55},
                "hold": {"bull": 0.25, "bear": 0.15, "sideways": 0.60},
                "sell": {"bull": 0.20, "bear": 0.20, "sideways": 0.60},
            },
        }

        R = {
            "bull":     {"buy": 0.08, "hold": 0.04, "sell": -0.02},
            "bear":     {"buy": -0.06, "hold": -0.02, "sell": 0.03},
            "sideways": {"buy": 0.01, "hold": 0.005, "sell": -0.005},
        }

        return FinancialMDP(states, actions, P, R, gamma)

    def analyze_optimal_strategy(self) -> Dict[str, Any]:
        """Run both DP methods and compare results."""
        print("=" * 60)
        print("FINANCIAL MDP ANALYSIS: Market Regime Trading")
        print("=" * 60)

        print("\n--- Value Iteration ---")
        V_vi, pi_vi = self.value_iteration()

        print("\n--- Policy Iteration ---")
        V_pi, pi_pi = self.policy_iteration()

        print("\n--- Optimal Policy ---")
        for s in self.states:
            print(f"  State: {s:>10s}  |  Action: {pi_vi[s]:>6s}  |  Value: {V_vi[s]:.4f}")

        print(f"\n--- Discount Factor: {self.gamma} ---")
        print("Interpretation: The value represents the expected discounted")
        print("cumulative return starting from each market regime.\n")

        return {"value_function": V_vi, "policy": pi_vi}


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_mdp_foundations():
    """
    Run a complete demonstration of MDP foundations in finance.

    This shows how dynamic programming can solve small financial MDPs
    when the transition model is known. In practice, we rarely know P
    exactly, motivating model-free RL methods covered in later modules.
    """
    print("=" * 70)
    print("  CHAPTER 1: MDP FOUNDATIONS FOR FINANCIAL AI")
    print("=" * 70)

    mdp = FinancialMDP.create_market_regime_mdp(gamma=0.95)
    results = mdp.analyze_optimal_strategy()

    # Sensitivity analysis on discount factor
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: Effect of Discount Factor")
    print("=" * 60)

    gammas = [0.5, 0.8, 0.9, 0.95, 0.99]
    all_policies = {}

    for g in gammas:
        mdp_g = FinancialMDP.create_market_regime_mdp(gamma=g)
        V, policy = mdp_g.value_iteration()
        all_policies[g] = policy
        print(f"\ngamma={g:.2f}: ", end="")
        for s in mdp_g.states:
            print(f"{s}={policy[s]}", end="  ")

    print("\n\nKey Insight: Lower discount factors make the agent more myopic,")
    print("favoring immediate rewards. Higher gamma values encourage the agent")
    print("to consider long-term regime dynamics.\n")

    return results


if __name__ == "__main__":
    demonstrate_mdp_foundations()
