"""
=============================================================================
MODULE 14: RL-Based Risk Management
=============================================================================

THEORY:
-------
Risk management is about controlling the distribution of returns, not just
maximizing the mean. RL provides a powerful framework for learning
risk-aware trading policies.

RISK MEASURES IN FINANCE:
  1. Value at Risk (VaR): P(loss > VaR) = alpha
     "The maximum loss at the alpha confidence level"

  2. Conditional VaR (CVaR / Expected Shortfall):
     CVaR = E[loss | loss > VaR]
     "Average loss in the worst alpha% of scenarios"

  3. Maximum Drawdown: max_t (peak_t - value_t) / peak_t
     "Largest peak-to-trough decline"

  4. Volatility: std(returns) * sqrt(252)
     "Annualized standard deviation of returns"

RISK-SENSITIVE RL:
  Standard RL maximizes E[sum gamma^t r_t].
  Risk-sensitive RL modifies this objective:

  1. MEAN-VARIANCE: max E[G] - lambda * Var[G]
  2. CVaR-CONSTRAINED: max E[G] s.t. CVaR_alpha(G) >= threshold
  3. ENTROPIC RISK: max -(1/beta) * log E[exp(-beta * G)]

  These can be implemented via:
  - Reward shaping (penalize risk in the reward function)
  - Constrained MDPs (Lagrangian relaxation of risk constraints)
  - Distributional RL (learn the full return distribution)
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class RiskMetrics:
    """Computation of standard financial risk metrics."""

    @staticmethod
    def var(returns: np.ndarray, alpha: float = 0.05) -> float:
        """Value at Risk at alpha confidence level."""
        return -np.percentile(returns, alpha * 100)

    @staticmethod
    def cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
        """Conditional Value at Risk (Expected Shortfall)."""
        var = RiskMetrics.var(returns, alpha)
        return -np.mean(returns[returns <= -var])

    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> float:
        """Maximum peak-to-trough drawdown."""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)

    @staticmethod
    def sortino_ratio(returns: np.ndarray, target: float = 0.0) -> float:
        """Sortino ratio: return / downside volatility."""
        excess = returns - target / 252
        downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * np.sqrt(252)
        annual_return = np.mean(returns) * 252
        return annual_return / (downside + 1e-8)

    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calmar ratio: annual return / max drawdown."""
        pv = np.cumprod(1 + returns)
        annual_return = (pv[-1]) ** (252 / len(returns)) - 1
        mdd = RiskMetrics.max_drawdown(pv)
        return annual_return / (mdd + 1e-8)

    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Omega ratio: probability-weighted gains / losses."""
        excess = returns - threshold / 252
        gains = np.sum(np.maximum(excess, 0))
        losses = np.sum(np.maximum(-excess, 0))
        return gains / (losses + 1e-8)


class RLRiskManager:
    """
    RL agent that manages portfolio risk dynamically.

    The risk manager observes the current portfolio state and market
    conditions, then decides on risk limits (position sizes, stop-losses,
    hedge ratios) to control downside risk while preserving upside.

    ARCHITECTURE:
      State: [portfolio_metrics, market_features, current_risk_budget]
      Action: [position_scale_factor, stop_loss_level, hedge_ratio]
      Reward: return - penalty * risk_violation

    CONSTRAINED MDP APPROACH:
      We use Lagrangian relaxation to handle risk constraints:
        L(theta, lambda) = E[G] + lambda * (CVaR_limit - CVaR(G))

      The Lagrange multiplier lambda is learned alongside the policy,
      automatically balancing return maximization with risk control.
    """

    def __init__(
        self,
        state_dim: int,
        risk_budget: float = 0.02,  # max 2% daily VaR
        max_drawdown_limit: float = 0.10,  # max 10% drawdown
        learning_rate: float = 1e-3,
    ):
        self.risk_budget = risk_budget
        self.max_dd_limit = max_drawdown_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Risk policy: maps state to risk parameters
        self.risk_policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [scale, stop_loss, hedge]
            nn.Sigmoid(),  # outputs in [0, 1]
        ).to(self.device)

        # Lagrange multipliers for constraints (learned)
        self.log_lambda_var = nn.Parameter(torch.tensor(0.0).to(self.device))
        self.log_lambda_dd = nn.Parameter(torch.tensor(0.0).to(self.device))

        all_params = list(self.risk_policy.parameters()) + [self.log_lambda_var, self.log_lambda_dd]
        self.optimizer = torch.optim.Adam(all_params, lr=learning_rate)

        self.risk_history: List[Dict] = []

    def get_risk_parameters(self, state: np.ndarray) -> Dict:
        """Compute risk management parameters from current state."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            params = self.risk_policy(state_t).squeeze().cpu().numpy()

        return {
            "position_scale": params[0],  # 0-1: fraction of max position allowed
            "stop_loss": 0.01 + params[1] * 0.09,  # 1%-10% stop loss
            "hedge_ratio": params[2],  # 0-1: fraction of portfolio to hedge
        }

    def compute_risk_adjusted_reward(
        self,
        base_reward: float,
        returns_history: np.ndarray,
        portfolio_values: np.ndarray,
    ) -> float:
        """
        Compute risk-adjusted reward with constraint penalties.

        reward = base_reward
                 - lambda_var * max(0, current_VaR - VaR_budget)
                 - lambda_dd * max(0, current_DD - DD_limit)
        """
        lambda_var = torch.exp(self.log_lambda_var).item()
        lambda_dd = torch.exp(self.log_lambda_dd).item()

        if len(returns_history) > 20:
            current_var = RiskMetrics.var(returns_history[-60:])
            var_violation = max(0, current_var - self.risk_budget)
        else:
            var_violation = 0

        if len(portfolio_values) > 1:
            current_dd = RiskMetrics.max_drawdown(portfolio_values)
            dd_violation = max(0, current_dd - self.max_dd_limit)
        else:
            dd_violation = 0

        adjusted_reward = base_reward - lambda_var * var_violation - lambda_dd * dd_violation

        self.risk_history.append({
            "base_reward": base_reward,
            "adjusted_reward": adjusted_reward,
            "var_violation": var_violation,
            "dd_violation": dd_violation,
            "lambda_var": lambda_var,
            "lambda_dd": lambda_dd,
        })

        return adjusted_reward

    def update_lagrange_multipliers(self, var_violations: List[float], dd_violations: List[float]):
        """
        Update Lagrange multipliers using dual gradient ascent.

        The dual update increases lambda when constraints are violated
        and decreases it when they are satisfied, automatically finding
        the right balance between return and risk.
        """
        avg_var_viol = np.mean(var_violations) if var_violations else 0
        avg_dd_viol = np.mean(dd_violations) if dd_violations else 0

        var_loss = -self.log_lambda_var.exp() * (avg_var_viol - 0)  # dual ascent
        dd_loss = -self.log_lambda_dd.exp() * (avg_dd_viol - 0)

        dual_loss = var_loss + dd_loss

        self.optimizer.zero_grad()
        dual_loss.backward()
        self.optimizer.step()


class DistributionalRiskEstimator:
    """
    Distributional RL for risk estimation.

    CONCEPT:
    Instead of learning E[G], learn the full distribution of G.
    This enables direct computation of VaR, CVaR, and other risk measures.

    QUANTILE REGRESSION DQN (QR-DQN):
    Represents the return distribution using N quantiles.
    Each quantile tau_i estimates Q_{tau_i}(s, a) such that:
      P(G <= Q_{tau_i}) = tau_i

    The quantile loss is:
      L(theta) = sum_i rho_{tau_i}(G - Q_{tau_i})
      where rho_tau(u) = u * (tau - I(u < 0))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 51,
        hidden_dim: int = 128,
    ):
        self.n_quantiles = n_quantiles
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Quantile values: evenly spaced between 0 and 1
        self.taus = torch.linspace(0, 1, n_quantiles + 2)[1:-1].to(self.device)

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * n_quantiles),
        ).to(self.device)

        self.action_dim = action_dim
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)

    def predict_quantiles(self, state: np.ndarray) -> np.ndarray:
        """Predict return distribution quantiles for each action."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            quantiles = self.network(state_t).view(self.action_dim, self.n_quantiles)
        return quantiles.cpu().numpy()

    def compute_risk_metrics(self, state: np.ndarray, action: int) -> Dict:
        """Compute risk metrics from the predicted distribution."""
        quantiles = self.predict_quantiles(state)[action]

        var_95 = -quantiles[int(0.05 * self.n_quantiles)]
        var_99 = -quantiles[int(0.01 * self.n_quantiles)]
        cvar_95 = -np.mean(quantiles[:int(0.05 * self.n_quantiles)])
        expected = np.mean(quantiles)
        std = np.std(quantiles)

        return {
            "expected_return": expected,
            "std": std,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
        }

    def quantile_huber_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantile Huber loss for distributional RL.

        Combines quantile regression with Huber loss for robustness.
        """
        diff = targets.unsqueeze(-1) - predictions.unsqueeze(-2)
        huber = torch.where(diff.abs() <= 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
        quantile_weight = (self.taus - (diff < 0).float()).abs()
        loss = (quantile_weight * huber).mean()
        return loss


def demonstrate_risk_management():
    """Demonstrate RL-based risk management."""
    print("=" * 70)
    print("  CHAPTER 14: RL-BASED RISK MANAGEMENT")
    print("=" * 70)

    np.random.seed(42)

    # Generate returns with fat tails (Student-t)
    n_days = 500
    returns = np.random.standard_t(df=4, size=n_days) * 0.01

    # Compute risk metrics
    print("\n--- Risk Metrics on Synthetic Returns ---")
    print(f"  VaR (95%):        {RiskMetrics.var(returns, 0.05) * 100:.2f}%")
    print(f"  CVaR (95%):       {RiskMetrics.cvar(returns, 0.05) * 100:.2f}%")
    print(f"  VaR (99%):        {RiskMetrics.var(returns, 0.01) * 100:.2f}%")
    print(f"  Sortino Ratio:    {RiskMetrics.sortino_ratio(returns):.3f}")

    portfolio_values = np.cumprod(1 + returns)
    print(f"  Max Drawdown:     {RiskMetrics.max_drawdown(portfolio_values) * 100:.2f}%")
    print(f"  Calmar Ratio:     {RiskMetrics.calmar_ratio(returns):.3f}")
    print(f"  Omega Ratio:      {RiskMetrics.omega_ratio(returns):.3f}")

    # Distributional risk estimation
    print("\n--- Distributional Risk Estimation (QR-DQN) ---")
    estimator = DistributionalRiskEstimator(state_dim=10, action_dim=5, n_quantiles=51)
    dummy_state = np.random.randn(10)

    for action in range(3):
        risk = estimator.compute_risk_metrics(dummy_state, action)
        print(f"  Action {action}: E[r]={risk['expected_return']:.4f}  "
              f"VaR95={risk['var_95']:.4f}  CVaR95={risk['cvar_95']:.4f}")

    print("\nKey Insights:")
    print("  1. CVaR is a coherent risk measure (VaR is not)")
    print("  2. Distributional RL directly models the return distribution")
    print("  3. Lagrangian relaxation automatically balances risk constraints")
    print("  4. Fat-tailed returns make risk management critical in practice\n")

    return estimator


if __name__ == "__main__":
    demonstrate_risk_management()
