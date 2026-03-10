"""
=============================================================================
MODULE 13: Mean-Variance Optimization Enhanced with RL
=============================================================================

THEORY:
-------
This module bridges classical portfolio theory with RL.

CLASSICAL MEAN-VARIANCE (Markowitz):
  max_w  w'mu - (lambda/2) * w'Sigma*w
  s.t.   1'w = 1

  Closed form: w* = (1/lambda) * Sigma^{-1} * (mu - r_f * 1) + w_min_var

PROBLEMS:
  1. Estimation of mu is extremely noisy (Merton, 1980)
  2. Sigma inversion amplifies estimation errors
  3. Optimal weights are unstable over time
  4. Ignores higher moments (skewness, kurtosis)

RL ENHANCEMENT:
  Use RL to learn a mapping from market state to "corrected" mean-variance
  weights. The agent observes the classical optimal weights and market
  features, then outputs adjustments:

    w_RL = f_theta(w_MV, market_state)

  This lets the agent learn when to trust the model and when to deviate.
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize


class MeanVarianceRL:
    """
    Combines classical mean-variance optimization with RL corrections.

    APPROACH:
    1. Compute classical MV-optimal weights using rolling estimates
    2. Use RL to learn state-dependent corrections
    3. The agent's policy adjusts weights based on market regime

    This "augmented" approach:
    - Starts from a reasonable baseline (MV)
    - Uses RL for fine-tuning rather than learning from scratch
    - Converges faster and is more stable
    """

    def __init__(
        self,
        n_assets: int,
        risk_free_rate: float = 0.02,
        estimation_window: int = 60,
        shrinkage_intensity: float = 0.5,
    ):
        self.n_assets = n_assets
        self.rf = risk_free_rate / 252  # daily
        self.window = estimation_window
        self.shrinkage = shrinkage_intensity

    def estimate_parameters(
        self, returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate mu and Sigma with Ledoit-Wolf shrinkage.

        Shrinkage reduces estimation error by pulling the sample covariance
        toward a structured target (identity matrix scaled by average variance).

        Sigma_shrunk = (1 - delta) * Sigma_sample + delta * target
        """
        mu = returns.mean(axis=0)
        sigma_sample = np.cov(returns.T)

        # Ledoit-Wolf shrinkage target: scaled identity
        avg_var = np.trace(sigma_sample) / self.n_assets
        target = avg_var * np.eye(self.n_assets)

        sigma_shrunk = (1 - self.shrinkage) * sigma_sample + self.shrinkage * target

        return mu, sigma_shrunk

    def classical_mv_weights(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: float = 1.0,
        long_only: bool = True,
    ) -> np.ndarray:
        """
        Compute classical mean-variance optimal weights.

        Solves: max_w  w'mu - (lambda/2) * w'Sigma*w
                s.t.   sum(w) = 1, w >= 0 (if long_only)
        """
        n = len(mu)

        def neg_utility(w):
            port_return = w @ mu
            port_risk = w @ sigma @ w
            return -(port_return - 0.5 * risk_aversion * port_risk)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n if long_only else [(-1, 1)] * n

        result = minimize(
            neg_utility,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else np.ones(n) / n

    def efficient_frontier(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Compute the efficient frontier.

        The efficient frontier is the set of portfolios that offer the
        maximum expected return for each level of risk.
        """
        target_returns = np.linspace(mu.min(), mu.max(), n_points)
        risks = []
        returns_ef = []
        weights_list = []

        for target in target_returns:
            n = len(mu)

            def port_variance(w):
                return w @ sigma @ w

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: w @ mu - target},
            ]
            bounds = [(0, 1)] * n

            result = minimize(
                port_variance,
                x0=np.ones(n) / n,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                risk = np.sqrt(result.fun * 252)
                ret = target * 252
                risks.append(risk)
                returns_ef.append(ret)
                weights_list.append(result.x)

        return np.array(risks), np.array(returns_ef), weights_list

    def black_litterman_views(
        self,
        sigma: np.ndarray,
        market_cap_weights: np.ndarray,
        views: np.ndarray,
        view_confidence: np.ndarray,
        tau: float = 0.05,
    ) -> np.ndarray:
        """
        Black-Litterman model for incorporating views into allocation.

        The BL model combines market equilibrium returns (from CAPM) with
        investor views to produce more stable expected return estimates.

        pi = lambda * Sigma * w_mkt  (equilibrium returns)
        E[r] = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1}
               * [(tau*Sigma)^{-1}*pi + P'*Omega^{-1}*q]

        where P = view matrix, q = view returns, Omega = view uncertainty
        """
        risk_aversion = 2.5
        pi = risk_aversion * sigma @ market_cap_weights

        n_views = len(views)
        P = np.eye(self.n_assets)[:n_views]
        q = views
        omega = np.diag(view_confidence ** 2)

        tau_sigma_inv = np.linalg.inv(tau * sigma)
        posterior_precision = tau_sigma_inv + P.T @ np.linalg.inv(omega) @ P
        posterior_cov = np.linalg.inv(posterior_precision)

        posterior_mean = posterior_cov @ (tau_sigma_inv @ pi + P.T @ np.linalg.inv(omega) @ q)

        return posterior_mean

    def rolling_backtest(
        self,
        returns: np.ndarray,
        rebalance_freq: int = 21,
        risk_aversion: float = 1.0,
    ) -> Dict:
        """
        Rolling window backtest of mean-variance optimization.

        This serves as the baseline to compare against RL approaches.
        """
        n_days = len(returns)
        portfolio_values = [1.0]
        weights_history = []
        turnover_history = []
        current_weights = np.ones(self.n_assets) / self.n_assets

        for t in range(self.window, n_days):
            if (t - self.window) % rebalance_freq == 0:
                # Re-estimate and reoptimize
                window_returns = returns[t - self.window:t]
                mu, sigma = self.estimate_parameters(window_returns)
                new_weights = self.classical_mv_weights(mu, sigma, risk_aversion)

                turnover = np.sum(np.abs(new_weights - current_weights))
                turnover_history.append(turnover)
                current_weights = new_weights

            # Apply returns
            port_return = current_weights @ returns[t]
            portfolio_values.append(portfolio_values[-1] * (1 + port_return))
            weights_history.append(current_weights.copy())

            # Account for drift
            current_weights = current_weights * (1 + returns[t])
            current_weights /= current_weights.sum()

        # Compute metrics
        pv = np.array(portfolio_values)
        daily_returns = np.diff(pv) / pv[:-1]
        total_return = pv[-1] / pv[0] - 1
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        annual_vol = daily_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        max_dd = np.max(1 - pv / np.maximum.accumulate(pv))

        return {
            "portfolio_values": pv,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "avg_turnover": np.mean(turnover_history) if turnover_history else 0,
            "weights_history": weights_history,
        }


def demonstrate_mean_variance():
    """Demonstrate classical MV optimization as baseline for RL comparison."""
    print("=" * 70)
    print("  CHAPTER 13: MEAN-VARIANCE WITH RL ENHANCEMENT")
    print("=" * 70)

    from utils.data_loader import FinancialDataLoader
    data = FinancialDataLoader.generate_correlated_assets(n_assets=5, n_days=2000)

    mv = MeanVarianceRL(n_assets=5)

    # Classical MV backtest
    print("\n--- Classical Mean-Variance Backtest ---")
    results = mv.rolling_backtest(data.returns, rebalance_freq=21)

    print(f"  Total Return:      {results['total_return'] * 100:.2f}%")
    print(f"  Annual Return:     {results['annual_return'] * 100:.2f}%")
    print(f"  Annual Volatility: {results['annual_vol'] * 100:.2f}%")
    print(f"  Sharpe Ratio:      {results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:      {results['max_drawdown'] * 100:.2f}%")
    print(f"  Avg Turnover:      {results['avg_turnover']:.4f}")

    # Equal weight comparison
    equal_returns = data.returns.mean(axis=1)
    equal_pv = np.cumprod(1 + equal_returns[60:])
    equal_total = equal_pv[-1] - 1
    print(f"\n  Equal Weight Return: {equal_total * 100:.2f}%")

    # Efficient frontier
    mu, sigma = mv.estimate_parameters(data.returns[-252:])
    risks, rets, _ = mv.efficient_frontier(mu, sigma)
    print(f"\n  Efficient Frontier: {len(risks)} portfolios computed")
    print(f"  Risk range: [{risks.min() * 100:.1f}%, {risks.max() * 100:.1f}%]")
    print(f"  Return range: [{rets.min() * 100:.1f}%, {rets.max() * 100:.1f}%]")

    print("\nKey Insight: Classical MV provides a reasonable baseline but")
    print("suffers from estimation error and cannot adapt to regimes.")
    print("RL can learn to correct these deficiencies dynamically.\n")

    return results


if __name__ == "__main__":
    demonstrate_mean_variance()
