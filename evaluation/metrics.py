"""
=============================================================================
MODULE 20: Financial Performance Metrics
=============================================================================

Comprehensive collection of metrics for evaluating RL trading strategies.
Covers return metrics, risk metrics, and RL-specific metrics.
=============================================================================
"""

import numpy as np
from typing import Dict, List, Optional


class FinancialMetrics:
    """Complete suite of financial performance metrics."""

    @staticmethod
    def compute_all(
        portfolio_values: np.ndarray,
        benchmark_values: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, float]:
        """Compute all metrics from a portfolio value series."""
        pv = portfolio_values
        returns = np.diff(pv) / pv[:-1]
        rf_daily = risk_free_rate / 252

        n_days = len(returns)
        total_return = pv[-1] / pv[0] - 1
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        annual_vol = returns.std() * np.sqrt(252)

        # Sharpe Ratio
        excess_returns = returns - rf_daily
        sharpe = excess_returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)

        # Sortino Ratio
        downside = returns[returns < rf_daily]
        downside_vol = np.sqrt(np.mean((downside - rf_daily) ** 2)) * np.sqrt(252) if len(downside) > 0 else 1e-8
        sortino = (annual_return - risk_free_rate) / (downside_vol + 1e-8)

        # Max Drawdown & Duration
        peak = np.maximum.accumulate(pv)
        drawdown = (peak - pv) / peak
        max_dd = np.max(drawdown)

        dd_duration = 0
        max_dd_duration = 0
        for d in drawdown:
            if d > 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # Calmar Ratio
        calmar = annual_return / (max_dd + 1e-8)

        # Omega Ratio
        threshold = rf_daily
        gains = np.sum(np.maximum(returns - threshold, 0))
        losses = np.sum(np.maximum(threshold - returns, 0))
        omega = gains / (losses + 1e-8)

        # Tail Ratio (right tail / left tail)
        p95 = np.percentile(returns, 95)
        p05 = np.abs(np.percentile(returns, 5))
        tail_ratio = p95 / (p05 + 1e-8)

        # Skewness and Kurtosis
        skewness = float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-8)) ** 3))
        kurtosis = float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-8)) ** 4)) - 3

        # VaR and CVaR
        var_95 = -np.percentile(returns, 5)
        var_99 = -np.percentile(returns, 1)
        cvar_95 = -np.mean(returns[returns <= -var_95]) if np.any(returns <= -var_95) else var_95

        # Win/Loss statistics
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        win_rate = len(winning_days) / max(len(returns), 1)
        avg_win = winning_days.mean() if len(winning_days) > 0 else 0
        avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
        profit_factor = abs(winning_days.sum() / (losing_days.sum() + 1e-8))

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_dd_duration_days': max_dd_duration,
            'calmar_ratio': calmar,
            'omega_ratio': omega,
            'tail_ratio': tail_ratio,
            'skewness': skewness,
            'excess_kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'n_days': n_days,
        }

        # Benchmark-relative metrics
        if benchmark_values is not None:
            bm_returns = np.diff(benchmark_values) / benchmark_values[:-1]
            min_len = min(len(returns), len(bm_returns))
            ret_slice = returns[:min_len]
            bm_slice = bm_returns[:min_len]

            # Alpha and Beta (CAPM)
            cov_matrix = np.cov(ret_slice, bm_slice)
            beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-8)
            alpha = (ret_slice.mean() - rf_daily - beta * (bm_slice.mean() - rf_daily)) * 252

            # Tracking error
            tracking_error = (ret_slice - bm_slice).std() * np.sqrt(252)

            # Information ratio
            info_ratio = (ret_slice.mean() - bm_slice.mean()) / ((ret_slice - bm_slice).std() + 1e-8) * np.sqrt(252)

            metrics.update({
                'alpha': alpha,
                'beta': beta,
                'tracking_error': tracking_error,
                'information_ratio': info_ratio,
            })

        return metrics

    @staticmethod
    def print_report(metrics: Dict[str, float], title: str = "Performance Report"):
        """Print a formatted performance report."""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

        sections = {
            'Return Metrics': ['total_return', 'annual_return', 'annual_volatility'],
            'Risk-Adjusted': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio'],
            'Risk Metrics': ['max_drawdown', 'max_dd_duration_days', 'var_95', 'cvar_95'],
            'Distribution': ['skewness', 'excess_kurtosis', 'tail_ratio'],
            'Trade Stats': ['win_rate', 'profit_factor', 'avg_win', 'avg_loss'],
        }

        if 'alpha' in metrics:
            sections['Benchmark-Relative'] = ['alpha', 'beta', 'tracking_error', 'information_ratio']

        for section, keys in sections.items():
            print(f"\n  --- {section} ---")
            for key in keys:
                if key in metrics:
                    val = metrics[key]
                    if 'return' in key or 'volatility' in key or 'drawdown' in key or key in ['var_95', 'cvar_95', 'tracking_error', 'alpha']:
                        print(f"    {key:>25s}: {val * 100:>10.2f}%")
                    elif key == 'max_dd_duration_days' or key == 'n_days':
                        print(f"    {key:>25s}: {val:>10.0f}")
                    else:
                        print(f"    {key:>25s}: {val:>10.4f}")

        print(f"\n{'=' * 60}\n")
