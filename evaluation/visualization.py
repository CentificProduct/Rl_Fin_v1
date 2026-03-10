"""
=============================================================================
MODULE 21: Performance Visualization
=============================================================================

Visualization tools for analyzing RL trading strategy performance.
Uses matplotlib for static plots suitable for research reports.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple


class PerformanceVisualizer:
    """Comprehensive visualization for RL trading strategies."""

    @staticmethod
    def plot_equity_curve(
        portfolio_values: np.ndarray,
        benchmark_values: Optional[np.ndarray] = None,
        title: str = "Portfolio Equity Curve",
        save_path: Optional[str] = None,
    ):
        """Plot portfolio value over time with optional benchmark."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

        # Equity curve
        ax = axes[0]
        ax.plot(portfolio_values, label='RL Strategy', color='#2196F3', linewidth=1.5)
        if benchmark_values is not None:
            scale = portfolio_values[0] / benchmark_values[0]
            ax.plot(benchmark_values * scale, label='Benchmark', color='#9E9E9E', linewidth=1, alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Drawdown
        ax = axes[1]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        ax.fill_between(range(len(drawdown)), -drawdown, 0, color='#F44336', alpha=0.3)
        ax.plot(-drawdown, color='#F44336', linewidth=0.8)
        ax.set_ylabel('Drawdown')
        ax.set_ylim([-max(drawdown) * 1.1, 0.01])
        ax.grid(True, alpha=0.3)

        # Daily returns
        ax = axes[2]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        colors = ['#4CAF50' if r > 0 else '#F44336' for r in returns]
        ax.bar(range(len(returns)), returns, color=colors, alpha=0.5, width=1.0)
        ax.set_ylabel('Daily Returns')
        ax.set_xlabel('Trading Days')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return fig

    @staticmethod
    def plot_training_progress(
        training_history: List[Dict],
        metrics: List[str] = ['total_reward', 'sharpe_ratio', 'total_return'],
        title: str = "RL Training Progress",
        save_path: Optional[str] = None,
    ):
        """Plot training metrics over episodes."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = [h.get(metric, 0) for h in training_history]
            episodes = range(len(values))

            ax.plot(episodes, values, alpha=0.3, color='#2196F3')

            # Moving average
            window = min(20, len(values) // 3)
            if window > 1:
                ma = np.convolve(values, np.ones(window) / window, mode='valid')
                ax.plot(range(window - 1, len(values)), ma, color='#F44336', linewidth=2)

            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)

        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Episode')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return fig

    @staticmethod
    def plot_action_distribution(
        actions: np.ndarray,
        action_names: Optional[List[str]] = None,
        title: str = "Action Distribution",
        save_path: Optional[str] = None,
    ):
        """Plot distribution of actions taken by the agent."""
        if action_names is None:
            action_names = [f"Action {i}" for i in range(int(actions.max()) + 1)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart
        unique, counts = np.unique(actions, return_counts=True)
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(unique)))
        axes[0].bar([action_names[int(u)] for u in unique], counts / len(actions),
                    color=colors)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Overall Distribution')
        axes[0].tick_params(axis='x', rotation=45)

        # Time series of actions
        axes[1].plot(actions, '.', markersize=1, alpha=0.3)
        axes[1].set_ylabel('Action')
        axes[1].set_xlabel('Time Step')
        axes[1].set_title('Actions Over Time')
        axes[1].set_yticks(range(len(action_names)))
        axes[1].set_yticklabels(action_names)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return fig

    @staticmethod
    def plot_risk_analysis(
        returns: np.ndarray,
        title: str = "Risk Analysis",
        save_path: Optional[str] = None,
    ):
        """Comprehensive risk analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Return distribution
        ax = axes[0, 0]
        ax.hist(returns, bins=50, density=True, alpha=0.7, color='#2196F3')
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, mu, sigma), 'r--', label='Normal fit')
        ax.axvline(np.percentile(returns, 5), color='orange', linestyle='--', label='VaR 95%')
        ax.set_title('Return Distribution')
        ax.legend()

        # Q-Q plot
        ax = axes[0, 1]
        sorted_returns = np.sort(returns)
        theoretical = norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
        ax.scatter(theoretical, sorted_returns, s=1, alpha=0.5)
        ax.plot([theoretical.min(), theoretical.max()], [theoretical.min() * sigma + mu, theoretical.max() * sigma + mu], 'r--')
        ax.set_title('Q-Q Plot (vs Normal)')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')

        # Rolling volatility
        ax = axes[1, 0]
        windows = [10, 30, 60]
        for w in windows:
            if len(returns) > w:
                rolling_vol = np.array([returns[max(0, i - w):i].std() * np.sqrt(252) for i in range(w, len(returns))])
                ax.plot(rolling_vol, label=f'{w}-day', alpha=0.8)
        ax.set_title('Rolling Volatility (Annualized)')
        ax.legend()

        # Rolling Sharpe
        ax = axes[1, 1]
        window = 60
        if len(returns) > window:
            rolling_sharpe = np.array([
                returns[max(0, i - window):i].mean() / (returns[max(0, i - window):i].std() + 1e-8) * np.sqrt(252)
                for i in range(window, len(returns))
            ])
            ax.plot(rolling_sharpe, color='#4CAF50')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.fill_between(range(len(rolling_sharpe)), rolling_sharpe, 0,
                          where=rolling_sharpe > 0, alpha=0.2, color='green')
            ax.fill_between(range(len(rolling_sharpe)), rolling_sharpe, 0,
                          where=rolling_sharpe < 0, alpha=0.2, color='red')
        ax.set_title(f'Rolling Sharpe Ratio ({window}-day)')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return fig

    @staticmethod
    def plot_portfolio_weights(
        weights_history: np.ndarray,
        asset_names: Optional[List[str]] = None,
        title: str = "Portfolio Weight Allocation",
        save_path: Optional[str] = None,
    ):
        """Plot portfolio weight evolution over time."""
        if isinstance(weights_history, list):
            weights_history = np.array(weights_history)

        n_assets = weights_history.shape[1] if weights_history.ndim > 1 else 1

        if asset_names is None:
            asset_names = [f"Asset {i}" for i in range(n_assets)]

        fig, ax = plt.subplots(figsize=(14, 6))

        colors = plt.cm.Set3(np.linspace(0, 1, n_assets))
        ax.stackplot(range(len(weights_history)), weights_history.T,
                    labels=asset_names, colors=colors, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Portfolio Weight')
        ax.set_xlabel('Time Step')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return fig

    @staticmethod
    def plot_algorithm_comparison(
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['sharpe_ratio', 'total_return', 'max_drawdown'],
        title: str = "Algorithm Comparison",
        save_path: Optional[str] = None,
    ):
        """Compare multiple algorithms across metrics."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        algorithms = list(results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

        for ax, metric in zip(axes, metrics):
            values = [results[algo].get(metric, 0) for algo in algorithms]
            bars = ax.bar(range(len(algorithms)), values, color=colors)
            ax.set_xticks(range(len(algorithms)))
            ax.set_xticklabels(algorithms, rotation=45, ha='right')
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return fig
