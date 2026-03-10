"""
===========================================================================
PORTFOLIO DEMO: RL vs Classical Portfolio Optimization
===========================================================================
Runtime: ~2 minutes  |  No external data needed  |  Generates charts

Compares:
  1. Equal-weight baseline (1/N)
  2. Classical Mean-Variance (Markowitz)
  3. Minimum Variance portfolio
  4. Risk Parity
  5. RL-optimized portfolio (Dirichlet policy via PPO)

Usage:
    python demos/demo_portfolio.py
    python demos/demo_portfolio.py --save   # save chart
===========================================================================
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import FinancialDataLoader, FeatureEngineering
from portfolio.mean_variance import MeanVarianceRL
from portfolio.optimization import RLPortfolioOptimizer
from evaluation.metrics import FinancialMetrics


ASSETS = ["US Equity", "Int'l Equity", "Bonds", "REITs", "Commodities"]
N_ASSETS = len(ASSETS)


def generate_data():
    """Produce synthetic correlated asset data (train + test split)."""
    data = FinancialDataLoader.generate_correlated_assets(
        n_assets=N_ASSETS, n_days=2500, correlation=0.4, seed=42,
    )
    split = 1800
    return data, split


def backtest_weights_strategy(returns, weights_fn, rebalance_freq=21):
    """Backtest a strategy that returns weights each rebalance day."""
    n_days = len(returns)
    pv = np.ones(n_days + 1)
    weights = np.ones(N_ASSETS) / N_ASSETS

    for t in range(n_days):
        if t % rebalance_freq == 0:
            weights = weights_fn(returns, t)

        port_ret = weights @ returns[t]
        pv[t + 1] = pv[t] * (1 + port_ret)

        weights = weights * (1 + returns[t])
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum

    return pv[1:]


# ----------------------------------------------------------------------
# Classical Strategies
# ----------------------------------------------------------------------
def equal_weight_fn(returns, t):
    return np.ones(N_ASSETS) / N_ASSETS


def mean_variance_fn(returns, t, window=120, risk_aversion=1.0):
    if t < window:
        return np.ones(N_ASSETS) / N_ASSETS
    mv = MeanVarianceRL(N_ASSETS)
    mu, sigma = mv.estimate_parameters(returns[t-window:t])
    return mv.classical_mv_weights(mu, sigma, risk_aversion)


def min_variance_fn(returns, t, window=120):
    if t < window:
        return np.ones(N_ASSETS) / N_ASSETS
    mv = MeanVarianceRL(N_ASSETS)
    _, sigma = mv.estimate_parameters(returns[t-window:t])
    return mv.classical_mv_weights(np.zeros(N_ASSETS), sigma, risk_aversion=100)


def risk_parity_fn(returns, t, window=120):
    """Risk Parity: allocate inversely proportional to volatility."""
    if t < window:
        return np.ones(N_ASSETS) / N_ASSETS
    vols = returns[t-window:t].std(axis=0)
    inv_vol = 1.0 / (vols + 1e-8)
    return inv_vol / inv_vol.sum()


# ----------------------------------------------------------------------
# RL Strategy
# ----------------------------------------------------------------------
def train_rl_optimizer(train_returns, train_features, n_episodes=50):
    """Train RL portfolio optimizer and return weight-selection function."""
    n_feature_per_asset = train_features.shape[1] // N_ASSETS if train_features.ndim > 1 else 12
    state_dim = n_feature_per_asset * N_ASSETS + N_ASSETS

    optimizer = RLPortfolioOptimizer(
        n_assets=N_ASSETS, state_dim=state_dim, hidden_dim=128,
        lr=3e-4, entropy_coef=0.02, cost_penalty=0.002,
    )

    print(f"  Training RL optimizer ({n_episodes} episodes) ...", end="", flush=True)
    t0 = time.time()
    optimizer.train_on_data(train_returns, train_features,
                            n_episodes=n_episodes, episode_length=200)
    print(f" {time.time()-t0:.1f}s")

    # Return a weight function that uses the trained policy
    def rl_weight_fn(returns_full, t, _opt=optimizer, _feat=train_features):
        # Use most recent features available
        idx = min(t, len(_feat) - 1)
        feat = _feat[idx] if idx >= 0 else np.zeros(_feat.shape[1])
        w_prev = np.ones(N_ASSETS) / N_ASSETS
        state = np.concatenate([feat, w_prev])
        return _opt.select_weights(state, training=False)

    return rl_weight_fn


# ----------------------------------------------------------------------
# Main demo
# ----------------------------------------------------------------------
def run_portfolio_demo(save_chart=False):
    print("""
+======================================================================+
|          PORTFOLIO OPTIMIZATION BENCHMARK                            |
|          RL vs Classical Strategies - 5 Correlated Assets            |
+======================================================================+
""")

    data, split = generate_data()

    train_returns  = data.returns[:split]
    test_returns   = data.returns[split:]
    train_features = data.features[:split]
    test_features  = data.features[split:]

    print(f"  Assets:        {', '.join(ASSETS)}")
    print(f"  Train period:  {split} days")
    print(f"  Test period:   {len(test_returns)} days")
    print(f"  Rebalance:     Monthly (21 days)")

    # Train RL
    rl_fn = train_rl_optimizer(train_returns, train_features, n_episodes=40)

    # Backtest all strategies on TEST data
    strategies = {
        "Equal Weight (1/N)":  equal_weight_fn,
        "Mean-Variance":       lambda r, t: mean_variance_fn(r, t),
        "Min Variance":        lambda r, t: min_variance_fn(r, t),
        "Risk Parity":         lambda r, t: risk_parity_fn(r, t),
        "RL Optimizer":        rl_fn,
    }

    results = {}
    pv_series = {}

    for name, fn in strategies.items():
        pv = backtest_weights_strategy(test_returns, fn, rebalance_freq=21)
        pv_series[name] = pv
        metrics = FinancialMetrics.compute_all(pv * 100_000)
        results[name] = metrics

    # Print table
    print(f"""
+==============================================================================+
|                 OUT-OF-SAMPLE PORTFOLIO BENCHMARK                              |
+====================+===========+==========+==========+==========+==============+
| Strategy           |  Return   |  Sharpe  |  Sortino |  Max DD  | Ann. Vol     |
+====================+===========+==========+==========+==========+==============+""")

    for name, m in results.items():
        ret = f"{m['total_return']*100:>+8.2f}%"
        sharpe = f"{m['sharpe_ratio']:>8.3f}"
        sortino = f"{m['sortino_ratio']:>8.3f}"
        mdd = f"{m['max_drawdown']*100:>7.2f}%"
        vol = f"{m['annual_volatility']*100:>11.2f}%"
        print(f"| {name:<18s} | {ret} | {sharpe} | {sortino} | {mdd} | {vol} |")

    print(f"+====================+===========+==========+==========+==========+==============+")

    best = max(results, key=lambda k: results[k]["sharpe_ratio"])
    print(f"\n  BEST BY SHARPE: {best}  ({results[best]['sharpe_ratio']:.3f})")

    if save_chart:
        _save_portfolio_chart(pv_series, results)

    return results


def _save_portfolio_chart(pv_series, results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                 gridspec_kw={'height_ratios': [3, 1]})

        colors = {'Equal Weight (1/N)': '#9E9E9E', 'Mean-Variance': '#FF9800',
                  'Min Variance': '#4CAF50', 'Risk Parity': '#2196F3',
                  'RL Optimizer': '#F44336'}

        ax = axes[0]
        for name, pv in pv_series.items():
            c = colors.get(name, '#333')
            lw = 2.5 if name == 'RL Optimizer' else 1.2
            ax.plot(pv, label=f"{name} (Sharpe={results[name]['sharpe_ratio']:.2f})",
                   color=c, linewidth=lw)
        ax.set_title("Portfolio Equity Curves (Out-of-Sample)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Growth of $1")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        names = list(results.keys())
        sharpes = [results[n]["sharpe_ratio"] for n in names]
        bar_colors = [colors.get(n, '#333') for n in names]
        ax.bar(range(len(names)), sharpes, color=bar_colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Risk-Adjusted Performance Comparison", fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(os.path.dirname(__file__), "portfolio_benchmark.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Chart saved to: {path}")
    except Exception as e:
        print(f"  (Chart not saved: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    run_portfolio_demo(args.save)
