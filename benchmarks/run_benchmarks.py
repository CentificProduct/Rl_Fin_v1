"""
===========================================================================
Configurable Benchmark Runner
===========================================================================

Single-command entry point that trains + evaluates every registered
strategy and prints a full comparison report.

USAGE:
  # Default: synthetic data, all strategies
  python benchmarks/run_benchmarks.py

  # Filter by category
  python benchmarks/run_benchmarks.py --categories baseline classical

  # Use real data (requires yfinance)
  python benchmarks/run_benchmarks.py --data yahoo --ticker AAPL

  # More training
  python benchmarks/run_benchmarks.py --timesteps 20000

  # Save chart
  python benchmarks/run_benchmarks.py --save

EXTENDING:
  1. Add a new strategy class in baselines.py (or any file) and decorate
     it with @BenchmarkRegistry.register("Name", category="xyz")
  2. It will appear in this runner automatically.
===========================================================================
"""

import sys, os, time, warnings, argparse
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmarks.baselines  # triggers @register decorators
from benchmarks.registry import BenchmarkRegistry, StrategyResult
from benchmarks.data_sources import DataSource
from evaluation.metrics import FinancialMetrics

N_EVAL = 5
EVAL_SEEDS = list(range(200, 200 + N_EVAL))


# ------------------------------------------------------------------
# Core engine
# ------------------------------------------------------------------

def evaluate_strategy(strategy, env, n_runs=N_EVAL) -> StrategyResult:
    """Run a strategy n_runs times and average results."""
    all_pv, all_ret, all_sharpe, all_dd, trades, costs = [], [], [], [], [], []

    for seed in EVAL_SEEDS[:n_runs]:
        strategy.reset()
        obs, info = env.reset(seed=seed)
        while True:
            action = strategy.predict(obs, info)
            obs, _, done, trunc, info = env.step(action)
            if done or trunc:
                break
        all_ret.append(info.get("total_return", 0))
        all_sharpe.append(info.get("sharpe_ratio", 0))
        all_dd.append(info.get("max_drawdown", 0))
        trades.append(info.get("trade_count", 0))
        costs.append(info.get("total_cost", 0))
        all_pv.append(info.get("portfolio_value", 0))

    return StrategyResult(
        name=strategy.name,
        category=strategy.category,
        total_return=float(np.mean(all_ret)),
        sharpe_ratio=float(np.mean(all_sharpe)),
        max_drawdown=float(np.mean(all_dd)),
        trade_count=int(np.mean(trades)),
        total_cost=float(np.mean(costs)),
    )


def run_full_benchmark(
    data_source: str = "synthetic",
    ticker: str = "SPY",
    csv_path: str = "",
    categories: list = None,
    total_timesteps: int = 10000,
    episodes: int = 15,
    n_eval: int = N_EVAL,
    save_chart: bool = False,
):
    """
    Main benchmark entry point.

    1. Load data (synthetic / yahoo / csv)
    2. Discover all registered strategies
    3. Train RL strategies
    4. Evaluate everything out-of-sample
    5. Print comparison table
    """
    print("""
+======================================================================+
|          FINANCIAL RL -- OPEN-SOURCE BENCHMARK RUNNER                 |
|          Extensible  -  Reproducible  -  Production-Ready            |
+======================================================================+
""")

    # --- Data ----------------------------------------------------------
    if data_source == "yahoo":
        train_env, test_env, meta = DataSource.from_yahoo(ticker=ticker)
    elif data_source == "csv":
        train_env, test_env, meta = DataSource.from_csv(csv_path)
    else:
        train_env, test_env, meta = DataSource.from_synthetic()

    print(f"  Data source:    {meta.get('source', 'synthetic')}")
    if meta.get("ticker"):
        print(f"  Ticker:         {meta['ticker']}")
    print(f"  Train days:     {meta.get('train_days', '?')}")
    print(f"  Test days:      {meta.get('test_days', '?')}")
    print(f"  Eval runs:      {n_eval}")
    print(f"  Train episodes: {episodes}")
    print()

    # --- Discover strategies -------------------------------------------
    all_entries = BenchmarkRegistry.list_all()
    if categories:
        all_entries = [e for e in all_entries if e["category"] in categories]

    cat_order = ["baseline", "classical", "rl", "open_source", "custom"]
    all_entries.sort(key=lambda e: (cat_order.index(e["category"])
                                    if e["category"] in cat_order else 99))

    print(f"  Strategies registered: {len(all_entries)}")
    for cat in cat_order:
        names = [e["name"] for e in all_entries if e["category"] == cat]
        if names:
            print(f"    [{cat:>12s}]  {', '.join(names)}")
    print()

    # --- Train + Evaluate ----------------------------------------------
    results = []
    config = {
        "total_timesteps": total_timesteps,
        "episodes": episodes,
        "iterations": max(episodes // 2, 5),
    }

    for i, entry in enumerate(all_entries, 1):
        name = entry["name"]
        strategy = BenchmarkRegistry.get(name)

        needs_training = entry["category"] in ("rl", "open_source")
        tag = f"[{i}/{len(all_entries)}]"

        if needs_training:
            print(f"  {tag} {name} (training + eval) ...", end="", flush=True)
            t0 = time.time()
            try:
                strategy.train(train_env, config)
            except Exception as e:
                print(f" TRAIN FAILED: {e}")
                continue
            train_time = time.time() - t0
        else:
            print(f"  {tag} {name} (eval) ...", end="", flush=True)
            t0 = time.time()
            train_time = 0

        try:
            result = evaluate_strategy(strategy, test_env, n_eval)
        except Exception as e:
            print(f" EVAL FAILED: {e}")
            continue
        result.train_time = train_time
        results.append(result)
        elapsed = time.time() - t0
        print(f" {elapsed:.1f}s")

    # --- Results -------------------------------------------------------
    print_results(results, meta)

    if save_chart:
        save_results_chart(results, meta)

    return results


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def print_results(results, meta):
    """Print a formatted benchmark table grouped by category."""
    if not results:
        print("\n  No results to display.")
        return

    print(f"""
+=====================================================================================+
|                     OUT-OF-SAMPLE BENCHMARK RESULTS                                 |
|                     Data: {meta.get('source','?'):>10s}   Test days: {meta.get('test_days','?'):>5}                          |
+======================+==========+=========+=========+=======+==========+=========+
| Strategy             |  Return  |  Sharpe | Max DD  | Trades|  Costs   | Time(s) |
+======================+==========+=========+=========+=======+==========+=========+""")

    current_cat = None
    for r in results:
        if r.category != current_cat:
            current_cat = r.category
            label = current_cat.upper()
            print(f"|--- {label:<82s}|")

        ret = f"{r.total_return*100:>+7.2f}%"
        sha = f"{r.sharpe_ratio:>7.3f}"
        mdd = f"{r.max_drawdown*100:>6.2f}%"
        trd = f"{r.trade_count:>5d}"
        cst = f"${r.total_cost:>7.0f}"
        ttm = f"{r.train_time:>6.1f}" if r.train_time > 0 else "    --"
        print(f"| {r.name:<20s} | {ret} | {sha} | {mdd} | {trd} | {cst} | {ttm} |")

    print(f"+======================+==========+=========+=========+=======+==========+=========+")
    print(f"|  Averaged over {N_EVAL} eval runs   |   Grouped by: baseline / classical / rl / open_source |")
    print(f"+=====================================================================================+")

    # Best in each category
    cats_present = sorted(set(r.category for r in results))
    print()
    for cat in cats_present:
        cat_results = [r for r in results if r.category == cat]
        best = max(cat_results, key=lambda r: r.sharpe_ratio)
        print(f"  Best {cat:>12s}: {best.name:<20s}  Sharpe={best.sharpe_ratio:.3f}  "
              f"Return={best.total_return*100:+.2f}%")

    overall_best = max(results, key=lambda r: r.sharpe_ratio)
    print(f"\n  OVERALL BEST:   {overall_best.name}  "
          f"(Sharpe={overall_best.sharpe_ratio:.3f})")


def save_results_chart(results, meta):
    """Save a comparison bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r.name for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        returns = [r.total_return * 100 for r in results]

        cat_colors = {
            "baseline": "#9E9E9E",
            "classical": "#FF9800",
            "rl": "#2196F3",
            "open_source": "#4CAF50",
            "custom": "#E91E63",
        }
        colors = [cat_colors.get(r.category, "#333") for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(names)), 8))

        ax1.bar(range(len(names)), returns, color=colors)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Total Return (%)")
        ax1.set_title("Return Comparison", fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(range(len(names)), sharpes, color=colors)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Risk-Adjusted Performance", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(0, color="red", linewidth=0.5)

        src = meta.get("source", "synthetic")
        fig.suptitle(f"Financial RL Benchmark ({src} data)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(os.path.dirname(__file__), "benchmark_chart.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Chart saved to: {path}")
    except Exception as e:
        print(f"\n  (Chart not saved: {e})")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Financial RL Benchmark Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data", choices=["synthetic", "yahoo", "csv"],
                        default="synthetic",
                        help="Data source (default: synthetic)")
    parser.add_argument("--ticker", default="SPY",
                        help="Yahoo Finance ticker (with --data yahoo)")
    parser.add_argument("--csv", default="",
                        help="CSV file path (with --data csv)")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Filter: baseline classical rl open_source")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="SB3 training timesteps")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Custom RL training episodes")
    parser.add_argument("--save", action="store_true",
                        help="Save comparison chart to PNG")
    args = parser.parse_args()

    run_full_benchmark(
        data_source=args.data,
        ticker=args.ticker,
        csv_path=args.csv,
        categories=args.categories,
        total_timesteps=args.timesteps,
        episodes=args.episodes,
        save_chart=args.save,
    )
