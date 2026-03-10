"""
===========================================================================
FULL BENCHMARK SUITE: All Algorithms Head-to-Head
===========================================================================
Runtime: ~5-8 minutes  |  No external data needed

Trains every algorithm on the same synthetic market data, evaluates on
held-out test data, and produces a comprehensive comparison report.

Usage:
    python demos/benchmark_suite.py
    python demos/benchmark_suite.py --episodes 200  # more training
    python demos/benchmark_suite.py --save           # save chart to file
===========================================================================
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import FinancialDataLoader, FeatureEngineering
from environments.stock_trading_env import StockTradingEnv
from agents.q_learning import QLearningTrader, DoubleQLearningTrader
from agents.dqn_agent import DQNAgent
from agents.policy_gradient import REINFORCEAgent
from agents.ppo_agent import PPOTrader
from agents.a2c_agent import A2CTrader
from evaluation.metrics import FinancialMetrics


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEFAULT_TRAIN_EPISODES = 100
N_EVAL_RUNS = 5
EVAL_SEEDS = list(range(100, 100 + N_EVAL_RUNS))


def build_train_and_test_envs():
    """Create separate train & test environments from synthetic data."""
    data = FinancialDataLoader.generate_synthetic_data(len_data=3000, seed=42)

    split = 2000
    train_prices = data.prices[:split]
    test_prices  = data.prices[split:]

    train_features = FeatureEngineering.compute_all_features(train_prices)
    test_features  = FeatureEngineering.compute_all_features(test_prices)

    train_env = StockTradingEnv(
        prices=train_prices, features=train_features,
        reward_type="sharpe", discrete_actions=True,
    )
    test_env = StockTradingEnv(
        prices=test_prices, features=test_features,
        reward_type="sharpe", discrete_actions=True,
    )
    return train_env, test_env


def evaluate_agent(agent, env, n_runs, is_q=False):
    """Run agent n_runs times on the env and return averaged metrics."""
    all_infos = []
    for seed in EVAL_SEEDS[:n_runs]:
        obs, _ = env.reset(seed=seed)
        if is_q:
            state = _q_discretize(agent, obs)
        while True:
            if is_q:
                action = agent.select_action(state, training=False)
            else:
                action = agent.select_action(obs, training=False)
            obs, _, done, trunc, info = env.step(action)
            if is_q:
                state = _q_discretize(agent, obs)
            if done or trunc:
                break
        all_infos.append(info)

    return {
        k: np.mean([i.get(k, 0) for i in all_infos])
        for k in ["portfolio_value", "total_return", "sharpe_ratio",
                   "max_drawdown", "trade_count", "total_cost"]
    }


def evaluate_fixed_strategy(env, action_value, n_runs):
    """Evaluate a fixed-action strategy (buy-hold, always-sell, etc.)."""
    all_infos = []
    for seed in EVAL_SEEDS[:n_runs]:
        obs, _ = env.reset(seed=seed)
        while True:
            obs, _, done, trunc, info = env.step(action_value)
            if done or trunc:
                break
        all_infos.append(info)
    return {
        k: np.mean([i.get(k, 0) for i in all_infos])
        for k in ["portfolio_value", "total_return", "sharpe_ratio",
                   "max_drawdown", "trade_count", "total_cost"]
    }


def _q_discretize(agent, obs):
    feats = np.array([
        obs[9] if len(obs) > 9 else 0,
        obs[5] if len(obs) > 5 else 0.15,
        obs[7] if len(obs) > 7 else 0.5,
        obs[-4] if len(obs) > 3 else 0,
    ])
    return agent.discretize_state(feats)


# ---------------------------------------------------------------------
# Training wrappers
# ---------------------------------------------------------------------
def train_q_learning(env, episodes):
    agent = QLearningTrader(alpha=0.1, gamma=0.99, epsilon_decay=0.99)
    for _ in range(episodes):
        agent.train_episode(env)
    return agent

def train_double_q(env, episodes):
    agent = DoubleQLearningTrader(alpha=0.1, gamma=0.99, epsilon_decay=0.99)
    for _ in range(episodes):
        agent.train_episode(env)
    return agent

def train_dqn(env, episodes, state_dim):
    agent = DQNAgent(
        state_dim=state_dim, action_dim=5,
        double_dqn=True, dueling=True,
        learning_rate=3e-4, epsilon_decay_steps=episodes * 500,
        buffer_size=50000, batch_size=64,
    )
    for _ in range(episodes):
        agent.train_episode(env)
    return agent

def train_reinforce(env, episodes, state_dim):
    agent = REINFORCEAgent(
        state_dim=state_dim, action_dim=5,
        gamma=0.99, entropy_coef=0.01, continuous=False,
    )
    for _ in range(episodes):
        agent.train_episode(env)
    return agent

def train_ppo(env, iters, state_dim):
    agent = PPOTrader(
        state_dim=state_dim, action_dim=5, hidden_dim=128,
        continuous=False, rollout_length=512, n_epochs=5,
    )
    for _ in range(iters):
        rollout = agent.collect_rollout(env)
        agent.update(rollout)
    return agent

def train_a2c(env, steps, state_dim):
    agent = A2CTrader(
        state_dim=state_dim, action_dim=5, n_steps=5, continuous=False,
    )
    obs, _ = env.reset()
    done_steps = 0
    while done_steps < steps:
        obs, _, metrics = agent.train_n_steps(env, obs)
        done_steps += 5
    return agent


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------
def run_benchmark(n_episodes=DEFAULT_TRAIN_EPISODES, save_chart=False):
    print("""
+======================================================================+
|            FINANCIAL RL BENCHMARK SUITE                             |
|            Training on synthetic regime-switching market data       |
+======================================================================+
""")

    train_env, test_env = build_train_and_test_envs()
    obs, _ = train_env.reset()
    state_dim = len(obs)

    print(f"  Config:")
    print(f"    Training data:    2,000 days (synthetic GBM with regime switching)")
    print(f"    Test data:        1,000 days (out-of-sample)")
    print(f"    State dimension:  {state_dim}")
    print(f"    Action space:     5 discrete (strong-sell -> strong-buy)")
    print(f"    Eval runs/algo:   {N_EVAL_RUNS}")
    print(f"    Train episodes:   {n_episodes}")

    results = {}

    # -- Baselines ----------------------------------------------------
    print(f"\n{'-'*70}")
    print("  BASELINES")
    print(f"{'-'*70}")

    print("  [1/8] Random agent ...", end="", flush=True)
    results["Random"] = evaluate_fixed_strategy(
        test_env, action_value=None, n_runs=N_EVAL_RUNS
    )
    # Random needs special handling:
    random_infos = []
    for seed in EVAL_SEEDS:
        obs, _ = test_env.reset(seed=seed)
        while True:
            obs, _, d, t, info = test_env.step(test_env.action_space.sample())
            if d or t:
                break
        random_infos.append(info)
    results["Random"] = {k: np.mean([i.get(k,0) for i in random_infos])
                         for k in ["portfolio_value","total_return","sharpe_ratio",
                                   "max_drawdown","trade_count","total_cost"]}
    print(" done")

    print("  [2/8] Buy & Hold ...", end="", flush=True)
    results["Buy & Hold"] = evaluate_fixed_strategy(test_env, 4, N_EVAL_RUNS)
    print(" done")

    # -- RL Algorithms -------------------------------------------------
    print(f"\n{'-'*70}")
    print("  RL ALGORITHMS (training ...)")
    print(f"{'-'*70}")

    algos = [
        ("Q-Learning",       lambda: train_q_learning(train_env, n_episodes), True),
        ("Double Q",         lambda: train_double_q(train_env, n_episodes), True),
        ("DQN (Double+Duel)",lambda: train_dqn(train_env, n_episodes, state_dim), False),
        ("REINFORCE",        lambda: train_reinforce(train_env, n_episodes, state_dim), False),
        ("PPO",              lambda: train_ppo(train_env, max(n_episodes//3, 20), state_dim), False),
        ("A2C",              lambda: train_a2c(train_env, n_episodes*200, state_dim), False),
    ]

    for idx, (name, train_fn, is_q) in enumerate(algos, start=3):
        print(f"  [{idx}/8] {name} ...", end="", flush=True)
        t0 = time.time()
        agent = train_fn()
        train_time = time.time() - t0

        if is_q:
            res = evaluate_agent(agent, test_env, N_EVAL_RUNS, is_q=True)
        elif hasattr(agent, 'select_action'):
            res = evaluate_agent(agent, test_env, N_EVAL_RUNS, is_q=False)
        else:
            # PPO uses network directly
            import torch
            ppo_infos = []
            for seed in EVAL_SEEDS:
                obs, _ = test_env.reset(seed=seed)
                while True:
                    st = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        a, _, _, _ = agent.network.get_action_and_value(st)
                    obs, _, d, t, info = test_env.step(a.item())
                    if d or t:
                        break
                ppo_infos.append(info)
            res = {k: np.mean([i.get(k,0) for i in ppo_infos])
                   for k in ["portfolio_value","total_return","sharpe_ratio",
                              "max_drawdown","trade_count","total_cost"]}

        res["train_time"] = train_time
        results[name] = res
        print(f" {train_time:.1f}s")

    # -- Results Table -------------------------------------------------
    _print_results_table(results)

    # -- Best Algorithm ------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["sharpe_ratio"])
    best = results[best_name]
    print(f"  BEST BY SHARPE: {best_name}")
    print(f"    Return: {best['total_return']*100:+.2f}%  "
          f"Sharpe: {best['sharpe_ratio']:.3f}  "
          f"MaxDD: {best['max_drawdown']*100:.2f}%")

    if save_chart:
        _save_comparison_chart(results)

    return results


def _print_results_table(results):
    print(f"""
+==========================================================================================+
|                         OUT-OF-SAMPLE BENCHMARK RESULTS                                  |
+===================+===========+==========+==========+========+==========+==========+
| Algorithm         |  Return   |  Sharpe  |  Max DD  | Trades |   Costs  | Time (s) |
+===================+===========+==========+==========+========+==========+==========+""")

    for name, m in results.items():
        ret = f"{m['total_return']*100:>+8.2f}%"
        sharpe = f"{m['sharpe_ratio']:>8.3f}"
        mdd = f"{m['max_drawdown']*100:>7.2f}%"
        trades = f"{m['trade_count']:>6.0f}"
        costs = f"${m['total_cost']:>7.0f}"
        tt = f"{m.get('train_time', 0):>7.1f}" if m.get('train_time') else "     --"
        print(f"| {name:<18s} | {ret} | {sharpe} | {mdd} | {trades} | {costs} | {tt} |")

    print(f"+===================+===========+==========+==========+========+==========+==========+")
    print(f"|  Evaluated on {N_EVAL_RUNS} out-of-sample runs  -  Synthetic regime-switching data            |")
    print(f"+==========================================================================================+")


def _save_comparison_chart(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        names = list(results.keys())
        sharpes = [results[n]["sharpe_ratio"] for n in names]
        returns = [results[n]["total_return"] * 100 for n in names]
        drawdowns = [results[n]["max_drawdown"] * 100 for n in names]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        colors = ['#9E9E9E', '#9E9E9E'] + ['#2196F3'] * (len(names) - 2)

        for ax, values, title, fmt in [
            (axes[0], returns, "Total Return (%)", "{:.1f}%"),
            (axes[1], sharpes, "Sharpe Ratio", "{:.3f}"),
            (axes[2], drawdowns, "Max Drawdown (%)", "{:.1f}%"),
        ]:
            bars = ax.bar(range(len(names)), values, color=colors)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       fmt.format(val), ha='center', va='bottom', fontsize=8)

        fig.suptitle("Financial RL Algorithm Benchmark (Out-of-Sample)", fontsize=15, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(os.path.dirname(__file__), "benchmark_results.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Chart saved to: {path}")
    except Exception as e:
        print(f"\n  (Chart not saved: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--save", action="store_true", help="Save comparison chart")
    args = parser.parse_args()
    run_benchmark(args.episodes, args.save)
