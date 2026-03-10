"""
===========================================================================
QUICK DEMO: Financial AI with Reinforcement Learning
===========================================================================
Runtime: ~2 minutes  |  No external data needed  |  Team presentation ready

Usage:
    python demos/quick_demo.py

This demo runs 5 headline results:
  1. Trains a DQN trader and shows it beating a random baseline
  2. Compares Q-Learning vs Double Q-Learning
  3. Shows PPO learning curve
  4. Runs a multi-agent market simulation
  5. Prints a summary benchmark table

Everything uses synthetic data -- no API keys or downloads required.
===========================================================================
"""

import sys
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.stock_trading_env import StockTradingEnv
from agents.dqn_agent import DQNAgent
from agents.q_learning import QLearningTrader, DoubleQLearningTrader
from agents.ppo_agent import PPOTrader
from advanced.multi_agent import (
    MultiAgentMarketSimulator, MomentumAgent, MeanReversionAgent, MarketMaker,
)


HEADER = """
+======================================================================+
|          FINANCIAL AI WITH REINFORCEMENT LEARNING                    |
|          Quick Demo  -  Synthetic Data  -  No Setup                  |
+======================================================================+
"""

SECTION = lambda title: (
    f"\n{'-' * 70}\n  {title}\n{'-' * 70}"
)


def _run_random_baseline(env, seed=42):
    """Run random agent as baseline."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if done or trunc:
            break
    return info


def _run_trained_agent(agent, env, seed=42):
    """Evaluate a trained agent (no exploration)."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    while True:
        action = agent.select_action(obs, training=False)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if done or trunc:
            break
    return info


# ----------------------------------------------------------------------
#  DEMO 1 : DQN vs Random Baseline
# ----------------------------------------------------------------------
def demo_dqn_vs_random():
    print(SECTION("DEMO 1 > DQN Agent vs Random Baseline"))

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)
    obs, _ = env.reset()
    state_dim = len(obs)

    # Random baseline
    random_info = _run_random_baseline(env)

    agent = DQNAgent(
        state_dim=state_dim, action_dim=5,
        hidden_dims=[64, 32],
        double_dqn=True, dueling=True,
        learning_rate=5e-4, epsilon_decay_steps=5000,
        buffer_size=20000, batch_size=32,
    )

    print("  Training DQN (15 episodes) ...", end="", flush=True)
    t0 = time.time()
    for ep in range(15):
        agent.train_episode(env)
    elapsed = time.time() - t0
    print(f" done in {elapsed:.1f}s")

    # Evaluate
    dqn_info = _run_trained_agent(agent, env)

    _print_comparison(
        labels=["Random Baseline", "DQN Agent"],
        infos=[random_info, dqn_info],
    )
    return agent


# ----------------------------------------------------------------------
#  DEMO 2 : Q-Learning vs Double Q-Learning
# ----------------------------------------------------------------------
def demo_q_learning_comparison():
    print(SECTION("DEMO 2 > Q-Learning vs Double Q-Learning"))

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)

    q_agent = QLearningTrader(alpha=0.1, gamma=0.99, epsilon_decay=0.97)
    dq_agent = DoubleQLearningTrader(alpha=0.1, gamma=0.99, epsilon_decay=0.97)

    print("  Training Q-Learning (50 eps) ...", end="", flush=True)
    t0 = time.time()
    for _ in range(50):
        q_agent.train_episode(env)
    print(f" {time.time()-t0:.1f}s")

    print("  Training Double Q-Learning (50 eps) ...", end="", flush=True)
    t0 = time.time()
    for _ in range(50):
        dq_agent.train_episode(env)
    print(f" {time.time()-t0:.1f}s")

    q_last = q_agent.training_history[-10:]
    dq_last = dq_agent.training_history[-10:]

    print(f"\n  {'Metric':<22s} {'Q-Learning':>14s} {'Double Q':>14s}")
    print(f"  {'-'*50}")
    print(f"  {'Avg Return (last 10)':<22s} "
          f"{np.mean([m['total_return'] for m in q_last])*100:>13.2f}% "
          f"{np.mean([m['total_return'] for m in dq_last])*100:>13.2f}%")
    print(f"  {'Avg Sharpe (last 10)':<22s} "
          f"{np.mean([m['sharpe_ratio'] for m in q_last]):>14.3f} "
          f"{np.mean([m['sharpe_ratio'] for m in dq_last]):>14.3f}")
    print(f"  {'States Discovered':<22s} "
          f"{len(q_agent.Q):>14d} "
          f"{len(dq_agent.Q):>14d}")


# ----------------------------------------------------------------------
#  DEMO 3 : PPO Learning Curve
# ----------------------------------------------------------------------
def demo_ppo_learning():
    print(SECTION("DEMO 3 > PPO Learning Curve"))

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)
    obs, _ = env.reset()
    state_dim = len(obs)

    agent = PPOTrader(
        state_dim=state_dim, action_dim=5, hidden_dim=64,
        continuous=False, rollout_length=256, n_epochs=3,
    )

    print("  Training PPO (15 iterations, 256 steps each) ...\n")
    print(f"  {'Iter':>6s} {'AvgReward':>12s} {'PolicyLoss':>12s} {'Entropy':>10s}")
    print(f"  {'-'*44}")

    for it in range(15):
        rollout = agent.collect_rollout(env)
        update_info = agent.update(rollout)
        avg_r = rollout["rewards"].mean()
        agent.training_history.append({**update_info, "avg_reward": avg_r})

        if (it + 1) % 5 == 0:
            print(f"  {it+1:>6d} {avg_r:>12.4f} "
                  f"{update_info['policy_loss']:>12.4f} {update_info['entropy']:>10.4f}")

    early = agent.training_history[:5]
    late  = agent.training_history[-5:]
    print(f"\n  Reward improvement: "
          f"{np.mean([m['avg_reward'] for m in early]):.4f} -> "
          f"{np.mean([m['avg_reward'] for m in late]):.4f}")


# ----------------------------------------------------------------------
#  DEMO 4 : Multi-Agent Market Simulation
# ----------------------------------------------------------------------
def demo_multi_agent():
    print(SECTION("DEMO 4 > Multi-Agent Market Simulation"))

    sim = MultiAgentMarketSimulator(fundamental_price=100.0, volatility=0.02)
    sim.add_agent(MomentumAgent(0, lookback=20))
    sim.add_agent(MomentumAgent(1, lookback=50))
    sim.add_agent(MeanReversionAgent(2, lookback=30))
    sim.add_agent(MeanReversionAgent(3, lookback=60))
    sim.add_agent(MarketMaker(4, spread_bps=10))
    sim.add_agent(MarketMaker(5, spread_bps=15))

    print("  Running 1,000-step simulation with 6 agents ...", end="", flush=True)
    t0 = time.time()
    results = sim.run_simulation(n_steps=1000)
    print(f" {time.time()-t0:.1f}s")

    prices = results["price_history"]
    vol = np.std(np.diff(prices)/np.array(prices[:-1])) * np.sqrt(252) * 100

    print(f"\n  Market Stats:")
    print(f"    Start -> End price: ${prices[0]:.2f} -> ${prices[-1]:.2f}")
    print(f"    Annualised Vol:     {vol:.1f}%")
    print(f"    Price Range:        ${min(prices):.2f} - ${max(prices):.2f}")

    agent_labels = [
        "Momentum-20", "Momentum-50", "MeanRev-30",
        "MeanRev-60", "MarketMaker-10", "MarketMaker-15",
    ]
    print(f"\n  {'Agent':<18s} {'P&L ($)':>10s} {'Inventory':>10s} {'Trades':>8s}")
    print(f"  {'-'*48}")
    for agent, label in zip(sim.agents, agent_labels):
        total = agent.state.cash + agent.state.inventory * prices[-1]
        pnl = total - 100_000
        print(f"  {label:<18s} {pnl:>+10.0f} {agent.state.inventory:>10.0f} "
              f"{agent.state.order_count:>8d}")


# ----------------------------------------------------------------------
#  DEMO 5 : Summary Benchmark Table
# ----------------------------------------------------------------------
def demo_summary_table(dqn_agent):
    print(SECTION("DEMO 5 > Algorithm Benchmark Summary"))

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)

    algorithms = {}

    # Random
    infos = [_run_random_baseline(env, seed=s) for s in range(5)]
    algorithms["Random"] = _avg_info(infos)

    # Buy & Hold (always action=4, strong buy)
    bh_infos = []
    for s in range(5):
        obs, _ = env.reset(seed=s)
        while True:
            obs, _, d, t, info = env.step(4)
            if d or t:
                break
        bh_infos.append(info)
    algorithms["Buy & Hold"] = _avg_info(bh_infos)

    # DQN (already trained)
    dqn_infos = [_run_trained_agent(dqn_agent, env, seed=s) for s in range(5)]
    algorithms["DQN (Double+Duel)"] = _avg_info(dqn_infos)

    # Q-Learning (quick train)
    q_agent = QLearningTrader(alpha=0.1, gamma=0.99, epsilon_decay=0.96)
    for _ in range(30):
        q_agent.train_episode(env)
    q_infos = []
    for s in range(5):
        obs, _ = env.reset(seed=s)
        relevant = _extract_q_features(obs)
        state = q_agent.discretize_state(relevant)
        while True:
            action = q_agent.select_action(state, training=False)
            obs, _, d, t, info = env.step(action)
            if d or t:
                break
            relevant = _extract_q_features(obs)
            state = q_agent.discretize_state(relevant)
        q_infos.append(info)
    algorithms["Q-Learning"] = _avg_info(q_infos)

    # PPO (quick train)
    obs, _ = env.reset()
    ppo_agent = PPOTrader(
        state_dim=len(obs), action_dim=5, hidden_dim=64,
        continuous=False, rollout_length=256, n_epochs=3,
    )
    for _ in range(8):
        rollout = ppo_agent.collect_rollout(env)
        ppo_agent.update(rollout)
    ppo_infos = []
    for s in range(5):
        obs, _ = env.reset(seed=s)
        while True:
            st = __import__('torch').FloatTensor(obs).unsqueeze(0)
            with __import__('torch').no_grad():
                a, _, _, _ = ppo_agent.network.get_action_and_value(st)
            obs, _, d, t, info = env.step(a.item())
            if d or t:
                break
        ppo_infos.append(info)
    algorithms["PPO"] = _avg_info(ppo_infos)

    # Print table
    print(f"\n  {'Algorithm':<20s} {'Return':>10s} {'Sharpe':>10s} "
          f"{'MaxDD':>10s} {'Trades':>8s} {'Costs':>10s}")
    print(f"  {'='*70}")
    for name, m in algorithms.items():
        print(f"  {name:<20s} {m['total_return']*100:>+9.2f}% "
              f"{m['sharpe_ratio']:>10.3f} {m['max_drawdown']*100:>9.2f}% "
              f"{m['trade_count']:>8.0f} ${m['total_cost']:>9.0f}")
    print(f"  {'='*70}")
    print("  (Results averaged over 5 evaluation runs with different seeds)")


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------
def _print_comparison(labels, infos):
    print(f"\n  {'Metric':<22s}", end="")
    for lbl in labels:
        print(f" {lbl:>18s}", end="")
    print()
    print(f"  {'-' * (22 + 19 * len(labels))}")

    rows = [
        ("Portfolio Value", "portfolio_value", "${:>14,.0f}", "${:>14,.0f}"),
        ("Total Return", "total_return", "{:>14.2f}%", "{:>14.2f}%"),
        ("Sharpe Ratio", "sharpe_ratio", "{:>15.3f}", "{:>15.3f}"),
        ("Max Drawdown", "max_drawdown", "{:>14.2f}%", "{:>14.2f}%"),
        ("Trades", "trade_count", "{:>15,d}", "{:>15,d}"),
    ]
    for label, key, *fmts in rows:
        print(f"  {label:<22s}", end="")
        for i, info in enumerate(infos):
            val = info.get(key, 0)
            if "Return" in label or "Drawdown" in label:
                val = val * 100
            fmt = fmts[min(i, len(fmts)-1)]
            print(f" {fmt.format(val):>18s}", end="")
        print()


def _avg_info(infos):
    keys = ["total_return", "sharpe_ratio", "max_drawdown", "trade_count", "total_cost", "portfolio_value"]
    return {k: np.mean([info.get(k, 0) for info in infos]) for k in keys}


def _extract_q_features(obs):
    return np.array([
        obs[9] if len(obs) > 9 else 0,
        obs[5] if len(obs) > 5 else 0.15,
        obs[7] if len(obs) > 7 else 0.5,
        obs[-4] if len(obs) > 3 else 0,
    ])


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main():
    print(HEADER)
    total_start = time.time()

    dqn_agent = demo_dqn_vs_random()
    demo_q_learning_comparison()
    demo_ppo_learning()
    demo_multi_agent()
    demo_summary_table(dqn_agent)

    elapsed = time.time() - total_start
    print(f"\n  Total demo time: {elapsed:.0f}s")
    print(f"\n{'=' * 70}")
    print(f"  DEMO COMPLETE  -  All results use synthetic data (reproducible)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
