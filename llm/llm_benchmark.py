"""
===========================================================================
LLM-Powered RL Benchmark
===========================================================================
Registers all LLM strategy variants into the benchmark framework and
provides a standalone demo that shows every LLM role in action.

Usage:
    python llm/llm_benchmark.py                     # auto-detect backend
    python llm/llm_benchmark.py --backend ollama     # force Ollama
    python llm/llm_benchmark.py --backend mock       # mock (no LLM needed)
    python llm/llm_benchmark.py --model mistral      # specific model

This demo:
  1. Tests provider connectivity
  2. Demonstrates all 4 LLM roles (reward, policy, state, world model)
  3. Benchmarks LLM strategies against baselines
  4. Prints formatted comparison table
===========================================================================
"""

import sys
import os
import time
import warnings
import argparse
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.providers import get_provider, LLMProvider
from llm.reward_model import LLMRewardModel
from llm.policy_agent import LLMDirectPolicy, LLMGuidedRL
from llm.state_encoder import LLMStateEncoder, LLMAugmentedTradingEnv
from llm.world_model import LLMWorldModel

from environments.stock_trading_env import StockTradingEnv
from benchmarks.registry import BenchmarkRegistry, BaseStrategy


# =====================================================================
# Register LLM strategies into the benchmark framework
# =====================================================================

@BenchmarkRegistry.register("LLM-Policy", category="llm")
class LLMPolicyBenchmark(BaseStrategy):
    """LLM directly decides trades (zero-shot)."""
    def __init__(self):
        self._policy = None

    def train(self, env, config):
        backend = config.get("llm_backend", "auto")
        model = config.get("llm_model", None)
        provider = get_provider(backend, model)
        self._policy = LLMDirectPolicy(provider=provider)

    def predict(self, obs, info=None):
        if self._policy is None:
            return 2
        return self._policy.predict(obs, info)

    def reset(self):
        if self._policy:
            self._policy.reset()


@BenchmarkRegistry.register("LLM-WorldModel", category="llm")
class LLMWorldModelBenchmark(BaseStrategy):
    """LLM predicts scenarios and plans the best action."""
    def __init__(self):
        self._wm = None

    def train(self, env, config):
        backend = config.get("llm_backend", "auto")
        model = config.get("llm_model", None)
        provider = get_provider(backend, model)
        self._wm = LLMWorldModel(provider=provider)

    def predict(self, obs, info=None):
        if self._wm is None:
            return 2
        action, _ = self._wm.plan_action(obs, info)
        return action

    def reset(self):
        if self._wm:
            self._wm.clear_cache()


@BenchmarkRegistry.register("LLM-Guided DQN", category="llm")
class LLMGuidedDQNBenchmark(BaseStrategy):
    """DQN agent with LLM teacher guidance during training."""
    def __init__(self):
        self._agent = None

    def train(self, env, config):
        from agents.dqn_agent import DQNAgent
        backend = config.get("llm_backend", "auto")
        model = config.get("llm_model", None)
        provider = get_provider(backend, model)

        obs, _ = env.reset()
        episodes = config.get("episodes", 10)

        self._agent = DQNAgent(
            state_dim=len(obs), action_dim=5,
            hidden_dims=[64, 32],
            double_dqn=True, dueling=True,
            learning_rate=5e-4,
            epsilon_decay_steps=episodes * 400,
            buffer_size=20000, batch_size=32,
        )
        guide = LLMGuidedRL(provider=provider, guidance_strength=0.4, decay_rate=0.95)

        for ep in range(episodes):
            obs, info = env.reset()
            for step in range(2000):
                agent_action = self._agent.select_action(obs, training=True)
                action = guide.select_action(agent_action, obs, info)

                next_obs, reward, done, trunc, info = env.step(action)
                self._agent.store_transition(obs, action, reward, next_obs, float(done or trunc))
                self._agent.train_step()
                obs = next_obs
                if done or trunc:
                    break
            guide.decay()

    def predict(self, obs, info=None):
        if self._agent is None:
            return 2
        return self._agent.select_action(obs, training=False)


@BenchmarkRegistry.register("LLM-Reward PPO", category="llm")
class LLMRewardPPOBenchmark(BaseStrategy):
    """PPO agent trained with LLM-augmented rewards."""
    def __init__(self):
        self._agent = None

    def train(self, env, config):
        from agents.ppo_agent import PPOTrader
        import torch

        backend = config.get("llm_backend", "auto")
        model_name = config.get("llm_model", None)
        provider = get_provider(backend, model_name)
        reward_model = LLMRewardModel(provider=provider, llm_weight=0.3, eval_every_n_steps=3)

        obs, _ = env.reset()
        iters = config.get("iterations", 5)

        self._agent = PPOTrader(
            state_dim=len(obs), action_dim=5,
            hidden_dim=64, continuous=False,
            rollout_length=256, n_epochs=3,
        )

        for _ in range(iters):
            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
            obs, info = env.reset()
            reward_model.reset()

            for _ in range(256):
                state_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, _, value = self._agent.network.get_action_and_value(state_t)
                action_np = action.item()

                next_obs, base_reward, terminated, truncated, info = env.step(action_np)
                blended = reward_model.blend_reward(base_reward, action_np, obs, info)

                states.append(obs)
                actions.append(action_np)
                rewards.append(blended)
                dones.append(float(terminated or truncated))
                log_probs.append(log_prob.item())
                values.append(value.item())

                obs = next_obs
                if terminated or truncated:
                    obs, info = env.reset()
                    reward_model.reset()

            with torch.no_grad():
                ns = torch.FloatTensor(obs).unsqueeze(0)
                _, _, _, nv = self._agent.network.get_action_and_value(ns)
                next_value = nv.item()

            advantages, returns = self._agent.compute_gae(
                np.array(rewards), np.array(values), np.array(dones), next_value,
            )
            rollout = {
                "states": np.array(states),
                "actions": np.array(actions),
                "log_probs": np.array(log_probs),
                "values": np.array(values),
                "advantages": advantages,
                "returns": returns,
                "rewards": np.array(rewards),
                "info": info,
            }
            self._agent.update(rollout)

    def predict(self, obs, info=None):
        if self._agent is None:
            return 2
        import torch
        st = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            a, _, _, _ = self._agent.network.get_action_and_value(st)
        return a.item()


# =====================================================================
# Standalone demo
# =====================================================================

def run_demo(backend: str = "auto", model_name: str = None):
    print("""
+======================================================================+
|          LLM-POWERED RL FINANCIAL TRADING DEMO                       |
|          4 LLM Roles  -  Benchmarked Against Baselines               |
+======================================================================+
""")
    total_start = time.time()

    # 1. Provider setup
    provider = get_provider(backend, model_name)
    print(f"  LLM Backend:  {provider.__class__.__name__}")
    print(f"  Model:        {provider.model_name}")
    print(f"  Available:    {provider.is_available()}")
    print()

    env = StockTradingEnv(reward_type="sharpe", discrete_actions=True)
    obs, info = env.reset(seed=42)

    # 2. Demo each role
    _demo_reward_model(provider, env)
    _demo_policy_agent(provider, env)
    _demo_state_encoder(provider, obs)
    _demo_world_model(provider, obs, info)

    # 3. Head-to-head benchmark
    _demo_benchmark(provider, env)

    elapsed = time.time() - total_start
    print(f"\n  Total demo time: {elapsed:.1f}s")
    print(f"\n{'='*70}")
    print(f"  LLM-RL DEMO COMPLETE")
    print(f"{'='*70}\n")


def _demo_reward_model(provider, env):
    print(f"{'-'*70}")
    print(f"  ROLE 1: LLM as Reward Model")
    print(f"{'-'*70}")

    rm = LLMRewardModel(provider=provider, eval_every_n_steps=1)
    obs, info = env.reset(seed=42)

    print(f"\n  Testing LLM reward on 5 actions in the same state:\n")
    print(f"  {'Action':<14s} {'Base Reward':>12s} {'LLM Score':>12s} {'Blended':>12s}")
    print(f"  {'-'*50}")

    for action in range(5):
        action_names = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
        rm.reset()
        _, base_r, _, _, step_info = env.step(action)
        env.reset(seed=42)
        llm_score = rm.score_trade(action, obs, info)
        blended = rm.blend_reward(base_r, action, obs, info)
        print(f"  {action_names[action]:<14s} {base_r:>+12.4f} {llm_score:>+12.4f} {blended:>+12.4f}")

    print()


def _demo_policy_agent(provider, env):
    print(f"{'-'*70}")
    print(f"  ROLE 2: LLM as Trading Policy")
    print(f"{'-'*70}")

    policy = LLMDirectPolicy(provider=provider)
    obs, info = env.reset(seed=42)

    total_reward = 0
    steps = 0
    while True:
        action = policy.predict(obs, info)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if done or trunc:
            break

    print(f"\n  LLM-Policy episode ({steps} steps):")
    print(f"    Portfolio:     ${info['portfolio_value']:,.0f}")
    print(f"    Return:        {info['total_return']*100:+.2f}%")
    print(f"    Sharpe:        {info['sharpe_ratio']:.3f}")
    print(f"    Max Drawdown:  {info['max_drawdown']*100:.2f}%")

    dist = policy._action_counts
    total = dist.sum()
    if total > 0:
        names = ["str_sell", "sell", "hold", "buy", "str_buy"]
        pcts = dist / total * 100
        print(f"    Actions:       {', '.join(f'{n}={p:.0f}%' for n, p in zip(names, pcts))}")
    print()


def _demo_state_encoder(provider, obs):
    print(f"{'-'*70}")
    print(f"  ROLE 3: LLM as State Encoder (Sentiment)")
    print(f"{'-'*70}")

    encoder = LLMStateEncoder(provider=provider, update_every_n=1)
    features = encoder.encode(obs)

    print(f"\n  Original state dim:  {len(obs)}")
    print(f"  LLM features:        {encoder.feature_names}")
    print(f"  Sentiment:           {features[0]:+.3f}  (-1=bearish, +1=bullish)")
    print(f"  Risk:                {features[1]:.3f}   (0=low, 1=extreme)")
    print(f"  Confidence:          {features[2]:.3f}   (0=uncertain, 1=certain)")
    print(f"  Augmented state dim: {len(obs) + encoder.feature_dim}")
    print()


def _demo_world_model(provider, obs, info):
    print(f"{'-'*70}")
    print(f"  ROLE 4: LLM as World Model (Scenario Planning)")
    print(f"{'-'*70}")

    wm = LLMWorldModel(provider=provider)
    scenarios = wm.generate_scenarios(obs, n_scenarios=3, info=info)

    print(f"\n  Generated {len(scenarios)} scenarios:\n")
    print(f"  {'#':<4s} {'Direction':<8s} {'Magnitude':>10s} {'Vol Change':>12s} {'Probability':>12s}")
    print(f"  {'-'*48}")
    for i, s in enumerate(scenarios):
        print(f"  {i+1:<4d} {s.direction:<8s} {s.magnitude:>+10.4f} {s.volatility_change:>+12.4f} {s.probability:>12.1%}")

    best_action, plan_info = wm.plan_action(obs, info)
    action_names = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
    print(f"\n  Planned action: {action_names[best_action]}")
    print(f"  Action scores:  {['%.2f' % s for s in plan_info['action_scores']]}")
    print()


def _demo_benchmark(provider, env):
    print(f"{'-'*70}")
    print(f"  BENCHMARK: LLM Strategies vs Baselines")
    print(f"{'-'*70}")

    N_EVAL = 3
    SEEDS = [42, 123, 456]

    strategies = {
        "Random": lambda obs, info: np.random.randint(5),
        "Buy & Hold": lambda obs, info: 4,
        "Always Cash": lambda obs, info: 2,
        "LLM-Policy": None,
        "LLM-WorldModel": None,
    }

    llm_policy = LLMDirectPolicy(provider=provider)
    wm = LLMWorldModel(provider=provider)

    results = {}

    for name, fn in strategies.items():
        returns, sharpes, drawdowns = [], [], []

        for seed in SEEDS:
            obs, info = env.reset(seed=seed)
            if name == "LLM-Policy":
                llm_policy.reset()
            if name == "LLM-WorldModel":
                wm.clear_cache()

            while True:
                if name == "LLM-Policy":
                    action = llm_policy.predict(obs, info)
                elif name == "LLM-WorldModel":
                    action, _ = wm.plan_action(obs, info)
                else:
                    action = fn(obs, info)

                obs, _, done, trunc, info = env.step(action)
                if done or trunc:
                    break

            returns.append(info.get("total_return", 0))
            sharpes.append(info.get("sharpe_ratio", 0))
            drawdowns.append(info.get("max_drawdown", 0))

        results[name] = {
            "return": np.mean(returns),
            "sharpe": np.mean(sharpes),
            "max_dd": np.mean(drawdowns),
        }

    print(f"\n  {'Strategy':<18s} {'Return':>10s} {'Sharpe':>10s} {'Max DD':>10s}")
    print(f"  {'='*50}")
    for name, m in results.items():
        tag = " *" if "LLM" in name else ""
        print(f"  {name:<18s} {m['return']*100:>+9.2f}% {m['sharpe']:>10.3f} {m['max_dd']*100:>9.2f}%{tag}")
    print(f"  {'='*50}")
    print(f"  (* = LLM-powered strategy)")
    print()


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-RL Financial Trading Demo")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "ollama", "huggingface", "mock"],
                        help="LLM backend (default: auto)")
    parser.add_argument("--model", default=None,
                        help="Model name (e.g., qwen2.5, llama3.2, mistral)")
    args = parser.parse_args()

    run_demo(backend=args.backend, model_name=args.model)
