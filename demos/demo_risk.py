"""
===========================================================================
RISK MANAGEMENT DEMO: Stress Testing & Risk-Aware RL
===========================================================================
Runtime: ~1 minute  |  No external data needed

Shows:
  1. Risk metrics on normal vs stressed market scenarios
  2. How reward shaping changes agent behaviour
  3. Distributional RL return quantiles
  4. Drawdown analysis under different regimes

Usage:
    python demos/demo_risk.py
===========================================================================
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.risk_management import RiskMetrics, DistributionalRiskEstimator
from risk.reward_shaping import FinancialRewardShaper, CurriculumRewardScheduler
from evaluation.metrics import FinancialMetrics


def generate_scenarios():
    """Generate return scenarios for stress testing."""
    np.random.seed(42)
    n = 500

    scenarios = {
        "Normal Market": np.random.normal(0.0004, 0.012, n),
        "Bull Market": np.random.normal(0.001, 0.008, n),
        "Bear Market": np.random.normal(-0.0005, 0.018, n),
        "High Volatility": np.random.normal(0.0002, 0.030, n),
        "Fat Tails (t=3)": np.random.standard_t(df=3, size=n) * 0.012,
        "Crash Scenario": _generate_crash_scenario(n),
    }
    return scenarios


def _generate_crash_scenario(n):
    """Normal market with embedded crash event."""
    returns = np.random.normal(0.0004, 0.012, n)
    crash_start = n // 2
    returns[crash_start:crash_start+5] = [-0.04, -0.06, -0.08, -0.03, -0.02]
    returns[crash_start+5:crash_start+20] = np.random.normal(-0.005, 0.025, 15)
    return returns


# ─────────────────────────────────────────────────────────────────────
# Demo 1: Risk Metrics Across Scenarios
# ─────────────────────────────────────────────────────────────────────
def demo_risk_metrics():
    print(f"\n{'-'*70}")
    print("  RISK METRICS ACROSS MARKET SCENARIOS")
    print(f"{'-'*70}")

    scenarios = generate_scenarios()

    header = (f"  {'Scenario':<20s} {'Ann.Ret':>8s} {'Ann.Vol':>8s} "
              f"{'Sharpe':>8s} {'VaR95':>8s} {'CVaR95':>8s} "
              f"{'MaxDD':>8s} {'Sortino':>8s}")
    print(header)
    print(f"  {'='*76}")

    for name, returns in scenarios.items():
        pv = np.cumprod(1 + returns)
        ann_ret = (pv[-1]) ** (252/len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-8)
        var95 = RiskMetrics.var(returns, 0.05)
        cvar95 = RiskMetrics.cvar(returns, 0.05)
        mdd = RiskMetrics.max_drawdown(pv)
        sortino = RiskMetrics.sortino_ratio(returns)

        print(f"  {name:<20s} {ann_ret*100:>+7.1f}% {ann_vol*100:>7.1f}% "
              f"{sharpe:>8.2f} {var95*100:>7.2f}% {cvar95*100:>7.2f}% "
              f"{mdd*100:>7.1f}% {sortino:>8.2f}")

    print(f"\n  Key: VaR95 = daily loss exceeded 5% of the time")
    print(f"       CVaR95 = average loss in worst 5% of days")
    print(f"       MaxDD  = largest peak-to-trough decline")


# ─────────────────────────────────────────────────────────────────────
# Demo 2: Reward Shaping Impact
# ─────────────────────────────────────────────────────────────────────
def demo_reward_shaping():
    print(f"\n{'-'*70}")
    print("  REWARD SHAPING: HOW OBJECTIVES CHANGE AGENT BEHAVIOUR")
    print(f"{'-'*70}")

    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.015, 252)

    configs = {
        "Raw Returns Only": {
            "return_weight": 1.0, "sharpe_weight": 0.0,
            "drawdown_penalty": 0.0, "turnover_penalty": 0.0,
        },
        "Sharpe-Focused": {
            "return_weight": 0.3, "sharpe_weight": 1.5,
            "drawdown_penalty": 0.0, "turnover_penalty": 0.0,
        },
        "Drawdown-Averse": {
            "return_weight": 0.5, "sharpe_weight": 0.5,
            "drawdown_penalty": 5.0, "turnover_penalty": 0.0,
        },
        "Low-Turnover": {
            "return_weight": 0.5, "sharpe_weight": 0.5,
            "drawdown_penalty": 1.0, "turnover_penalty": 1.0,
        },
        "Full Composite": {
            "return_weight": 1.0, "sharpe_weight": 0.5,
            "drawdown_penalty": 2.0, "turnover_penalty": 0.1,
            "position_penalty": 0.05,
        },
    }

    print(f"\n  {'Reward Config':<22s} {'Total Reward':>14s} {'Mean Step':>12s} "
          f"{'Std Step':>10s} {'Min Step':>10s}")
    print(f"  {'-'*68}")

    for name, cfg in configs.items():
        shaper = FinancialRewardShaper(**cfg)
        step_rewards = []
        pv = 1.0

        for t in range(252):
            pv *= (1 + returns[t])
            pos_change = np.random.uniform(-0.2, 0.2)
            pos = np.random.uniform(-0.5, 0.5)
            r = shaper.compute_reward(returns[t], pos_change, pos, pv)
            step_rewards.append(r)

        total = sum(step_rewards)
        arr = np.array(step_rewards)
        print(f"  {name:<22s} {total:>+14.2f} {arr.mean():>12.4f} "
              f"{arr.std():>10.4f} {arr.min():>10.4f}")

    print(f"\n  Takeaway: Different reward shapes produce very different learning")
    print(f"  signals from the SAME market data. The agent's behavior follows.")


# ─────────────────────────────────────────────────────────────────────
# Demo 3: Distributional Risk Estimation
# ─────────────────────────────────────────────────────────────────────
def demo_distributional_risk():
    print(f"\n{'-'*70}")
    print("  DISTRIBUTIONAL RL: RETURN QUANTILE ESTIMATION")
    print(f"{'-'*70}")

    estimator = DistributionalRiskEstimator(state_dim=10, action_dim=5, n_quantiles=51)

    np.random.seed(42)
    state = np.random.randn(10)

    action_names = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]

    print(f"\n  {'Action':<14s} {'E[Return]':>10s} {'Std':>8s} "
          f"{'VaR 95%':>10s} {'CVaR 95%':>10s} {'Upside':>10s}")
    print(f"  {'-'*62}")

    for a, name in enumerate(action_names):
        risk = estimator.compute_risk_metrics(state, a)
        quantiles = estimator.predict_quantiles(state)[a]
        upside = np.mean(quantiles[int(0.9*51):])

        print(f"  {name:<14s} {risk['expected_return']:>+10.4f} {risk['std']:>8.4f} "
              f"{risk['var_95']:>+10.4f} {risk['cvar_95']:>+10.4f} {upside:>+10.4f}")

    print(f"\n  Note: QR-DQN learns the FULL return distribution, enabling")
    print(f"  direct computation of any risk measure (VaR, CVaR, tail ratios).")
    print(f"  The untrained model shows random quantiles; after training these")
    print(f"  would reflect the true conditional return distribution.")


# ─────────────────────────────────────────────────────────────────────
# Demo 4: Drawdown Stress Test
# ─────────────────────────────────────────────────────────────────────
def demo_drawdown_stress_test():
    print(f"\n{'-'*70}")
    print("  DRAWDOWN STRESS TEST")
    print(f"{'-'*70}")

    np.random.seed(42)

    crash_sizes = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    recovery_rates = [0.005, 0.001, 0.0005]

    print(f"\n  Recovery time (days) to break even after sudden crash:")
    print(f"  {'Crash Size':<12s}", end="")
    for rr in recovery_rates:
        print(f" {'r='+str(rr*100)+'%/d':>12s}", end="")
    print()
    print(f"  {'-'*48}")

    for crash in crash_sizes:
        print(f"  {crash*100:>10.0f}%  ", end="")
        for rr in recovery_rates:
            days = int(np.log(1/(1-crash)) / np.log(1+rr))
            years = days / 252
            if years < 1:
                print(f"  {days:>5d} days  ", end="")
            else:
                print(f"  {years:>5.1f} yrs  ", end="")
        print()

    print(f"\n  Key Insight: A 40% crash at 0.05%/day recovery takes ~4 years.")
    print(f"  This is why drawdown control is critical -- prevention > cure.")

    # Show VaR scaling with confidence level
    print(f"\n  VaR scaling with confidence level (normal market returns):")
    returns = np.random.normal(0.0004, 0.012, 5000)
    print(f"  {'Confidence':>12s} {'VaR':>10s} {'CVaR':>10s} {'Ratio':>8s}")
    print(f"  {'-'*40}")
    for alpha in [0.10, 0.05, 0.025, 0.01, 0.005]:
        var = RiskMetrics.var(returns, alpha)
        cvar = RiskMetrics.cvar(returns, alpha)
        ratio = cvar / (var + 1e-8)
        print(f"  {(1-alpha)*100:>11.1f}% {var*100:>9.3f}% {cvar*100:>9.3f}% {ratio:>8.2f}x")


# ─────────────────────────────────────────────────────────────────────
# Demo 5: Curriculum Learning Schedule
# ─────────────────────────────────────────────────────────────────────
def demo_curriculum():
    print(f"\n{'-'*70}")
    print("  CURRICULUM LEARNING: PROGRESSIVE TRAINING SCHEDULE")
    print(f"{'-'*70}")

    scheduler = CurriculumRewardScheduler(total_episodes=200)

    print(f"\n  {'Episode Range':<20s} {'Phase':<20s} {'Focus':>30s}")
    print(f"  {'-'*70}")

    phase_info = {
        "Survival":       "Heavy drawdown penalty, survival bonus",
        "BasicProfits":   "Return-focused, moderate risk penalty",
        "RiskAdjusted":   "Sharpe ratio optimization",
        "FullObjective":  "All components: return + risk + costs",
    }

    prev_phase = ""
    for ep in range(200):
        scheduler.advance()
        phase = scheduler.current_phase_name
        if phase != prev_phase:
            info = phase_info.get(phase, "")
            range_str = f"  {ep+1:>3d} -> "
            prev_phase = phase
            start_ep = ep + 1

    # Print all phases cleanly
    scheduler2 = CurriculumRewardScheduler(total_episodes=200)
    prev = ""
    start = 1
    for ep in range(200):
        scheduler2.advance()
        phase = scheduler2.current_phase_name
        if phase != prev:
            if prev:
                end = ep
                info = phase_info.get(prev, "")
                print(f"  {start:>3d} - {end:<3d}            {prev:<20s} {info:>30s}")
            prev = phase
            start = ep + 1
    info = phase_info.get(prev, "")
    print(f"  {start:>3d} - 200            {prev:<20s} {info:>30s}")

    print(f"\n  Analogy: Like training a junior trader --")
    print(f"  first learn not to blow up, then learn to make money,")
    print(f"  then learn to manage risk, then handle the full job.")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("""
+======================================================================+
|          RISK MANAGEMENT & STRESS TESTING DEMO                       |
|          Synthetic Data  -  No Setup Required                        |
+======================================================================+
""")

    demo_risk_metrics()
    demo_reward_shaping()
    demo_distributional_risk()
    demo_drawdown_stress_test()
    demo_curriculum()

    print(f"\n{'='*70}")
    print(f"  RISK DEMO COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
