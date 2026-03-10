"""
=============================================================================
FINANCIAL AI WITH REINFORCEMENT LEARNING
Complete Research Project - Main Entry Point
=============================================================================

This script runs demonstrations of all modules in sequence,
providing a comprehensive tour of RL concepts applied to finance.

Run: python main.py
=============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all_demonstrations():
    """Run all module demonstrations in order."""
    print("\n" + "=" * 70)
    print("  FINANCIAL AI WITH REINFORCEMENT LEARNING")
    print("  Complete Research Demonstration")
    print("=" * 70)

    modules = [
        ("Part I: Foundations", [
            ("Chapter 1: MDP Foundations", "core.mdp_foundations", "demonstrate_mdp_foundations"),
            ("Chapter 2: Value Functions & TD Learning", "core.value_functions", "demonstrate_td_learning"),
            ("Chapter 3: Bellman Equations", "core.bellman_equations", "demonstrate_bellman_equations"),
        ]),
        ("Part II: Environments", [
            ("Chapter 4: Stock Trading Environment", "environments.stock_trading_env", "demonstrate_trading_environment"),
            ("Chapter 5: Portfolio Allocation Environment", "environments.portfolio_env", "demonstrate_portfolio_env"),
            ("Chapter 6: Options Hedging Environment", "environments.options_pricing_env", "demonstrate_options_env"),
        ]),
        ("Part III: RL Algorithms", [
            ("Chapter 7: Q-Learning", "agents.q_learning", "demonstrate_q_learning"),
            ("Chapter 8: Deep Q-Network (DQN)", "agents.dqn_agent", "demonstrate_dqn"),
            ("Chapter 9: REINFORCE", "agents.policy_gradient", "demonstrate_reinforce"),
            ("Chapter 10: PPO", "agents.ppo_agent", "demonstrate_ppo"),
            ("Chapter 11: A2C", "agents.a2c_agent", "demonstrate_a2c"),
        ]),
        ("Part IV: Applications", [
            ("Chapter 12: RL Portfolio Optimization", "portfolio.optimization", "demonstrate_portfolio_optimization"),
            ("Chapter 14: Risk Management", "risk.risk_management", "demonstrate_risk_management"),
            ("Chapter 15: Reward Shaping", "risk.reward_shaping", "demonstrate_reward_shaping"),
        ]),
        ("Part V: Advanced Topics", [
            ("Chapter 16: Multi-Agent RL", "advanced.multi_agent", "demonstrate_multi_agent"),
            ("Chapter 17: Model-Based RL", "advanced.model_based", "demonstrate_model_based"),
            ("Chapter 18: Inverse RL", "advanced.inverse_rl", "demonstrate_inverse_rl"),
        ]),
        ("Part VI: Evaluation", [
            ("Chapter 19: Backtesting", "evaluation.backtester", "demonstrate_backtesting"),
        ]),
    ]

    for part_name, chapters in modules:
        print(f"\n\n{'#' * 70}")
        print(f"  {part_name}")
        print(f"{'#' * 70}")

        for chapter_name, module_path, func_name in chapters:
            print(f"\n>>> Running {chapter_name}...")
            try:
                module = __import__(module_path, fromlist=[func_name])
                func = getattr(module, func_name)
                func()
            except Exception as e:
                print(f"  [SKIPPED] {chapter_name}: {e}")

    print("\n" + "=" * 70)
    print("  ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Financial AI with RL - Research Demonstrations")
    parser.add_argument("--chapter", type=int, help="Run a specific chapter (1-19)")
    parser.add_argument("--all", action="store_true", help="Run all demonstrations")
    args = parser.parse_args()

    if args.all or args.chapter is None:
        run_all_demonstrations()
    else:
        chapter_map = {
            1: ("core.mdp_foundations", "demonstrate_mdp_foundations"),
            2: ("core.value_functions", "demonstrate_td_learning"),
            3: ("core.bellman_equations", "demonstrate_bellman_equations"),
            4: ("environments.stock_trading_env", "demonstrate_trading_environment"),
            5: ("environments.portfolio_env", "demonstrate_portfolio_env"),
            6: ("environments.options_pricing_env", "demonstrate_options_env"),
            7: ("agents.q_learning", "demonstrate_q_learning"),
            8: ("agents.dqn_agent", "demonstrate_dqn"),
            9: ("agents.policy_gradient", "demonstrate_reinforce"),
            10: ("agents.ppo_agent", "demonstrate_ppo"),
            11: ("agents.a2c_agent", "demonstrate_a2c"),
            12: ("portfolio.optimization", "demonstrate_portfolio_optimization"),
            14: ("risk.risk_management", "demonstrate_risk_management"),
            15: ("risk.reward_shaping", "demonstrate_reward_shaping"),
            16: ("advanced.multi_agent", "demonstrate_multi_agent"),
            17: ("advanced.model_based", "demonstrate_model_based"),
            18: ("advanced.inverse_rl", "demonstrate_inverse_rl"),
            19: ("evaluation.backtester", "demonstrate_backtesting"),
        }

        if args.chapter in chapter_map:
            module_path, func_name = chapter_map[args.chapter]
            module = __import__(module_path, fromlist=[func_name])
            func = getattr(module, func_name)
            func()
        else:
            print(f"Chapter {args.chapter} not found. Available: {sorted(chapter_map.keys())}")
