"""
=============================================================================
MODULE 19: Backtesting Framework for RL Trading Strategies
=============================================================================

THEORY:
-------
Backtesting evaluates a strategy on historical data. For RL strategies,
this is particularly important because:

1. OVERFITTING RISK: RL agents can overfit to training data patterns
2. LOOK-AHEAD BIAS: Must ensure no future information leaks
3. SURVIVORSHIP BIAS: Historical data may not include delisted assets
4. REGIME DEPENDENCE: Strategy may work in one regime but not another

BEST PRACTICES:
  - Walk-forward validation: train on [0, T], test on [T, T+k], repeat
  - Out-of-sample testing: strict train/test split
  - Cross-validation: k-fold with time-aware splits
  - Monte Carlo simulation: bootstrap confidence intervals
  - Paper trading: forward testing before live deployment
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    """Container for backtest results."""
    portfolio_values: np.ndarray
    returns: np.ndarray
    trades: List[Dict]
    metrics: Dict[str, float]
    daily_positions: List[float] = field(default_factory=list)


class RLBacktester:
    """
    Comprehensive backtesting framework for RL trading strategies.

    Supports:
    - Walk-forward optimization
    - Monte Carlo bootstrap for confidence intervals
    - Multiple benchmark comparisons
    - Transaction cost sensitivity analysis
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate / 252

    def run_backtest(
        self,
        prices: np.ndarray,
        strategy_fn: Callable,
        features: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run a backtest on historical price data.

        strategy_fn: takes (state_dict) -> action
          state_dict includes: prices, features, position, portfolio_value, etc.
        """
        n_days = len(prices)
        portfolio_values = np.zeros(n_days)
        portfolio_values[0] = self.initial_capital
        returns_arr = np.zeros(n_days)
        positions = []
        trades = []

        cash = self.initial_capital
        position = 0.0
        peak = self.initial_capital

        for t in range(1, n_days):
            state = {
                'price': prices[t],
                'prev_price': prices[t - 1],
                'position': position,
                'cash': cash,
                'portfolio_value': portfolio_values[t - 1],
                'features': features[t] if features is not None else None,
                'step': t,
            }

            action = strategy_fn(state)

            # action: target position fraction [-1, 1]
            target_value = action * portfolio_values[t - 1]
            target_shares = target_value / prices[t]
            shares_to_trade = target_shares - position

            if abs(shares_to_trade) > 1e-8:
                direction = np.sign(shares_to_trade)
                exec_price = prices[t] * (1 + direction * self.slippage)
                cost = abs(shares_to_trade) * exec_price * self.transaction_cost

                cash -= shares_to_trade * exec_price + cost
                position = target_shares

                trades.append({
                    'step': t,
                    'side': 'buy' if shares_to_trade > 0 else 'sell',
                    'shares': abs(shares_to_trade),
                    'price': exec_price,
                    'cost': cost,
                })

            portfolio_values[t] = cash + position * prices[t]
            returns_arr[t] = (portfolio_values[t] - portfolio_values[t - 1]) / portfolio_values[t - 1]
            positions.append(position * prices[t] / portfolio_values[t] if portfolio_values[t] > 0 else 0)

        # Compute metrics
        metrics = self._compute_metrics(portfolio_values, returns_arr, trades)

        return BacktestResult(
            portfolio_values=portfolio_values,
            returns=returns_arr,
            trades=trades,
            metrics=metrics,
            daily_positions=positions,
        )

    def walk_forward_test(
        self,
        prices: np.ndarray,
        train_fn: Callable,
        test_fn: Callable,
        features: Optional[np.ndarray] = None,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 63,
    ) -> List[BacktestResult]:
        """
        Walk-forward optimization and testing.

        1. Train on [t, t+train_window]
        2. Test on [t+train_window, t+train_window+test_window]
        3. Step forward by step_size
        4. Repeat

        This simulates realistic strategy deployment:
        periodically retrain on new data, test on unseen future data.
        """
        n_days = len(prices)
        all_results = []
        t = 0

        while t + train_window + test_window <= n_days:
            train_prices = prices[t:t + train_window]
            test_prices = prices[t + train_window:t + train_window + test_window]

            train_features = features[t:t + train_window] if features is not None else None
            test_features = features[t + train_window:t + train_window + test_window] if features is not None else None

            # Train
            trained_strategy = train_fn(train_prices, train_features)

            # Test
            result = self.run_backtest(test_prices, trained_strategy, test_features)
            all_results.append(result)

            t += step_size

        return all_results

    def monte_carlo_bootstrap(
        self,
        returns: np.ndarray,
        n_simulations: int = 1000,
        simulation_length: int = 252,
    ) -> Dict:
        """
        Bootstrap simulation for confidence intervals.

        Samples returns with replacement to estimate the distribution
        of performance metrics under different market realizations.
        """
        simulated_sharpes = []
        simulated_returns = []
        simulated_drawdowns = []

        for _ in range(n_simulations):
            sampled_returns = np.random.choice(returns, size=simulation_length, replace=True)
            sim_pv = self.initial_capital * np.cumprod(1 + sampled_returns)

            annual_ret = (sim_pv[-1] / self.initial_capital) ** (252 / simulation_length) - 1
            annual_vol = sampled_returns.std() * np.sqrt(252)
            sharpe = (annual_ret - self.risk_free_rate * 252) / (annual_vol + 1e-8)
            max_dd = np.max(1 - sim_pv / np.maximum.accumulate(sim_pv))

            simulated_sharpes.append(sharpe)
            simulated_returns.append(annual_ret)
            simulated_drawdowns.append(max_dd)

        return {
            'sharpe_mean': np.mean(simulated_sharpes),
            'sharpe_5th': np.percentile(simulated_sharpes, 5),
            'sharpe_95th': np.percentile(simulated_sharpes, 95),
            'return_mean': np.mean(simulated_returns),
            'return_5th': np.percentile(simulated_returns, 5),
            'return_95th': np.percentile(simulated_returns, 95),
            'drawdown_mean': np.mean(simulated_drawdowns),
            'drawdown_95th': np.percentile(simulated_drawdowns, 95),
        }

    def _compute_metrics(
        self,
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        trades: List[Dict],
    ) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        pv = portfolio_values[portfolio_values > 0]
        rets = returns[1:]  # skip first zero

        total_return = pv[-1] / pv[0] - 1
        n_days = len(rets)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        annual_vol = rets.std() * np.sqrt(252)
        sharpe = (annual_return - self.risk_free_rate * 252) / (annual_vol + 1e-8)

        # Sortino
        downside = rets[rets < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
        sortino = annual_return / (downside_vol + 1e-8)

        # Max drawdown
        peak = np.maximum.accumulate(pv)
        drawdown = (peak - pv) / peak
        max_dd = np.max(drawdown)

        # Calmar
        calmar = annual_return / (max_dd + 1e-8)

        # Trade statistics
        n_trades = len(trades)
        total_costs = sum(t['cost'] for t in trades)
        avg_trade_cost = total_costs / max(n_trades, 1)

        # Win rate
        if n_trades > 0:
            trade_pnls = []
            for i in range(1, len(trades)):
                if trades[i]['side'] != trades[i-1]['side']:
                    pnl = (trades[i]['price'] - trades[i-1]['price']) * trades[i-1]['shares']
                    if trades[i-1]['side'] == 'sell':
                        pnl = -pnl
                    trade_pnls.append(pnl)
            win_rate = sum(1 for p in trade_pnls if p > 0) / max(len(trade_pnls), 1)
        else:
            win_rate = 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'n_trades': n_trades,
            'total_costs': total_costs,
            'avg_trade_cost': avg_trade_cost,
            'win_rate': win_rate,
        }


def demonstrate_backtesting():
    """Demonstrate the backtesting framework."""
    print("=" * 70)
    print("  CHAPTER 19: BACKTESTING FRAMEWORK")
    print("=" * 70)

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_loader import FinancialDataLoader

    data = FinancialDataLoader.generate_synthetic_data(len_data=1000)
    prices = data.prices

    backtester = RLBacktester(initial_capital=100000)

    # Strategy 1: Buy and Hold
    def buy_hold(state):
        return 1.0

    # Strategy 2: Simple Momentum
    momentum_prices = []
    def momentum_strategy(state):
        momentum_prices.append(state['price'])
        if len(momentum_prices) < 20:
            return 0.0
        avg = np.mean(momentum_prices[-20:])
        return 1.0 if state['price'] > avg else -0.5

    # Strategy 3: Mean Reversion
    mr_prices = []
    def mean_reversion_strategy(state):
        mr_prices.append(state['price'])
        if len(mr_prices) < 50:
            return 0.0
        avg = np.mean(mr_prices[-50:])
        std = np.std(mr_prices[-50:])
        z = (state['price'] - avg) / (std + 1e-8)
        return np.clip(-z * 0.5, -1, 1)

    strategies = {
        "Buy & Hold": buy_hold,
        "Momentum": momentum_strategy,
        "Mean Reversion": mean_reversion_strategy,
    }

    print(f"\n--- Backtest Results (1000 days) ---")
    print(f"{'Strategy':>20s} {'Return':>10s} {'Sharpe':>8s} {'MaxDD':>8s} {'Sortino':>8s} {'Trades':>8s}")
    print("-" * 72)

    for name, strategy_fn in strategies.items():
        momentum_prices.clear()
        mr_prices.clear()
        result = backtester.run_backtest(prices, strategy_fn)
        m = result.metrics
        print(f"{name:>20s} {m['total_return'] * 100:9.2f}% {m['sharpe_ratio']:8.3f} "
              f"{m['max_drawdown'] * 100:7.2f}% {m['sortino_ratio']:8.3f} {m['n_trades']:8d}")

    # Monte Carlo bootstrap
    print(f"\n--- Monte Carlo Bootstrap (Momentum Strategy, 1000 sims) ---")
    momentum_prices.clear()
    result = backtester.run_backtest(prices, momentum_strategy)
    mc_results = backtester.monte_carlo_bootstrap(result.returns[1:])

    print(f"  Sharpe: {mc_results['sharpe_mean']:.3f} [{mc_results['sharpe_5th']:.3f}, {mc_results['sharpe_95th']:.3f}]")
    print(f"  Return: {mc_results['return_mean'] * 100:.1f}% [{mc_results['return_5th'] * 100:.1f}%, {mc_results['return_95th'] * 100:.1f}%]")
    print(f"  MaxDD:  {mc_results['drawdown_mean'] * 100:.1f}% (95th: {mc_results['drawdown_95th'] * 100:.1f}%)")

    print("\nKey Insights:")
    print("  1. Walk-forward testing prevents look-ahead bias")
    print("  2. Bootstrap gives confidence intervals, not point estimates")
    print("  3. Multiple metrics needed (Sharpe alone is insufficient)")
    print("  4. Transaction costs can dramatically change rankings\n")

    return result


if __name__ == "__main__":
    demonstrate_backtesting()
