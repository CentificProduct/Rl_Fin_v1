# Financial RL: Scenarios and Verifiers for UI Integration

**Prepared by:** Supreeth Mysore
**For:** Kausalya Rani (UI Team), Ananya, Abhishek
**Date:** March 2026

---

## Company Context

**Company Type:** Single Stock Trading Company
**Three Workflows:** Stock Trading, Portfolio Allocation, Options Pricing & Hedging

---

## Workflow 1: Stock Trading

### Scenarios (Tasks)

| Scenario ID | Scenario Name | Description | Difficulty | Steps |
|-------------|--------------|-------------|------------|-------|
| ST-001 | **Buy the Dip** | RSI drops below 0.30 (oversold). Agent must identify the buying opportunity, enter a position, and exit when RSI normalizes. | Easy | 3-5 |
| ST-002 | **Trend Following** | 5-day MA crosses above 20-day MA (golden cross). Agent must enter long position and ride the trend until reversal signal. | Easy | 4-6 |
| ST-003 | **Mean Reversion Trade** | Price deviates >2 standard deviations from 20-day mean (Bollinger breakout). Agent must fade the move and capture reversion. | Medium | 3-5 |
| ST-004 | **Momentum Scalping** | Strong 10-day momentum (>3%). Agent must take quick directional trades and manage position sizing with transaction costs. | Medium | 5-8 |
| ST-005 | **Volatile Market Navigation** | Volatility spikes above 30% annualized. Agent must reduce exposure, avoid excessive trading, and preserve capital. | Medium | 5-10 |
| ST-006 | **Drawdown Recovery** | Portfolio has declined 15% from peak. Agent must recover without increasing risk. No aggressive positions allowed. | Hard | 10-20 |
| ST-007 | **Regime Change Adaptation** | Market switches from bull to bear mid-episode. Agent must detect the shift and reverse from long to short/cash. | Hard | 8-15 |
| ST-008 | **Cost-Aware Trading** | High transaction cost environment (50 bps). Agent must minimize trade frequency while still capturing directional moves. | Hard | 5-10 |
| ST-009 | **Full Episode Optimization** | Complete trading episode from start to finish (~500 steps). Maximize Sharpe ratio over the full period. | Hard | 400-500 |
| ST-010 | **Earnings Shock Response** | Sudden 5% price gap (simulated). Agent must decide: hold through, cut losses, or add to position. | Expert | 3-5 |

### Verifiers for Stock Trading

| Verifier ID | Verifier Name | Check | Threshold | Type |
|-------------|--------------|-------|-----------|------|
| STV-001 | **Return Threshold** | Total return >= minimum | >= -25% | Outcome |
| STV-002 | **Sharpe Ratio Check** | Annualized Sharpe ratio acceptable | >= -1.0 | Outcome |
| STV-003 | **Max Drawdown Limit** | Peak-to-trough drawdown within limit | <= 40% | Risk |
| STV-004 | **Position Limit** | Position fraction within allowed range | abs(position) <= 100% | Constraint |
| STV-005 | **Trade Frequency** | Trades per episode within bounds | <= 2x episode length | Efficiency |
| STV-006 | **Transaction Cost Budget** | Total costs within budget | <= 5% of initial capital | Cost |
| STV-007 | **Action Sequence Valid** | Actions are valid integers (0-4) | All actions in {0,1,2,3,4} | Format |
| STV-008 | **No Ruin** | Portfolio never hits ruin threshold | Portfolio value > 50% of initial | Survival |
| STV-009 | **Reward Consistency** | Average reward is not degenerate | abs(mean reward) < 100 | Sanity |
| STV-010 | **Episode Completion** | Agent completes the full episode | terminated or truncated = true | Completion |

---

## Workflow 2: Portfolio Allocation

### Scenarios (Tasks)

| Scenario ID | Scenario Name | Description | Difficulty | Steps |
|-------------|--------------|-------------|------------|-------|
| PA-001 | **Equal Weight Baseline** | Maintain equal weights [0.20, 0.20, 0.20, 0.20, 0.20] across 5 assets. Baseline for comparison. | Easy | 100-200 |
| PA-002 | **Concentrated Bet** | Identify the best-performing asset and allocate 80%+ to it. High risk, high reward. | Easy | 50-100 |
| PA-003 | **Risk Parity Allocation** | Allocate inversely proportional to each asset's volatility. Lower-vol assets get more weight. | Medium | 100-200 |
| PA-004 | **Correlation Regime Shift** | Asset correlations change mid-episode (e.g., from 0.3 to 0.8). Agent must detect and diversify accordingly. | Medium | 150-250 |
| PA-005 | **Low-Turnover Optimization** | Maximize return while keeping monthly turnover below 20%. Transaction costs are elevated (20 bps). | Medium | 100-200 |
| PA-006 | **Tail Risk Hedging** | One asset has fat-tailed returns (Student-t, df=3). Agent must limit exposure to the risky asset. | Hard | 100-200 |
| PA-007 | **Multi-Objective: Return + ESG** | Maximize return while keeping allocation to "green" assets (assets 1,2) above 40%. | Hard | 100-200 |
| PA-008 | **Crisis Allocation** | Bear market across all assets. Agent must find the least-bad allocation and minimize losses. | Hard | 100-200 |
| PA-009 | **Dynamic Rebalancing** | Full episode with monthly rebalancing. Agent adapts weights based on rolling 60-day metrics. | Expert | 200-400 |
| PA-010 | **Markowitz vs RL** | Benchmark scenario: compare RL weights against mean-variance optimal on same data. | Expert | 200-400 |

### Verifiers for Portfolio Allocation

| Verifier ID | Verifier Name | Check | Threshold | Type |
|-------------|--------------|-------|-----------|------|
| PAV-001 | **Weights Sum to 1** | Portfolio weights sum to approximately 1.0 | abs(sum - 1.0) < 0.01 | Constraint |
| PAV-002 | **Non-Negative Weights** | All weights are non-negative (long-only) | All weights >= 0 | Constraint |
| PAV-003 | **Max Single Asset** | No single asset exceeds concentration limit | max(weight) <= 80% | Risk |
| PAV-004 | **Portfolio Return** | Total portfolio return above minimum | >= -30% | Outcome |
| PAV-005 | **Sharpe Ratio** | Risk-adjusted return acceptable | >= -1.5 | Outcome |
| PAV-006 | **Max Drawdown** | Drawdown within limit | <= 50% | Risk |
| PAV-007 | **Turnover Budget** | Total turnover across episode within limit | <= 500% cumulative | Cost |
| PAV-008 | **Diversification** | Effective number of assets > minimum | Herfindahl index < 0.5 | Quality |
| PAV-009 | **Cost Efficiency** | Transaction costs within budget | Total costs <= 3% of initial | Cost |
| PAV-010 | **Episode Completion** | Agent completes the full allocation period | Status = completed | Completion |

---

## Workflow 3: Options Pricing & Hedging

### Scenarios (Tasks)

| Scenario ID | Scenario Name | Description | Difficulty | Steps |
|-------------|--------------|-------------|------------|-------|
| OP-001 | **Full Delta Hedge** | Maintain hedge ratio = Black-Scholes delta at every step. Baseline strategy. | Easy | 20-30 |
| OP-002 | **Static Hedge** | Set hedge ratio once and hold. No rebalancing. Compare P&L variance against dynamic. | Easy | 20-30 |
| OP-003 | **Cost-Aware Hedging** | Transaction cost = 50 bps. Agent must hedge less frequently to save costs while managing risk. | Medium | 20-30 |
| OP-004 | **Stochastic Volatility** | Implied volatility follows Heston dynamics (mean-reverting with vol-of-vol). Agent must adapt hedge ratio to changing vol. | Medium | 20-30 |
| OP-005 | **Deep In-the-Money** | Option is deep ITM (S/K = 1.20). Delta is near 1.0. Agent must maintain tight hedge. | Medium | 20-30 |
| OP-006 | **Deep Out-of-Money** | Option is deep OTM (S/K = 0.80). Delta is near 0. Agent must avoid over-hedging. | Medium | 20-30 |
| OP-007 | **Gamma Scalping** | Near expiry (T < 5 days), gamma spikes. Agent must dynamically adjust hedge as delta changes rapidly. | Hard | 5-10 |
| OP-008 | **Volatility Spike** | Sudden vol increase from 20% to 50% mid-episode. Agent must react by increasing hedge aggressively. | Hard | 20-30 |
| OP-009 | **Multi-Option Book** | Hedge a portfolio of options with different strikes. Agent manages net delta across the book. | Expert | 20-30 |
| OP-010 | **P&L Variance Minimization** | Full hedging episode. Minimize the variance of the daily P&L, not just terminal P&L. | Expert | 20-30 |

### Verifiers for Options Hedging

| Verifier ID | Verifier Name | Check | Threshold | Type |
|-------------|--------------|-------|-----------|------|
| OPV-001 | **Hedge Ratio Bounds** | Hedge ratio within valid range | -0.5 <= ratio <= 1.5 | Constraint |
| OPV-002 | **P&L Above Minimum** | Total hedging P&L above floor | P&L >= -$50,000 | Outcome |
| OPV-003 | **Hedging Error** | Hedge ratio close to BS delta on average | avg abs(hedge - delta) < 0.5 | Quality |
| OPV-004 | **P&L Variance** | Variance of daily P&L below threshold | Variance < $10,000^2 | Risk |
| OPV-005 | **Transaction Costs** | Total hedging costs within budget | Costs <= 10% of option premium | Cost |
| OPV-006 | **Smooth Hedging** | Hedge ratio changes are not erratic | avg abs(delta_hedge) < 0.3 per step | Quality |
| OPV-007 | **Terminal Payoff Correct** | At expiry, final P&L reflects correct option payoff | abs(P&L - theoretical) < $5,000 | Accuracy |
| OPV-008 | **No Naked Exposure** | Agent never leaves position fully unhedged for >3 steps | Consecutive unhedged steps <= 3 | Risk |
| OPV-009 | **Greeks Awareness** | Agent adjusts more when gamma is high (near expiry/ATM) | Correlation(gamma, hedge_change) > 0 | Quality |
| OPV-010 | **Episode Completion** | Agent completes through option expiry | All steps executed | Completion |

---

## Summary for UI Integration

### Total Counts

| Component | Stock Trading | Portfolio Allocation | Options Hedging | Total |
|-----------|:------------:|:-------------------:|:---------------:|:-----:|
| **Scenarios** | 10 | 10 | 10 | **30** |
| **Verifiers** | 10 | 10 | 10 | **30** |

### Verifier Categories

| Category | Description | Examples |
|----------|-------------|---------|
| **Outcome** | End-of-episode performance metrics | Return >= -25%, Sharpe >= -1.0 |
| **Risk** | Risk limit checks | Max drawdown <= 40%, P&L variance |
| **Constraint** | Hard constraints on actions/state | Weights sum to 1, position limits |
| **Cost** | Transaction cost budgets | Total costs <= 5% of capital |
| **Quality** | Behavioral quality metrics | Diversification, hedging smoothness |
| **Completion** | Episode lifecycle checks | Agent completed all steps |
| **Format** | Data format validation | Actions are valid integers |
| **Sanity** | Degenerate behavior detection | Rewards not stuck at extreme values |

### API Endpoints (Already Implemented)

```
POST /envs/create           -- Create environment with scenario config
POST /envs/{id}/step        -- Execute one step (returns verifier_result)
POST /envs/{id}/train       -- Train RL agent (DQN for stock, PPO for portfolio/options)
POST /envs/{id}/agent-step  -- Step using trained agent
GET  /envs/{id}/rollout     -- Get rollout with all steps + verifier results
GET  /rollouts              -- List all completed rollouts
GET  /rollouts/{id}         -- Full rollout detail (for HIL review)
```

### Verifier Response Format (per step)

```json
{
  "verifier_result": {
    "verifier_type": "financial",
    "enabled": true,
    "score": 0.67,
    "checks": [
      {"name": "total_return", "passed": true, "value": 0.05},
      {"name": "max_drawdown", "passed": true, "value": 0.12},
      {"name": "sharpe_ratio", "passed": false, "value": -0.5}
    ],
    "reward_observed": 0.023
  }
}
```

### How Scenarios Map to Training

```
Scenario selected in UI
    |
    v
Environment created with scenario-specific config
(e.g., ST-005 sets volatility to high, PA-006 sets fat-tailed returns)
    |
    v
Agent trains on this scenario (DQN or PPO)
    |
    v
Rollout recorded with per-step verifier checks
    |
    v
HIL reviewer sees: rollout state diagram + verifier pass/fail + metrics
```
