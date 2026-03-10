"""
=============================================================================
MODULE 16: Multi-Agent RL for Market Simulation
=============================================================================

THEORY:
-------
Financial markets are inherently multi-agent systems. Prices emerge from
the interactions of many agents with different strategies, horizons,
and information.

WHY MULTI-AGENT RL (MARL) IN FINANCE?
  1. MARKET IMPACT: Agent's actions affect the market (and other agents)
  2. GAME THEORY: Trading is competitive; one agent's gain is another's loss
  3. EMERGENT DYNAMICS: Complex market behavior (bubbles, crashes, flash crashes)
     emerges from simple agent interactions
  4. STRATEGY ROBUSTNESS: Testing against diverse opponents reveals weaknesses

TYPES OF MARL:
  1. COOPERATIVE: Agents work together (e.g., multi-strategy fund)
  2. COMPETITIVE: Zero-sum (e.g., trading against other agents)
  3. MIXED: General-sum (most realistic for markets)

KEY CONCEPTS:
  - Nash Equilibrium: No agent can improve by unilateral deviation
  - Best Response: Optimal policy given fixed opponent policies
  - Population Dynamics: How strategy distributions evolve over time
  - Market Making: Providing liquidity as a strategic interaction

CHALLENGES:
  - Non-stationarity: Environment changes as agents learn
  - Scalability: Exponential state/action space with more agents
  - Credit Assignment: Who caused the market move?
  - Equilibrium Selection: Multiple equilibria may exist
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Order:
    """Represents a market order."""
    agent_id: int
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: int
    order_type: str = 'limit'  # 'limit' or 'market'


@dataclass
class AgentState:
    """State of an individual agent."""
    cash: float
    inventory: float
    realized_pnl: float
    unrealized_pnl: float
    order_count: int = 0


class OrderBook:
    """
    Simple limit order book for multi-agent market simulation.

    The order book is the mechanism through which agents interact.
    It aggregates all buy (bid) and sell (ask) orders and matches them.
    """

    def __init__(self):
        self.bids: List[Order] = []  # buy orders, sorted by price desc
        self.asks: List[Order] = []  # sell orders, sorted by price asc
        self.trade_history: List[Dict] = []

    def add_order(self, order: Order) -> List[Dict]:
        """Add an order and return list of trades (matches)."""
        trades = []

        if order.side == 'buy':
            # Match against asks
            while self.asks and order.quantity > 0:
                best_ask = self.asks[0]
                if order.order_type == 'market' or order.price >= best_ask.price:
                    traded_qty = min(order.quantity, best_ask.quantity)
                    trade = {
                        'buyer': order.agent_id,
                        'seller': best_ask.agent_id,
                        'price': best_ask.price,
                        'quantity': traded_qty,
                        'timestamp': order.timestamp,
                    }
                    trades.append(trade)
                    self.trade_history.append(trade)

                    order.quantity -= traded_qty
                    best_ask.quantity -= traded_qty

                    if best_ask.quantity <= 0:
                        self.asks.pop(0)
                else:
                    break

            if order.quantity > 0 and order.order_type == 'limit':
                self.bids.append(order)
                self.bids.sort(key=lambda x: -x.price)

        elif order.side == 'sell':
            while self.bids and order.quantity > 0:
                best_bid = self.bids[0]
                if order.order_type == 'market' or order.price <= best_bid.price:
                    traded_qty = min(order.quantity, best_bid.quantity)
                    trade = {
                        'buyer': best_bid.agent_id,
                        'seller': order.agent_id,
                        'price': best_bid.price,
                        'quantity': traded_qty,
                        'timestamp': order.timestamp,
                    }
                    trades.append(trade)
                    self.trade_history.append(trade)

                    order.quantity -= traded_qty
                    best_bid.quantity -= traded_qty

                    if best_bid.quantity <= 0:
                        self.bids.pop(0)
                else:
                    break

            if order.quantity > 0 and order.order_type == 'limit':
                self.asks.append(order)
                self.asks.sort(key=lambda x: x.price)

        return trades

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        elif self.bids:
            return self.bids[0].price
        elif self.asks:
            return self.asks[0].price
        return 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return float('inf')


class TradingAgent:
    """Base class for trading agents in the multi-agent simulation."""

    def __init__(self, agent_id: int, initial_cash: float = 100000.0):
        self.agent_id = agent_id
        self.state = AgentState(
            cash=initial_cash,
            inventory=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
        )

    def decide(self, market_state: Dict) -> Optional[Order]:
        raise NotImplementedError

    def update(self, trades: List[Dict], current_price: float):
        """Update agent state based on executed trades."""
        for trade in trades:
            if trade['buyer'] == self.agent_id:
                self.state.cash -= trade['price'] * trade['quantity']
                self.state.inventory += trade['quantity']
            elif trade['seller'] == self.agent_id:
                self.state.cash += trade['price'] * trade['quantity']
                self.state.inventory -= trade['quantity']
            self.state.order_count += 1

        self.state.unrealized_pnl = self.state.inventory * current_price - \
            abs(self.state.inventory) * current_price  # simplified


class MomentumAgent(TradingAgent):
    """Agent that follows price momentum."""

    def __init__(self, agent_id: int, lookback: int = 20, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.lookback = lookback
        self.price_history: List[float] = []

    def decide(self, market_state: Dict) -> Optional[Order]:
        price = market_state['mid_price']
        self.price_history.append(price)

        if len(self.price_history) < self.lookback:
            return None

        momentum = (price - self.price_history[-self.lookback]) / self.price_history[-self.lookback]

        if momentum > 0.01 and self.state.inventory < 100:
            return Order(
                agent_id=self.agent_id,
                side='buy',
                quantity=10,
                price=price * 1.001,
                timestamp=market_state['timestamp'],
                order_type='limit',
            )
        elif momentum < -0.01 and self.state.inventory > -100:
            return Order(
                agent_id=self.agent_id,
                side='sell',
                quantity=10,
                price=price * 0.999,
                timestamp=market_state['timestamp'],
                order_type='limit',
            )
        return None


class MeanReversionAgent(TradingAgent):
    """Agent that trades mean reversion."""

    def __init__(self, agent_id: int, lookback: int = 50, threshold: float = 1.5, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.lookback = lookback
        self.threshold = threshold
        self.price_history: List[float] = []

    def decide(self, market_state: Dict) -> Optional[Order]:
        price = market_state['mid_price']
        self.price_history.append(price)

        if len(self.price_history) < self.lookback:
            return None

        recent = self.price_history[-self.lookback:]
        mean = np.mean(recent)
        std = np.std(recent)
        z_score = (price - mean) / (std + 1e-8)

        if z_score < -self.threshold and self.state.inventory < 100:
            return Order(
                agent_id=self.agent_id,
                side='buy',
                quantity=10,
                price=price * 1.002,
                timestamp=market_state['timestamp'],
            )
        elif z_score > self.threshold and self.state.inventory > -100:
            return Order(
                agent_id=self.agent_id,
                side='sell',
                quantity=10,
                price=price * 0.998,
                timestamp=market_state['timestamp'],
            )
        return None


class MarketMaker(TradingAgent):
    """
    Market maker agent that provides liquidity.

    Market makers profit from the bid-ask spread while managing
    inventory risk. They quote both buy and sell prices.
    """

    def __init__(self, agent_id: int, spread_bps: float = 10, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.spread_bps = spread_bps

    def decide(self, market_state: Dict) -> List[Order]:
        price = market_state['mid_price']
        half_spread = price * self.spread_bps / 10000

        # Skew quotes based on inventory
        inventory_skew = -self.state.inventory * 0.001

        orders = [
            Order(
                agent_id=self.agent_id,
                side='buy',
                quantity=5,
                price=price - half_spread + inventory_skew,
                timestamp=market_state['timestamp'],
            ),
            Order(
                agent_id=self.agent_id,
                side='sell',
                quantity=5,
                price=price + half_spread + inventory_skew,
                timestamp=market_state['timestamp'],
            ),
        ]
        return orders


class MultiAgentMarketSimulator:
    """
    Multi-agent market simulation environment.

    Simulates a market with heterogeneous agents interacting through
    a central limit order book. Emergent price dynamics arise from
    the collective behavior of agents.
    """

    def __init__(self, fundamental_price: float = 100.0, volatility: float = 0.001):
        self.fundamental_price = fundamental_price
        self.volatility = volatility
        self.order_book = OrderBook()
        self.agents: List[TradingAgent] = []
        self.price_history: List[float] = [fundamental_price]
        self.timestamp = 0

    def add_agent(self, agent: TradingAgent):
        self.agents.append(agent)

    def step(self) -> Dict:
        """Run one simulation step."""
        self.timestamp += 1

        # Update fundamental with noise
        self.fundamental_price *= np.exp(
            np.random.normal(0, self.volatility)
        )

        market_state = {
            'mid_price': self.order_book.mid_price if self.order_book.mid_price > 0 else self.fundamental_price,
            'spread': self.order_book.spread,
            'fundamental': self.fundamental_price,
            'timestamp': self.timestamp,
        }

        all_trades = []

        # Each agent makes decisions
        for agent in self.agents:
            decision = agent.decide(market_state)

            if decision is None:
                continue

            if isinstance(decision, list):
                for order in decision:
                    trades = self.order_book.add_order(order)
                    all_trades.extend(trades)
            else:
                trades = self.order_book.add_order(decision)
                all_trades.extend(trades)

        # Update all agents
        current_price = self.order_book.mid_price if self.order_book.mid_price > 0 else self.fundamental_price
        for agent in self.agents:
            agent.update(all_trades, current_price)

        self.price_history.append(current_price)

        return {
            'price': current_price,
            'fundamental': self.fundamental_price,
            'spread': self.order_book.spread,
            'n_trades': len(all_trades),
            'timestamp': self.timestamp,
        }

    def run_simulation(self, n_steps: int = 1000) -> Dict:
        """Run the full simulation."""
        results = []
        for _ in range(n_steps):
            step_result = self.step()
            results.append(step_result)

        return {
            'price_history': self.price_history,
            'step_results': results,
            'agent_states': {a.agent_id: a.state for a in self.agents},
        }


def demonstrate_multi_agent():
    """Run a multi-agent market simulation."""
    print("=" * 70)
    print("  CHAPTER 16: MULTI-AGENT RL IN FINANCIAL MARKETS")
    print("=" * 70)

    sim = MultiAgentMarketSimulator(fundamental_price=100.0, volatility=0.001)

    sim.add_agent(MomentumAgent(0, lookback=20))
    sim.add_agent(MomentumAgent(1, lookback=50))
    sim.add_agent(MeanReversionAgent(2, lookback=30))
    sim.add_agent(MeanReversionAgent(3, lookback=60))
    sim.add_agent(MarketMaker(4, spread_bps=10))
    sim.add_agent(MarketMaker(5, spread_bps=15))

    print(f"\nMarket Simulation: 6 agents, 1000 steps")
    print(f"  Momentum agents:     2")
    print(f"  Mean-reversion:      2")
    print(f"  Market makers:       2")

    results = sim.run_simulation(n_steps=1000)

    prices = results['price_history']
    print(f"\n--- Simulation Results ---")
    print(f"  Start price:     ${prices[0]:.2f}")
    print(f"  End price:       ${prices[-1]:.2f}")
    print(f"  Price range:     ${min(prices):.2f} - ${max(prices):.2f}")
    print(f"  Volatility:      {np.std(np.diff(prices) / prices[:-1]) * np.sqrt(252) * 100:.1f}%")

    print(f"\n--- Agent P&L ---")
    agent_types = ["Momentum-20", "Momentum-50", "MeanRev-30", "MeanRev-60", "MM-10", "MM-15"]
    for agent, name in zip(sim.agents, agent_types):
        total_value = agent.state.cash + agent.state.inventory * prices[-1]
        pnl = total_value - 100000
        print(f"  {name:>14s}: P&L = ${pnl:8.2f}  Inventory = {agent.state.inventory:6.0f}  Trades = {agent.state.order_count}")

    print("\nKey Insights:")
    print("  1. Market dynamics emerge from agent interactions")
    print("  2. Momentum and mean-reversion naturally compete")
    print("  3. Market makers profit from the spread")
    print("  4. RL can learn optimal strategies against this population\n")

    return results


if __name__ == "__main__":
    demonstrate_multi_agent()
