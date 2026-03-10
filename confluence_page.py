"""
===========================================================================
Confluence Page Publisher (Atlassian Cloud)
===========================================================================
Creates or updates the "RL-Financial Env" subpage under the RL Env Studio
parent page on Confluence Cloud.

Usage:
    python confluence_page.py --email you@company.com --token YOUR_API_TOKEN
    python confluence_page.py --dry-run

    # Or via environment variables:
    set CONFLUENCE_EMAIL=you@company.com
    set CONFLUENCE_TOKEN=YOUR_API_TOKEN
    python confluence_page.py

Get your API token at:
    https://id.atlassian.com/manage-profile/security/api-tokens

Requires: requests  (pip install requests)
===========================================================================
"""

import argparse
import base64
import json
import os
import sys
import textwrap

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required.  Install with:  pip install requests")
    sys.exit(1)

# -------------------------------------------------------------------------
# Confluence target
# -------------------------------------------------------------------------
CONFLUENCE_BASE = "https://centific-idc.atlassian.net"
SPACE_KEY = "CAR"
PARENT_PAGE_ID = "832897089"
PAGE_TITLE = "RL-Financial Env: Reinforcement Learning for Quantitative Finance"

# -------------------------------------------------------------------------
# Page content in Confluence Storage Format (XHTML)
# -------------------------------------------------------------------------

PAGE_BODY = textwrap.dedent(r"""
<ac:structured-macro ac:name="toc">
  <ac:parameter ac:name="maxLevel">2</ac:parameter>
</ac:structured-macro>

<h1>RL-Financial Env: Reinforcement Learning for Quantitative Finance</h1>

<ac:structured-macro ac:name="info">
  <ac:rich-text-body>
    <p><strong>Status:</strong> Research Demo &mdash; Ready to Run</p>
    <p><strong>Runtime:</strong> All demos run on CPU with synthetic data. No API keys, no GPU, no external data needed.</p>
    <p><strong>Stack:</strong> Python 3.10+ &middot; PyTorch &middot; Gymnasium &middot; Stable-Baselines3</p>
  </ac:rich-text-body>
</ac:structured-macro>

<h2>1. Overview</h2>

<p>This environment demonstrates how <strong>Reinforcement Learning (RL)</strong> can be applied to core problems in <strong>quantitative finance</strong>: stock trading, portfolio allocation, options hedging, and risk management.</p>

<p>The project contains <strong>21 research modules</strong>, <strong>3 Gymnasium environments</strong>, <strong>5 RL algorithm implementations</strong>, a <strong>15-strategy benchmark suite</strong>, and <strong>4 ready-to-run team demos</strong>.</p>

<h3>Why RL for Finance?</h3>

<table>
  <thead>
    <tr><th>Aspect</th><th>Traditional Approach</th><th>RL Approach</th></tr>
  </thead>
  <tbody>
    <tr><td>Decision making</td><td>Static rules</td><td>Adaptive, learned policies</td></tr>
    <tr><td>Market model</td><td>Assumed (GBM, etc.)</td><td>Learned from data</td></tr>
    <tr><td>Risk management</td><td>Separate overlay</td><td>Integrated into objective</td></tr>
    <tr><td>Transaction costs</td><td>Post-hoc adjustment</td><td>Part of the optimization</td></tr>
    <tr><td>Non-stationarity</td><td>Periodic recalibration</td><td>Continuous adaptation</td></tr>
    <tr><td>Multi-period</td><td>Myopic approximation</td><td>Native sequential optimization</td></tr>
  </tbody>
</table>

<h2>2. Architecture</h2>

<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">text</ac:parameter>
  <ac:parameter ac:name="title">Project Structure</ac:parameter>
  <ac:plain-text-body><![CDATA[financial_ai_rl/
|
|-- core/                        # RL Foundations (MDP, value functions, Bellman)
|-- environments/                # Gymnasium Environments (trading, portfolio, options)
|-- agents/                      # RL Algorithms (Q-Learning, DQN, REINFORCE, PPO, A2C)
|-- portfolio/                   # Portfolio Optimization (RL + Markowitz)
|-- risk/                        # Risk Management (VaR, CVaR, reward shaping)
|-- advanced/                    # Multi-Agent RL, Model-Based RL, Inverse RL
|-- evaluation/                  # Backtesting, Metrics (20+), Visualization
|-- benchmarks/                  # Extensible Benchmark Framework (15 strategies)
|-- demos/                       # Ready-to-Run Team Demos (4 scripts)
|-- utils/                       # Data loaders, feature engineering, replay buffers
|-- main.py                      # Run all chapter demonstrations
|-- requirements.txt             # Dependencies
]]></ac:plain-text-body>
</ac:structured-macro>

<h3>Data Flow</h3>

<table>
  <thead>
    <tr><th>Layer</th><th>Components</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Data Sources</strong></td><td>Synthetic GBM, Yahoo Finance, CSV</td><td>Regime-switching price generation or real market data</td></tr>
    <tr><td><strong>Feature Engineering</strong></td><td>12 technical indicators per asset</td><td>RSI, MACD, Bollinger, momentum, volatility, moving averages</td></tr>
    <tr><td><strong>Environments</strong></td><td>3 Gymnasium envs</td><td>Stock trading, portfolio allocation, options hedging</td></tr>
    <tr><td><strong>Agents</strong></td><td>5 custom + 3 SB3 + 6 classical + 3 baselines</td><td>Train on env, evaluated out-of-sample</td></tr>
    <tr><td><strong>Evaluation</strong></td><td>Backtester + 20 metrics + visualization</td><td>Walk-forward, Monte Carlo bootstrap, formatted reports</td></tr>
  </tbody>
</table>

<h2>3. MDP-to-Finance Mapping</h2>

<p>The Markov Decision Process (MDP) framework maps naturally to financial trading:</p>

<table>
  <thead>
    <tr><th>MDP Component</th><th>Financial Interpretation</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>State <em>s</em></strong></td><td>Market features (prices, volumes, indicators) + portfolio positions + risk metrics</td></tr>
    <tr><td><strong>Action <em>a</em></strong></td><td>Trade execution: buy/sell/hold, position sizing, portfolio weight allocation</td></tr>
    <tr><td><strong>Transition <em>P(s'|s,a)</em></strong></td><td>Market dynamics, regime changes, impact of agent's own trades</td></tr>
    <tr><td><strong>Reward <em>R</em></strong></td><td>Differential Sharpe ratio, Sortino, utility, risk-penalized returns</td></tr>
    <tr><td><strong>Discount <em>&gamma;</em></strong></td><td>Time value of money and planning horizon</td></tr>
  </tbody>
</table>

<h2>4. Environments</h2>

<table>
  <thead>
    <tr><th>Environment</th><th>State Dim</th><th>Action Space</th><th>Reward Options</th><th>Key Features</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Stock Trading</strong></td>
      <td>16</td>
      <td>5 discrete or continuous [-1,1]</td>
      <td>Simple, Sharpe, Sortino, Calmar</td>
      <td>Transaction costs, slippage, drawdown termination</td>
    </tr>
    <tr>
      <td><strong>Portfolio Allocation</strong></td>
      <td>N*20 + N + 4</td>
      <td>Continuous (N weights, sum=1)</td>
      <td>Return, Sharpe, CRRA utility</td>
      <td>N correlated assets, cost-aware rebalancing</td>
    </tr>
    <tr>
      <td><strong>Options Hedging</strong></td>
      <td>7</td>
      <td>Continuous [-0.5, 1.5]</td>
      <td>Hedging error minimization</td>
      <td>Black-Scholes benchmark, stochastic vol</td>
    </tr>
  </tbody>
</table>

<p>All environments follow the <strong>Gymnasium API</strong> (<code>reset()</code>, <code>step()</code>, <code>observation_space</code>, <code>action_space</code>) and are directly compatible with Stable-Baselines3.</p>

<h2>5. RL Algorithms Implemented</h2>

<table>
  <thead>
    <tr><th>Algorithm</th><th>Type</th><th>Action Space</th><th>Key Innovation</th><th>Best For</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Q-Learning</strong></td><td>Value-based, Off-policy</td><td>Discrete</td><td>Tabular + Double Q variant</td><td>Simple discrete trading</td></tr>
    <tr><td><strong>DQN</strong></td><td>Value-based, Off-policy</td><td>Discrete</td><td>Double + Dueling + Experience Replay</td><td>Feature-rich trading</td></tr>
    <tr><td><strong>REINFORCE</strong></td><td>Policy-based, On-policy</td><td>Both</td><td>Learned baseline + entropy reg.</td><td>Research baseline</td></tr>
    <tr><td><strong>PPO</strong></td><td>Actor-Critic, On-policy</td><td>Both</td><td>Clipped objective + GAE</td><td>Portfolio allocation</td></tr>
    <tr><td><strong>A2C</strong></td><td>Actor-Critic, On-policy</td><td>Both</td><td>N-step returns</td><td>Real-time trading</td></tr>
  </tbody>
</table>

<p>Additionally, <strong>Model-Based RL</strong> (world model ensembles + MPC), <strong>Multi-Agent RL</strong> (limit order book simulation), and <strong>Inverse RL</strong> (recovering hidden reward functions) are implemented in the <code>advanced/</code> module.</p>

<h2>6. Benchmark Framework</h2>

<p>The <code>benchmarks/</code> module provides a <strong>plug-in registry</strong> with 15 pre-built strategies evaluated head-to-head on out-of-sample data:</p>

<table>
  <thead>
    <tr><th>Category</th><th>Strategies</th><th>Training?</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Baseline</strong> (3)</td><td>Random, Buy &amp; Hold, Always Cash</td><td>No</td></tr>
    <tr><td><strong>Classical Quant</strong> (6)</td><td>SMA Crossover, RSI Mean-Reversion, Bollinger Breakout, Momentum, MACD Signal, Volatility Regime</td><td>No</td></tr>
    <tr><td><strong>Open-Source RL</strong> (3)</td><td>Stable-Baselines3 PPO, A2C, DQN</td><td>Yes</td></tr>
    <tr><td><strong>Custom RL</strong> (3)</td><td>Our DQN (Double+Dueling), Our PPO, Our Q-Learning</td><td>Yes</td></tr>
  </tbody>
</table>

<h3>Sample Benchmark Output</h3>

<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">text</ac:parameter>
  <ac:parameter ac:name="title">Out-of-Sample Results (synthetic data, 5 eval runs)</ac:parameter>
  <ac:plain-text-body><![CDATA[
+======================+==========+=========+=========+=======+==========+
| Strategy             |  Return  |  Sharpe | Max DD  | Trades|  Costs   |
+======================+==========+=========+=========+=======+==========+
|--- BASELINE                                                            |
| Random               |  -50.40% |  -2.921 |  50.46% |   500 | $ 31426  |
| Buy & Hold           |   +9.53% |   0.227 |   3.65% |     4 | $   100  |
| Always Cash          |   +0.00% |   0.000 |   0.00% |     0 | $     0  |
|--- CLASSICAL                                                           |
| RSI Mean-Reversion   |   +6.38% |   0.436 |   0.65% |   241 | $  6543  |
| SMA Crossover        |  -15.36% |  -0.429 |  18.18% |   315 | $ 19663  |
|--- OPEN-SOURCE RL                                                      |
| SB3-PPO              |   -2.23% |   0.045 |  12.95% |    37 | $  1670  |
| SB3-A2C              |   +9.53% |   0.227 |   3.65% |     4 | $   100  |
| SB3-DQN              |   -1.27% |   0.049 |   8.22% |   517 | $  5626  |
|--- CUSTOM RL                                                           |
| Custom DQN           |  -15.45% |  -0.213 |  29.66% |   540 | $ 11186  |
| Custom PPO           |  -50.42% |  -2.387 |  50.44% |   610 | $ 34092  |
+======================+==========+=========+=========+=======+==========+
]]></ac:plain-text-body>
</ac:structured-macro>

<h3>Data Sources</h3>

<table>
  <thead>
    <tr><th>Source</th><th>Command Flag</th><th>Requirement</th></tr>
  </thead>
  <tbody>
    <tr><td>Synthetic (GBM + regime switching)</td><td><code>--data synthetic</code> (default)</td><td>None</td></tr>
    <tr><td>Yahoo Finance</td><td><code>--data yahoo --ticker AAPL</code></td><td><code>pip install yfinance</code></td></tr>
    <tr><td>Local CSV</td><td><code>--data csv --csv file.csv</code></td><td>CSV with "Close" column</td></tr>
  </tbody>
</table>

<h2>7. How to Run</h2>

<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">bash</ac:parameter>
  <ac:parameter ac:name="title">Quick Start</ac:parameter>
  <ac:plain-text-body><![CDATA[# 1. Install dependencies
cd financial_ai_rl
pip install -r requirements.txt

# 2. Fastest demo (risk analysis, ~12 seconds)
python demos/demo_risk.py

# 3. Full benchmark: 15 strategies head-to-head (~3 min)
python benchmarks/run_benchmarks.py

# 4. Team presentation demo (~3 min)
python demos/quick_demo.py

# 5. Portfolio optimization: RL vs Markowitz vs Risk Parity (~30 sec)
python demos/demo_portfolio.py

# 6. Use real market data instead of synthetic
python benchmarks/run_benchmarks.py --data yahoo --ticker AAPL
]]></ac:plain-text-body>
</ac:structured-macro>

<h3>Team Demos</h3>

<table>
  <thead>
    <tr><th>Demo</th><th>Runtime</th><th>Command</th><th>What It Shows</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Risk Analysis</strong></td><td>~12s</td><td><code>python demos/demo_risk.py</code></td><td>Risk metrics across 6 scenarios, reward shaping, stress testing, curriculum learning</td></tr>
    <tr><td><strong>Portfolio Benchmark</strong></td><td>~30s</td><td><code>python demos/demo_portfolio.py</code></td><td>RL vs Equal Weight vs Markowitz vs Min Variance vs Risk Parity</td></tr>
    <tr><td><strong>Quick Demo</strong></td><td>~3 min</td><td><code>python demos/quick_demo.py</code></td><td>DQN vs Random, Q-Learning comparison, PPO curves, multi-agent sim, summary table</td></tr>
    <tr><td><strong>Full Benchmark</strong></td><td>~5 min</td><td><code>python benchmarks/run_benchmarks.py</code></td><td>All 15 strategies trained + evaluated out-of-sample</td></tr>
  </tbody>
</table>

<h2>8. Extending the Framework</h2>

<p>Add any new strategy with one decorator &mdash; it appears in every benchmark run automatically:</p>

<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">python</ac:parameter>
  <ac:parameter ac:name="title">Adding a Custom Strategy</ac:parameter>
  <ac:plain-text-body><![CDATA[from benchmarks.registry import BenchmarkRegistry, BaseStrategy

@BenchmarkRegistry.register("My Alpha Strategy", category="custom")
class MyAlpha(BaseStrategy):
    def train(self, env, config):
        # your training logic here
        pass

    def predict(self, obs, info=None):
        # return action (0-4 for discrete)
        return 4 if obs[8] > 0.02 else 2
]]></ac:plain-text-body>
</ac:structured-macro>

<p>No other files need to change. The runner discovers new strategies via the registry automatically.</p>

<h2>9. Key References</h2>

<table>
  <thead>
    <tr><th>Year</th><th>Paper</th><th>Contribution</th></tr>
  </thead>
  <tbody>
    <tr><td>1952</td><td>Markowitz &mdash; Portfolio Selection</td><td>Mean-variance framework</td></tr>
    <tr><td>2001</td><td>Moody &amp; Saffell &mdash; RL for Trading</td><td>Differential Sharpe ratio</td></tr>
    <tr><td>2013</td><td>Mnih et al. &mdash; DQN</td><td>Deep RL with experience replay</td></tr>
    <tr><td>2017</td><td>Schulman et al. &mdash; PPO</td><td>Clipped policy optimization</td></tr>
    <tr><td>2020</td><td>Yang et al. &mdash; FinRL</td><td>RL library for finance</td></tr>
    <tr><td>2021</td><td>Liu et al. &mdash; FinRL-Meta</td><td>Universe of market environments</td></tr>
  </tbody>
</table>
""").strip()


# -------------------------------------------------------------------------
# API helpers
# -------------------------------------------------------------------------

def find_existing_page(session: requests.Session, title: str, space_key: str):
    """Check if a page with this title already exists in the space."""
    url = f"{CONFLUENCE_BASE}/wiki/api/v2/spaces/{space_key}/pages"
    params = {"title": title, "limit": 5}
    resp = session.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        for page in data.get("results", []):
            if page.get("title") == title:
                return page.get("id")
    return None


def create_page(session: requests.Session, title: str, body: str,
                space_key: str, parent_id: str):
    """Create a new Confluence page as a child of parent_id."""
    url = f"{CONFLUENCE_BASE}/wiki/api/v2/pages"
    payload = {
        "spaceId": _get_space_id(session, space_key),
        "status": "current",
        "title": title,
        "parentId": parent_id,
        "body": {
            "representation": "storage",
            "value": body,
        },
    }
    resp = session.post(url, json=payload)
    return resp


def update_page(session: requests.Session, page_id: str, title: str,
                body: str):
    """Update an existing Confluence page."""
    # First get current version number
    url = f"{CONFLUENCE_BASE}/wiki/api/v2/pages/{page_id}"
    resp = session.get(url)
    resp.raise_for_status()
    current_version = resp.json()["version"]["number"]

    url = f"{CONFLUENCE_BASE}/wiki/api/v2/pages/{page_id}"
    payload = {
        "id": page_id,
        "status": "current",
        "title": title,
        "body": {
            "representation": "storage",
            "value": body,
        },
        "version": {
            "number": current_version + 1,
            "message": "Updated via confluence_page.py",
        },
    }
    resp = session.put(url, json=payload)
    return resp


def _get_space_id(session: requests.Session, space_key: str) -> str:
    """Resolve space key to space ID (required by v2 API)."""
    url = f"{CONFLUENCE_BASE}/wiki/api/v2/spaces"
    params = {"keys": space_key, "limit": 1}
    resp = session.get(url, params=params)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"Space '{space_key}' not found")
    return results[0]["id"]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Publish RL-Financial Env page to Confluence Cloud",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("CONFLUENCE_EMAIL", ""),
        help="Your Atlassian account email\n"
             "(or set CONFLUENCE_EMAIL env var)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CONFLUENCE_TOKEN", ""),
        help="Atlassian Cloud API token\n"
             "(or set CONFLUENCE_TOKEN env var)\n"
             "Create at: https://id.atlassian.com/manage-profile/security/api-tokens",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the page HTML to stdout without publishing",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update the page if it already exists (default: fail)",
    )
    args = parser.parse_args()

    # -- Dry run: just dump HTML ----------------------------------------
    if args.dry_run:
        print("=" * 70)
        print("  DRY RUN -- Confluence Storage Format HTML")
        print("=" * 70)
        print()
        print(PAGE_BODY)
        print()
        print("=" * 70)
        print(f"  Title:     {PAGE_TITLE}")
        print(f"  Space:     {SPACE_KEY}")
        print(f"  Parent ID: {PARENT_PAGE_ID}")
        print(f"  Length:    {len(PAGE_BODY)} characters")
        print("=" * 70)
        return

    # -- Validate credentials --------------------------------------------
    if not args.email or not args.token:
        print("ERROR: Both --email and --token are required for Confluence Cloud.")
        print()
        print("  python confluence_page.py --email you@company.com --token YOUR_API_TOKEN")
        print()
        print("  Or set environment variables:")
        print("    set CONFLUENCE_EMAIL=you@company.com")
        print("    set CONFLUENCE_TOKEN=YOUR_API_TOKEN")
        print()
        print("  Get your API token at:")
        print("    https://id.atlassian.com/manage-profile/security/api-tokens")
        print()
        print("  Use --dry-run to preview the HTML without publishing.")
        sys.exit(1)

    # -- Publish ---------------------------------------------------------
    # Atlassian Cloud uses Basic Auth: base64(email:api_token)
    credentials = base64.b64encode(f"{args.email}:{args.token}".encode()).decode()
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })

    print(f"  Target:  {CONFLUENCE_BASE}/wiki/spaces/{SPACE_KEY}")
    print(f"  Title:   {PAGE_TITLE}")
    print(f"  Parent:  {PARENT_PAGE_ID}")
    print()

    # Check if page already exists
    existing_id = find_existing_page(session, PAGE_TITLE, SPACE_KEY)

    if existing_id:
        if args.update:
            print(f"  Page exists (id={existing_id}). Updating...")
            resp = update_page(session, existing_id, PAGE_TITLE, PAGE_BODY)
        else:
            print(f"  Page already exists (id={existing_id}).")
            print(f"  Use --update to overwrite, or change the title.")
            page_url = f"{CONFLUENCE_BASE}/wiki/spaces/{SPACE_KEY}/pages/{existing_id}"
            print(f"  URL: {page_url}")
            return
    else:
        print("  Creating new page...")
        resp = create_page(session, PAGE_TITLE, PAGE_BODY,
                           SPACE_KEY, PARENT_PAGE_ID)

    if resp.status_code in (200, 201):
        page_data = resp.json()
        page_id = page_data.get("id", "?")
        page_url = f"{CONFLUENCE_BASE}/wiki/spaces/{SPACE_KEY}/pages/{page_id}"
        print()
        print("  SUCCESS!")
        print(f"  Page ID:  {page_id}")
        print(f"  URL:      {page_url}")
    else:
        print()
        print(f"  FAILED (HTTP {resp.status_code})")
        print(f"  Response: {resp.text[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
