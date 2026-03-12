"""
FastAPI server wrapping the Financial AI RL Gymnasium environments.

Exposes StockTradingEnv, PortfolioAllocationEnv, and OptionsPricingEnv
as HTTP endpoints compatible with RL Env Studio.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8090
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from environments.stock_trading_env import StockTradingEnv
from environments.portfolio_env import PortfolioAllocationEnv
from environments.options_pricing_env import OptionsPricingEnv
from utils.data_loader import FinancialDataLoader
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOTrader


ENV_REGISTRY = {
    "stock-trading": {
        "class": StockTradingEnv,
        "display_name": "Apex Equities Desk",
        "description": "Directional equity trading using momentum, mean-reversion, and regime detection signals",
        "tools": [
            {"name": "get_market_state", "description": "Fetch price, volume, and technical indicators"},
            {"name": "execute_trade", "description": "Buy/sell/hold with position sizing"},
            {"name": "get_portfolio_status", "description": "Current P&L, position, drawdown"},
        ],
        "observation_dim": 16,
        "action_type": "discrete",
        "action_dim": 5,
    },
    "portfolio-allocation": {
        "class": PortfolioAllocationEnv,
        "display_name": "Apex Multi-Asset Allocator",
        "description": "Dynamic portfolio weight optimization across 5 correlated asset classes with cost-aware rebalancing",
        "tools": [
            {"name": "get_asset_returns", "description": "Multi-asset return history with lookback window"},
            {"name": "rebalance_portfolio", "description": "Set target portfolio weights (sum to 1)"},
            {"name": "get_risk_metrics", "description": "Sharpe ratio, volatility, max drawdown"},
        ],
        "observation_dim": "n_assets * lookback + n_assets + 4",
        "action_type": "continuous",
        "action_dim": "n_assets",
    },
    "options-pricing": {
        "class": OptionsPricingEnv,
        "display_name": "Apex Derivatives Hedging",
        "description": "Intelligent delta hedging for options book with stochastic volatility and transaction cost optimization",
        "tools": [
            {"name": "get_option_greeks", "description": "Delta, gamma, and current hedge ratio"},
            {"name": "adjust_hedge", "description": "Set hedge ratio (0=none, 1=full delta)"},
            {"name": "get_pnl_status", "description": "Current P&L and hedging error variance"},
        ],
        "observation_dim": 7,
        "action_type": "continuous",
        "action_dim": 1,
    },
}

COMPANY_REGISTRY = {
    "apex-capital": {
        "id": "apex-capital",
        "name": "Apex Capital Management",
        "description": "Quantitative investment firm specializing in algorithmic trading, multi-asset portfolio management, and derivatives risk solutions",
        "icon": "building",
        "environments": ["stock-trading", "portfolio-allocation", "options-pricing"],
        "workflows": [
            {
                "id": "stock-trading",
                "name": "Apex Equities Desk",
                "description": "Directional equity trading using technical signals -- momentum, mean-reversion, and regime detection",
                "algorithm": "DQN (Double + Dueling)",
                "scenarios_count": 10,
                "verifiers_count": 10,
            },
            {
                "id": "portfolio-allocation",
                "name": "Apex Multi-Asset Allocator",
                "description": "Dynamic portfolio weight optimization across correlated asset classes with cost-aware rebalancing",
                "algorithm": "PPO (Continuous)",
                "scenarios_count": 10,
                "verifiers_count": 10,
            },
            {
                "id": "options-pricing",
                "name": "Apex Derivatives Hedging",
                "description": "Intelligent delta hedging for options book with stochastic volatility and transaction cost optimization",
                "algorithm": "PPO (Continuous)",
                "scenarios_count": 10,
                "verifiers_count": 10,
            },
        ],
    },
}

active_envs: dict[str, dict[str, Any]] = {}
rollout_store: dict[str, dict[str, Any]] = {}
server_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_start_time
    server_start_time = time.time()
    yield
    active_envs.clear()


app = FastAPI(
    title="Financial AI RL Gym Server",
    description="Gymnasium environment server for RL Env Studio",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def serve_testbed():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Request/Response Models ---

class CreateEnvRequest(BaseModel):
    env_type: str = Field(..., description="One of: stock-trading, portfolio-allocation, options-pricing")
    config: dict = Field(default_factory=dict, description="Optional environment configuration overrides")

class CreateEnvResponse(BaseModel):
    env_id: str
    env_type: str
    observation_shape: list[int]
    action_space_info: dict
    rollout_id: str

class StepRequest(BaseModel):
    action: Any = Field(..., description="Action to take (int for discrete, list[float] for continuous)")

class StepResponse(BaseModel):
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict

class ResetResponse(BaseModel):
    observation: list[float]
    info: dict

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    active_environments: int
    total_connections: int
    available_env_types: list[str]


class VerifierConfigRequest(BaseModel):
    verifier_type: str = Field(default="financial", description="Verifier type")
    enabled: bool = Field(default=True, description="Enable verifier checks")
    thresholds: dict = Field(default_factory=dict, description="Verifier thresholds")

class TrainRequest(BaseModel):
    algorithm: str = Field(default="dqn", description="RL algorithm: dqn")
    episodes: int = Field(default=15, ge=1, le=100, description="Number of training episodes")


# --- Helper ---

def _ndarray_to_list(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ndarray_to_list(v) for v in obj]
    return obj


def _to_iso_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _serialize_action(action: Any) -> Any:
    if isinstance(action, (int, float, str, bool)) or action is None:
        return action
    if isinstance(action, np.ndarray):
        return action.tolist()
    if isinstance(action, (list, tuple)):
        return [_serialize_action(v) for v in action]
    return str(action)


def _default_verifier_config(env_type: str) -> Dict[str, Any]:
    if env_type in ("stock-trading", "portfolio-allocation"):
        return {
            "verifier_type": "financial",
            "enabled": True,
            "thresholds": {
                "min_total_return": -0.25,
                "max_drawdown": 0.40,
                "min_sharpe_ratio": -1.0,
            },
        }
    return {
        "verifier_type": "financial",
        "enabled": True,
        "thresholds": {
            "min_pnl": -50000.0,
            "max_hedge_error": 0.5,
        },
    }


def _build_verifier_result(
    env_type: str, info: Dict[str, Any], reward: float, verifier_config: Dict[str, Any]
) -> Dict[str, Any]:
    if not verifier_config.get("enabled", True):
        return {
            "verifier_type": verifier_config.get("verifier_type", "financial"),
            "enabled": False,
            "score": 0.0,
            "checks": [],
        }

    thresholds = verifier_config.get("thresholds", {})
    checks = []

    if env_type in ("stock-trading", "portfolio-allocation"):
        total_return = float(info.get("total_return", 0.0))
        max_drawdown = float(info.get("max_drawdown", 0.0))
        sharpe_ratio = float(info.get("sharpe_ratio", 0.0))
        checks = [
            {
                "name": "total_return",
                "passed": total_return >= float(thresholds.get("min_total_return", -0.25)),
                "value": total_return,
            },
            {
                "name": "max_drawdown",
                "passed": max_drawdown <= float(thresholds.get("max_drawdown", 0.40)),
                "value": max_drawdown,
            },
            {
                "name": "sharpe_ratio",
                "passed": sharpe_ratio >= float(thresholds.get("min_sharpe_ratio", -1.0)),
                "value": sharpe_ratio,
            },
        ]
    else:
        pnl = float(info.get("pnl", 0.0))
        hedge_position = float(info.get("hedge_position", 0.0))
        bs_delta = float(info.get("bs_delta", 0.0))
        n_options = float(info.get("_n_options", 100.0))
        normalized_hedge = hedge_position / max(n_options, 1.0)
        hedge_error = abs(normalized_hedge - bs_delta)
        checks = [
            {
                "name": "pnl",
                "passed": pnl >= float(thresholds.get("min_pnl", -50000.0)),
                "value": pnl,
            },
            {
                "name": "hedge_error",
                "passed": hedge_error <= float(thresholds.get("max_hedge_error", 0.5)),
                "value": hedge_error,
            },
        ]

    passed_count = sum(1 for c in checks if c["passed"])
    score = float(passed_count) / float(len(checks) or 1)
    return {
        "verifier_type": verifier_config.get("verifier_type", "financial"),
        "enabled": True,
        "score": score,
        "checks": checks,
        "reward_observed": float(reward),
    }


def _build_env(env_type: str, config: dict):
    """Instantiate a Gymnasium environment with synthetic data."""
    entry = ENV_REGISTRY[env_type]
    cls = entry["class"]

    if env_type == "stock-trading":
        n_steps = config.get("n_steps", 500)
        data = FinancialDataLoader.generate_synthetic_data(len_data=n_steps)
        return cls(
            prices=data.prices,
            features=data.features,
            initial_balance=config.get("initial_balance", 100_000),
            discrete_actions=config.get("discrete_actions", True),
            reward_type=config.get("reward_type", "sharpe"),
        )
    elif env_type == "portfolio-allocation":
        n_assets = config.get("n_assets", 5)
        n_steps = config.get("n_steps", 500)
        prices = np.cumsum(np.random.randn(n_steps, n_assets) * 0.02, axis=0) + 100
        return cls(
            prices=prices,
            n_assets=n_assets,
            transaction_cost=config.get("transaction_cost", 0.001),
        )
    elif env_type == "options-pricing":
        return cls(
            S0=config.get("S0", 100.0),
            K=config.get("K", 100.0),
            T=config.get("T", 30 / 252),
            sigma=config.get("sigma", 0.2),
            r=config.get("r", 0.05),
        )
    raise ValueError(f"Unknown env_type: {env_type}")


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="online",
        uptime_seconds=round(time.time() - server_start_time, 1),
        active_environments=len(active_envs),
        total_connections=len(active_envs),
        available_env_types=list(ENV_REGISTRY.keys()),
    )


@app.get("/companies")
async def list_companies():
    """List all companies and their environment workflows."""
    return {
        cid: {
            "id": company["id"],
            "name": company["name"],
            "description": company["description"],
            "icon": company.get("icon", "building"),
            "workflows": company["workflows"],
            "total_environments": len(company["environments"]),
            "total_scenarios": sum(w["scenarios_count"] for w in company["workflows"]),
            "total_verifiers": sum(w["verifiers_count"] for w in company["workflows"]),
        }
        for cid, company in COMPANY_REGISTRY.items()
    }


@app.get("/companies/{company_id}")
async def get_company(company_id: str):
    """Get a specific company with its environment list."""
    if company_id not in COMPANY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Company '{company_id}' not found")
    company = COMPANY_REGISTRY[company_id]
    envs = []
    for env_id in company["environments"]:
        entry = ENV_REGISTRY[env_id]
        envs.append({
            "id": env_id,
            "display_name": entry["display_name"],
            "description": entry["description"],
            "tools": entry["tools"],
            "observation_dim": entry["observation_dim"],
            "action_type": entry["action_type"],
            "action_dim": entry["action_dim"],
        })
    return {
        "id": company["id"],
        "name": company["name"],
        "description": company["description"],
        "workflows": company["workflows"],
        "environments": envs,
    }


@app.get("/companies/{company_id}/environments")
async def list_company_environments(company_id: str):
    """List environments for a specific company as a selectable list."""
    if company_id not in COMPANY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Company '{company_id}' not found")
    company = COMPANY_REGISTRY[company_id]
    result = []
    for env_id in company["environments"]:
        entry = ENV_REGISTRY[env_id]
        workflow = next((w for w in company["workflows"] if w["id"] == env_id), {})
        result.append({
            "id": env_id,
            "display_name": entry["display_name"],
            "description": entry["description"],
            "algorithm": workflow.get("algorithm", ""),
            "action_type": entry["action_type"],
            "action_dim": entry["action_dim"],
            "observation_dim": entry["observation_dim"],
            "scenarios_count": workflow.get("scenarios_count", 0),
            "verifiers_count": workflow.get("verifiers_count", 0),
            "tools": entry["tools"],
        })
    return {"company": company["name"], "environments": result}


@app.get("/envs/list")
async def list_env_types():
    """List all environments (flat view, backward compatible)."""
    return {
        name: {
            "display_name": entry["display_name"],
            "description": entry["description"],
            "tools": entry["tools"],
            "observation_dim": entry["observation_dim"],
            "action_type": entry["action_type"],
            "action_dim": entry["action_dim"],
        }
        for name, entry in ENV_REGISTRY.items()
    }


@app.get("/tools")
async def list_tools():
    tools = []
    for env_name, entry in ENV_REGISTRY.items():
        for tool in entry["tools"]:
            tools.append({**tool, "environment": env_name})
    return tools


@app.post("/envs/create", response_model=CreateEnvResponse)
async def create_env(req: CreateEnvRequest):
    if req.env_type not in ENV_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown env_type '{req.env_type}'. Choose from: {list(ENV_REGISTRY.keys())}",
        )

    env = _build_env(req.env_type, req.config)
    env_id = str(uuid.uuid4())[:8]
    rollout_id = str(uuid.uuid4())[:12]
    verifier_config = _default_verifier_config(req.env_type)

    obs, info = env.reset()
    active_envs[env_id] = {
        "env": env,
        "env_type": req.env_type,
        "created_at": time.time(),
        "steps": 0,
        "last_obs": obs,
        "verifier_config": verifier_config,
        "rollout_id": rollout_id,
    }
    rollout_store[rollout_id] = {
        "id": rollout_id,
        "env_id": env_id,
        "env_type": req.env_type,
        "source": "manual",
        "policy": "human",
        "status": "in_progress",
        "created_at": _to_iso_timestamp(),
        "ended_at": None,
        "steps": [],
        "total_reward": 0.0,
        "total_steps": 0,
        "verifier_config": verifier_config,
        "initial_info": _ndarray_to_list(info),
        "initial_observation": _ndarray_to_list(obs),
        "final_outcome": None,
    }

    return CreateEnvResponse(
        env_id=env_id,
        env_type=req.env_type,
        observation_shape=list(env.observation_space.shape),
        action_space_info=_ndarray_to_list({
            "type": ENV_REGISTRY[req.env_type]["action_type"],
            "dim": env.action_space.n if hasattr(env.action_space, "n") else list(env.action_space.shape),
        }),
        rollout_id=rollout_id,
    )


@app.post("/envs/{env_id}/reset", response_model=ResetResponse)
async def reset_env(env_id: str, seed: Optional[int] = None):
    if env_id not in active_envs:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")

    entry = active_envs[env_id]
    obs, info = entry["env"].reset(seed=seed)
    entry["last_obs"] = obs
    entry["steps"] = 0
    verifier_config = entry.get("verifier_config", _default_verifier_config(entry["env_type"]))
    rollout_id = str(uuid.uuid4())[:12]
    entry["rollout_id"] = rollout_id
    source = "manual"
    policy = "human"
    if entry.get("is_trained"):
        source = "agent"
        policy = f"DQN ({entry.get('agent_type', 'dqn')})"

    rollout_store[rollout_id] = {
        "id": rollout_id,
        "env_id": env_id,
        "env_type": entry["env_type"],
        "source": source,
        "policy": policy,
        "status": "in_progress",
        "created_at": _to_iso_timestamp(),
        "ended_at": None,
        "steps": [],
        "total_reward": 0.0,
        "total_steps": 0,
        "verifier_config": verifier_config,
        "initial_info": _ndarray_to_list(info),
        "initial_observation": _ndarray_to_list(obs),
        "final_outcome": None,
    }

    return ResetResponse(
        observation=_ndarray_to_list(obs),
        info=_ndarray_to_list(info),
    )


@app.post("/envs/{env_id}/step", response_model=StepResponse)
async def step_env(env_id: str, req: StepRequest):
    if env_id not in active_envs:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")

    entry = active_envs[env_id]
    raw_action = req.action
    action = raw_action
    if isinstance(raw_action, list):
        action = np.array(raw_action, dtype=np.float32)

    obs, reward, terminated, truncated, info = entry["env"].step(action)
    entry["last_obs"] = obs
    entry["steps"] += 1

    verifier_config = entry.get("verifier_config", _default_verifier_config(entry["env_type"]))
    info_for_verifier = dict(info) if isinstance(info, dict) else {}
    info_for_verifier["_n_options"] = float(getattr(entry["env"], "n_options", 100.0))
    verifier_result = _build_verifier_result(
        env_type=entry["env_type"],
        info=info_for_verifier,
        reward=float(reward),
        verifier_config=verifier_config,
    )
    info_out = dict(info) if isinstance(info, dict) else {}
    info_out["verifier_result"] = verifier_result

    rollout_id = entry.get("rollout_id")
    rollout = rollout_store.get(rollout_id) if rollout_id else None
    if rollout is not None:
        action_names_map = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
        action_label = action_names_map[int(raw_action)] if isinstance(raw_action, (int, float)) and 0 <= int(raw_action) < 5 else str(raw_action)
        step_record = {
            "step": int(entry["steps"]),
            "action": _serialize_action(raw_action),
            "action_label": action_label,
            "reward": float(reward),
            "observation": _ndarray_to_list(obs),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _ndarray_to_list(info_out),
        }
        rollout["steps"].append(step_record)
        rollout["total_reward"] = float(rollout.get("total_reward", 0.0)) + float(reward)
        rollout["total_steps"] = int(entry["steps"])
        if terminated or truncated:
            rollout["status"] = "completed"
            rollout["ended_at"] = _to_iso_timestamp()
            rollout["final_outcome"] = {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "last_reward": float(reward),
                "last_info": _ndarray_to_list(info_out),
            }

    return StepResponse(
        observation=_ndarray_to_list(obs),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=_ndarray_to_list(info_out),
    )


@app.get("/envs/{env_id}/state")
async def get_env_state(env_id: str):
    if env_id not in active_envs:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")

    entry = active_envs[env_id]
    return {
        "env_id": env_id,
        "env_type": entry["env_type"],
        "steps_taken": entry["steps"],
        "created_at": entry["created_at"],
        "last_observation": _ndarray_to_list(entry["last_obs"]),
        "rollout_id": entry.get("rollout_id"),
        "verifier_config": entry.get("verifier_config", {}),
    }


@app.delete("/envs/{env_id}")
async def delete_env(env_id: str):
    if env_id not in active_envs:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")

    entry = active_envs[env_id]
    rollout_id = entry.get("rollout_id")
    rollout = rollout_store.get(rollout_id) if rollout_id else None
    if rollout is not None and rollout.get("status") == "in_progress":
        rollout["status"] = "aborted"
        rollout["ended_at"] = _to_iso_timestamp()
        rollout["final_outcome"] = {"reason": "environment_deleted"}
    del active_envs[env_id]
    return {"status": "deleted", "env_id": env_id}


@app.post("/envs/{env_id}/verifier")
async def configure_env_verifier(env_id: str, req: VerifierConfigRequest):
    if env_id not in active_envs:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")
    entry = active_envs[env_id]
    entry["verifier_config"] = {
        "verifier_type": req.verifier_type,
        "enabled": req.enabled,
        "thresholds": req.thresholds or {},
    }
    rollout = rollout_store.get(entry.get("rollout_id", ""))
    if rollout is not None:
        rollout["verifier_config"] = entry["verifier_config"]
    return {"env_id": env_id, "verifier_config": entry["verifier_config"]}


@app.get("/envs/{env_id}/rollout")
async def get_current_rollout(env_id: str):
    if env_id not in active_envs:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")
    rollout_id = active_envs[env_id].get("rollout_id")
    rollout = rollout_store.get(rollout_id) if rollout_id else None
    if rollout is None:
        raise HTTPException(status_code=404, detail="Rollout not found")
    return rollout


@app.get("/rollouts")
async def list_rollouts(env_type: Optional[str] = None, status: Optional[str] = None):
    rollouts = list(rollout_store.values())
    if env_type:
        rollouts = [r for r in rollouts if r.get("env_type") == env_type]
    if status:
        rollouts = [r for r in rollouts if r.get("status") == status]
    rollouts = sorted(rollouts, key=lambda r: r.get("created_at", ""), reverse=True)
    return {"count": len(rollouts), "rollouts": rollouts}


@app.get("/rollouts/{rollout_id}")
async def get_rollout(rollout_id: str):
    rollout = rollout_store.get(rollout_id)
    if rollout is None:
        raise HTTPException(status_code=404, detail=f"Rollout '{rollout_id}' not found")
    return rollout


# --- RL Training Endpoints ---

@app.post("/envs/{env_id}/train")
async def train_agent(env_id: str, req: TrainRequest):
    """Train an RL agent on this environment. Uses DQN for discrete, PPO for continuous."""
    if env_id not in active_envs:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")

    entry = active_envs[env_id]
    env = entry["env"]
    env_type = entry["env_type"]
    obs, _ = env.reset()
    state_dim = len(obs)

    is_discrete = hasattr(env.action_space, "n")
    algo_label = "DQN" if is_discrete else "PPO"

    if is_discrete:
        action_dim = env.action_space.n
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[64, 32],
            double_dqn=True, dueling=True,
            learning_rate=5e-4,
            epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay_steps=req.episodes * 400,
            buffer_size=20000, batch_size=32,
        )
    else:
        action_dim = env.action_space.shape[0]
        agent = PPOTrader(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            continuous=True,
            rollout_length=256,
            n_epochs=3,
            lr=3e-4,
        )

    training_history = []
    for ep in range(req.episodes):
        if is_discrete:
            metrics = agent.train_episode(env)
            ep_data = {
                "episode": ep + 1,
                "total_return": float(metrics.get("total_return", 0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0)),
                "portfolio_value": float(metrics.get("portfolio_value", 0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0)),
                "avg_loss": float(metrics.get("avg_loss", 0)),
                "epsilon": float(metrics.get("epsilon", 0)),
                "steps": int(metrics.get("steps", 0)),
            }
        else:
            rollout_data = agent.collect_rollout(env)
            update_info = agent.update(rollout_data)
            avg_reward = float(rollout_data["rewards"].mean())

            # Run a quick evaluation episode to get real env metrics
            eval_obs, eval_info = env.reset()
            eval_total_reward = 0
            eval_steps = 0
            while True:
                st = torch.FloatTensor(eval_obs).unsqueeze(0)
                with torch.no_grad():
                    act, _, _, _ = agent.network.get_action_and_value(st)
                act_np = act.cpu().numpy().flatten()
                if env_type == "portfolio-allocation":
                    act_np = np.clip(act_np, 0, None)
                    s = act_np.sum()
                    if s > 0:
                        act_np = act_np / s
                    else:
                        act_np = np.ones_like(act_np) / len(act_np)
                eval_obs, r, term, trunc, eval_info = env.step(act_np)
                eval_total_reward += r
                eval_steps += 1
                if term or trunc or eval_steps > 500:
                    break

            ep_data = {
                "episode": ep + 1,
                "total_return": float(eval_info.get("total_return", avg_reward)),
                "sharpe_ratio": float(eval_info.get("sharpe_ratio", 0)),
                "portfolio_value": float(eval_info.get("portfolio_value", 0)),
                "max_drawdown": float(eval_info.get("max_drawdown", 0)),
                "avg_loss": float(update_info.get("value_loss", 0)),
                "epsilon": 0.0,
                "policy_loss": float(update_info.get("policy_loss", 0)),
                "entropy": float(update_info.get("entropy", 0)),
                "avg_reward": float(eval_info.get("total_return", avg_reward)),
                "steps": eval_steps,
            }
        training_history.append(ep_data)

        train_rollout_id = f"train-{env_id}-ep{ep+1}"
        rollout_store[train_rollout_id] = {
            "id": train_rollout_id,
            "env_id": env_id,
            "env_type": env_type,
            "source": "training",
            "policy": f"{algo_label} (episode {ep+1}/{req.episodes})",
            "status": "completed",
            "created_at": _to_iso_timestamp(),
            "ended_at": _to_iso_timestamp(),
            "steps": [],
            "total_reward": float(ep_data.get("avg_reward", ep_data.get("total_return", 0))),
            "total_steps": int(ep_data.get("steps", 0)),
            "verifier_config": entry.get("verifier_config", {}),
            "initial_info": {},
            "final_outcome": {k: v for k, v in ep_data.items() if k != "episode"},
        }

    entry["agent"] = agent
    entry["is_trained"] = True
    entry["is_continuous"] = not is_discrete
    entry["training_history"] = training_history
    entry["agent_type"] = algo_label.lower()

    env.reset()
    obs, info = env.reset()
    entry["last_obs"] = obs
    entry["steps"] = 0

    best_ep = max(training_history, key=lambda x: x.get("sharpe_ratio", x.get("avg_reward", 0)))

    return {
        "status": "trained",
        "algorithm": algo_label,
        "episodes_trained": req.episodes,
        "training_history": training_history,
        "best_episode": best_ep,
        "final_epsilon": training_history[-1].get("epsilon", 0),
    }


@app.post("/envs/{env_id}/agent-step")
async def agent_step(env_id: str):
    """Step the environment using the trained RL agent's policy."""
    if env_id not in active_envs:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")

    entry = active_envs[env_id]
    if not entry.get("is_trained") or entry.get("agent") is None:
        raise HTTPException(status_code=400, detail="No trained agent. Call /train first.")

    agent = entry["agent"]
    env = entry["env"]
    obs = entry["last_obs"]
    is_continuous = entry.get("is_continuous", False)

    if is_continuous:
        import torch
        state_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _, _ = agent.network.get_action_and_value(state_t)
        action_np = action_t.cpu().numpy().flatten()
        # Ensure valid weights for portfolio (softmax-like clipping)
        if entry["env_type"] == "portfolio-allocation":
            action_np = np.clip(action_np, 0, None)
            s = action_np.sum()
            if s > 0:
                action_np = action_np / s
            else:
                action_np = np.ones_like(action_np) / len(action_np)
        action_for_env = action_np
        action_label = "[" + ", ".join(f"{v:.3f}" for v in action_np[:5]) + "]"
        action_serialized = action_np.tolist()
    else:
        action = agent.select_action(obs, training=False)
        action_int = int(action)
        action_for_env = action_int
        discrete_names = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
        action_label = discrete_names[action_int] if action_int < 5 else str(action_int)
        action_serialized = action_int

    obs_new, reward, terminated, truncated, info = env.step(action_for_env)
    entry["last_obs"] = obs_new
    entry["steps"] += 1

    info_out = dict(info) if isinstance(info, dict) else {}
    info_out["agent_action"] = action_label
    info_out["agent_action_raw"] = _ndarray_to_list(action_serialized)

    verifier_config = entry.get("verifier_config", _default_verifier_config(entry["env_type"]))
    info_for_v = dict(info) if isinstance(info, dict) else {}
    info_for_v["_n_options"] = float(getattr(env, "n_options", 100.0))
    info_out["verifier_result"] = _build_verifier_result(
        entry["env_type"], info_for_v, float(reward), verifier_config,
    )

    algo_label = entry.get("agent_type", "agent").upper()
    rollout_id = entry.get("rollout_id")
    rollout = rollout_store.get(rollout_id) if rollout_id else None
    if rollout is not None:
        if rollout.get("source") != "agent":
            rollout["source"] = "agent"
            rollout["policy"] = f"{algo_label} (trained)"
        rollout["steps"].append({
            "step": int(entry["steps"]),
            "action": _serialize_action(action_serialized),
            "action_label": action_label,
            "reward": float(reward),
            "observation": _ndarray_to_list(obs_new),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _ndarray_to_list(info_out),
        })
        rollout["total_reward"] = float(rollout.get("total_reward", 0)) + float(reward)
        rollout["total_steps"] = int(entry["steps"])
        if terminated or truncated:
            rollout["status"] = "completed"
            rollout["ended_at"] = _to_iso_timestamp()

    return {
        "observation": _ndarray_to_list(obs_new),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": _ndarray_to_list(info_out),
        "agent_action": action_label,
        "agent_action_raw": _ndarray_to_list(action_serialized),
    }


@app.get("/envs/{env_id}/training-status")
async def training_status(env_id: str):
    """Get training status and history for this environment's agent."""
    if env_id not in active_envs:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")

    entry = active_envs[env_id]
    is_trained = entry.get("is_trained", False)
    history = entry.get("training_history", [])

    result = {
        "is_trained": is_trained,
        "agent_type": entry.get("agent_type", None),
        "episodes_trained": len(history),
        "training_history": history,
    }

    if history:
        best = max(history, key=lambda x: x["sharpe_ratio"])
        last = history[-1]
        result["best_episode"] = best
        result["latest_metrics"] = last

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
