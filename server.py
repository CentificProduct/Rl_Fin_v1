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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from environments.stock_trading_env import StockTradingEnv
from environments.portfolio_env import PortfolioAllocationEnv
from environments.options_pricing_env import OptionsPricingEnv
from utils.data_loader import FinancialDataLoader


ENV_REGISTRY = {
    "stock-trading": {
        "class": StockTradingEnv,
        "display_name": "Stock Trading Environment",
        "description": "Single-asset trading with discrete/continuous actions",
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
        "display_name": "Portfolio Allocation Environment",
        "description": "Multi-asset portfolio weight optimization",
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
        "display_name": "Options Pricing & Hedging Environment",
        "description": "Dynamic delta hedging for options positions",
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


@app.get("/envs/list")
async def list_env_types():
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
        "status": "in_progress",
        "created_at": _to_iso_timestamp(),
        "ended_at": None,
        "steps": [],
        "total_reward": 0.0,
        "total_steps": 0,
        "verifier_config": verifier_config,
        "initial_info": _ndarray_to_list(info),
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
    rollout_store[rollout_id] = {
        "id": rollout_id,
        "env_id": env_id,
        "env_type": entry["env_type"],
        "status": "in_progress",
        "created_at": _to_iso_timestamp(),
        "ended_at": None,
        "steps": [],
        "total_reward": 0.0,
        "total_steps": 0,
        "verifier_config": verifier_config,
        "initial_info": _ndarray_to_list(info),
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
        step_record = {
            "step": int(entry["steps"]),
            "action": _serialize_action(raw_action),
            "reward": float(reward),
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
