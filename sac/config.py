from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from sac import utils


@dataclass
class SACConfig:
    env: str
    seed: int
    total_steps: int
    gamma: float
    tau: float
    lr_actor: float
    lr_critic: float
    lr_alpha: float
    alpha_init: float
    target_entropy: float | None
    batch_size: int
    replay_buffer_size: int
    gradient_clip_norm: float
    hidden_dim: int
    hidden_layers: int
    activation: str
    warmup_steps: int
    train_freq: int
    train_log_freq: int
    updates_per_step: int
    eval_freq: int
    eval_episodes: int
    checkpoint_freq: int
    timestamp: str
    log_dir: Path
    checkpoint_dir: Path


def load_yaml(yaml_path: str | Path) -> dict[str, Any]:
    path = Path(yaml_path)
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping")
    return data


def build_config(yaml_path: str | Path, overrides: dict[str, Any]) -> SACConfig:
    raw = load_yaml(yaml_path)
    merged: dict[str, Any] = dict(raw)

    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    values = _validate_and_normalize(merged)
    timestamp = utils.get_timestamp()
    values["timestamp"] = timestamp
    values["log_dir"] = utils.make_log_dir(values["env"], timestamp)
    values["checkpoint_dir"] = utils.make_checkpoint_dir(values["env"], timestamp)
    return SACConfig(**values)


def save_config(cfg: SACConfig, path: str | Path) -> None:
    data = asdict(cfg)
    data["log_dir"] = str(cfg.log_dir)
    data["checkpoint_dir"] = str(cfg.checkpoint_dir)
    out_path = Path(path)

    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _validate_and_normalize(values: dict[str, Any]) -> dict[str, Any]:
    required_keys = {
        "env",
        "seed",
        "total_steps",
        "gamma",
        "tau",
        "lr_actor",
        "lr_critic",
        "lr_alpha",
        "alpha_init",
        "target_entropy",
        "batch_size",
        "replay_buffer_size",
        "gradient_clip_norm",
        "hidden_dim",
        "hidden_layers",
        "activation",
        "warmup_steps",
        "train_freq",
        "train_log_freq",
        "updates_per_step",
        "eval_freq",
        "eval_episodes",
        "checkpoint_freq",
    }
    missing = required_keys - values.keys()
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing config keys: {missing_list}")

    env = _require_str(values["env"], "env")
    seed = _require_int(values["seed"], "seed", min_value=0)
    total_steps = _require_int(values["total_steps"], "total_steps", min_value=1)
    gamma = _require_float(values["gamma"], "gamma", min_value=0.0, max_value=1.0)
    tau = _require_float(values["tau"], "tau", min_value=0.0, max_value=1.0)
    lr_actor = _require_float(values["lr_actor"], "lr_actor", min_value=0.0)
    lr_critic = _require_float(values["lr_critic"], "lr_critic", min_value=0.0)
    lr_alpha = _require_float(values["lr_alpha"], "lr_alpha", min_value=0.0)
    alpha_init = _require_float(values["alpha_init"], "alpha_init", min_value=0.0)

    target_entropy = values["target_entropy"]
    if target_entropy is not None:
        target_entropy = _require_float(target_entropy, "target_entropy")

    batch_size = _require_int(values["batch_size"], "batch_size", min_value=1)
    replay_buffer_size = _require_int(
        values["replay_buffer_size"], "replay_buffer_size", min_value=1
    )
    gradient_clip_norm = _require_float(
        values["gradient_clip_norm"], "gradient_clip_norm", min_value=0.0
    )
    hidden_dim = _require_int(values["hidden_dim"], "hidden_dim", min_value=1)
    hidden_layers = _require_int(values["hidden_layers"], "hidden_layers", min_value=1)
    activation = _require_str(values["activation"], "activation")
    if activation not in {"relu", "tanh", "elu", "silu"}:
        raise ValueError(f"Unsupported activation: {activation}")

    warmup_steps = _require_int(values["warmup_steps"], "warmup_steps", min_value=0)
    train_freq = _require_int(values["train_freq"], "train_freq", min_value=1)
    train_log_freq = _require_int(values["train_log_freq"], "train_log_freq", min_value=1)
    updates_per_step = _require_int(values["updates_per_step"], "updates_per_step", min_value=1)
    eval_freq = _require_int(values["eval_freq"], "eval_freq", min_value=1)
    eval_episodes = _require_int(values["eval_episodes"], "eval_episodes", min_value=1)
    checkpoint_freq = _require_int(values["checkpoint_freq"], "checkpoint_freq", min_value=1)

    return {
        "env": env,
        "seed": seed,
        "total_steps": total_steps,
        "gamma": gamma,
        "tau": tau,
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "lr_alpha": lr_alpha,
        "alpha_init": alpha_init,
        "target_entropy": target_entropy,
        "batch_size": batch_size,
        "replay_buffer_size": replay_buffer_size,
        "gradient_clip_norm": gradient_clip_norm,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "warmup_steps": warmup_steps,
        "train_freq": train_freq,
        "train_log_freq": train_log_freq,
        "updates_per_step": updates_per_step,
        "eval_freq": eval_freq,
        "eval_episodes": eval_episodes,
        "checkpoint_freq": checkpoint_freq,
    }


def _require_int(value: Any, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return value


def _require_float(
    value: Any,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a float")
    float_value = float(value)
    if min_value is not None and float_value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and float_value > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return float_value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
