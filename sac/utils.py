from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

_CACHED_TIMESTAMP: str | None = None


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")

    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def init_weights_orthogonal(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_uniform(module: nn.Module, scale: float) -> None:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -scale, scale)
        if module.bias is not None:
            nn.init.uniform_(module.bias, -scale, scale)


def get_timestamp() -> str:
    global _CACHED_TIMESTAMP
    if _CACHED_TIMESTAMP is None:
        _CACHED_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return _CACHED_TIMESTAMP


def sanitize_env_name(env_id: str) -> str:
    return env_id.replace("-", "_")


def make_log_dir(env_id: str, timestamp: str) -> Path:
    return Path("logs") / sanitize_env_name(env_id) / f"run_{timestamp}"


def make_checkpoint_dir(env_id: str, timestamp: str) -> Path:
    return Path("checkpoints") / sanitize_env_name(env_id) / f"run_{timestamp}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_activation(name: str) -> Callable[[], nn.Module]:
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]


