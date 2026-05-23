from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size replay buffer with pre-allocated NumPy storage."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        size: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if size <= 0:
            raise ValueError("size must be positive")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._max_size = size
        self.device = torch.device(device)

        self._obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self._actions_buf = np.zeros((size, action_dim), dtype=np.float32)
        self._rewards_buf = np.zeros((size, 1), dtype=np.float32)
        self._dones_buf = np.zeros((size, 1), dtype=np.float32)
        self._next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None:
        obs_arr = self._ensure_1d(obs, self.obs_dim, "obs")
        action_arr = self._ensure_1d(action, self.action_dim, "action")
        next_obs_arr = self._ensure_1d(next_obs, self.obs_dim, "next_obs")

        self._obs_buf[self._ptr] = obs_arr
        self._actions_buf[self._ptr] = action_arr
        self._rewards_buf[self._ptr] = float(reward)
        self._dones_buf[self._ptr] = float(done)
        self._next_obs_buf[self._ptr] = next_obs_arr

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")

        idxs = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.from_numpy(self._obs_buf[idxs]).to(self.device),
            "actions": torch.from_numpy(self._actions_buf[idxs]).to(self.device),
            "rewards": torch.from_numpy(self._rewards_buf[idxs]).to(self.device),
            "dones": torch.from_numpy(self._dones_buf[idxs]).to(self.device),
            "next_obs": torch.from_numpy(self._next_obs_buf[idxs]).to(self.device),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            obs=self._obs_buf,
            actions=self._actions_buf,
            rewards=self._rewards_buf,
            dones=self._dones_buf,
            next_obs=self._next_obs_buf,
            ptr=self._ptr,
            size=self._size,
            max_size=self._max_size,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "ReplayBuffer":
        data = np.load(Path(path), allow_pickle=False)

        obs_buf = data["obs"]
        actions_buf = data["actions"]
        rewards_buf = data["rewards"]
        dones_buf = data["dones"]
        next_obs_buf = data["next_obs"]
        max_size = int(data["max_size"])

        buffer = cls(
            obs_dim=obs_buf.shape[1],
            action_dim=actions_buf.shape[1],
            size=max_size,
            device=device,
        )
        buffer._obs_buf[:] = obs_buf
        buffer._actions_buf[:] = actions_buf
        buffer._rewards_buf[:] = rewards_buf
        buffer._dones_buf[:] = dones_buf
        buffer._next_obs_buf[:] = next_obs_buf
        buffer._ptr = int(data["ptr"])
        buffer._size = int(data["size"])

        return buffer

    @staticmethod
    def _ensure_1d(value: np.ndarray, dim: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] != dim:
            raise ValueError(f"{name} must have shape ({dim},)")
        return arr
