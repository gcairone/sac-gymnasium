from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sac.actor import GaussianActor
from sac.config import SACConfig
from sac.critic import TwinCritic
from sac.replay_buffer import ReplayBuffer
from sac import utils


class SACAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: SACConfig) -> None:
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = utils.get_device()

        self.actor = GaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
        ).to(self.device)
        self.critic = TwinCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
        ).to(self.device)
        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            size=cfg.replay_buffer_size,
            device=self.device,
        )

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        critic_params = list(self.critic.q1.parameters()) + list(self.critic.q2.parameters())
        self.critic_optim = torch.optim.Adam(critic_params, lr=cfg.lr_critic)

        log_alpha = math.log(cfg.alpha_init)
        self.log_alpha = nn.Parameter(torch.tensor(log_alpha, dtype=torch.float32, device=self.device))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        self.target_entropy = cfg.target_entropy
        if self.target_entropy is None:
            self.target_entropy = -float(action_dim)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().detach().cpu().item())

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = self._prepare_obs(obs)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(obs_tensor)
            else:
                action, _ = self.actor.sample(obs_tensor)
        return action.squeeze(0).cpu().numpy()

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None:
        self.replay_buffer.add(obs, action, reward, done, next_obs)

    def train_step(self) -> dict[str, float]:
        batch = self.replay_buffer.sample(self.cfg.batch_size)

        critic_loss = self._compute_critic_loss(batch)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.cfg.gradient_clip_norm > 0.0:
            params = list(self.critic.q1.parameters()) + list(self.critic.q2.parameters())
            torch.nn.utils.clip_grad_norm_(params, self.cfg.gradient_clip_norm)
        self.critic_optim.step()

        actor_loss, log_prob = self._compute_actor_loss(batch)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = self._compute_alpha_loss(log_prob)
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.critic.soft_update_targets(self.cfg.tau)

        with torch.no_grad():
            q1, q2 = self.critic(batch["obs"], batch["actions"])
            mean_q = torch.min(q1, q2).mean().item()
            entropy = (-log_prob).mean().item()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": self.alpha,
            "mean_q": float(mean_q),
            "entropy": float(entropy),
        }

    def _compute_critic_loss(self, batch: dict[str, Tensor]) -> Tensor:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch["next_obs"])
            q1_t, q2_t = self.critic.target_values(batch["next_obs"], next_action)
            alpha = self.log_alpha.exp().detach()
            q_target = torch.min(q1_t, q2_t) - alpha * next_log_prob
            target = batch["rewards"] + self.cfg.gamma * (1.0 - batch["dones"]) * q_target

        q1, q2 = self.critic(batch["obs"], batch["actions"])
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        return loss

    def _compute_actor_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        action, log_prob = self.actor.sample(batch["obs"])
        q1, q2 = self.critic(batch["obs"], action)
        q = torch.min(q1, q2)
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - q).mean()
        return actor_loss, log_prob.detach()

    def _compute_alpha_loss(self, log_prob: Tensor) -> Tensor:
        target = log_prob + self.target_entropy
        loss = -(self.log_alpha * target.detach()).mean()
        return loss

    def save(
        self,
        checkpoint_path: str | Path,
        save_buffer: bool = False,
        step: int = 0,
        episode: int = 0,
    ) -> None:
        path = Path(checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor.state_dict(), path / "actor.pt")
        torch.save(
            {"q1": self.critic.q1.state_dict(), "q2": self.critic.q2.state_dict()},
            path / "critic.pt",
        )
        torch.save(
            {
                "q1_target": self.critic.q1_target.state_dict(),
                "q2_target": self.critic.q2_target.state_dict(),
            },
            path / "critic_target.pt",
        )
        torch.save(self.log_alpha.detach().cpu(), path / "log_alpha.pt")
        torch.save(self.actor_optim.state_dict(), path / "actor_optim.pt")
        torch.save(self.critic_optim.state_dict(), path / "critic_optim.pt")
        torch.save(self.alpha_optim.state_dict(), path / "alpha_optim.pt")

        meta = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "env_id": self.cfg.env,
            "timestamp": self.cfg.timestamp,
            "step": int(step),
            "episode": int(episode),
            "config": self._serialize_config(self.cfg),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if save_buffer:
            self.replay_buffer.save(path / "replay_buffer.npz")

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> "SACAgent":
        path = Path(checkpoint_path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        cfg = cls._deserialize_config(meta["config"])

        agent = cls(obs_dim=meta["obs_dim"], action_dim=meta["action_dim"], cfg=cfg)
        agent.actor.load_state_dict(torch.load(path / "actor.pt", map_location=agent.device))

        critic_state = torch.load(path / "critic.pt", map_location=agent.device)
        agent.critic.q1.load_state_dict(critic_state["q1"])
        agent.critic.q2.load_state_dict(critic_state["q2"])

        target_state = torch.load(path / "critic_target.pt", map_location=agent.device)
        agent.critic.q1_target.load_state_dict(target_state["q1_target"])
        agent.critic.q2_target.load_state_dict(target_state["q2_target"])

        log_alpha = torch.load(path / "log_alpha.pt", map_location=agent.device)
        agent.log_alpha.data.copy_(log_alpha)

        agent.actor_optim.load_state_dict(torch.load(path / "actor_optim.pt", map_location=agent.device))
        agent.critic_optim.load_state_dict(torch.load(path / "critic_optim.pt", map_location=agent.device))
        agent.alpha_optim.load_state_dict(torch.load(path / "alpha_optim.pt", map_location=agent.device))

        return agent

    @staticmethod
    def _serialize_config(cfg: SACConfig) -> dict[str, Any]:
        data = asdict(cfg)
        data["log_dir"] = str(cfg.log_dir)
        data["checkpoint_dir"] = str(cfg.checkpoint_dir)
        return data

    @staticmethod
    def _deserialize_config(data: dict[str, Any]) -> SACConfig:
        data = dict(data)
        data["log_dir"] = Path(data["log_dir"])
        data["checkpoint_dir"] = Path(data["checkpoint_dir"])
        return SACConfig(**data)

    def _prepare_obs(self, obs: np.ndarray) -> Tensor:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            if obs_arr.shape[0] != self.obs_dim:
                raise ValueError(f"obs must have shape ({self.obs_dim},)")
            obs_arr = obs_arr[None, :]
        elif obs_arr.ndim == 2:
            if obs_arr.shape != (1, self.obs_dim):
                raise ValueError(f"obs must have shape (1, {self.obs_dim})")
        else:
            raise ValueError("obs must be 1D or 2D")
        return torch.from_numpy(obs_arr).to(self.device)
