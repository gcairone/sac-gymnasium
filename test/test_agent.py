from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from sac import config, utils
from sac.agent import SACAgent
from sac.config import SACConfig


def _make_cfg(tmp_path: Path, tau: float = 0.5) -> SACConfig:
    return SACConfig(
        env="InvertedPendulum-v5",
        seed=0,
        total_steps=100,
        gamma=0.99,
        tau=tau,
        lr_actor=1e-3,
        lr_critic=1e-3,
        lr_alpha=1e-3,
        alpha_init=0.2,
        target_entropy=None,
        reward_scale_enabled=False,
        reward_scale=1.0,
        reward_clip=0.0,
        batch_size=4,
        replay_buffer_size=100,
        gradient_clip_norm=0.5,
        hidden_dim=32,
        hidden_layers=2,
        activation="relu",
        warmup_steps=0,
        train_freq=1,
        train_log_freq=1,
        updates_per_step=1,
        eval_freq=10,
        eval_episodes=2,
        checkpoint_freq=10,
        timestamp="20240101_000000",
        log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "checkpoints",
    )


def _fill_buffer(agent: SACAgent, n: int = 10) -> None:
    for _ in range(n):
        obs = np.random.randn(agent.obs_dim).astype(np.float32)
        action = np.tanh(np.random.randn(agent.action_dim)).astype(np.float32)
        reward = float(np.random.randn())
        done = bool(np.random.rand() > 0.5)
        next_obs = obs + np.random.randn(agent.obs_dim).astype(np.float32) * 0.1
        agent.store_transition(obs, action, reward, done, next_obs)


def _base_config_dict() -> dict:
    return {
        "env": "InvertedPendulum-v5",
        "seed": 0,
        "total_steps": 100,
        "gamma": 0.99,
        "tau": 0.005,
        "lr_actor": 1e-3,
        "lr_critic": 1e-3,
        "lr_alpha": 1e-3,
        "alpha_init": 0.2,
        "target_entropy": None,
        "batch_size": 4,
        "replay_buffer_size": 100,
        "gradient_clip_norm": 0.5,
        "hidden_dim": 32,
        "hidden_layers": 2,
        "activation": "relu",
        "warmup_steps": 0,
        "train_freq": 1,
        "train_log_freq": 1,
        "updates_per_step": 1,
        "eval_freq": 10,
        "eval_episodes": 2,
        "checkpoint_freq": 10,
    }


def test_agent_train_step_updates_targets_and_metrics(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, tau=0.5)
    agent = SACAgent(obs_dim=3, action_dim=2, cfg=cfg)
    _fill_buffer(agent, n=12)

    with torch.no_grad():
        for param in agent.critic.q1.parameters():
            param.fill_(1.0)
        for param in agent.critic.q2.parameters():
            param.fill_(1.0)
        for param in agent.critic.q1_target.parameters():
            param.zero_()
        for param in agent.critic.q2_target.parameters():
            param.zero_()

    metrics = agent.train_step()

    assert set(metrics.keys()) == {
        "actor_loss",
        "critic_loss",
        "alpha_loss",
        "alpha",
        "mean_q",
        "entropy",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["alpha"] > 0.0

    target_sum = 0.0
    for param in agent.critic.q1_target.parameters():
        target_sum += float(param.abs().sum().item())
    for param in agent.critic.q2_target.parameters():
        target_sum += float(param.abs().sum().item())
    assert target_sum > 0.0


def test_reward_scaling_applies_before_clip(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.reward_scale_enabled = True
    cfg.reward_scale = 2.0
    cfg.reward_clip = 1.0
    agent = SACAgent(obs_dim=3, action_dim=2, cfg=cfg)

    obs = np.zeros((3,), dtype=np.float32)
    action = np.zeros((2,), dtype=np.float32)
    agent.store_transition(obs, action, 0.8, False, obs)

    stored = agent.replay_buffer.sample(1)["rewards"].item()
    assert np.isclose(stored, 1.0)


def test_agent_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    agent = SACAgent(obs_dim=4, action_dim=3, cfg=cfg)
    _fill_buffer(agent, n=6)

    checkpoint_dir = tmp_path / "checkpoint"
    agent.save(checkpoint_dir, save_buffer=True, step=12, episode=3)

    assert (checkpoint_dir / "actor.pt").exists()
    assert (checkpoint_dir / "critic.pt").exists()
    assert (checkpoint_dir / "critic_target.pt").exists()
    assert (checkpoint_dir / "log_alpha.pt").exists()
    assert (checkpoint_dir / "actor_optim.pt").exists()
    assert (checkpoint_dir / "critic_optim.pt").exists()
    assert (checkpoint_dir / "alpha_optim.pt").exists()
    assert (checkpoint_dir / "meta.json").exists()
    assert (checkpoint_dir / "replay_buffer.npz").exists()

    meta = json.loads((checkpoint_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["step"] == 12
    assert meta["episode"] == 3
    assert meta["env_id"] == agent.cfg.env

    loaded = SACAgent.load(checkpoint_dir)

    for key, value in agent.actor.state_dict().items():
        assert torch.allclose(value, loaded.actor.state_dict()[key])

    for key, value in agent.critic.q1.state_dict().items():
        assert torch.allclose(value, loaded.critic.q1.state_dict()[key])
    for key, value in agent.critic.q2.state_dict().items():
        assert torch.allclose(value, loaded.critic.q2.state_dict()[key])
    for key, value in agent.critic.q1_target.state_dict().items():
        assert torch.allclose(value, loaded.critic.q1_target.state_dict()[key])
    for key, value in agent.critic.q2_target.state_dict().items():
        assert torch.allclose(value, loaded.critic.q2_target.state_dict()[key])

    assert torch.allclose(agent.log_alpha, loaded.log_alpha)
    assert loaded.cfg.env == agent.cfg.env


def test_actor_loss_does_not_backprop_alpha(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    agent = SACAgent(obs_dim=3, action_dim=2, cfg=cfg)
    _fill_buffer(agent, n=8)

    batch = agent.replay_buffer.sample(cfg.batch_size)
    agent.log_alpha.grad = None
    actor_loss, _ = agent._compute_actor_loss(batch)
    actor_loss.backward()

    assert agent.log_alpha.grad is None


def test_agent_with_build_config_action_bounds(tmp_path: Path) -> None:
    utils._CACHED_TIMESTAMP = "20240101_000000"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_base_config_dict(), sort_keys=False), encoding="utf-8")

    cfg = config.build_config(cfg_path, {})
    agent = SACAgent(obs_dim=3, action_dim=2, cfg=cfg)

    obs = np.ones((3,), dtype=np.float32)
    deterministic = agent.select_action(obs, deterministic=True)
    stochastic = agent.select_action(obs, deterministic=False)

    assert deterministic.shape == (2,)
    assert stochastic.shape == (2,)
    assert np.all(deterministic <= 1.0) and np.all(deterministic >= -1.0)
    assert np.all(stochastic <= 1.0) and np.all(stochastic >= -1.0)
    assert agent.target_entropy == -2.0
    assert cfg.log_dir.as_posix() == "logs/InvertedPendulum_v5/run_20240101_000000"


def test_train_step_updates_actor_critic_and_alpha(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    agent = SACAgent(obs_dim=3, action_dim=2, cfg=cfg)
    _fill_buffer(agent, n=12)

    actor_before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    critic_before = {k: v.clone() for k, v in agent.critic.state_dict().items()}
    alpha_before = agent.log_alpha.detach().clone()

    agent.train_step()

    actor_changed = any(
        not torch.allclose(value, agent.actor.state_dict()[key])
        for key, value in actor_before.items()
    )
    critic_changed = any(
        not torch.allclose(value, agent.critic.state_dict()[key])
        for key, value in critic_before.items()
    )

    assert actor_changed
    assert critic_changed
    assert not torch.allclose(alpha_before, agent.log_alpha.detach())
