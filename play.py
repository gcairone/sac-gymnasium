from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sac.agent import SACAgent
from sac.config import SACConfig, build_config, load_yaml
from sac import utils


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/default.yaml")
    config_args, _ = config_parser.parse_known_args()

    raw_cfg = load_yaml(config_args.config)
    env_default = raw_cfg.get("env") if isinstance(raw_cfg, dict) else None

    parser = argparse.ArgumentParser(description="Render a SAC agent.")
    parser.add_argument("--agent", required=True, help="'zero' | 'random' | PATH to checkpoint")
    parser.add_argument("--config", default=config_args.config, help="Path to YAML config")
    parser.add_argument("--env", default=env_default, help="Override environment id")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to render")
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--policy",
        choices=["stochastic", "deterministic"],
        default="deterministic",
        help="Policy mode for rendering",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (e.g. 2.0, 0.5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_agent_and_env(agent_arg: str, cfg: SACConfig) -> tuple[SACAgent | str, gym.Env]:
    path = Path(agent_arg)
    if path.exists():
        meta_path = path / "meta.json"
        if not meta_path.exists():
            raise ValueError(f"Checkpoint path missing meta.json: {path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        env_id = meta.get("env_id")
        if not env_id:
            raise ValueError("meta.json missing env_id")
        env = gym.make(env_id, render_mode="human")
        agent = load_agent_checkpoint(path)
        return agent, env

    if agent_arg not in {"zero", "random"}:
        raise ValueError("--agent must be 'zero', 'random', or a checkpoint path")

    env = gym.make(cfg.env, render_mode="human")
    return agent_arg, env


def load_agent_checkpoint(path: Path) -> SACAgent:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    cfg = SACAgent._deserialize_config(meta["config"])
    agent = SACAgent(obs_dim=meta["obs_dim"], action_dim=meta["action_dim"], cfg=cfg)

    actor_state = torch.load(path / "actor.pt", map_location=agent.device)
    if "log_2" in actor_state:
        actor_state = dict(actor_state)
        actor_state.pop("log_2", None)
    agent.actor.load_state_dict(actor_state)

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


def _get_env_dt(env: gym.Env) -> float:
    unwrapped = getattr(env, "unwrapped", env)
    dt = getattr(unwrapped, "dt", None)
    if isinstance(dt, (int, float)) and dt > 0.0:
        return float(dt)
    return 1.0 / 60.0


def run_episode(
    agent: SACAgent | str,
    env: gym.Env,
    max_steps: int,
    deterministic: bool,
    speed: float,
    seed: int | None = None,
) -> float:
    if speed <= 0.0:
        raise ValueError("--speed must be positive")

    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    dt = _get_env_dt(env)
    target_step_time = dt / speed

    for _ in range(max_steps):
        if isinstance(agent, str):
            if agent == "zero":
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=deterministic)

        step_start = time.perf_counter()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        elapsed = time.perf_counter() - step_start
        if target_step_time > 0.0:
            time.sleep(max(0.0, target_step_time - elapsed))

        if terminated or truncated:
            break

    return total_reward


def main() -> None:
    args = parse_args()
    cfg = build_config(args.config, {"env": args.env})

    agent, env = load_agent_and_env(args.agent, cfg)

    # Seed RNGs for reproducibility
    utils.set_seed(args.seed)
    try:
        env.action_space.seed(args.seed)
    except Exception:
        # Some envs may not support seeding of the action space.
        pass

    try:
        for episode_idx in range(args.episodes):
            run_episode(
                agent=agent,
                env=env,
                max_steps=args.max_steps_per_episode,
                deterministic=args.policy == "deterministic",
                speed=args.speed,
                seed=(args.seed + episode_idx) if args.seed is not None else None,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
