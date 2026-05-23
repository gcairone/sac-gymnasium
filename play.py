from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from sac.agent import SACAgent
from sac.config import SACConfig, build_config, load_yaml


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
        agent = SACAgent.load(path)
        return agent, env

    if agent_arg not in {"zero", "random"}:
        raise ValueError("--agent must be 'zero', 'random', or a checkpoint path")

    env = gym.make(cfg.env, render_mode="human")
    return agent_arg, env


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
) -> float:
    if speed <= 0.0:
        raise ValueError("--speed must be positive")

    obs, _ = env.reset()
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

    try:
        for _ in range(args.episodes):
            run_episode(
                agent=agent,
                env=env,
                max_steps=args.max_steps_per_episode,
                deterministic=args.policy == "deterministic",
                speed=args.speed,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
