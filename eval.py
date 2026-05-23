from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from sac.agent import SACAgent
from sac.config import SACConfig, build_config, load_yaml
from sac import utils


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/default.yaml")
    config_args, _ = config_parser.parse_known_args()

    raw_cfg = load_yaml(config_args.config)
    env_default = raw_cfg.get("env") if isinstance(raw_cfg, dict) else None

    parser = argparse.ArgumentParser(description="Evaluate a SAC agent.")
    parser.add_argument("--agent", required=True, help="'zero' | 'random' | PATH to checkpoint")
    parser.add_argument("--config", default=config_args.config, help="Path to YAML config")
    parser.add_argument("--env", default=env_default, help="Override environment id")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument(
        "--max_steps_per_env",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--policy",
        choices=["stochastic", "deterministic"],
        default="deterministic",
        help="Policy mode for evaluation",
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
        env = gym.make(env_id)
        agent = SACAgent.load(path)
        return agent, env

    if agent_arg not in {"zero", "random"}:
        raise ValueError("--agent must be 'zero', 'random', or a checkpoint path")

    env = gym.make(cfg.env)
    return agent_arg, env


def run_evaluation(
    agent: SACAgent | str,
    env: gym.Env,
    n_episodes: int,
    max_steps: int,
    deterministic: bool,
) -> dict[str, float]:
    returns: list[float] = []
    durations: list[int] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            if isinstance(agent, str):
                if agent == "zero":
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action = env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        returns.append(total_reward)
        durations.append(steps)

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    min_return = float(np.min(returns))
    max_return = float(np.max(returns))
    mean_duration = float(np.mean(durations))

    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "min_return": min_return,
        "max_return": max_return,
        "n_episodes": float(len(returns)),
        "mean_episode_duration": mean_duration,
    }


def print_summary(metrics: dict[str, float]) -> None:
    print("Evaluation summary")
    print("-------------------")
    print(
        "Return: mean={mean:.2f} std={std:.2f} min={min:.2f} max={max:.2f}".format(
            mean=metrics["mean_return"],
            std=metrics["std_return"],
            min=metrics["min_return"],
            max=metrics["max_return"],
        )
    )
    print("Episodes: {n:.0f}".format(n=metrics["n_episodes"]))
    print("Mean episode duration: {dur:.1f} steps".format(dur=metrics["mean_episode_duration"]))


def main() -> None:
    args = parse_args()

    utils.set_seed(args.seed)
    cfg = build_config(args.config, {"env": args.env})

    agent, env = load_agent_and_env(args.agent, cfg)
    env.action_space.seed(args.seed)

    try:
        metrics = run_evaluation(
            agent=agent,
            env=env,
            n_episodes=args.episodes,
            max_steps=args.max_steps_per_env,
            deterministic=args.policy == "deterministic",
        )
    finally:
        env.close()

    print_summary(metrics)


if __name__ == "__main__":
    main()
