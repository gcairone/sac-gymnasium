from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from sac.agent import SACAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a SAC policy rollout as a GIF.")
    parser.add_argument("--agent", required=True, help="Path to checkpoint directory")
    parser.add_argument("--out", required=True, help="Output GIF path")
    parser.add_argument("--steps", type=int, default=300, help="Maximum environment steps to record")
    parser.add_argument("--frame_skip", type=int, default=2, help="Keep one frame every N steps")
    parser.add_argument("--size", type=int, default=320, help="Frame width and height in pixels")
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed")
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=None,
        help="Override the env's default camera distance",
    )
    parser.add_argument(
        "--camera_lookat_z",
        type=float,
        default=0.0,
        help="Camera target height, used with --camera_distance",
    )
    return parser.parse_args()


def record(agent: SACAgent, env: gym.Env, steps: int, frame_skip: int, seed: int) -> tuple[list[np.ndarray], float]:
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0

    for step in range(1, steps + 1):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if step % frame_skip == 0:
            frames.append(env.render())
        if terminated or truncated:
            break

    return frames, total_reward


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.frame_skip <= 0:
        raise ValueError("--frame_skip must be positive")

    path = Path(args.agent)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    agent = SACAgent.load(path)
    env_kwargs = {"render_mode": "rgb_array", "width": args.size, "height": args.size}
    if args.camera_distance is not None:
        env_kwargs["default_camera_config"] = {
            "trackbodyid": 0,
            "distance": args.camera_distance,
            "lookat": [0.0, 0.0, args.camera_lookat_z],
        }
    env = gym.make(meta["env_id"], **env_kwargs)

    try:
        frames, total_reward = record(agent, env, args.steps, args.frame_skip, args.seed)
        # Play back in real time: each kept frame spans frame_skip env steps.
        duration_ms = int(round(1000.0 * env.unwrapped.dt * args.frame_skip))
    finally:
        env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True)
    print(f"Saved {out} ({len(images)} frames, {duration_ms} ms/frame, return={total_reward:.2f})")


if __name__ == "__main__":
    main()
