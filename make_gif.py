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
    parser.add_argument("--steps", type=int, default=300, help="Maximum environment steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of consecutive episodes to record")
    parser.add_argument("--frame_skip", type=int, default=2, help="Keep one frame every N steps")
    parser.add_argument("--size", type=int, default=320, help="Frame width and height in pixels")
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed")
    parser.add_argument(
        "--camera",
        default=None,
        help='JSON overriding the env\'s default_camera_config, e.g. \'{"distance": 2.2, "lookat": [0, 0, 0.6]}\'',
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Keep the camera centred on the root body in the horizontal plane",
    )
    return parser.parse_args()


def render_frame(env: gym.Env, track: bool) -> np.ndarray:
    if track:
        # Gymnasium renders with a free camera, which ignores trackbodyid; follow the root body by hand.
        unwrapped = env.unwrapped
        viewer = unwrapped.mujoco_renderer._get_viewer("rgb_array")
        viewer.cam.lookat[:2] = unwrapped.data.xpos[1][:2]
    return env.render()


def record(
    agent: SACAgent,
    env: gym.Env,
    steps: int,
    episodes: int,
    frame_skip: int,
    seed: int,
    track: bool,
) -> tuple[list[np.ndarray], list[float]]:
    frames: list[np.ndarray] = []
    returns: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        frames.append(render_frame(env, track))
        total_reward = 0.0

        for step in range(1, steps + 1):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if step % frame_skip == 0:
                frames.append(render_frame(env, track))
            if terminated or truncated:
                break

        returns.append(total_reward)

    return frames, returns


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.frame_skip <= 0:
        raise ValueError("--frame_skip must be positive")

    path = Path(args.agent)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    agent = SACAgent.load(path)
    env_kwargs = {"render_mode": "rgb_array", "width": args.size, "height": args.size}
    if args.camera is not None:
        camera = json.loads(args.camera)
        if "lookat" in camera:
            camera["lookat"] = np.asarray(camera["lookat"], dtype=np.float64)
        env_kwargs["default_camera_config"] = camera
    env = gym.make(meta["env_id"], **env_kwargs)

    try:
        frames, returns = record(agent, env, args.steps, args.episodes, args.frame_skip, args.seed, args.track)
        # Play back in real time: each kept frame spans frame_skip env steps.
        duration_ms = int(round(1000.0 * env.unwrapped.dt * args.frame_skip))
    finally:
        env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # One global palette for every frame: per-frame palettes on cropped delta frames render badly in some viewers.
    sample = np.concatenate(frames[:: max(1, len(frames) // 8)], axis=0)
    palette = Image.fromarray(sample).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    images = [Image.fromarray(frame).quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True)
    print(f"Saved {out} ({len(images)} frames, {duration_ms} ms/frame, returns={[round(r, 2) for r in returns]})")


if __name__ == "__main__":
    main()
