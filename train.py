from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from sac import utils
from sac.agent import SACAgent
from sac.config import SACConfig, build_config, load_yaml, save_config
from sac.logger import RunLogger
from sac.replay_buffer import ReplayBuffer

from torch import set_num_threads

set_num_threads(1)

@dataclass
class TrainState:
    step: int = 0
    episode: int = 0
    interrupted: bool = False


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/default.yaml")
    config_args, _ = config_parser.parse_known_args()

    raw_cfg = load_yaml(config_args.config)
    env_default = raw_cfg.get("env") if isinstance(raw_cfg, dict) else None
    seed_default = raw_cfg.get("seed") if isinstance(raw_cfg, dict) else None
    total_steps_default = raw_cfg.get("total_steps") if isinstance(raw_cfg, dict) else None

    parser = argparse.ArgumentParser(description="Train a SAC agent.")
    parser.add_argument("--config", default=config_args.config, help="Path to YAML config")
    parser.add_argument("--env", default=env_default, help="Override environment id")
    parser.add_argument("--seed", type=int, default=seed_default, help="Override random seed")
    parser.add_argument(
        "--total_steps",
        type=int,
        default=total_steps_default,
        help="Override total environment steps",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint directory to resume")
    return parser.parse_args()


def run_eval(
    agent: SACAgent,
    env_id: str,
    n_episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    env = gym.make(env_id)
    env.action_space.seed(seed)

    returns: list[float] = []
    durations: list[int] = []

    try:
        for episode in range(n_episodes):
            obs = _reset_env(env, seed=seed + episode)
            total_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                action = agent.select_action(obs, deterministic=True)
                action = _clip_action(action, env.action_space)
                obs, reward, terminated, truncated, _ = _step_env(env, action)
                total_reward += float(reward)
                steps += 1

                if terminated or truncated:
                    break

            returns.append(total_reward)
            durations.append(steps)
    finally:
        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "n_episodes": float(len(returns)),
        "mean_episode_duration": float(np.mean(durations)),
    }


def setup_signal_handler(state: TrainState) -> None:
    def _handler(_signum: int, _frame: Any) -> None:
        state.interrupted = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


def train(cfg: SACConfig | None, resume_path: Path | None) -> None:
    if resume_path is None:
        if cfg is None:
            raise ValueError("cfg is required when resume_path is not provided")
        state = TrainState()
        agent = None
    else:
        agent, cfg, state = _load_resume(resume_path)

    utils.set_seed(cfg.seed)

    env = gym.make(cfg.env)
    env.action_space.seed(cfg.seed)
    obs_dim = _space_dim(env.observation_space, "observation_space")
    action_dim = _space_dim(env.action_space, "action_space")

    if agent is None:
        agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, cfg=cfg)
    else:
        if agent.obs_dim != obs_dim or agent.action_dim != action_dim:
            raise ValueError("Checkpoint dimensions do not match the environment")

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = cfg.log_dir / "config.yaml"
    if not config_path.exists():
        save_config(cfg, config_path)

    logger = RunLogger(cfg.log_dir)
    logger.log_event(
        "startup",
        {
            "env": cfg.env,
            "seed": cfg.seed,
            "resume": str(resume_path) if resume_path else None,
            "total_steps": cfg.total_steps,
        },
    )

    try:
        setup_signal_handler(state)

        obs = _reset_env(env, seed=cfg.seed if state.step == 0 else None)
        episode_return = 0.0
        episode_steps = 0
        max_episode_steps = _get_env_max_steps(env)

        logging.info("Training started: env=%s total_steps=%d", cfg.env, cfg.total_steps)
        start_time = time.perf_counter()

        try:
            for _ in range(state.step, cfg.total_steps):
                if state.step < cfg.warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs, deterministic=False)
                action = _clip_action(action, env.action_space)

                next_obs, reward, terminated, truncated, _ = _step_env(env, action)
                done_for_buffer = bool(terminated)
                agent.store_transition(obs, action, reward, done_for_buffer, next_obs)

                obs = next_obs
                episode_return += float(reward)
                episode_steps += 1
                state.step += 1

                if terminated or truncated:
                    logger.log_event(
                        "episode_end",
                        {
                            "step": state.step,
                            "episode": state.episode,
                            "episode_return": episode_return,
                            "episode_length": episode_steps,
                        },
                    )
                    state.episode += 1
                    obs = _reset_env(env)
                    episode_return = 0.0
                    episode_steps = 0

                if state.step >= cfg.warmup_steps and state.step % cfg.train_freq == 0:
                    if len(agent.replay_buffer) >= cfg.batch_size:
                        metrics: dict[str, float] | None = None
                        for _ in range(cfg.updates_per_step):
                            metrics = agent.train_step()
                        # Log training metrics only every cfg.train_log_freq env steps
                        if metrics is not None and (state.step % cfg.train_log_freq == 0):
                            logger.log_train(state.step, metrics)

                if state.step % cfg.eval_freq == 0:
                    eval_metrics = run_eval(
                        agent,
                        cfg.env,
                        cfg.eval_episodes,
                        max_steps=max_episode_steps,
                        seed=cfg.seed + 100,
                    )
                    logger.log_eval(state.step, eval_metrics)
                    logger.flush()
                    logging.info(
                        "Eval step %d: mean_return=%.2f std=%.2f",
                        state.step,
                        eval_metrics["mean_return"],
                        eval_metrics["std_return"],
                    )

                if state.step % cfg.checkpoint_freq == 0:
                    _save_checkpoint(agent, cfg.checkpoint_dir, state.step, state.episode)
                    logger.flush()
                    logging.info("Checkpoint saved at step %d", state.step)

        except KeyboardInterrupt:
            state.interrupted = True
            logging.info("Training interrupted at step %d", state.step)

        elapsed = time.perf_counter() - start_time
        logging.info("Training finished in %.1f seconds", elapsed)

        if not state.interrupted and state.step > 0:
            eval_metrics = run_eval(
                agent,
                cfg.env,
                cfg.eval_episodes,
                max_steps=max_episode_steps,
                seed=cfg.seed + 100,
            )
            logger.log_eval(state.step, eval_metrics)

        latest_path = cfg.checkpoint_dir / "latest"
        agent.save(
            latest_path,
            save_buffer=state.interrupted,
            step=state.step,
            episode=state.episode,
        )
    finally:
        logger.flush()
        logger.close()
        env.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        train(None, resume_path)
        return

    cfg = build_config(
        args.config,
        {
            "env": args.env,
            "seed": args.seed,
            "total_steps": args.total_steps,
        },
    )
    train(cfg, None)


def _load_resume(resume_path: Path) -> tuple[SACAgent, SACConfig, TrainState]:
    if not resume_path.exists():
        raise ValueError(f"Resume path not found: {resume_path}")

    meta_path = resume_path / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Checkpoint missing meta.json: {resume_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    step = int(meta.get("step", 0))
    episode = int(meta.get("episode", 0))

    agent = SACAgent.load(resume_path)
    buffer_path = resume_path / "replay_buffer.npz"
    if not buffer_path.exists():
        raise ValueError("Replay buffer not found. Resume from a latest checkpoint.")
    agent.replay_buffer = ReplayBuffer.load(buffer_path, device=agent.device)

    utils._CACHED_TIMESTAMP = agent.cfg.timestamp
    state = TrainState(step=step, episode=episode)
    return agent, agent.cfg, state


def _save_checkpoint(
    agent: SACAgent,
    checkpoint_dir: Path,
    step: int,
    episode: int,
) -> None:
    latest_path = checkpoint_dir / "latest"
    agent.save(latest_path, save_buffer=False, step=step, episode=episode)

    step_path = checkpoint_dir / f"step_{step}"
    if not step_path.exists():
        agent.save(step_path, save_buffer=False, step=step, episode=episode)


def _reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    if seed is None:
        result = env.reset()
    else:
        result = env.reset(seed=seed)

    if isinstance(result, tuple):
        obs = result[0]
    else:
        obs = result
    return np.asarray(obs, dtype=np.float32)


def _step_env(
    env: gym.Env,
    action: np.ndarray,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    else:
        obs, reward, done, info = result
        terminated = bool(done)
        truncated = False
    info_dict = info if isinstance(info, dict) else {}
    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info_dict


def _space_dim(space: gym.Space, name: str) -> int:
    if not isinstance(space, gym.spaces.Box):
        raise ValueError(f"{name} must be a Box space")
    if space.shape is None or len(space.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    return int(space.shape[0])


def _clip_action(action: np.ndarray, action_space: gym.Space) -> np.ndarray:
    if not isinstance(action_space, gym.spaces.Box):
        return action
    low = action_space.low
    high = action_space.high
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        return action
    return np.clip(action, low, high)


def _get_env_max_steps(env: gym.Env) -> int:
    spec = getattr(env, "spec", None)
    max_steps = getattr(spec, "max_episode_steps", None) if spec else None
    if isinstance(max_steps, int) and max_steps > 0:
        return max_steps
    return 1000


if __name__ == "__main__":
    main()
