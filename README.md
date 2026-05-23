# sac-gymnasium

Overview
--------
`sac-gymnasium` is a compact, research-focused implementation of Soft Actor-Critic (SAC)
written in Python 3.10+ and PyTorch (CPU-only). The project emphasizes algorithmic
correctness, stability, and reproducibility for Gymnasium continuous-control benchmarks.

Project layout
--------------
- `sac/` contains the reusable SAC implementation (actor, critic, agent, replay buffer,
  config utilities, logging, and helpers).
- `train.py`, `eval.py`, `play.py`, and `log_analysis.py` are the entry points.
- `configs/default.yaml` defines the default hyperparameters.
- `logs/` and `checkpoints/` are created at runtime (kept in git via .gitkeep).

Benchmarks Gymnasium
--------------------
From simpler to harder:
"InvertedPendulum-v5"
"InvertedDoublePendulum-v5"
"Reacher-v5"
"Pusher-v5"
"HalfCheetah-v5"
"Hopper-v5"
"Walker2d-v5"
"Swimmer-v5"
"Ant-v5"
"Humanoid-v5"
"HumanoidStandup-v5"



Quickstart (Docker)
-------------------
Build the Docker image from the project root:

```bash
docker build -t sac-gymnasium .
```

Start a container with the project mounted (convenience script):

```bash
./run.sh
```

Inside the container the project is mounted at `/sac-gymnasium`.

Run tests
---------
Execute the test suite (inside container or on host with deps installed):

```bash
pytest test/
```

Training
--------
Run training from the project root:

```bash
python train.py --config configs/default.yaml --env InvertedPendulum-v5 --total_steps 100000
```

CLI options for `train.py`:
- `--config` PATH (default: `configs/default.yaml`)
- `--env` ENV_ID (override env in YAML)
- `--seed` INT (override seed in YAML)
- `--total_steps` INT (override total_steps in YAML)
- `--resume` PATH (checkpoint directory to resume from)

Evaluation (headless)
---------------------
Evaluate a trained agent or a baseline policy. The `--agent` argument is required and
can be `zero`, `random`, or a checkpoint directory path containing `meta.json`.

```bash
python eval.py --agent zero --config configs/default.yaml --env InvertedPendulum-v5
python eval.py --agent checkpoints/InvertedPendulum_v5/run_<ts>/latest --episodes 10
```

Key options:
- `--agent` zero | random | PATH
- `--config` PATH (used only for zero/random agents)
- `--env` ENV_ID (override env for zero/random agents)
- `--episodes` INT (default: 10)
- `--max_steps_per_env` INT (default: 500)
- `--policy` deterministic | stochastic (default: deterministic)
- `--seed` INT (default: 42)

Playback (rendered)
-------------------
Render episodes with a trained policy or baseline policy. Rendering uses
`render_mode="human"` and requires a graphical environment.

```bash
python play.py --agent random --config configs/default.yaml --env InvertedPendulum-v5
python play.py --agent checkpoints/InvertedPendulum_v5/run_<ts>/latest --episodes 3
```

Key options:
- `--agent` zero | random | PATH
- `--config` PATH (used only for zero/random agents)
- `--env` ENV_ID (override env for zero/random agents)
- `--episodes` INT (default: 3)
- `--max_steps_per_episode` INT (default: 1000)
- `--policy` deterministic | stochastic (default: deterministic)
- `--speed` FLOAT (default: 1.0)

Logs and checkpoints
--------------------
Logs are written under `logs/{env_name}/run_{timestamp}/` and include:
- `train.csv` with `step, actor_loss, critic_loss, alpha_loss, alpha, mean_q, entropy`
- `eval.csv` with `step, mean_return, std_return, min_return, max_return, n_episodes,
  mean_episode_duration`
- `events.jsonl` with structured events

Checkpoints are written under `checkpoints/{env_name}/run_{timestamp}/` with:
- `latest/` (overwritten) and `step_{N}/` snapshots
- `meta.json` (env id, step, episode, config snapshot)
- `replay_buffer.npz` stored only for `latest/` and on SIGINT for full resumability

Resume training
---------------
To resume, pass a `latest/` checkpoint directory:

```bash
python train.py --resume checkpoints/InvertedPendulum_v5/run_<ts>/latest
```

Log analysis
------------
Plot training and evaluation curves from a run directory:

```bash
python log_analysis.py --log_dir logs/InvertedPendulum_v5/run_<ts> --smooth 10
python log_analysis.py --log_dir logs/InvertedPendulum_v5/run_<ts> --save
```

Notes
-----
- Configuration files live under `configs/` and use YAML + CLI override precedence.
- The codebase is intentionally compact and avoids framework-like abstractions.
- See `AGENTS.md` for the full project specification and constraints.

