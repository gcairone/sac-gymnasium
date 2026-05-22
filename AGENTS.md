# AGENTS — sac-gymnasium

This document is a cleaned and well-formatted version of the original `AGENTS.md`. It captures the complete specification for the `sac-gymnasium` project and is organized for readability.

## Project Identity

- **Repository name:** sac-gymnasium
- **Language:** Python 3.10+
- **Framework:** PyTorch (CPU-only)
- **RL algorithm:** Soft Actor-Critic (SAC), implemented from scratch
- **Environments:** Gymnasium MuJoCo continuous control benchmarks

## Directory Structure

```
sac-gymnasium/
├── sac/
│   ├── __init__.py
│   ├── agent.py
│   ├── actor.py
│   ├── critic.py
│   ├── replay_buffer.py
│   ├── config.py
│   ├── logger.py
│   └── utils.py
├── configs/
│   └── default.yaml
├── logs/
│   └── .gitkeep
├── checkpoints/
│   └── .gitkeep
├── test/
│   ├── conftest.py
│   ├── test_agent.py
│   ├── test_actor.py
│   ├── test_critic.py
│   ├── test_replay_buffer.py
│   ├── test_config.py
│   └── test_logger.py
├── train.py
├── eval.py
├── play.py
├── log_analysis.py
├── run.sh
├── Dockerfile
├── README.md
└── .gitignore
```

## Rationale

`sac/` is the only package and contains all reusable algorithmic code, separated by concern:

- `sac/actor.py` and `sac/critic.py` isolate policy and value function implementations.
- `sac/config.py` handles configuration parsing (YAML + CLI merge).
- `sac/logger.py` handles structured logging (CSV, JSONL, terminal).
- `sac/utils.py` contains path generation, timestamp caching, soft updates, tensor helpers, and initialization.

Root-level scripts (`train.py`, `eval.py`, `play.py`, `log_analysis.py`) are entry points and import from `sac/`.

Logs and checkpoints are created at runtime; `.gitkeep` files are present to keep directories in git.

**Research focus:** This project targets research-grade SAC experiments — not a toy demo. Implementations must be stable and convergent in practice, applying well-known stabilization "tricks" where appropriate (see "Stability measures"). The aim is to provide a solid, reproducible SAC implementation that is algorithmically rich while remaining compact and focused.

## Import Strategy

Always run scripts from the project root. `sac/` is a normal package with `__init__.py`. No `sys.path` manipulation is used.

Example:

```bash
cd sac-gymnasium
python train.py --env HalfCheetah-v4
```

`pytest` is run from the project root:

```bash
cd sac-gymnasium
pytest test/
```

## CLI Interface

All scripts follow the `python <script>.py [--arg value ...]` pattern.

### train.py

```
python train.py [options]

Options:
  --config       PATH           Path to YAML config file          [default: configs/default.yaml]
  --env          ENV_ID         Override env from config          [default: none — use YAML value]
  --seed         INT            Override seed from config         [default: none — use YAML value]
  --total_steps  INT            Override total_steps from config  [default: none — use YAML value]
  --resume       PATH           Path to checkpoint dir to resume  [default: None]
```

At least one of YAML config or `--env` must supply a valid Gymnasium env id.
Clear progress in terminal

### eval.py

```
python eval.py [options]

Options:
  --checkpoint   PATH           Path to checkpoint directory      [required]
  --env          ENV_ID         Override env (inferred from meta.json if not provided)
  --episodes     INT            Number of evaluation episodes     [default: 10]
  --seed         INT            Random seed                       [default: 42]
```

### play.py

```
python play.py [options]

Options:
  --checkpoint   PATH           Path to checkpoint directory      [required]
  --env          ENV_ID         Override env (inferred from meta.json if not provided)
  --episodes     INT            Number of episodes to render      [default: 3]
  --noisy                        Enable stochastic policy          [flag, default: deterministic]
```

Rendering is enabled automatically in `play.py`.

### log_analysis.py

```
python log_analysis.py [options]

Options:
  --log_dir      PATH           Path to a run directory           [required]
  --smooth       INT            Smoothing window for curves       [default: 10]
  --save                        Save plots instead of showing     [flag]
```

## Config + CLI precedence

1. Defaults in `argparse` (lowest priority)
2. Values from YAML config file
3. Explicit CLI `--arg` overrides (highest priority)

YAML values are loaded first and used as defaults when registering argparse arguments, so CLI args always win.

## Log and Checkpoint Paths

Logs are stored under `logs/{env_name}/run_{timestamp}/` where `env_name` replaces `-` with `_` and `timestamp` is `YYYYMMDD_HHMMSS` UTC.

Typical run directory contents:

```
logs/HalfCheetah_v4/run_20240315_143022/
├── config.yaml
├── train.csv
├── eval.csv
└── events.jsonl
```

`train.csv` columns: `step, actor_loss, critic_loss, alpha_loss, alpha, mean_q, entropy`

`eval.csv` columns: `step, mean_return, std_return, min_return, max_return, n_episodes`

Checkpoints
checkpoints/{env_name}/run_{timestamp}/
Same env_name and timestamp as the corresponding log run (generated together, stored in config at run start).

Contents:

```text
checkpoints/HalfCheetah_v4/run_20240315_143022/
├── latest/
│   ├── actor.pt
│   ├── critic.pt
│   ├── critic_target.pt
│   ├── log_alpha.pt
│   ├── actor_optim.pt
│   ├── critic_optim.pt
│   ├── alpha_optim.pt
│   ├── replay_buffer.npz     # entire replay buffer state (observations, actions, rewards, dones, next_obs)
│   └── meta.json             # step, episode, timestamp, env_id, config snapshot
└── step_100000/
  └── ... (same structure)

```

`latest/` is overwritten at every checkpoint save interval. `step_{N}/` snapshots are saved at configurable intervals and never overwritten.
`replay_buffer.npz` is saved only in case of stopping like ctrl+C, not in every step

## Resume and Ctrl+C Safety

On SIGINT the training loop must perform a final checkpoint save to `latest/` before exiting. The entire replay buffer must be saved (`replay_buffer.npz`).

Resuming from `--resume checkpoints/{env_name}/run_{timestamp}/latest` must restore:

- All network weights and optimizer states from `.pt` files.
- The entire replay buffer from `replay_buffer.npz`.
- Step and episode counters from `meta.json`.
- Log files are appended (not overwritten).

Full resumability is required: every checkpoint is a complete snapshot.

## Algorithm Scope

Implemented from scratch:

- Stochastic actor: Gaussian policy with tanh squashing (outputs mean and log_std).
- Twin critics (Q1, Q2): separate MLPs; the minimum is used for target computations.
- Target critic networks: soft-updated copies.
- Automatic entropy tuning: learned `log_alpha`, `target_entropy = -dim(action_space)`.
- Replay buffer: uniform sampling, pre-allocated NumPy arrays, no per-step Python allocation.

- Research-level stability (policy): prioritize stable, convergent algorithmic choices. Advanced options (observation/reward normalization, additional regularizers or ensembles) may be provided as opt-in toggles but are disabled by default to keep experiments reproducible and the code focused.

## Stability measures

- Soft target updates: `tau = 0.005` (configurable).
- `log_std` clamped.
- Gradient clipping on critic: max norm `= 1.0` (configurable; `0` = disabled).
- Orthogonal initialization for hidden layers; small uniform init for output layers.
- No batch norm or layer norm by default.

- Target policy smoothing / clipped target actions (optional configurable noise added to targets).
- Clipped double-Q (use the minimum of twin critics and optionally clipping target values) to reduce overestimation bias.
- Optional, opt-in observation normalization / reward scaling utilities for difficult tasks; disabled by default to preserve reproducibility.

## Exclusions

- No prioritized replay
- No n-step returns
- No distributional critics
- No discrete action support
- No multi-agent support
- No environment wrappers or reward shaping

## Environment Usage

Use Gymnasium continuous control environments as provided. No reward scaling, observation normalization, or action repeat. Use `gymnasium.make(env_id)` only. `render_mode` is set only in `play.py`.

Recommended benchmark progression: `Pendulum-v1` → `HalfCheetah-v5` → `Hopper-v5` → `Walker2d-v5` → `Humanoid-v5`.

## Network Architecture

- Hidden layers: `2 × 256` (configurable).
- Activation: `ReLU` by default (configurable: `relu | tanh | elu | silu`).
- No dropout.
- Actor output: mean and log_std via separate linear heads on a shared trunk.
- Critic input: concatenated observation and action.

## Hyperparameter Config (configs/default.yaml)

```yaml
# configs/default.yaml
env: Pendulum-v1
seed: 0
total_steps: 1_000_000

# SAC
gamma: 0.99
tau: 0.005
lr_actor: 3.0e-4
lr_critic: 3.0e-4
lr_alpha: 3.0e-4
alpha_init: 0.2
target_entropy: null  # null = auto: -dim(action_space)
batch_size: 256
replay_buffer_size: 1_000_000
gradient_clip_norm: 1.0  # 0 = disabled

# Network
hidden_dim: 256
hidden_layers: 2
activation: relu

# Training schedule
warmup_steps: 5_000
train_freq: 1
updates_per_step: 1

# Evaluation
eval_freq: 10_000
eval_episodes: 10

# Checkpointing
checkpoint_freq: 50_000
```

## Compute Constraints

- CPU-only. No `.cuda()` or GPU logic; device is `torch.device("cpu")`.
- No vectorized environments.
- Single-environment, single-process training loop.
- MLP networks only (no CNN, no RNN).

Target training times: Pendulum ~10 min, HalfCheetah ~1 h, Hopper ~1–2 h, Humanoid ~8–12 h.

## Performance Constraints

- Replay buffer: pre-allocated NumPy arrays; sampling uses NumPy/Torch batch indexing.
- `train_step()` performs one forward + backward pass per call with minimal tensor copies.
- Log writes are buffered and flushed at configurable intervals.
- No debug assertions in hot paths (assert only in init/config validation).

- Attention to performance: avoid unnecessary Python loops and excessive tensor copies, prefer vectorized operations and efficient indexing. That said, do not overengineer for production-level throughput — prioritize readable, correct, and debuggable implementations that are fast enough for research experiments.

## Code Style

- English identifiers, comments, docstrings, and log messages.
- Type hints on all function signatures.
- No inline comments; short technical docstrings only when needed.
- Use Python logging in `sac/` (no `print()` calls in library code).
- Max line length: 100 characters.

- Clean and Pythonic: prefer simple, explicit functions and modules over deep class hierarchies. Avoid "wrapper-of-wrapper" patterns and unnecessary abstraction layers (no class just to hold a single function). Keep configuration handling straightforward — no layered wrappers around wrappers. Clarity and correctness take precedence over speculative extensibility.

## Logging & Experiment Tracking

- Terminal logging with `logging` (INFO default, DEBUG with `--verbose`).
- `train.csv` and `eval.csv` written with the `csv` stdlib module (no pandas).
- `events.jsonl` written with `json.dumps` + newline per event.
- No TensorBoard or Weights & Biases.
- `log_analysis.py` uses `matplotlib`; plots can be saved as `.png` with `--save`.

## Testing (test/)

Run tests with `pytest test/` from project root. Tests are designed to be fast and CPU-only.

Guidelines:

- No test requires rendering, display, or GPU.
- No test runs more than ~100 env steps or ~10 gradient updates.
- Tests use `tmp_path` for file I/O and do not write to `logs/` or `checkpoints/`.

Target coverage areas:

- `test_actor.py`: forward pass shapes, action sampling, log_prob computation.
- `test_critic.py`: forward pass shapes, Q-value computations.
- `test_replay_buffer.py`: push, sample, overflow wrap-around, returned tensor shapes.
- `test_agent.py`: agent init, `select_action`, `train_step` runs without error, output shapes.
- `test_utils.py`: soft update correctness, config loading/merge, path utilities.

`test/conftest.py` contains shared fixtures only (no `sys.path` manipulation).

## Docker

A `Dockerfile` is provided in the project root.

## Setup and run

Use the convenience script `run.sh` at the project root to run the Docker container. It mounts the project root into `/workspace` inside the container for live I/O.

## Project Scope & Phases

1. Implementation: Build the complete SAC algorithm from scratch following constraints.
2. Testing: Cover functionality through the pytest suite.
3. Ablation studies & analysis: Run experiments and analyze metrics using `log_analysis.py`.

Note on scope: many projects in this area grow into large, framework-like codebases with many files, classes and extension points. This repository intentionally avoids that: the goal is to train SAC reliably and reproducibly, not to provide a generic, highly-extensible framework. This means keeping the codebase compact, algorithm-focused, and free of unnecessary indirection while still providing research-grade stability and useful opt-in features.

## README

`README.md` should contain the technical README, including: project structure, requirements, Docker setup, training/eval/play examples, log analysis, checkpointing & resume, and running tests.

---

This file preserves the full specification from the original `AGENTS.md` while improving structure and readability.
