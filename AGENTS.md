## Project: sac-gymnasium (SAC from Scratch in PyTorch for Gymnasium MuJoCo)

You are tasked with implementing a **Soft Actor-Critic (SAC)** reinforcement learning agent from scratch in **PyTorch (CPU-only)** and evaluating it on **Gymnasium MuJoCo continuous control environments** of increasing difficulty (e.g., *Pendulum → HalfCheetah → Hopper → Humanoid*).

The project and repository name must be exactly **`sac-gymnasium`**.

The goal is a **research-grade but minimal and pragmatic implementation** that successfully solves standard benchmarks without relying on Stable-Baselines3 or similar libraries.

---

## Core Requirements

### 1. Algorithm Scope

* Implement **SAC (Soft Actor-Critic)** fully from scratch.
* No Stable-Baselines or external RL frameworks.
* Must include all key stability improvements used in modern SAC implementations:
* Target networks with soft updates
* Automatic entropy tuning (α learning)
* Proper initialization and normalization where appropriate
* Stable replay buffer design
* Gradient clipping if needed


* Focus on **continuous control tasks only**.

---

### 2. Environment Usage

* Use **Gymnasium MuJoCo environments as-is** (no reward tuning, no environment modification).
* Environments are strictly **benchmarks**, not part of the research scope.
* Do not modify observation space, reward functions, or dynamics.

---

### 3. Compute Constraints

* CPU-only training is assumed.
* No vectorized environments (no VecEnv) to avoid overhead.
* Small MLP networks only (GPU-level acceleration not required).
* Training times of up to ~2 hours per run are acceptable.
* Expect convergence scales similar to SAC literature (e.g., Humanoid ~10M steps, simpler tasks ~1M steps).

---

### 4. Performance Constraints

* Python implementation must be efficient:
* Avoid unnecessary Python loops in training hot paths.
* Avoid redundant tensor copies.
* Keep data flow clean and minimal.


* No production-level overengineering, but also no naïve slow implementations.

---

### 5. Code Architecture Principles

* Clean, research-grade Python API.
* Avoid excessive abstraction:
* No deep class hierarchies
* No factory-over-factory patterns
* No unnecessary generalization beyond SAC + MLP policy/value networks


* Prefer clarity and directness over extensibility.

---

### 6. Code Style

* All code, comments, logs, variable names in **English**.
* Minimal comments:
* Only short technical comments above classes/functions when necessary.
* No inline noise comments.


* Use proper type hints:
* `def func(x: Type) -> ReturnType`


* Pythonic, research-style structure.

---

### 7. Logging & Experiment Tracking

* Terminal logs can be informal.
* Persistent logs must be structured and analysis-ready:
* Episode rewards
* Loss curves (actor, critic, entropy)
* Q-values / entropy stats
* Step counters and evaluation metrics


* Save logs in files suitable for later plotting/analysis.
* Include checkpointing system:
* Periodic saves
* Resume training from checkpoint seamlessly



---

### 8. Hyperparameters & Configuration

* Environment defaults must remain unchanged.
* SAC hyperparameters must be stored in a **YAML config file**.
* CLI arguments (`--arg`) override YAML values.
* All CLI args must have defaults.
* No hardcoded hyperparameters in code.

---

## Required Scripts

### `train.py`

* Main training loop for SAC.
* Supports:
* Resume from checkpoint
* Continuous logging
* Periodic evaluation during training


* No rendering.
* Clean progress reporting in terminal.

---

### `eval.py`

* Loads a trained checkpoint.
* Runs evaluation episodes without exploration noise.
* Computes performance metrics:
* Success rate (if applicable)
* Average return
* Variance across episodes



---

### `play.py`

* Loads a trained checkpoint.
* Runs policy in **rendered mode** for visualization.
* Deterministic evaluation mode (no training, no noise unless specified).

---

### `log_analysis.py`

* Loads training logs.
* Produces structured analysis of learning curves.
* Designed for offline inspection of training quality.

---

## Project Structure & Testing Requirements

* Simple, standard Python project layout.
* Clear separation of:
* source code
* `test/` (for `pytest` execution)
* logs
* checkpoints
* configs


* **Testing (`test/`) constraints:**
* Must be fully compatible with `pytest`.
* No tests requiring graphics/rendering.
* No tests requiring long training runs (tests must be fast).
* Tests must be completely clean: they must not leave dirty files, logs, or traces behind (use temp directories or mock components).


* **Git integration:**
* Provide a rigorous and clean `.gitignore` file.
* Ensure logs, checkpoints, `__pycache__`, OS artifacts, and anything generated during testing or training are explicitly ignored to avoid polluting the shared repository.


* Avoid PYTHONPATH hacks or complex packaging systems.
* Choose one consistent import strategy and stick to it.
* Ensure reproducibility without environment-specific path fixes.

---

## Docker & VSCode Integration Setup

* Provide a Dockerfile with:
* PyTorch (CPU-only)
* Gymnasium + MuJoCo dependencies
* Basic utilities (`tree`, `nano`)


* **Host-Container Integration:** For clean VSCode management, the project folder (including the `.git` directory) must be mounted as a volume so the repository is fully shared between the host and the Docker container.
* No Docker Compose. Use pure Docker commands.
* No conda, no poetry, no venv complexity inside the container.
* Fully reproducible container-based setup.

---

## Documentation

* Include a **technical README.md in English**:
* Project structure
* Exact `docker build` and `docker run` commands (explicitly avoiding `docker compose`) showing how to mount the volume correctly
* How to train
* How to evaluate
* How to visualize
* How logging/checkpoints work


* No marketing language, no unnecessary explanation.

---

## Final Objective

The implementation must:

* Successfully train SAC on standard MuJoCo benchmarks
* Be stable across multiple random seeds
* Be simple enough to inspect and modify
* Be efficient enough for CPU training
* Produce clean, analyzable experimental logs
* Pass all fast, isolated tests cleanly via `pytest`