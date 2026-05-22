# sac-gymnasium — README

Overview
--------
`sac-gymnasium` is a compact, research-focused implementation of Soft Actor-Critic (SAC) written in Python 3.10+ and PyTorch (CPU-only). The project emphasizes algorithmic correctness and experimental stability (convergence)

Quickstart (Docker)
--------------------
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

Minimal CLI examples
--------------------
Train, evaluate and run playback using the provided scripts. Examples:

```bash
python train.py --config configs/default.yaml --env Pendulum-v1 --total_steps 100000
python eval.py --checkpoint checkpoints/Pendulum_v1/run_<ts>/latest --episodes 10
python play.py --checkpoint checkpoints/Pendulum_v1/run_<ts>/latest --episodes 3
```

Notes
-----
- Configuration files live under `configs/`.
- The codebase is intentionally compact: focus is on stable, reproducible SAC experiments rather than extensibility or framework features.
- See `AGENTS.md` for detailed project and implementation guidelines.

