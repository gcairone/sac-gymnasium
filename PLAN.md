# Structured Implementation Plan — `sac-gymnasium`

## Implementation Order

```
Phase 1 — Foundations (no internal dependencies)
  sac/utils.py          ← already implemented
  sac/config.py         ← already implemented

Phase 2 — Neural networks (depend only on utils)
  sac/actor.py          ← already implemented
  sac/critic.py         ← already implemented

Phase 3 — Algorithm core (depends on actor, critic, replay_buffer, config, utils)
  sac/replay_buffer.py  ← already implemented
  sac/agent.py          ← already implemented

Phase 4 — I/O and logging (depends on config, utils)
  sac/logger.py         ← already implemented

Phase 5 — Entry points (depend on everything above)
  eval.py               ← already implemented
  play.py               ← already implemented
  train.py              ← already implemented
  log_analysis.py
```

---

## Phase 1 — `sac/utils.py`

```python
# No internal dependencies

def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None
# Polyak update: θ_target ← τ·θ_source + (1−τ)·θ_target

def init_weights_orthogonal(module: nn.Module) -> None
# Orthogonal initialization for Linear layers in hidden layers.

def init_weights_uniform(module: nn.Module, scale: float) -> None
# Small uniform initialization for output-layer Linear weights and biases.

def get_timestamp() -> str
# Returns a UTC timestamp string in YYYYMMDD_HHMMSS format, cached per process.

def sanitize_env_name(env_id: str) -> str
# Replaces hyphens with underscores: "HalfCheetah-v5" → "HalfCheetah_v5"

def make_log_dir(env_id: str, timestamp: str) -> Path
# Returns: logs/{sanitized_env_id}/run_{timestamp}/

def make_checkpoint_dir(env_id: str, timestamp: str) -> Path
# Returns: checkpoints/{sanitized_env_id}/run_{timestamp}/

def set_seed(seed: int) -> None
# Sets seeds for Python random, NumPy, and PyTorch.

def get_device() -> torch.device
# Always returns torch.device("cpu").
```

---

## Phase 1 — `sac/config.py`

```python
# No internal dependencies

@dataclass
class SACConfig:
    # Environment
    env: str
    seed: int
    total_steps: int
    # SAC hyperparameters
    gamma: float
    tau: float
    lr_actor: float
    lr_critic: float
    lr_alpha: float
    alpha_init: float
    target_entropy: float | None       # None → auto: −dim(action_space)
    batch_size: int
    replay_buffer_size: int
    gradient_clip_norm: float          # 0.0 = disabled
    # Network architecture
    hidden_dim: int
    hidden_layers: int
    activation: str
    # Training schedule
    warmup_steps: int
    train_freq: int
    updates_per_step: int
    # Evaluation
    eval_freq: int
    eval_episodes: int
    # Checkpointing
    checkpoint_freq: int
    # Runtime fields (not in YAML; populated at run start)
    timestamp: str
    log_dir: Path
    checkpoint_dir: Path


def load_yaml(yaml_path: str | Path) -> dict
# Reads a YAML file and returns the raw dict.

def build_config(yaml_path: str | Path, overrides: dict) -> SACConfig
# Merge order: YAML defaults → CLI overrides.
# Populates runtime fields (timestamp, log_dir, checkpoint_dir).
# Validates field types and value ranges; raises ValueError on invalid input.

def save_config(cfg: SACConfig, path: str | Path) -> None
# Serializes SACConfig to YAML; written into the run's log directory at startup.
```

---

## Phase 2 — `sac/actor.py`

```python
# Depends on: sac.utils

LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0

class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
    ) -> None
    # Shared trunk: MLP built from Linear + activation blocks that produces a
    # `hidden_dim`-sized output.
    # Apply `init_weights_orthogonal` to the trunk and initialize
    # `mean_head` / `log_std_head` with `init_weights_uniform`.
    # mean_head:    nn.Linear(hidden_dim, action_dim)
    # log_std_head: nn.Linear(hidden_dim, action_dim)
    # Hidden layers: orthogonal init. Output heads: small uniform init.

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]
    # → (mean, log_std) where log_std is clamped to [LOG_STD_MIN, LOG_STD_MAX].

    def sample(self, obs: Tensor) -> tuple[Tensor, Tensor]
    # Reparameterization trick: z = mean + std * ε, ε ~ N(0, I).
    # Applies tanh squashing; corrects log_prob for the change of variables.
    # → (action ∈ (−1,1)^d, log_prob)

    def deterministic_action(self, obs: Tensor) -> Tensor
    # → tanh(mean); used during evaluation and rendering.
    # Wrapped in torch.no_grad().
```

---

## Phase 2 — `sac/critic.py`

```python
# Depends on: sac.utils

class QNetwork(nn.Module):
    """Single Q-network. Input: concatenation of obs and action."""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
    ) -> None
    # MLP built from Linear + activation blocks ending in a 1-unit output layer.
    # Apply `init_weights_orthogonal` to all layers, then overwrite the final
    # linear layer with `init_weights_uniform(..., scale=3e-3)` for Q-value outputs.

    def forward(self, obs: Tensor, action: Tensor) -> Tensor
    # Concatenates obs and action along dim=1.
    # → Q-value, shape (B, 1)


class TwinCritic(nn.Module):
    """Clipped double-Q critic: two independent Q-networks plus their target copies."""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
    ) -> None
    # Attributes: q1, q2, q1_target, q2_target (all QNetwork instances).
    # Target networks initialized as hard copies; gradients disabled on targets.

    def forward(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]
    # → (q1(obs, action), q2(obs, action))

    def target_values(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]
    # → (q1_target(obs, action), q2_target(obs, action))
    # Called inside train_step with torch.no_grad().

    def soft_update_targets(self, tau: float) -> None
    # Calls soft_update(q1_target, q1, tau) and soft_update(q2_target, q2, tau).
```

---

## Phase 3 — `sac/agent.py`

```python
# Depends on: sac.actor, sac.critic, sac.replay_buffer, sac.config, sac.utils

class SACAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: SACConfig) -> None
    # Constructs: actor (GaussianActor), critic (TwinCritic), replay_buffer (ReplayBuffer).
    # Optimizers: actor_optim, critic_optim (both networks share one optimizer), alpha_optim.
    # log_alpha: nn.Parameter initialized to log(cfg.alpha_init).
    # target_entropy: cfg.target_entropy if set, else −action_dim.

    @property
    def alpha(self) -> float
    # Returns exp(log_alpha).detach().item()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray
    # Converts obs to tensor; calls actor.deterministic_action or actor.sample.
    # Returns action as a NumPy array. No gradient tracking.

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None
    # Delegates to replay_buffer.add.

    def train_step(self) -> dict[str, float]
    # Samples one batch from replay_buffer.
    # Calls _compute_critic_loss, _compute_actor_loss, _compute_alpha_loss.
    # Updates critic, then actor, then alpha.
    # Applies gradient clipping on critic if cfg.gradient_clip_norm > 0.
    # Calls critic.soft_update_targets(cfg.tau).
    # → {"actor_loss", "critic_loss", "alpha_loss", "alpha", "mean_q", "entropy"}

    def _compute_critic_loss(self, batch: dict[str, Tensor]) -> Tensor
    # Computes Bellman targets:
    #   next_action, next_log_prob = actor.sample(next_obs)
    #   q_target = min(q1_target, q2_target) − alpha * next_log_prob
    #   y = reward + gamma * (1 − done) * q_target
    # MSE loss over both Q-networks: (q1 − y)² + (q2 − y)².

    def _compute_actor_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]
    # action, log_prob = actor.sample(obs)
    # q = min(q1, q2) for sampled actions (no target networks)
    # actor_loss = mean(alpha * log_prob − q)
    # → (actor_loss, log_prob.detach())

    def _compute_alpha_loss(self, log_prob: Tensor) -> Tensor
    # alpha_loss = mean(−log_alpha * (log_prob + target_entropy))

    def save(self, checkpoint_path: str | Path, save_buffer: bool = False) -> None
    # Saves: actor.pt, critic.pt, critic_target.pt, log_alpha.pt,
    #        actor_optim.pt, critic_optim.pt, alpha_optim.pt, meta.json.
    # If `save_buffer` is True, also writes `replay_buffer.npz` (heavy).
    # Use `save_buffer=False` for periodic snapshot checkpoints (step_N/),
    # and `save_buffer=True` for the continuously-updated `latest/` checkpoint
    # and the SIGINT handler to guarantee resumability without creating
    # many large files.

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> "SACAgent"
    # Reads meta.json to reconstruct SACConfig and dimension info.
    # Loads all .pt files into the corresponding networks and optimizers.
    # Returns a fully restored SACAgent.
    # Replay buffer loading is handled separately in train.py.
```

---

## Phase 4 — `sac/logger.py`

```python
# Depends on: sac.config, sac.utils

class RunLogger:
    def __init__(self, log_dir: Path) -> None
    # Opens (or appends to) train.csv, eval.csv, events.jsonl in log_dir.
    # Writes CSV headers if files are newly created.
    # train.csv columns: step, actor_loss, critic_loss, alpha_loss, alpha, mean_q, entropy
    # eval.csv columns:  step, mean_return, std_return, min_return, max_return,
    #                    n_episodes, mean_episode_duration

    def log_train(self, step: int, metrics: dict[str, float]) -> None
    # Writes one row to train.csv.

    def log_eval(self, step: int, metrics: dict[str, float]) -> None
    # Writes one row to eval.csv.

    def log_event(self, event_type: str, payload: dict) -> None
    # Appends one JSON line to events.jsonl:
    # {"timestamp": ..., "event": event_type, **payload}

    def flush(self) -> None
    # Flushes all open file handles.

    def close(self) -> None
    # Flushes and closes all file handles.
```

---

## Phase 5 — `train.py`

```python
# Depends on: sac.agent, sac.config, sac.logger, sac.utils

def parse_args() -> argparse.Namespace
# Registers: --config, --env, --seed, --total_steps, --resume.
# Loads YAML first; CLI flags override YAML values.

def run_eval(
    agent: SACAgent,
    env_id: str,
    n_episodes: int,
    max_steps: int,
) -> dict[str, float]
# Runs n_episodes with deterministic policy; no rendering.
# → {"mean_return", "std_return", "min_return", "max_return",
#    "n_episodes", "mean_episode_duration"}

def setup_signal_handler(
    agent: SACAgent,
    checkpoint_path: Path,
    step: int,
    episode: int,
) -> None
# Registers SIGINT handler.
# On interrupt: calls agent.save_full(latest/), closes logger, exits cleanly.

def train(cfg: SACConfig, resume_path: Path | None) -> None
# Main training loop:
#   1. Build env, agent, logger.
#   2. If resume_path: restore agent weights + replay buffer + step/episode counters.
#   3. Warmup phase: collect cfg.warmup_steps transitions with random actions.
#   4. Training loop (step → store → train every train_freq steps):
#      - Evaluate every eval_freq steps → log_eval.
#      - Save checkpoint (latest/ + step_N/) every checkpoint_freq steps.
#   5. Final evaluation and checkpoint on loop exit.

def main() -> None
```

---

## Phase 5 — `eval.py`

```python
# Depends on: sac.agent, sac.config, sac.utils

def parse_args() -> argparse.Namespace
# Registers: --agent, --config, --env, --episodes, --max_steps_per_env, --policy, --seed.

def load_agent_and_env(
    agent_arg: str,
    cfg: SACConfig,
) -> tuple[SACAgent | str, gym.Env]
# Validation logic:
# - If `agent_arg` is a checkpoint PATH: ignore `cfg` (and any CLI `--config`/`--env`) and reconstruct the environment id and agent state from `meta.json` in the checkpoint directory.
# - If `agent_arg` is `zero` or `random`: use the provided `cfg` (which was loaded from `--config`, defaulting to `configs/default.yaml`), and apply an explicit `--env` override when present.
# Returns: (`SACAgent` or stub, `gym.Env`)

def run_evaluation(
    agent,
    env: gym.Env,
    n_episodes: int,
    max_steps: int,
    deterministic: bool,
) -> dict[str, float]
# → {"mean_return", "std_return", "min_return", "max_return",
#    "n_episodes", "mean_episode_duration"}

def print_summary(metrics: dict[str, float]) -> None
# Prints a formatted terminal summary table.

def main() -> None
```

---

## Phase 5 — `play.py`

```python
# Depends on: sac.agent, sac.config, sac.utils

def parse_args() -> argparse.Namespace
# Registers: --agent, --config, --env, --episodes, --max_steps_per_episode, --policy, --speed.

def load_agent_and_env(
    agent_arg: str,
    cfg: SACConfig,
) -> tuple[SACAgent | str, gym.Env]
# Validation logic:
# - If `agent_arg` is a checkpoint PATH: ignore `cfg` (and any CLI `--config`/`--env`) and reconstruct the environment id and agent state from `meta.json` in the checkpoint directory.
# - If `agent_arg` is `zero` or `random`: use the provided `cfg` (which was loaded from `--config`, defaulting to `configs/default.yaml`), and apply an explicit `--env` override when present.
# Always creates env with `render_mode="human"`.

def run_episode(
    agent,
    env: gym.Env,
    max_steps: int,
    deterministic: bool,
    speed: float,
) -> float
# Runs one rendered episode.
# time.sleep(dt / speed) between steps to respect the speed multiplier.
# Does not modify env physics or timestep.
# → total_reward

def main() -> None
```

---

## Phase 5 — `log_analysis.py`

```python
# Depends on: stdlib only + matplotlib

def parse_args() -> argparse.Namespace
# Registers: --log_dir (required), --smooth (default 10), --save (flag).

def load_csv(path: Path) -> dict[str, list[float]]
# Reads train.csv or eval.csv using the csv stdlib module.
# → column_name → list of float values.

def smooth(values: list[float], window: int) -> list[float]
# Rolling mean with edge padding; returns a list of the same length.

def plot_train_metrics(
    data: dict[str, list[float]],
    smooth_w: int,
    save: bool,
    out_dir: Path,
) -> None
# Subplots: actor_loss, critic_loss, alpha_loss, alpha, entropy.
# Saves to out_dir/train_metrics.png if --save, otherwise plt.show().

def plot_eval_metrics(
    data: dict[str, list[float]],
    smooth_w: int,
    save: bool,
    out_dir: Path,
) -> None
# Subplots: mean_return with ±std shading.
# Saves to out_dir/eval_metrics.png if --save, otherwise plt.show().

def main() -> None
```

---

## Dependency graph

```
utils.py        (no deps)
config.py   ←── utils
actor.py    ←── utils
critic.py   ←── utils
replay_buffer.py (no internal deps; already implemented)
agent.py    ←── actor, critic, replay_buffer, config, utils
logger.py   ←── config, utils

eval.py     ←── agent, config, utils
play.py     ←── agent, config, utils
train.py    ←── agent, config, logger, utils
log_analysis.py ←── stdlib + matplotlib only
```