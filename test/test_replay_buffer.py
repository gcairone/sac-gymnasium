import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sac.replay_buffer import ReplayBuffer


def _make_transition(obs_dim, action_dim, value):
    obs = np.full((obs_dim,), value, dtype=np.float32)
    action = np.full((action_dim,), value + 1, dtype=np.float32)
    reward = float(value)
    done = bool(value % 2)
    next_obs = obs + 0.5
    return obs, action, reward, done, next_obs


def test_replay_buffer_add_and_sample():
    buffer = ReplayBuffer(obs_dim=3, action_dim=2, size=10)

    for i in range(6):
        buffer.add(*_make_transition(3, 2, i))

    assert len(buffer) == 6

    batch = buffer.sample(batch_size=4)
    assert set(batch.keys()) == {"obs", "actions", "rewards", "dones", "next_obs"}

    assert batch["obs"].shape == (4, 3)
    assert batch["actions"].shape == (4, 2)
    assert batch["rewards"].shape == (4, 1)
    assert batch["dones"].shape == (4, 1)
    assert batch["next_obs"].shape == (4, 3)

    assert batch["obs"].device.type == "cpu"
    assert batch["actions"].dtype == torch.float32


def test_replay_buffer_overwrite_on_capacity():
    buffer = ReplayBuffer(obs_dim=2, action_dim=1, size=3)

    for i in range(5):
        buffer.add(*_make_transition(2, 1, i))

    assert len(buffer) == 3
    assert buffer._ptr == 2

    expected = np.stack(
        [
            np.full((2,), 2, dtype=np.float32),
            np.full((2,), 3, dtype=np.float32),
            np.full((2,), 4, dtype=np.float32),
        ],
        axis=0,
    )
    ordered = np.concatenate(
        [buffer._obs_buf[buffer._ptr :], buffer._obs_buf[: buffer._ptr]],
        axis=0,
    )
    assert np.allclose(ordered, expected)


def test_replay_buffer_save_and_load(tmp_path):
    buffer = ReplayBuffer(obs_dim=4, action_dim=3, size=5)

    for i in range(4):
        buffer.add(*_make_transition(4, 3, i))

    save_path = tmp_path / "replay_buffer.npz"
    buffer.save(save_path)

    loaded = ReplayBuffer.load(save_path)

    assert len(loaded) == 4
    assert loaded._ptr == buffer._ptr
    assert np.allclose(loaded._obs_buf, buffer._obs_buf)
    assert np.allclose(loaded._actions_buf, buffer._actions_buf)
    assert np.allclose(loaded._rewards_buf, buffer._rewards_buf)
    assert np.allclose(loaded._dones_buf, buffer._dones_buf)
    assert np.allclose(loaded._next_obs_buf, buffer._next_obs_buf)
