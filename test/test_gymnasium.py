import pytest

gym = pytest.importorskip("gymnasium")
import numpy as np


ENV_IDS = [
    "InvertedPendulum-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_headless_basic_api(env_id):
    try:
        env = gym.make(env_id)
    except Exception as e:
        pytest.skip(f"Could not create environment {env_id}: {e}")

    try:
        obs, info = env.reset(seed=0)
    except TypeError:
        obs = env.reset()
        info = {}

    assert obs is not None

    for _ in range(5):
        action = env.action_space.sample()
        step_result = env.step(action)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            # legacy gym: (obs, reward, done, info)
            obs, reward, done, info = step_result

        assert reward is not None
        assert isinstance(info, dict)

    env.close()


def test_make_env_rescales_actions():
    from sac import utils

    try:
        plain = utils.make_env("InvertedPendulum-v5")
        scaled = utils.make_env("InvertedPendulum-v5", rescale_actions=True)
    except Exception as e:
        pytest.skip(f"Could not create environment: {e}")

    # InvertedPendulum-v5 has bounds [-3, 3]; the rescaled env exposes [-1, 1].
    assert np.allclose(plain.action_space.high, 3.0)
    assert np.allclose(scaled.action_space.low, -1.0)
    assert np.allclose(scaled.action_space.high, 1.0)

    plain.reset(seed=0)
    scaled.reset(seed=0)
    obs_plain, *_ = plain.step(np.array([3.0], dtype=np.float32))
    obs_scaled, *_ = scaled.step(np.array([1.0], dtype=np.float32))
    assert np.allclose(obs_plain, obs_scaled)

    plain.close()
    scaled.close()
