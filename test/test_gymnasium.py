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
