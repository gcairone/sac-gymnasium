import pytest

torch = pytest.importorskip("torch")

from sac.actor import GaussianActor, HEAD_INIT_SCALE, LOG_STD_MAX, LOG_STD_MIN


def test_actor_forward_shapes_and_clamp():
    actor = GaussianActor(
        obs_dim=4,
        action_dim=2,
        hidden_dim=16,
        hidden_layers=2,
        activation="relu",
    )
    obs = torch.randn(5, 4)
    mean, log_std = actor(obs)

    assert mean.shape == (5, 2)
    assert log_std.shape == (5, 2)
    assert torch.all(log_std <= LOG_STD_MAX)
    assert torch.all(log_std >= LOG_STD_MIN)


def test_actor_sample_action_range_and_log_prob():
    actor = GaussianActor(
        obs_dim=3,
        action_dim=2,
        hidden_dim=32,
        hidden_layers=2,
        activation="tanh",
    )
    obs = torch.randn(7, 3)
    action, log_prob = actor.sample(obs)

    assert action.shape == (7, 2)
    assert log_prob.shape == (7, 1)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert torch.isfinite(log_prob).all()


def test_actor_deterministic_action_matches_mean():
    actor = GaussianActor(
        obs_dim=2,
        action_dim=1,
        hidden_dim=8,
        hidden_layers=1,
        activation="elu",
    )
    obs = torch.randn(4, 2)
    mean, _ = actor(obs)
    deterministic = actor.deterministic_action(obs)

    assert deterministic.shape == (4, 1)
    assert not deterministic.requires_grad
    assert torch.allclose(deterministic, torch.tanh(mean))


def test_actor_head_init_scale():
    actor = GaussianActor(
        obs_dim=2,
        action_dim=2,
        hidden_dim=8,
        hidden_layers=2,
        activation="silu",
    )
    assert actor.mean_head.weight.abs().max().item() <= HEAD_INIT_SCALE + 1e-6
    assert actor.log_std_head.weight.abs().max().item() <= HEAD_INIT_SCALE + 1e-6


def test_actor_log_std_clamping_extremes():
    actor = GaussianActor(
        obs_dim=2,
        action_dim=2,
        hidden_dim=8,
        hidden_layers=2,
        activation="relu",
    )
    obs = torch.zeros(3, 2)

    actor.log_std_head.weight.data.zero_()
    actor.log_std_head.bias.data.fill_(10.0)
    _, log_std_max = actor(obs)
    assert torch.allclose(log_std_max, torch.full_like(log_std_max, LOG_STD_MAX))

    actor.log_std_head.bias.data.fill_(-10.0)
    _, log_std_min = actor(obs)
    assert torch.allclose(log_std_min, torch.full_like(log_std_min, LOG_STD_MIN))
