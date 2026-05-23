import pytest

torch = pytest.importorskip("torch")

from sac.critic import QNetwork, Q_INIT_SCALE, TwinCritic


def test_qnetwork_output_shape_and_init_scale():
    critic = QNetwork(
        obs_dim=3,
        action_dim=2,
        hidden_dim=16,
        hidden_layers=2,
        activation="relu",
    )
    obs = torch.randn(5, 3)
    action = torch.randn(5, 2)
    q = critic(obs, action)

    assert q.shape == (5, 1)

    linear_layers = [layer for layer in critic.net if isinstance(layer, torch.nn.Linear)]
    assert linear_layers[-1].weight.abs().max().item() <= Q_INIT_SCALE + 1e-6


def test_twin_critic_targets_frozen_and_independent():
    critic = TwinCritic(
        obs_dim=4,
        action_dim=3,
        hidden_dim=8,
        hidden_layers=2,
        activation="tanh",
    )

    for source, target in zip(critic.q1.parameters(), critic.q1_target.parameters()):
        assert torch.allclose(source, target)
    for source, target in zip(critic.q2.parameters(), critic.q2_target.parameters()):
        assert torch.allclose(source, target)

    with torch.no_grad():
        next(critic.q1.parameters()).add_(1.0)

    assert any(
        not torch.allclose(source, target)
        for source, target in zip(critic.q1.parameters(), critic.q1_target.parameters())
    )

    assert all(not param.requires_grad for param in critic.q1_target.parameters())
    assert all(not param.requires_grad for param in critic.q2_target.parameters())


def test_soft_update_targets_moves_weights():
    critic = TwinCritic(
        obs_dim=2,
        action_dim=1,
        hidden_dim=8,
        hidden_layers=2,
        activation="elu",
    )

    with torch.no_grad():
        for param in critic.q1.parameters():
            param.fill_(1.0)
        for param in critic.q2.parameters():
            param.fill_(1.0)
        for param in critic.q1_target.parameters():
            param.zero_()
        for param in critic.q2_target.parameters():
            param.zero_()

    critic.soft_update_targets(tau=0.5)

    for param in critic.q1_target.parameters():
        assert torch.allclose(param, torch.full_like(param, 0.5))
    for param in critic.q2_target.parameters():
        assert torch.allclose(param, torch.full_like(param, 0.5))


def test_target_values_disable_grad_on_inputs():
    critic = TwinCritic(
        obs_dim=3,
        action_dim=2,
        hidden_dim=8,
        hidden_layers=2,
        activation="silu",
    )
    obs = torch.randn(4, 3, requires_grad=True)
    action = torch.randn(4, 2, requires_grad=True)

    q1_target, q2_target = critic.target_values(obs, action)

    assert not q1_target.requires_grad
    assert not q2_target.requires_grad
