from __future__ import annotations

import torch
from torch import Tensor, nn

from sac import utils

Q_INIT_SCALE: float = 3e-3


class QNetwork(nn.Module):
    """Single Q-network: Q(s, a)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if hidden_layers <= 0:
            raise ValueError("hidden_layers must be positive")

        activation_fn = utils.get_activation(activation)
        layers: list[nn.Module] = []
        in_dim = obs_dim + action_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.net.apply(utils.init_weights_orthogonal)
        utils.init_weights_uniform(self.net[-1], Q_INIT_SCALE)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([obs, action], dim=1)
        return self.net(x)


class TwinCritic(nn.Module):
    """Clipped double-Q critic with target networks."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dim, hidden_layers, activation)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dim, hidden_layers, activation)
        self.q1_target = QNetwork(obs_dim, action_dim, hidden_dim, hidden_layers, activation)
        self.q2_target = QNetwork(obs_dim, action_dim, hidden_dim, hidden_layers, activation)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False

    def forward(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def target_values(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            return self.q1_target(obs, action), self.q2_target(obs, action)

    def soft_update_targets(self, tau: float) -> None:
        utils.soft_update(self.q1_target, self.q1, tau)
        utils.soft_update(self.q2_target, self.q2, tau)
