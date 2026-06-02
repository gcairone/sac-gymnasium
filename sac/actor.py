from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from sac import utils

LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0
HEAD_INIT_SCALE: float = 3e-3


class GaussianActor(nn.Module):
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
        in_dim = obs_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.trunk.apply(utils.init_weights_orthogonal)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        utils.init_weights_uniform(self.mean_head, HEAD_INIT_SCALE)
        utils.init_weights_uniform(self.log_std_head, HEAD_INIT_SCALE)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z).sum(dim=1, keepdim=True)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        log_prob = log_prob - correction
        return action, log_prob

    def deterministic_action(self, obs: Tensor) -> Tensor:
        with torch.no_grad():
            mean, _ = self.forward(obs)
            return torch.tanh(mean)
