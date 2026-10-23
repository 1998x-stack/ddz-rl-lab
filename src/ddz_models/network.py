"""Batchable, mask-free Q(s,a): legal actions come from authoritative Game Core."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from ddz_env.encoding import action_size, observation_size, action_size_v2, observation_size_v2


@dataclass(frozen=True)
class NetworkSpec:
    state_dim: int = observation_size()
    action_dim: int = action_size()
    hidden: int = 128
    schema: str = "ddz-candidate-mlp-v1"

    def __post_init__(self) -> None:
        expected = {
            "ddz-candidate-mlp-v1": (observation_size(), action_size()),
            "ddz-candidate-mlp-v2": (observation_size_v2(), action_size_v2()),
        }
        if (self.hidden <= 0 or self.schema not in expected or
                (self.state_dim, self.action_dim) != expected[self.schema]):
            raise ValueError("invalid or incompatible network specification")

    @classmethod
    def v2(cls, hidden: int = 128) -> "NetworkSpec":
        return cls(observation_size_v2(), action_size_v2(), hidden,
                   "ddz-candidate-mlp-v2")

    @classmethod
    def from_dict(cls, value: dict) -> "NetworkSpec":
        return cls(state_dim=int(value["state_dim"]), action_dim=int(value["action_dim"]),
                   hidden=int(value["hidden"]), schema=str(value["schema"]))


class CandidateQNetwork(nn.Module):
    """Encode state once, batch-score current candidates; outputs uncalibrated Q-values."""

    def __init__(self, spec: NetworkSpec | None = None):
        super().__init__()
        self.spec = spec or NetworkSpec()
        h = self.spec.hidden
        self.state_encoder = nn.Sequential(nn.Linear(self.spec.state_dim, h), nn.ReLU(),
                                           nn.Linear(h, h), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(self.spec.action_dim, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or actions.ndim != 2 or states.shape[0] != actions.shape[0]:
            raise ValueError("states/actions must be aligned 2-D batches")
        return self.head(torch.cat([self.state_encoder(states),
                                    self.action_encoder(actions)], dim=-1)).squeeze(-1)

    def score_legal(self, state: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        """1-D state with K x action-dim candidates. No illegal action scored."""
        if state.ndim != 1 or legal.ndim != 2 or legal.shape[0] == 0:
            raise ValueError("expected one state and nonempty legal-action matrix")
        x = self.state_encoder(state.unsqueeze(0)).expand(legal.shape[0], -1)
        y = self.action_encoder(legal)
        return self.head(torch.cat([x, y], dim=-1)).squeeze(-1)
