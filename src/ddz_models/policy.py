"""Role routing, safe redacted-observation inference, seeded exploration."""
from __future__ import annotations
import random
from dataclasses import dataclass
import torch
from ddz_core.state import Phase, PlayerObservation
from ddz_env.encoding import encode_action, encode_observation
from .network import CandidateQNetwork, NetworkSpec

ROLES = ("bid", "landlord", "farmer_up", "farmer_down")


def role_for_observation(obs: PlayerObservation) -> str:
    if obs.phase in (Phase.CALL, Phase.ROB):
        return "bid"
    if obs.phase != Phase.PLAY or obs.landlord is None:
        raise ValueError("cannot select playing policy until landlord is known")
    if obs.seat == obs.landlord:
        return "landlord"
    # Seat order progresses clockwise +1; landlord+1 is DOWN, +2 is UP.
    return "farmer_down" if obs.seat == (obs.landlord + 1) % 3 else "farmer_up"


@dataclass(frozen=True)
class Decision:
    local_index: int
    role: str
    q_value: float
    explored: bool


class RoleBundle:
    def __init__(self, spec: NetworkSpec | None = None, device: str = "cpu"):
        self.spec = spec or NetworkSpec()
        self.device = torch.device(device)
        self.models = {role: CandidateQNetwork(self.spec).to(self.device) for role in ROLES}

    def act(self, obs: PlayerObservation, rng: random.Random,
            epsilon: float = 0.0) -> Decision:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        if not obs.legal_actions:
            raise ValueError("no legal moves for observation")
        role = role_for_observation(obs)
        model = self.models[role]
        model.eval()
        state = torch.tensor(encode_observation(obs), dtype=torch.float32, device=self.device)
        candidates = torch.tensor([encode_action(a) for a in obs.legal_actions],
                                  dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            scores = model.score_legal(state, candidates)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("nonfinite action scores")
        exploratory = rng.random() < epsilon
        idx = rng.randrange(len(obs.legal_actions)) if exploratory else int(scores.argmax().item())
        return Decision(idx, role, float(scores[idx].item()), exploratory)

    def state_dict(self) -> dict:
        return {role: net.state_dict() for role, net in self.models.items()}

    def load_state_dict(self, data: dict) -> None:
        if set(data) != set(ROLES):
            raise ValueError("checkpoint role set mismatch")
        for role in ROLES:
            self.models[role].load_state_dict(data[role], strict=True)
