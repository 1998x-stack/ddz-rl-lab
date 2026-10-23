"""Role routing, safe redacted-observation inference, seeded exploration."""
from __future__ import annotations
import random
from dataclasses import dataclass
import torch
from ddz_core.actions import ActionKind
from ddz_core.state import Phase, PlayerObservation
from ddz_env.baselines import reasonable_action_index, reasonable_action_score
from ddz_env.encoding import (encode_action, encode_observation,
                              encode_action_v2, encode_observation_v2)
from .network import CandidateQNetwork, NetworkSpec

ROLES = ("bid", "landlord", "farmer_up", "farmer_down")


from ddz_core.roles import role_for_observation


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

    def encode_observation(self, obs: PlayerObservation) -> tuple[float, ...]:
        return (encode_observation_v2(obs) if self.spec.schema == "ddz-candidate-mlp-v2"
                else encode_observation(obs))

    def encode_action(self, obs: PlayerObservation, action) -> tuple[float, ...]:
        return (encode_action_v2(obs, action) if self.spec.schema == "ddz-candidate-mlp-v2"
                else encode_action(action))

    def act(self, obs: PlayerObservation, rng: random.Random,
            epsilon: float = 0.0) -> Decision:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        if not obs.legal_actions:
            raise ValueError("no legal moves for observation")
        role = role_for_observation(obs)
        model = self.models[role]
        model.eval()
        state = torch.tensor(self.encode_observation(obs), dtype=torch.float32, device=self.device)
        candidates = torch.tensor([self.encode_action(obs, a) for a in obs.legal_actions],
                                  dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            scores = model.score_legal(state, candidates)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("nonfinite action scores")
        exploratory = rng.random() < epsilon
        if self.spec.schema == "ddz-candidate-mlp-v2" and obs.phase == Phase.PLAY:
            actions = obs.legal_actions
            finish = [i for i, action in enumerate(actions)
                      if action.kind == ActionKind.PLAY and action.counts == obs.own_counts]
            if finish:
                idx = max(finish, key=lambda i: float(scores[i]))
            elif (obs.landlord is not None and obs.seat != obs.landlord and
                  obs.last_play_seat is not None and obs.last_play_seat != obs.landlord and
                  obs.remaining_counts[obs.landlord] > 2 and
                  any(a.kind == ActionKind.PASS for a in actions)):
                idx = next(i for i, a in enumerate(actions) if a.kind == ActionKind.PASS)
            elif exploratory:
                idx = reasonable_action_index(obs, rng)
            else:
                adjusted = [float(score) + .15 * reasonable_action_score(obs, action)
                            for score, action in zip(scores, actions)]
                opponents = [seat for seat in range(3) if seat != obs.seat and
                             (obs.seat == obs.landlord or seat == obs.landlord)]
                urgent = (obs.last_play_seat is not None and
                          any(obs.remaining_counts[seat] <= 2 for seat in opponents))
                playable = [i for i, action in enumerate(actions)
                            if action.kind == ActionKind.PLAY]
                eligible = playable if urgent and playable else range(len(actions))
                idx = max(eligible, key=adjusted.__getitem__)
        else:
            idx = rng.randrange(len(obs.legal_actions)) if exploratory else int(scores.argmax().item())
        return Decision(idx, role, float(scores[idx].item()), exploratory)

    def state_dict(self) -> dict:
        return {role: net.state_dict() for role, net in self.models.items()}

    def load_state_dict(self, data: dict) -> None:
        if set(data) != set(ROLES):
            raise ValueError("checkpoint role set mismatch")
        for role in ROLES:
            self.models[role].load_state_dict(data[role], strict=True)
