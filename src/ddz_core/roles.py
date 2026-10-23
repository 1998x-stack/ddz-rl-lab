"""Role identity is a pure game-state function, never a PyTorch dependency."""
from __future__ import annotations
from .state import Phase, PlayerObservation

def role_for_observation(obs: PlayerObservation) -> str:
    if obs.phase in (Phase.CALL, Phase.ROB):
        return "bid"
    if obs.phase != Phase.PLAY or obs.landlord is None:
        raise ValueError("playing policy requires a known landlord")
    if obs.seat == obs.landlord:
        return "landlord"
    return "farmer_down" if obs.seat == (obs.landlord + 1) % 3 else "farmer_up"
