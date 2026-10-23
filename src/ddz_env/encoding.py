"""Deterministic, versioned input/output encodings for the RL learner.

Do not pass GameState into this module: PlayerObservation is its sole input.
All features are publicly observable to the selected seat, or belong to its hand.
"""
from __future__ import annotations

from ddz_core.actions import Action, ActionKind, PatternKind
from ddz_core.state import Phase, PlayerObservation

ENCODER_VERSION = "ddz-observation-v1"
ACTION_ENCODER_VERSION = "ddz-action-v1"
PHASES = tuple(Phase)
KINDS = tuple(ActionKind)
PATTERNS = tuple(PatternKind)


def observation_size() -> int:
    # hand(15), played(45), bottom(15), lead(15), remaining(3),
    # seat(3), acting seat(4), landlord(4), caller(4), claimant(4),
    # first bidder(3), last play seat(4), phase(4), passes(1), redeals(1).
    return 15 + 45 + 15 + 15 + 3 + 3 + 4 + 4 + 4 + 4 + 3 + 4 + 4 + 1 + 1


def action_size() -> int:
    # kind(6), rank counts(15), pattern kind(14), main rank(15),
    # chain length(1).
    return len(KINDS) + 15 + len(PATTERNS) + 15 + 1


def _onehot(idx: int | None, size: int) -> list[float]:
    out = [0.0] * size
    out[size - 1 if idx is None else idx] = 1.0
    return out


def encode_observation(obs: PlayerObservation) -> tuple[float, ...]:
    values: list[float] = []
    values.extend(x / 4 for x in obs.own_counts)
    for counts in obs.played_counts_by_seat:
        values.extend(x / 4 for x in counts)
    values.extend(x / 4 for x in obs.public_bottom_counts)
    values.extend(x / 4 for x in obs.last_play_counts)
    values.extend(x / 20 for x in obs.remaining_counts)
    values.extend(_onehot(obs.seat, 3))
    values.extend(_onehot(obs.current_seat, 4))
    values.extend(_onehot(obs.landlord, 4))
    values.extend(_onehot(obs.caller, 4))
    values.extend(_onehot(obs.claimant, 4))
    values.extend(_onehot(obs.first_bidder, 3))
    values.extend(_onehot(obs.last_play_seat, 4))
    values.extend(_onehot(PHASES.index(obs.phase), len(PHASES)))
    # A lead trick has zero passes; after one pass, record 1.
    # Public events are available separately for sequence-aware future models.
    passes = 0
    for event in reversed(obs.public_events):
        if event.kind in ("PLAYED", "TRICK_RESET", "LANDLORD_GRANTED", "DEALT"):
            break
        if event.kind == "PASS":
            passes += 1
    values.append(float(min(passes, 1)))
    values.append(float(min(obs.deal_index, 100)) / 100)
    if len(values) != observation_size():
        raise AssertionError(f"observation length mismatch: {len(values)}")
    if not all(0 <= x <= 1 for x in values):
        raise AssertionError("observation contains out-of-range feature")
    return tuple(values)


def encode_action(action: Action) -> tuple[float, ...]:
    values = _onehot(KINDS.index(action.kind), len(KINDS))
    values.extend(x / 4 for x in action.counts)
    pattern = None if action.pattern_key is None else PatternKind(action.pattern_key[0])
    values.extend([float(p == pattern) for p in PATTERNS])
    rank = None if action.pattern_key is None else action.pattern_key[1]
    values.extend([float(i == rank) for i in range(15)])
    chain = 0 if action.pattern_key is None else action.pattern_key[2]
    values.append(min(chain, 12) / 12)
    if len(values) != action_size():
        raise AssertionError("action length mismatch")
    return tuple(values)
