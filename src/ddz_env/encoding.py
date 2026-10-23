"""Deterministic, versioned input/output encodings for the RL learner.

Do not pass GameState into this module: PlayerObservation is its sole input.
All features are publicly observable to the selected seat, or belong to its hand.
"""
from __future__ import annotations

from ddz_core.actions import Action, ActionKind, PatternKind
from ddz_core.cards import FULL_COUNTS
from ddz_core.state import Phase, PlayerObservation

ENCODER_VERSION = "ddz-observation-v1"
ACTION_ENCODER_VERSION = "ddz-action-v1"
ENCODER_VERSION_V2 = "ddz-observation-v2"
ACTION_ENCODER_VERSION_V2 = "ddz-action-v2"
PHASES = tuple(Phase)
KINDS = tuple(ActionKind)
PATTERNS = tuple(PatternKind)


def encoder_ids(model_schema: str) -> dict[str, str]:
    if model_schema == "ddz-candidate-mlp-v1":
        return {"observation": ENCODER_VERSION, "action": ACTION_ENCODER_VERSION}
    if model_schema == "ddz-candidate-mlp-v2":
        return {"observation": ENCODER_VERSION_V2, "action": ACTION_ENCODER_VERSION_V2}
    raise ValueError("unsupported model schema")


def observation_size() -> int:
    # hand(15), played(45), bottom(15), lead(15), remaining(3),
    # seat(3), acting seat(4), landlord(4), caller(4), claimant(4),
    # first bidder(3), last play seat(4), phase(4), passes(1), redeals(1).
    return 15 + 45 + 15 + 15 + 3 + 3 + 4 + 4 + 4 + 4 + 3 + 4 + 4 + 1 + 1


def action_size() -> int:
    # kind(6), rank counts(15), pattern kind(14), main rank(15),
    # chain length(1).
    return len(KINDS) + 15 + len(PATTERNS) + 15 + 1


def observation_size_v2() -> int:
    # v1 + last pattern (including no lead), rank, chain, teammate lead,
    # unseen rank counts, call and rob counts in the current deal.
    return observation_size() + len(PATTERNS) + 1 + 1 + 1 + 1 + 15 + 2


def action_size_v2() -> int:
    # v1 + resulting hand rank counts + eight action-result features.
    return action_size() + 15 + 8


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


def encode_observation_v2(obs: PlayerObservation) -> tuple[float, ...]:
    values = list(encode_observation(obs))
    pattern = obs.last_play
    values.extend(_onehot(PATTERNS.index(pattern.kind) if pattern else None,
                          len(PATTERNS) + 1))
    values.append(pattern.main_rank / 14 if pattern else 0.0)
    values.append(pattern.chain_length / 12 if pattern else 0.0)
    teammate_leads = (obs.landlord is not None and obs.seat != obs.landlord and
                      obs.last_play_seat is not None and
                      obs.last_play_seat != obs.landlord)
    values.append(float(teammate_leads))
    played = [sum(seat[r] for seat in obs.played_counts_by_seat) for r in range(15)]
    values.extend(max(0, total - own - used) / 4
                  for total, own, used in zip(FULL_COUNTS, obs.own_counts, played))
    values.extend((min(obs.call_count + int(obs.caller is not None), 3) / 3,
                   min(obs.rob_count, 3) / 3))
    if len(values) != observation_size_v2() or not all(0 <= x <= 1 for x in values):
        raise AssertionError("v2 observation feature mismatch")
    return tuple(values)


def encode_action_v2(obs: PlayerObservation, action: Action) -> tuple[float, ...]:
    values = list(encode_action(action))
    after = tuple(have - used for have, used in zip(obs.own_counts, action.counts))
    if any(left < 0 for left in after):
        raise ValueError("candidate exceeds observed hand")
    values.extend(left / 4 for left in after)
    values.extend((sum(left == 1 for left in after) / 15,
                   sum(left == 2 for left in after) / 15,
                   sum(left == 3 for left in after) / 15,
                   sum(left == 4 for left in after) / 15,
                   sum(have >= 2 and used > 0 and left == 1
                       for have, used, left in zip(obs.own_counts, action.counts, after)) / 15,
                   sum(action.counts) / 20,
                   float(action.kind == ActionKind.PLAY and not any(after)),
                   (max(0, action.pattern_key[1] - obs.last_play.main_rank) / 14
                    if action.pattern_key and obs.last_play and
                    action.pattern_key[0] == obs.last_play.kind.value else 0.0)))
    if len(values) != action_size_v2() or not all(0 <= x <= 1 for x in values):
        raise AssertionError("v2 action feature mismatch")
    return tuple(values)
