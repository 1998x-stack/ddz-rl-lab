"""Authoritative state and redacted player observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum
from typing import Any

from .actions import Action, Pattern
from .cards import EMPTY_COUNTS


class Phase(str, Enum):
    CALL = "CALL"
    ROB = "ROB"
    PLAY = "PLAY"
    FINISHED = "FINISHED"


class Team(str, Enum):
    LANDLORD = "LANDLORD"
    FARMERS = "FARMERS"


@dataclass(frozen=True, slots=True)
class GameEvent:
    sequence: int
    kind: str
    actor: int | None
    public: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "kind": self.kind, "actor": self.actor, "public": deepcopy(self.public)}


@dataclass(slots=True)
class GameState:
    seed: int
    rule_id: str
    deal_index: int
    phase: Phase
    hands: list[list[int]]
    bottom_cards: list[int]
    first_bidder: int
    current_seat: int | None
    call_passed: list[bool] = field(default_factory=lambda: [False] * 3)
    call_count: int = 0
    caller: int | None = None
    claimant: int | None = None
    rob_turns: list[int] = field(default_factory=list)
    rob_cursor: int = 0
    rob_count: int = 0
    landlord: int | None = None
    last_play: Pattern | None = None
    last_play_counts: tuple[int, ...] = EMPTY_COUNTS
    last_play_seat: int | None = None
    passes: int = 0
    played_by: list[list[int]] = field(default_factory=lambda: [[], [], []])
    winner: Team | None = None
    truncated: bool = False
    winner_seat: int | None = None
    step_index: int = 0
    events: list[GameEvent] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """Intentionally no other player's hand or unrevealed bottom cards."""
    seat: int
    phase: Phase
    current_seat: int | None
    own_physical_cards: tuple[int, ...]
    own_counts: tuple[int, ...]
    remaining_counts: tuple[int, int, int]
    played_counts_by_seat: tuple[tuple[int, ...], ...]
    public_bottom_counts: tuple[int, ...]
    landlord: int | None
    caller: int | None
    claimant: int | None
    first_bidder: int
    last_play: Pattern | None
    last_play_counts: tuple[int, ...]
    last_play_seat: int | None
    public_events: tuple[GameEvent, ...]
    legal_actions: tuple[Action, ...]
    deal_index: int
    step_index: int
    winner: Team | None
