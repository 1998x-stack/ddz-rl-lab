"""Stable, rank-based action schema for future RL training."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .cards import EMPTY_COUNTS, check_counts


class ActionKind(str, Enum):
    CALL = "CALL"
    DECLINE_CALL = "DECLINE_CALL"
    ROB = "ROB"
    DECLINE_ROB = "DECLINE_ROB"
    PLAY = "PLAY"
    PASS = "PASS"


class PatternKind(str, Enum):
    SINGLE = "SINGLE"
    PAIR = "PAIR"
    TRIPLE = "TRIPLE"
    TRIPLE_SINGLE = "TRIPLE_SINGLE"
    TRIPLE_PAIR = "TRIPLE_PAIR"
    STRAIGHT = "STRAIGHT"
    PAIR_CHAIN = "PAIR_CHAIN"
    TRIPLE_CHAIN = "TRIPLE_CHAIN"
    AIRPLANE_SINGLE = "AIRPLANE_SINGLE"
    AIRPLANE_PAIR = "AIRPLANE_PAIR"
    FOUR_TWO_SINGLE = "FOUR_TWO_SINGLE"
    FOUR_TWO_PAIR = "FOUR_TWO_PAIR"
    BOMB = "BOMB"
    ROCKET = "ROCKET"


@dataclass(frozen=True, slots=True)
class Pattern:
    kind: PatternKind
    main_rank: int
    chain_length: int
    total_cards: int

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.kind.value, self.main_rank, self.chain_length)


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    counts: tuple[int, ...] = EMPTY_COUNTS
    pattern_key: tuple[str, int, int] | None = None

    @classmethod
    def play(cls, counts: tuple[int, ...], pattern: Pattern) -> "Action":
        return cls(ActionKind.PLAY, check_counts(counts), pattern.key)

    @classmethod
    def simple(cls, kind: ActionKind) -> "Action":
        if kind == ActionKind.PLAY:
            raise ValueError("PLAY must provide rank counts and a pattern")
        return cls(kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "counts": list(self.counts),
            "pattern_key": list(self.pattern_key) if self.pattern_key else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        kind = ActionKind(data["kind"])
        counts = check_counts(data.get("counts", EMPTY_COUNTS))
        key = data.get("pattern_key")
        if key is not None:
            if len(key) != 3:
                raise ValueError("pattern_key must have 3 entries")
            key = (str(key[0]), int(key[1]), int(key[2]))
        if kind != ActionKind.PLAY and (counts != EMPTY_COUNTS or key is not None):
            raise ValueError("non-PLAY actions must not specify cards")
        if kind == ActionKind.PLAY and (sum(counts) == 0 or key is None):
            raise ValueError("PLAY action needs cards and a pattern")
        return cls(kind, counts, key)
