"""Rules and exhaustive suit-invariant candidate generator for classic Dou Dizhu.

Intentional house-rule choices are documented in docs/RULE_SPEC.md. They are NOT
claims that every historical Tencent variant uses the same wing constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, combinations_with_replacement

from .actions import Action, Pattern, PatternKind
from .cards import EMPTY_COUNTS, check_counts


@dataclass(frozen=True, slots=True)
class RuleConfig:
    rule_id: str = "classic_call_rob_2016_reference_v1"
    repeated_airplane_single_wings: bool = True
    repeated_four_two_singles: bool = True
    forbid_double_joker_as_wings: bool = True
    max_redeals: int = 100  # guard to protect demo/training; exceeding aborts, not a loss

    def __post_init__(self) -> None:
        if not self.rule_id or self.max_redeals < 0:
            raise ValueError("invalid rule configuration")


def beats(candidate: Pattern, previous: Pattern) -> bool:
    if previous.kind == PatternKind.ROCKET:
        return False
    if candidate.kind == PatternKind.ROCKET:
        return True
    if candidate.kind == PatternKind.BOMB:
        return previous.kind != PatternKind.BOMB or candidate.main_rank > previous.main_rank
    if previous.kind == PatternKind.BOMB:
        return False
    return (candidate.kind == previous.kind and
            candidate.total_cards == previous.total_cards and
            candidate.chain_length == previous.chain_length and
            candidate.main_rank > previous.main_rank)


def _chains(counts: tuple[int, ...], multiplicity: int, minimum: int):
    for start in range(12):
        for length in range(minimum, 13 - start):
            if all(counts[i] >= multiplicity for i in range(start, start + length)):
                yield start, length
            else:
                break


def _wing_singles(rem: tuple[int, ...], number: int, repeated: bool,
                  forbid_double_joker: bool):
    available = [i for i, count in enumerate(rem) if count]
    choices = (combinations_with_replacement(available, number)
               if repeated else combinations(available, number))
    for ranks in choices:
        if forbid_double_joker and 13 in ranks and 14 in ranks:
            continue
        if any(ranks.count(i) > rem[i] for i in set(ranks)):
            continue
        yield ranks


def _wing_pairs(rem: tuple[int, ...], number: int):
    available = [i for i in range(13) if rem[i] >= 2]
    yield from combinations(available, number)


@lru_cache(maxsize=8192)
def _all_plays(counts: tuple[int, ...], config: RuleConfig) -> tuple[Action, ...]:
    """Enumerate EVERY rank-multiset + interpretation; never prune legal moves.

    Duplicates are collapsed by (15-rank action counts, pattern interpretation).
    Results are deterministic and sorted so training and replay get stable order.
    """
    result: set[Action] = set()

    def add(kind: PatternKind, rank: int, chain: int, selected: list[int]) -> None:
        if any(value > have for value, have in zip(selected, counts)):
            return
        pattern = Pattern(kind, rank, chain, sum(selected))
        result.add(Action.play(tuple(selected), pattern))

    for r, n in enumerate(counts):
        for qty, kind in ((1, PatternKind.SINGLE), (2, PatternKind.PAIR),
                          (3, PatternKind.TRIPLE), (4, PatternKind.BOMB)):
            if n >= qty and not (r >= 13 and qty >= 2):
                arr = [0] * 15
                arr[r] = qty
                add(kind, r, 1, arr)
        if r >= 13 or n < 3:
            continue
        for wing in range(15):
            if wing == r:
                continue
            if counts[wing] >= 1:
                arr = [0] * 15
                arr[r], arr[wing] = 3, 1
                add(PatternKind.TRIPLE_SINGLE, r, 1, arr)
            if counts[wing] >= 2 and wing < 13:
                arr = [0] * 15
                arr[r], arr[wing] = 3, 2
                add(PatternKind.TRIPLE_PAIR, r, 1, arr)
        if n >= 4:
            rem = list(counts)
            rem[r] -= 4
            for wings in _wing_singles(tuple(rem), 2, config.repeated_four_two_singles,
                                       config.forbid_double_joker_as_wings):
                arr = [0] * 15
                arr[r] = 4
                for wing in wings:
                    arr[wing] += 1
                add(PatternKind.FOUR_TWO_SINGLE, r, 1, arr)
            for wings in _wing_pairs(tuple(rem), 2):
                arr = [0] * 15
                arr[r] = 4
                for wing in wings:
                    arr[wing] += 2
                add(PatternKind.FOUR_TWO_PAIR, r, 1, arr)

    if counts[13] and counts[14]:
        arr = [0] * 15
        arr[13] = arr[14] = 1
        add(PatternKind.ROCKET, 14, 1, arr)

    for multiplicity, min_chain, kind in (
        (1, 5, PatternKind.STRAIGHT),
        (2, 3, PatternKind.PAIR_CHAIN),
        (3, 2, PatternKind.TRIPLE_CHAIN),
    ):
        for start, length in _chains(counts, multiplicity, min_chain):
            arr = [0] * 15
            for r in range(start, start + length):
                arr[r] = multiplicity
            add(kind, start + length - 1, length, arr)
            if multiplicity != 3:
                continue
            rem = list(counts)
            for r in range(start, start + length):
                # Wings cannot share main triplet ranks, including spare fourth cards.
                rem[r] = 0
            for wings in _wing_singles(tuple(rem), length,
                                       config.repeated_airplane_single_wings,
                                       config.forbid_double_joker_as_wings):
                chosen = arr.copy()
                for wing in wings:
                    chosen[wing] += 1
                add(PatternKind.AIRPLANE_SINGLE, start + length - 1, length, chosen)
            for wings in _wing_pairs(tuple(rem), length):
                chosen = arr.copy()
                for wing in wings:
                    chosen[wing] += 2
                add(PatternKind.AIRPLANE_PAIR, start + length - 1, length, chosen)

    # Single-stage ordering independent of Python set/hash randomization.
    return tuple(sorted(result, key=lambda action: (
        sum(action.counts), action.pattern_key or (), action.counts)))


def generate_plays(hand_counts: tuple[int, ...],
                   last_play: Pattern | None = None,
                   config: RuleConfig | None = None) -> tuple[Action, ...]:
    """Full non-PASS legal play candidates; caller adds PASS only when following."""
    config = config or RuleConfig()
    available = _all_plays(check_counts(hand_counts), config)
    if last_play is None:
        return available
    return tuple(a for a in available if beats(pattern_from_action(a), last_play))


def pattern_from_action(action: Action) -> Pattern:
    if action.pattern_key is None:
        raise ValueError("action has no pattern")
    kind, main_rank, chain = action.pattern_key
    return Pattern(PatternKind(kind), main_rank, chain, sum(action.counts))


def recognize_patterns(cards: tuple[int, ...], config: RuleConfig | None = None) -> tuple[Pattern, ...]:
    """Return all valid interpretations of a rank multiset (can be ambiguous)."""
    target = check_counts(cards)
    return tuple(pattern_from_action(move) for move in generate_plays(target, config=config)
                 if move.counts == target)


def validate_completeness(hand_counts: tuple[int, ...], actions: tuple[Action, ...]) -> None:
    """Development-time check, not a replacement for exhaustive tests."""
    hand = check_counts(hand_counts)
    for action in actions:
        if action.pattern_key is None or not any(action.counts):
            raise AssertionError("invalid action in PLAY candidate set")
        if any(a > b for a, b in zip(action.counts, hand)):
            raise AssertionError("action uses unavailable cards")
