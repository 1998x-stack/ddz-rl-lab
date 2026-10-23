"""52 suit cards plus two unique jokers; ranks 3..2, small/big joker."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

RANKS = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "SJ", "BJ")
RANK_TO_INDEX = {rank: i for i, rank in enumerate(RANKS)}
EMPTY_COUNTS: tuple[int, ...] = (0,) * 15
FULL_COUNTS: tuple[int, ...] = (4,) * 13 + (1, 1)


def full_deck() -> tuple[int, ...]:
    return tuple(range(54))


def card_rank(card_id: int) -> int:
    if not isinstance(card_id, int) or isinstance(card_id, bool) or not (0 <= card_id < 54):
        raise ValueError(f"invalid physical card ID: {card_id!r}")
    return card_id % 13 if card_id < 52 else (13 if card_id == 52 else 14)


def card_name(card_id: int) -> str:
    r = card_rank(card_id)
    return ("♠", "♥", "♣", "♦")[card_id // 13] + RANKS[r] if r < 13 else RANKS[r]


def check_counts(counts: Iterable[int], *, deck_limit: bool = True) -> tuple[int, ...]:
    result = tuple(counts)
    if len(result) != 15:
        raise ValueError("card counts must have exactly 15 ranks")
    if any(not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in result):
        raise ValueError("card counts must be nonnegative integers")
    if deck_limit and any(c > max_c for c, max_c in zip(result, FULL_COUNTS)):
        raise ValueError("card counts exceed a standard 54-card deck")
    return result


def cards_to_counts(cards: Iterable[int]) -> tuple[int, ...]:
    physical = tuple(cards)
    if len(set(physical)) != len(physical):
        raise ValueError("duplicate physical card")
    counts = [0] * 15
    for card in physical:
        counts[card_rank(card)] += 1
    return check_counts(counts)


def counts_from_ranks(*ranks: str) -> tuple[int, ...]:
    unknown = set(ranks) - set(RANK_TO_INDEX)
    if unknown:
        raise ValueError(f"unknown ranks: {unknown}")
    counter = Counter(ranks)
    return check_counts(tuple(counter.get(rank, 0) for rank in RANKS))


def take_physical_cards(hand: Iterable[int], required: Iterable[int]) -> tuple[int, ...]:
    """Resolve suitless model action to deterministic physical IDs from the actor's own hand."""
    counts = check_counts(required)
    candidates = sorted(hand)
    chosen: list[int] = []
    for rank, quantity in enumerate(counts):
        matching = [cid for cid in candidates if card_rank(cid) == rank]
        if len(matching) < quantity:
            raise ValueError(f"hand lacks {quantity} copies of {RANKS[rank]}")
        chosen.extend(matching[:quantity])
    return tuple(chosen)
