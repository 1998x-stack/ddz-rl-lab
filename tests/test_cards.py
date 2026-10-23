import pytest

from ddz_core.cards import (
    FULL_COUNTS, RANKS, card_name, card_rank, cards_to_counts,
    check_counts, counts_from_ranks, full_deck, take_physical_cards,
)


def test_full_deck():
    deck = full_deck()
    assert len(deck) == len(set(deck)) == 54
    assert cards_to_counts(deck) == FULL_COUNTS
    assert card_rank(52) == 13
    assert card_rank(53) == 14
    assert card_name(53) == "BJ"
    assert RANKS[0] == "3"


def test_count_validation():
    assert counts_from_ranks("3", "3", "BJ")[0] == 2
    with pytest.raises(ValueError):
        counts_from_ranks("Z")
    with pytest.raises(ValueError):
        check_counts([0] * 14)
    with pytest.raises(ValueError):
        check_counts([5] + [0] * 14)
    with pytest.raises(ValueError):
        cards_to_counts([1, 1])
    with pytest.raises(ValueError):
        card_rank(54)


def test_physical_selection_is_deterministic():
    ids = (0, 13, 26, 39, 52)
    desired = counts_from_ranks("3", "3", "SJ")
    assert take_physical_cards(ids, desired) == (0, 13, 52)
    with pytest.raises(ValueError):
        take_physical_cards([0], desired)
