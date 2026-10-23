import pytest

from ddz_core import RuleConfig, counts_from_ranks, recognize_patterns, generate_plays, beats
from ddz_core.actions import PatternKind
from ddz_core.rules import pattern_from_action, validate_completeness


def labels(text: str):
    return counts_from_ranks(*text.split())


def kinds(text):
    return {p.kind for p in recognize_patterns(labels(text))}


@pytest.mark.parametrize("cards,expected", [
    ("3", PatternKind.SINGLE),
    ("3 3", PatternKind.PAIR),
    ("7 7 7", PatternKind.TRIPLE),
    ("7 7 7 9", PatternKind.TRIPLE_SINGLE),
    ("7 7 7 9 9", PatternKind.TRIPLE_PAIR),
    ("3 4 5 6 7", PatternKind.STRAIGHT),
    ("3 3 4 4 5 5", PatternKind.PAIR_CHAIN),
    ("3 3 3 4 4 4", PatternKind.TRIPLE_CHAIN),
    ("3 3 3 4 4 4 7 8", PatternKind.AIRPLANE_SINGLE),
    ("3 3 3 4 4 4 7 7 8 8", PatternKind.AIRPLANE_PAIR),
    ("5 5 5 5", PatternKind.BOMB),
    ("SJ BJ", PatternKind.ROCKET),
    ("5 5 5 5 3 8", PatternKind.FOUR_TWO_SINGLE),
    ("5 5 5 5 3 3 8 8", PatternKind.FOUR_TWO_PAIR),
])
def test_each_pattern(cards, expected):
    assert expected in kinds(cards)


@pytest.mark.parametrize("cards", [
    "2 3 4 5 6", "10 J Q K A 2", "3 3 4 4 2 2",
    "3 3 3 4 4 4 2 2 2", "3 3 4 4", "SJ BJ 5 5 5 5",
    "4 4 4 4 3 3 3", "3 3 3 SJ BJ", "3 3 3 4 4 4 SJ BJ",
])
def test_invalid_patterns(cards):
    assert not recognize_patterns(labels(cards))


def test_beats():
    def pat(x):
        return recognize_patterns(labels(x))[0]
    assert beats(pat("9 9 9 9"), pat("2"))
    assert beats(pat("SJ BJ"), pat("2 2 2 2"))
    assert beats(pat("10 J Q K A"), pat("9 10 J Q K"))
    assert not beats(pat("9 10 J Q K"), pat("10 J Q K A"))
    assert not beats(pat("3 3 3 3"), pat("SJ BJ"))
    assert not beats(pat("4 4"), pat("3"))
    assert not beats(pat("3 3 3 3"), pat("4 4 4 4"))


def test_generator_exhaustiveness_simple_hand():
    hand = labels("3 3 3 4 4 4 5 5 5 6 7 8")
    actions = generate_plays(hand)
    validate_completeness(hand, actions)
    assert all(sum(action.counts) > 0 for action in actions)
    assert len(actions) == len(set(actions))
    target = recognize_patterns(labels("3 3 3 4 4 4"))[0]
    replies = generate_plays(hand, target)
    assert all(beats(pattern_from_action(a), target) for a in replies)


def test_explicit_ambiguous_interpretation_and_wings_variants():
    cards = labels("3 3 3 4 4 4 5 5")
    assert PatternKind.AIRPLANE_SINGLE in {p.kind for p in recognize_patterns(cards)}
    config = RuleConfig(repeated_airplane_single_wings=False)
    assert PatternKind.AIRPLANE_SINGLE not in {
        p.kind for p in recognize_patterns(cards, config)
    }
    quad = labels("5 5 5 5 8 8")
    assert PatternKind.FOUR_TWO_SINGLE in {p.kind for p in recognize_patterns(quad)}
    strict = RuleConfig(repeated_four_two_singles=False)
    assert PatternKind.FOUR_TWO_SINGLE not in {p.kind for p in recognize_patterns(quad, strict)}


def test_ambiguous_play_pattern_hints_remain_distinct():
    # A chain of 4 triples is also a chain of 3 triples + 3 identical single wings
    # under repeated-wing rules; retain both semantic interpretations.
    cards = labels("3 3 3 4 4 4 5 5 5 6 6 6")
    observed = {p.kind for p in recognize_patterns(cards)}
    assert PatternKind.TRIPLE_CHAIN in observed
    assert PatternKind.AIRPLANE_SINGLE in observed
