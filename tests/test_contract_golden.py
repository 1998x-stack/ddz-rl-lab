"""Frozen rule/action contract: every accepted move belongs to the engine's exact legal set."""
import random
import pytest
from ddz_core import (Action, ActionKind, GameEngine, IllegalAction, beats,
                      counts_from_ranks, generate_plays, recognize_patterns, replay)
from ddz_core.state import Phase

def pattern(*cards):
    return recognize_patterns(counts_from_ranks(*cards))[0]

@pytest.mark.parametrize("candidate,leader,expected", [
    (("Q","Q"),("2","2"),False),
    (("A","A"),("2","2"),False),
    (("2","2"),("Q","Q"),True),
    (("3","3","3","3"),("2","2"),True),
    (("SJ","BJ"),("2","2"),True),
])
def test_rank_golden_cases(candidate,leader,expected):
    assert beats(pattern(*candidate), pattern(*leader)) is expected

def test_invalid_follow_cannot_enter_action_candidates():
    lead = pattern("2","2")
    hand = counts_from_ranks("Q","Q","A","A","3","3","3","3")
    legal=generate_plays(hand, lead)
    assert all(a.counts != counts_from_ranks("Q","Q") for a in legal)
    assert all(a.counts != counts_from_ranks("A","A") for a in legal)
    assert any(a.counts == counts_from_ranks("3","3","3","3") for a in legal)

def test_illegal_actions_do_not_mutate_game():
    engine=GameEngine(2026)
    first=engine.state_digest()
    with pytest.raises(IllegalAction):
        engine.step(Action.simple(ActionKind.PASS))
    assert engine.state_digest()==first
    assert not engine.action_history

@pytest.mark.parametrize("seed", [3,7,17,31,42,91,2026,2093])
def test_full_replay_action_contract(seed):
    engine=GameEngine(seed)
    rng=random.Random(seed ^ 0x512)
    for _ in range(800):
        if engine.state.phase == Phase.FINISHED:
            break
        actions=engine.legal_actions()
        assert actions
        action=rng.choice(actions)
        old_digest=engine.state_digest()
        engine.step(action)
        assert engine.state_digest()!=old_digest
    assert engine.state.phase==Phase.FINISHED
    restored=replay(engine.export_replay())
    assert restored.state_digest()==engine.state_digest()
