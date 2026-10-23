import json
import random

import pytest

from ddz_core import Action, ActionKind, GameEngine, IllegalAction, RuleConfig, replay
from ddz_core.cards import EMPTY_COUNTS, cards_to_counts
from ddz_core.state import Phase, Team


def simple(kind):
    return Action.simple(kind)


def get_playing_engine(seed=121):
    e = GameEngine(seed)
    caller = e.state.current_seat
    e.step(simple(ActionKind.CALL))
    while e.state.phase == Phase.ROB:
        e.step(simple(ActionKind.DECLINE_ROB))
    assert e.state.landlord == caller
    return e


def test_deal_and_bidding_visibility():
    e = GameEngine(7)
    obs = e.observe(e.state.current_seat)
    assert len(obs.own_physical_cards) == 17
    assert obs.public_bottom_counts == EMPTY_COUNTS
    assert sum(obs.own_counts) == 17
    assert all("bottom" not in str(event.public).lower() or
               "bottom_count" in event.public for event in obs.public_events)
    e = get_playing_engine(7)
    obs = e.observe(1)
    assert sum(obs.public_bottom_counts) == 3
    assert sum(obs.own_counts) == (20 if e.state.landlord == 1 else 17)
    assert obs.landlord is not None
    assert e.state.current_seat == e.state.landlord


def test_all_decline_redeals_reproducibly():
    e = GameEngine(11)
    first = [tuple(h) for h in e.state.hands]
    for _ in range(3):
        e.step(simple(ActionKind.DECLINE_CALL))
    assert e.state.phase == Phase.CALL
    assert e.state.deal_index == 1
    assert first != [tuple(h) for h in e.state.hands]
    assert any(ev.kind == "ALL_DECLINED" for ev in e.state.events)
    assert replay(e.export_replay()).state_digest() == e.state_digest()


def test_no_rob_after_declining_call():
    e = GameEngine(17)
    passed_seat = e.state.current_seat
    e.step(simple(ActionKind.DECLINE_CALL))
    caller = e.state.current_seat
    e.step(simple(ActionKind.CALL))
    while e.state.phase == Phase.ROB:
        assert e.state.current_seat != passed_seat
        e.step(simple(ActionKind.DECLINE_ROB))
    assert e.state.landlord == caller


def test_rob_claimant_changes():
    e = GameEngine(18)
    caller = e.state.current_seat
    e.step(simple(ActionKind.CALL))
    robber = e.state.current_seat
    e.step(simple(ActionKind.ROB))
    assert e.state.claimant == robber
    while e.state.phase == Phase.ROB:
        e.step(simple(ActionKind.DECLINE_ROB))
    assert e.state.landlord == robber
    assert e.state.landlord != caller


def test_invalid_action_is_atomic():
    e = get_playing_engine()
    digest = e.state_digest()
    with pytest.raises(IllegalAction):
        e.step(simple(ActionKind.PASS))
    with pytest.raises(IllegalAction):
        e.step(simple(ActionKind.ROB))
    assert digest == e.state_digest()


def test_two_passes_restore_leading_turn():
    e = get_playing_engine()
    leader = e.state.current_seat
    play = next(a for a in e.legal_actions() if a.kind == ActionKind.PLAY)
    e.step(play)
    e.step(simple(ActionKind.PASS))
    e.step(simple(ActionKind.PASS))
    assert e.state.current_seat == leader
    assert e.state.last_play is None
    assert Action.simple(ActionKind.PASS) not in e.legal_actions()
    assert any(ev.kind == "TRICK_RESET" for ev in e.state.events)


def test_nonacting_player_cannot_see_legal_actions():
    e = get_playing_engine()
    inactive = (e.state.current_seat + 1) % 3
    assert e.observe(inactive).legal_actions == ()
    assert e.observe(e.state.current_seat).legal_actions


def test_clone_isolated():
    e = get_playing_engine()
    x = e.clone()
    x.step(x.legal_actions()[0])
    assert x.state_digest() != e.state_digest()


def test_fixed_seed_determinism_and_replay():
    from ddz_core.cli import simulate
    first = simulate(2026)
    second = simulate(2026)
    assert first.state_digest() == second.state_digest()
    record = json.loads(json.dumps(first.export_replay()))
    restored = replay(record)
    assert first.state_digest() == restored.state_digest()
    with pytest.raises(ValueError, match="mismatch"):
        corrupted = {**record, "state_digest": "123"}
        replay(corrupted)


def test_finished_and_reward_winner_rule():
    from ddz_core.cli import simulate
    for seed in range(12):
        e = simulate(seed)
        assert e.state.phase == Phase.FINISHED
        assert e.state.winner in (Team.LANDLORD, Team.FARMERS)
        seat = e.state.winner_seat
        assert seat is not None and e.state.hands[seat] == []
        assert e.state.winner == (Team.LANDLORD if seat == e.state.landlord else Team.FARMERS)
        assert e.legal_actions() == ()
        assert replay(e.export_replay()).state_digest() == e.state_digest()


def test_redraw_guard_truncates_without_awarding_win():
    e = GameEngine(15, RuleConfig(max_redeals=0))
    for _ in range(3):
        e.step(simple(ActionKind.DECLINE_CALL))
    assert e.state.phase == Phase.FINISHED
    assert e.state.truncated
    assert e.state.winner is None
    assert e.legal_actions() == ()
    assert replay(e.export_replay()).state_digest() == e.state_digest()


def test_observation_event_mutations_cannot_corrupt_engine():
    e = GameEngine(99)
    original = e.state_digest()
    event = e.observe(e.state.current_seat).public_events[0]
    event.public["first_bidder"] = -999
    assert e.state_digest() == original
    assert e.observe(e.state.current_seat).public_events[0].public["first_bidder"] >= 0
