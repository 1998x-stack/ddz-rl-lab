import random

import pytest

from ddz_core.actions import ActionKind
from ddz_core.engine import IllegalAction, replay
from ddz_core.state import Phase
from ddz_env import DDZEnv, encode_action, encode_observation, observation_size, action_size
from ddz_env.baselines import play_episode, random_action_index
from ddz_env.catalogue import action_catalogue, global_action_id, global_legal_ids


def test_encoder_fixed_sizes_and_bounded_features():
    env = DDZEnv(2026)
    for seat in range(3):
        view = env.observe(seat)
        features = encode_observation(view)
        assert len(features) == observation_size()
        assert all(0 <= v <= 1 for v in features)
        if seat != env.acting_seat:
            assert not view.legal_actions
    for action in env.legal_actions():
        vector = encode_action(action)
        assert len(vector) == action_size()
        assert all(0 <= v <= 1 for v in vector)


def test_no_private_cards_before_landlord_assignment():
    env = DDZEnv(2026)
    for seat in range(3):
        view = env.observe(seat)
        assert sum(view.own_counts) == 17
        assert sum(view.public_bottom_counts) == 0
        assert not any(ev.kind == "LANDLORD_GRANTED" for ev in view.public_events)
        assert len(encode_observation(view)) == observation_size()


def test_legal_global_catalogue_contains_all_local_choices():
    catalogue = action_catalogue()
    assert len(catalogue) == len(set(catalogue))
    env = DDZEnv(2026)
    rng = random.Random(2026)
    for _ in range(100):
        if env.done:
            break
        legal = env.legal_actions()
        indices = global_legal_ids(legal)
        assert len(indices) == len(legal)
        assert all(catalogue[i] == action for i, action in zip(indices, legal))
        assert all(global_action_id(a) < len(catalogue) for a in legal)
        env.step(random_action_index(env, rng))


def test_illegal_index_rejected_without_mutation():
    env = DDZEnv(31)
    original = env.engine.state_digest()
    with pytest.raises(IllegalAction):
        env.step(50)
    with pytest.raises(IllegalAction):
        env.step(-1)
    with pytest.raises(IllegalAction):
        env.step(True)
    assert env.engine.state_digest() == original


def test_training_rewards_zero_before_terminal_and_team_aligned():
    env = DDZEnv(4)
    assert env.rewards() == (0.0, 0.0, 0.0)
    rng = random.Random(4)
    for _ in range(1200):
        if env.done:
            break
        transition = env.step(random_action_index(env, rng))
        if not transition.terminated and not transition.truncated:
            assert transition.rewards == (0.0, 0.0, 0.0)
    assert env.done
    if not env.engine.state.truncated:
        landlord = env.engine.state.landlord
        assert landlord is not None
        values = env.rewards()
        assert sum(values) == 0
        assert values[(landlord + 1) % 3] == values[(landlord + 2) % 3]
        with pytest.raises(RuntimeError):
            env.step(0)


def test_seeded_repeatability_and_replay():
    first, n1 = play_episode(999)
    second, n2 = play_episode(999)
    assert n1 == n2
    assert first.engine.state_digest() == second.engine.state_digest()
    record = first.export_replay()
    assert replay(record).state_digest() == first.engine.state_digest()
    obs = second.reset(999)
    assert obs.phase == Phase.CALL
    assert second.engine.state.step_index == 0


def test_many_complete_episodes():
    for seed in range(15):
        env, steps = play_episode(seed)
        assert env.done and steps < 1200
        assert env.engine.state.winner is not None or env.engine.state.truncated
        env.engine.assert_invariants()


def test_action_mask_only_acting_player():
    env = DDZEnv(300)
    current = env.acting_seat
    assert current in (0, 1, 2)
    assert len(env.observe(current).legal_actions) == 2
    for seat in range(3):
        if seat != current:
            assert env.observe(seat).legal_actions == ()
