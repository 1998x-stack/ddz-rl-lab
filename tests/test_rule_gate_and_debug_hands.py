"""Cross-layer rule regression for illegal follow moves."""
import json
import random
import pytest
from ddz_core import counts_from_ranks, generate_plays, recognize_patterns, beats
from ddz_core.actions import ActionKind
from ddz_core.engine import GameEngine, IllegalAction
from ddz_core.rules import pattern_from_action
from ddz_env.environment import DDZEnv

def pat(*ranks):
    return recognize_patterns(counts_from_ranks(*ranks))[0]

def test_pair_two_beats_pair_queen_and_pair_ace():
    target = pat('2','2')
    assert not beats(pat('Q','Q'), target)
    assert not beats(pat('A','A'), target)
    assert beats(pat('3','3','3','3'), target)
    assert beats(pat('SJ','BJ'), target)

def test_generator_rejects_lower_pairs_and_accepts_bombs():
    hand = counts_from_ranks('Q','Q','A','A','3','3','3','3','SJ','BJ')
    choices = generate_plays(hand, pat('2','2'))
    assert not any(a.counts == counts_from_ranks('Q','Q') for a in choices)
    assert not any(a.counts == counts_from_ranks('A','A') for a in choices)
    assert any(a.counts == counts_from_ranks('3','3','3','3') for a in choices)
    assert any(a.counts == counts_from_ranks('SJ','BJ') for a in choices)
    assert all(beats(pattern_from_action(a), pat('2','2')) for a in choices)

@pytest.mark.parametrize('seed', [4,23,97])
def test_every_candidate_is_a_legal_follow_move(seed):
    env = DDZEnv(seed)
    rng = random.Random(seed)
    for _ in range(1200):
        if env.done:
            break
        lead = env.engine.state.last_play
        choices = env.legal_actions()
        assert choices
        assert all(a.kind != ActionKind.PLAY or lead is None
                   or beats(pattern_from_action(a), lead) for a in choices)
        action = choices[rng.randrange(len(choices))]
        env.step_action(action, include_digest=False)
        if action.kind == ActionKind.PLAY and lead is not None:
            assert beats(pattern_from_action(action), lead)
    else:
        pytest.fail('seeded game failed to terminate')

def test_authoritative_engine_rejects_unauthorized_action_immutably():
    engine = GameEngine(42)
    before = engine.state_digest()
    action = next(a for a in generate_plays(counts_from_ranks('Q','Q'))
                  if a.counts == counts_from_ranks('Q','Q'))
    with pytest.raises(IllegalAction):
        engine.step(action)
    assert engine.state_digest() == before

def test_debug_hands_sorted_and_not_in_public_snapshot(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from ddz_core.cards import card_rank
    from ddz_web.api import create_app
    for enabled in [False, True]:
        app = create_app(tmp_path / ('registry' + str(enabled)),
                         tmp_path / ('replays' + str(enabled)),
                         allow_baseline=True, allow_debug_hands=enabled)
        with TestClient(app) as client:
            game = client.post('/api/games',json={'model_ids':['@random']*3,'seed':33})
            assert game.status_code == 201, game.text
            gid = game.json()['game_id']
            assert client.get('/api/debug/config').json()['enabled'] is enabled
            result = client.get('/api/games/' + gid + '/debug/hands')
            assert result.status_code == (200 if enabled else 404)
            if enabled:
                cards = result.json()['hands']
                assert [len(h) for h in cards] == [17,17,17]
                assert len(set(card for hand in cards for card in hand)) == 51
                assert all([card_rank(c) for c in hand]
                           == sorted([card_rank(c) for c in hand],reverse=True)
                           for hand in cards)
            public = client.get('/api/games/' + gid).json()
            assert public['seed'] is None
            assert 'hands' not in json.dumps(public)
            assert client.get('/api/games/' + gid + '/debug/hands',
                              headers={'Host':'untrusted.example'}).status_code == 400
