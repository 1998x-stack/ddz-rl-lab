"""Human seat uses a private hand view and authoritative legal actions."""
import json
import pytest

pytest.importorskip('fastapi')
pytest.importorskip('torch')

from fastapi.testclient import TestClient

from ddz_core.cards import card_rank
from ddz_web.api import create_app
from ddz_web.arena import ArenaManager


def test_human_turn_and_recovery(tmp_path):
    registry = tmp_path / 'registry'
    replays = tmp_path / 'replays'
    app = create_app(registry, replays, allow_baseline=True)
    with TestClient(app) as client:
        created = client.post('/api/games', json={
            'model_ids': ['@human', '@reasonable', '@reasonable'], 'seed': 2026,
        }).json()
        game_id = created['game_id']
        public = client.get(f'/api/games/{game_id}').json()
        assert 'own_counts' not in json.dumps(public)
        for _ in range(4):
            hand = client.get(f'/api/games/{game_id}/hand').json()
            if hand['your_turn']:
                break
            state = client.post(f'/api/games/{game_id}/control', json={'action': 'step'}).json()
            assert state['step_index'] > 0
        assert hand['your_turn'] and sum(hand['own_counts']) == 17
        assert hand['legal_actions']
        assert client.post(f'/api/games/{game_id}/action', json={
            'index': 999999, 'expected_step': hand['step_index'],
        }).status_code == 409
        assert client.post(f'/api/games/{game_id}/action', json={
            'index': 0, 'expected_step': hand['step_index'] - 1,
        }).status_code == 409
        moved = client.post(f'/api/games/{game_id}/action', json={
            'index': 0, 'expected_step': hand['step_index'],
        })
        assert moved.status_code == 200
        assert moved.json()['step_index'] >= hand['step_index'] + 1
    restored = ArenaManager(registry, replays, allow_baseline=True)
    assert restored.get(game_id).snapshot()['step_index'] >= hand['step_index'] + 1


def test_human_seat_restrictions(tmp_path):
    app = create_app(tmp_path / 'registry', tmp_path / 'replays', allow_baseline=True)
    with TestClient(app) as client:
        assert client.get('/').headers['cache-control'] == 'no-store'
        assert client.get('/static/handview.js').headers['cache-control'] == 'no-store'
        assert client.post('/api/games', json={
            'model_ids': ['@reasonable', '@human', '@reasonable'],
        }).status_code == 400
        game = client.post('/api/games', json={
            'model_ids': ['@reasonable'] * 3,
        }).json()
        assert client.get(f"/api/games/{game['game_id']}/hand").status_code == 404
        spectator = client.get(f"/api/games/{game['game_id']}/spectator/hands")
        assert spectator.status_code == 200
        payload = spectator.json()
        assert payload['schema'] == 'ddz-spectator-hands-v1'
        assert [len(cards) for cards in payload['hands']] == [17, 17, 17]
        for cards in payload['hands']:
            assert cards == sorted(cards, key=lambda card: (-card_rank(card), card))


def test_human_game_cannot_reveal_opponent_hands(tmp_path):
    app = create_app(tmp_path / 'registry', tmp_path / 'replays', allow_baseline=True)
    with TestClient(app) as client:
        game = client.post('/api/games', json={
            'model_ids': ['@human', '@reasonable', '@reasonable'],
        }).json()
        assert client.get(f"/api/games/{game['game_id']}/spectator/hands").status_code == 403
