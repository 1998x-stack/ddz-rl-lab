"""Spectator API, registered inference, concurrency guards and replay invariants."""
from __future__ import annotations

import asyncio
import json
import random

import pytest
pytest.importorskip('torch')
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from ddz_core.engine import replay
from ddz_core.actions import ActionKind
from ddz_core.state import Phase
from ddz_env.environment import DDZEnv
from ddz_models.network import NetworkSpec
from ddz_models.policy import Decision, RoleBundle
from ddz_registry import ModelRegistry
from ddz_training.checkpoints import save_checkpoint
from ddz_web.api import create_app
from ddz_web.arena import ArenaGame, ArenaManager, public_snapshot


def test_public_snapshot_does_not_reveal_hands_or_seed():
    env = DDZEnv(2026)
    snapshot = public_snapshot(env, 'abc', ('@random',) * 3,
                               running=False, delay=.5, status='paused')
    raw = json.dumps(snapshot)
    assert snapshot['seed'] is None
    assert 'hands' not in raw and 'own_physical_cards' not in raw
    assert 'bottom_cards' not in raw
    assert snapshot['bottom_counts'] is None


def test_no_unregistered_or_unenabled_baseline(tmp_path):
    mgr = ArenaManager(tmp_path / 'registry', tmp_path / 'replays')
    with pytest.raises(ValueError, match='disabled'):
        mgr.create(('@random',) * 3)
    with pytest.raises(KeyError):
        mgr.create(('unknown',) * 3)
    with pytest.raises(ValueError):
        mgr.create(('@random',), seed=7)


def test_create_and_websocket_step_and_origin(tmp_path):
    app = create_app(tmp_path / 'registry', tmp_path / 'replays', allow_baseline=True)
    with TestClient(app) as client:
        assert client.get('/api/health').json()['status'] == 'ok'
        assert '<html' in client.get('/').text.lower()
        bad = client.post('/api/games', json={'model_ids': ['@random'] * 3},
                          headers={'Origin': 'https://evil.example'})
        assert bad.status_code == 403
        new = client.post('/api/games', json={'model_ids': ['@random'] * 3,
                                               'seed': 2026, 'random_bidding': True})
        assert new.status_code == 201
        state = new.json()
        assert state['seed'] is None
        assert state['random_bidding'] is True
        game_id = state['game_id']
        assert client.get(f'/api/games/{game_id}/replay').status_code == 409
        with client.websocket_connect(f'/api/games/{game_id}/events') as ws:
            init = ws.receive_json()
            assert init['game_id'] == game_id
            assert init['sequence'] == state['sequence']
            step = client.post(f'/api/games/{game_id}/control', json={'action': 'step'})
            assert step.status_code == 200
            event = ws.receive_json()
            assert event['step_index'] == 1
            assert event['sequence'] > init['sequence']
        assert client.post(f'/api/games/{game_id}/control',
                           json={'action': 'speed', 'delay': 0}).status_code == 400
        assert client.get('/api/games/bad').status_code == 404


@pytest.mark.parametrize('model_id', ['@random', '@reasonable'])
def test_baselines_finish_with_verified_replay(tmp_path, model_id):
    async def run():
        manager = ArenaManager(tmp_path / 'registry', tmp_path / 'replays',
                               allow_baseline=True)
        game = manager.create((model_id,) * 3, seed=11)
        for _ in range(1200):
            if game.env.done:
                break
            await game.step_once()
        assert game.env.done
        assert game.snapshot()['status'] in {'finished', 'truncated'}
        stored = json.loads((tmp_path / 'replays' / f'{game.game_id}.json').read_text())
        assert replay(stored).state_digest() == game.env.engine.state_digest()
        assert game.snapshot()['seed'] == 11
        await manager.close()
    asyncio.run(run())


@pytest.mark.parametrize('schema', ['ddz-candidate-mlp-v1', 'ddz-candidate-mlp-v2'])
def test_registered_checkpoint_loaded_and_serves_decision(tmp_path, schema):
    registry = ModelRegistry(tmp_path / 'registry')
    source = tmp_path / 'source'
    source.mkdir()
    weight = source / 'test.weights.pt'
    spec = NetworkSpec.v2(16) if schema.endswith('v2') else NetworkSpec(hidden=16)
    bundle = RoleBundle(spec)
    save_checkpoint(weight, bundle, {}, episodes_seen=0, updates=0,
                    rng=random.Random(5), metrics={}, config={}, inference_only=True)
    registry.register('baseline', weight, trusted_source_dir=source)
    manager = ArenaManager(tmp_path / 'registry', tmp_path / 'replays')
    game = manager.create(('baseline',) * 3, seed=12)
    assert game.policies[0] is not None
    async def step():
        result = await game.step_once()
        assert result['step_index'] == 1
        await manager.close()
    asyncio.run(step())


def test_explicit_random_bidding_reaches_play_with_declining_models(tmp_path):
    class DecliningPolicy:
        def act(self, obs, _rng, _epsilon):
            decline = (ActionKind.DECLINE_CALL if obs.phase == Phase.CALL
                       else ActionKind.DECLINE_ROB)
            idx = next(i for i, action in enumerate(obs.legal_actions)
                       if action.kind == decline)
            return Decision(idx, 'bid', 0.0, False)

    def game(random_bidding):
        policy = DecliningPolicy()
        return ArenaGame('test', DDZEnv(2026), ('declines',) * 3,
                         (policy,) * 3,
                         tuple(random.Random(2026 + i * 1009) for i in range(3)),
                         tmp_path / 'replays', random_bidding=random_bidding)

    async def run():
        pure = game(False)
        for _ in range(9):
            await pure.step_once()
        assert pure.env.engine.state.phase == Phase.CALL
        assert pure.env.engine.state.landlord is None
        mixed = game(True)
        for _ in range(100):
            if mixed.env.engine.state.phase == Phase.PLAY:
                break
            await mixed.step_once()
        assert mixed.env.engine.state.phase == Phase.PLAY
        assert mixed.snapshot()['random_bidding'] is True

    asyncio.run(run())


def test_session_limit_and_control_guards(tmp_path):
    manager = ArenaManager(tmp_path / 'registry', tmp_path / 'replays',
                           allow_baseline=True, max_sessions=1)
    game = manager.create(('@random',) * 3, seed=1)
    game.running = True
    with pytest.raises(ValueError, match='running'):
        manager.create(('@random',) * 3)
    game.running = False
    newer = manager.create(('@random',) * 3, seed=2)
    assert len(manager.games) == 1 and newer.game_id in manager.games
    assert manager.get(game.game_id).env.engine.state_digest() == game.env.engine.state_digest()
    assert len(manager.games) == 1
    game = manager.get(game.game_id)
    async def exercise():
        await game.control('play')
        with pytest.raises(ValueError, match='pause'):
            await game.control('step')
        await game.control('pause')
        await game.control('step')
        assert game.snapshot()['step_index'] >= 1
        await manager.close()
    asyncio.run(exercise())


def test_invalid_host_and_websocket_origin(tmp_path):
    app = create_app(tmp_path / 'registry', tmp_path / 'replays', allow_baseline=True)
    with TestClient(app) as client:
        r = client.get('/api/health', headers={'host': 'evil.example'})
        assert r.status_code == 400
        game = client.post('/api/games', json={'model_ids': ['@random'] * 3}).json()
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/api/games/{game['game_id']}/events",
                                          headers={'origin': 'https://evil.example'}) as ws:
                ws.receive_json()
