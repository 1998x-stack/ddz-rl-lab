"""v8 read-only dashboard, replay integrity and hostile filesystem fixtures."""
import json
import random
from pathlib import Path

import pytest
pytest.importorskip('torch')
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from ddz_env.environment import DDZEnv
from ddz_env.baselines import random_action_index
from ddz_web.api import create_app
from ddz_web.lab import LabError, LabStore


def fixtures(tmp_path: Path):
    for name in ('runs', 'reports', 'replays', 'logs', 'registry'):
        (tmp_path / name).mkdir()
    run = tmp_path / 'runs' / 'experiment-1'
    run.mkdir()
    (run / 'throughput.jsonl').write_text('\n'.join([
        '{"episodes_seen":2,"episodes_per_s":3.2,"samples_per_s":100}',
        'not json',
        '{"episodes_seen":4,"episodes_per_s":4.2,"samples_per_s":120}',
    ])+'\n')
    (run / 'evaluations.jsonl').write_text('{"episodes_seen":4,"balanced_score":0.55,"best_score":0.6}\n')
    report = {'schema': 'ddz-tournament-report-v1', 'incumbent_id': 'old',
              'recommended_id': None, 'candidates': {'old': {'score': {'mean': 0.4}}},
              'deals_per_seed': 5}
    (tmp_path / 'reports' / 'report-1.json').write_text(json.dumps(report))
    log = {'record': {'time': {'repr': '2026-10-23'}, 'level': {'name': 'INFO'},
                      'message': 'healthy', 'exception': {'text': 'sensitive internal state'}}}
    (tmp_path / 'logs' / 'app-2026-10-23.jsonl').write_text(json.dumps(log) + '\n')
    env = DDZEnv(2026)
    rng = random.Random(3)
    for _ in range(1200):
        if env.done:
            break
        env.step(random_action_index(env, rng))
    assert env.done and not env.engine.state.truncated
    replay = env.export_replay()
    replay['model_ids'] = ['@random'] * 3
    (tmp_path / 'replays' / 'game-a.json').write_text(json.dumps(replay))
    return LabStore(tmp_path/'runs', tmp_path/'reports', tmp_path/'replays', tmp_path/'logs')


def test_readonly_lab_summaries(tmp_path):
    store=fixtures(tmp_path)
    assert store.runs()[0]['run_id'] == 'experiment-1'
    metrics=store.metrics('experiment-1')
    assert len(metrics['throughput']['records']) == 2
    assert metrics['throughput']['invalid_lines'] == 1
    assert store.reports()[0]['incumbent_id'] == 'old'
    assert store.report('report-1')['schema'] == 'ddz-tournament-report-v1'
    assert store.logs()['lines'][0]['message'] == 'healthy'
    assert 'sensitive internal state' not in str(store.logs())


def test_completed_public_replay_and_integrity(tmp_path):
    store=fixtures(tmp_path)
    assert store.replays()[0]['steps'] > 10
    record=store.replay('game-a')
    assert record['verified'] is True
    assert len(record['frames']) == store.replays()[0]['steps'] + 1
    assert record['frames'][-1]['winner'] in ('LANDLORD', 'FARMERS')
    # Never return seed, hands, actual unrevealed bottom, or full checkpoint data.
    for item in record['frames']:
        assert 'hands' not in item and 'seed' not in item and 'bottom_cards' not in item
    path=tmp_path/'replays'/'game-a.json'
    raw=json.loads(path.read_text())
    raw['state_digest']='0'*64
    path.write_text(json.dumps(raw))
    with pytest.raises(LabError, match='integrity'):
        store.replay('game-a')
    raw['terminated']=False
    path.write_text(json.dumps(raw))
    assert not store.replays()
    with pytest.raises(LabError, match='completed'):
        store.replay('game-a')


def test_traversal_and_symlink_rejected(tmp_path):
    store=fixtures(tmp_path)
    for invalid in ('../etc', '%2e%2e', '..', 'a/b', 'bad\\name'):
        with pytest.raises((LabError, FileNotFoundError)):
            store.metrics(invalid)
    (tmp_path/'reports'/'outside.json').symlink_to(tmp_path/'runs'/'experiment-1'/'evaluations.jsonl')
    assert all(x['report_id'] != 'outside' for x in store.reports())
    with pytest.raises(FileNotFoundError):
        store.report('outside')
    (tmp_path/'runs'/'linked').symlink_to(tmp_path/'runs'/'experiment-1', target_is_directory=True)
    assert all(x['run_id'] != 'linked' for x in store.runs())


def test_jsonl_bounded_tail(tmp_path):
    store=fixtures(tmp_path)
    target=tmp_path/'runs'/'experiment-1'/'throughput.jsonl'
    with target.open('a') as out:
        for i in range(9000):
            out.write(json.dumps({'episodes_seen':i, 'padding':'x'*70})+'\n')
    row=store.metrics('experiment-1', limit=40)['throughput']
    assert len(row['records']) == 40
    assert row['truncated'] is True
    assert row['records'][-1]['episodes_seen'] == 8999


def test_dashboard_http_and_security(tmp_path):
    fixtures(tmp_path)
    app=create_app(tmp_path/'registry',tmp_path/'replays',allow_baseline=True,
                   runs_dir=tmp_path/'runs',reports_dir=tmp_path/'reports',log_dir=tmp_path/'logs')
    with TestClient(app) as c:
        r=c.get('/api/lab/overview')
        assert r.status_code==200
        assert r.json()['schema']=='ddz-lab-v1'
        assert r.json()['runs'][0]['run_id']=='experiment-1'
        assert c.get('/api/lab/runs/experiment-1/metrics').status_code==200
        assert c.get('/api/lab/reports/report-1').status_code==200
        assert c.get('/api/lab/replays/game-a').json()['verified'] is True
        assert c.get('/api/lab/logs').json()['lines'][0]['message']=='healthy'
        assert c.get('/static/lab.js').status_code==200
        assert b'lab-tabs' in c.get('/').content
        assert c.get('/api/lab/runs/invalid/metrics').status_code==404
        assert c.get('/api/lab/runs/experiment-1/metrics?limit=999999').status_code==422
        assert c.get('/api/lab/reports/none').status_code==404
        assert c.post('/api/games', headers={'Origin':'https://evil.invalid'},
                      json={'model_ids':['@random']*3}).status_code==403
