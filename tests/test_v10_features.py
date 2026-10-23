"""v10 persistence, scheduler, lineage and role-matrix regression suite."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


def experiment_spec(path: Path, name: str = 'newrun') -> Path:
    path.write_text(json.dumps({'schema':'ddz-experiment-v1','id':name,
            'training':{'episodes':1,'batch_episodes':1,'eval_every':1,
                        'eval_games':1,'hidden':16,'history_capacity':0}}))
    return path


def test_arena_restores_exact_state_and_rng(tmp_path):
    pytest.importorskip('torch')
    from ddz_web.arena import ArenaManager
    a = ArenaManager(tmp_path/'registry', tmp_path/'replays', allow_baseline=True)
    game = a.create(('@random',)*3, seed=2026)
    for _ in range(7):
        asyncio.run(game.step_once())
    digest = game.env.engine.state_digest()
    assert game.snapshot()['seed'] is None
    journal = tmp_path/'replays'/'sessions'/f'{game.game_id}.json'
    assert journal.exists()
    assert journal.stat().st_mode & 0o077 == 0
    other = ArenaManager(tmp_path/'registry', tmp_path/'replays', allow_baseline=True)
    recovered = other.get(game.game_id)
    assert recovered.env.engine.state_digest() == digest
    assert recovered.status == 'paused' and not recovered.running
    asyncio.run(game.step_once())
    asyncio.run(recovered.step_once())
    assert game.env.engine.state_digest() == recovered.env.engine.state_digest()


def test_arena_tampered_journal_skipped(tmp_path):
    pytest.importorskip('torch')
    from ddz_web.arena import ArenaManager
    a = ArenaManager(tmp_path/'registry', tmp_path/'replays', allow_baseline=True)
    g = a.create(('@random',)*3, seed=42)
    asyncio.run(g.step_once())
    path = tmp_path/'replays'/'sessions'/f'{g.game_id}.json'
    value = json.loads(path.read_text())
    value['state_digest'] = 'wrong'
    path.write_text(json.dumps(value))
    restored = ArenaManager(tmp_path/'registry', tmp_path/'replays', allow_baseline=True)
    assert g.game_id not in restored.games
    with pytest.raises(KeyError):
        restored.get(g.game_id)


def test_arena_dangling_journal_symlink_does_not_block_recovery(tmp_path):
    pytest.importorskip('torch')
    from ddz_web.arena import ArenaManager
    a = ArenaManager(tmp_path/'registry', tmp_path/'replays', allow_baseline=True)
    valid = a.create(('@random',)*3, seed=42)
    bad = tmp_path/'replays'/'sessions'/('0'*24 + '.json')
    bad.symlink_to(tmp_path/'missing.json')
    restored = ArenaManager(tmp_path/'registry', tmp_path/'replays', allow_baseline=True)
    assert valid.game_id in restored.games


def test_recovered_arena_http_requires_no_private_seed(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('fastapi');pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from ddz_web.api import create_app
    settings = {'registry_dir':tmp_path/'registry','replay_dir':tmp_path/'replays',
                'allow_baseline':True}
    with TestClient(create_app(**settings)) as client:
        response=client.post('/api/games',json={'model_ids':['@random']*3,'seed':730})
        assert response.status_code == 201
        gid=response.json()['game_id']
        client.post(f'/api/games/{gid}/control',json={'action':'step'})
    with TestClient(create_app(**settings)) as client:
        resp=client.get(f'/api/games/{gid}')
        assert resp.status_code==200 and resp.json()['status']=='paused'
        assert resp.json()['seed'] is None
        assert 'hands' not in resp.text and 'rng_states' not in resp.text


def test_queue_prevents_duplicate_and_records_parent(tmp_path):
    pytest.importorskip('torch')
    from ddz_experiments.jobs import JobQueue, JobError
    q=JobQueue(tmp_path/'jobs',tmp_path/'runs')
    file=experiment_spec(tmp_path/'spec.json')
    job=q.submit(file)
    assert job['status']=='queued'
    assert q.get(job['job_id'])['source_sha256']
    with pytest.raises(JobError,match='already'):
        q.submit(file)
    with pytest.raises(JobError,match='parent'):
        q.submit(experiment_spec(tmp_path/'child.json','child'),parent_experiment='missing')
    assert len(q.list()) == 1
    assert q.lineage('newrun')['related_jobs'][0]['job_id'] == job['job_id']


def test_queue_worker_process_and_resume_lineage(tmp_path):
    pytest.importorskip('torch')
    from ddz_experiments.jobs import JobQueue
    q=JobQueue(tmp_path/'jobs',tmp_path/'runs')
    p=experiment_spec(tmp_path/'run.json')
    first=q.submit(p)
    result=q.work(parallel=1,once=True,poll_seconds=.1)
    assert result=={'completed':1,'failed':0}
    assert q.get(first['job_id'])['status']=='succeeded'
    assert (tmp_path/'runs'/'newrun'/'latest.pt').is_file()
    p2=experiment_spec(tmp_path/'descendant.json','childrun')
    desc=q.submit(p2,parent_experiment='newrun')
    assert q.get(desc['job_id'])['parent_experiment']=='newrun'
    assert len(q.lineage('newrun')['related_jobs'])==2


def test_interrupted_running_jobs_not_requeued(tmp_path):
    pytest.importorskip("torch")
    from ddz_experiments.jobs import JobQueue
    q=JobQueue(tmp_path/'jobs',tmp_path/'runs')
    job=q.submit(experiment_spec(tmp_path/'a.json'))
    row=q._claim_one()
    assert row is not None and row['job_id']==job['job_id']
    assert q.work(once=True)=={'completed':0,'failed':0}
    assert q.get(job['job_id'])['status']=='interrupted'


def test_matrix_crossplay_reproducible_and_bounded():
    pytest.importorskip('torch')
    from ddz_models.network import NetworkSpec
    from ddz_models.policy import RoleBundle
    from ddz_eval.matrix import evaluate_matrix
    models={'a':RoleBundle(NetworkSpec(hidden=16)),'b':RoleBundle(NetworkSpec(hidden=16))}
    params=dict(landlords=['a','b'],farmer_ups=['a'],farmer_downs=['b'],seed=2193,deals=2,draws=100)
    one=evaluate_matrix(models,**params)
    two=evaluate_matrix(models,**params)
    assert one==two
    assert len(one['rows'])==2 and len(one['paired_differences'])==1
    assert one['paired_differences'][0]['paired_deals'] <= 2
    with pytest.raises(ValueError):evaluate_matrix(models,landlords=['a']*2,
        farmer_ups=['a'],farmer_downs=['a'],deals=1)


def test_dashboard_jobs_and_matrix_read_only(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('fastapi');pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from ddz_experiments.jobs import JobQueue
    from ddz_web.api import create_app
    from ddz_web.lab import LabStore
    q=JobQueue(tmp_path/'jobs',tmp_path/'runs')
    q.submit(experiment_spec(tmp_path/'x.json'))
    matrix={'schema':'ddz-crossplay-matrix-v1','seed':1,'deals':1,
            'rows':[{'id':'combo-00','models':{'landlord':'a','farmer_up':'b','farmer_down':'c'},
                     'landlord_win_rate':.5,'completed':1}], 'paired_differences':[]}
    reports=tmp_path/'reports';reports.mkdir()
    (reports/'test-matrix.json').write_text(json.dumps(matrix))
    with TestClient(create_app(tmp_path/'registry',tmp_path/'replays',allow_baseline=True,
                               runs_dir=tmp_path/'runs',reports_dir=reports,jobs_dir=tmp_path/'jobs',
                               operator_token='v10-valid-operator-token-123456')) as client:
        assert client.get('/api/lab/jobs').status_code==401
        client.post('/api/auth/login',headers={'X-Operator-Token':'v10-valid-operator-token-123456'})
        jobs=client.get('/api/lab/jobs').json()['jobs']
        assert len(jobs)==1 and jobs[0]['status']=='queued'
        assert 'spec_json' not in json.dumps(jobs)
        assert client.get('/api/lab/matrices').json()['matrices'][0]['report_id']=='test-matrix'
        assert client.get('/api/lab/matrices/test-matrix').json()['deals']==1
        assert client.get('/api/lab/matrices/../../etc').status_code in (400,404)
