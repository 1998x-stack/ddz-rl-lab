"""v11 queue migration, retry, resource control, timeout and statistical validity."""
import json
import os
from pathlib import Path
import sqlite3

import pytest

pytest.importorskip('torch')
from ddz_experiments.jobs import JobQueue, JobError, SCHEMA
from ddz_experiments.runner import ExperimentRunner


def spec(path, name='v11', *, episodes=2, device='cpu'):
    path.write_text(json.dumps({'schema':'ddz-experiment-v1','id':name,
        'training':{'episodes':episodes,'batch_episodes':1,'eval_every':1,
                    'eval_games':1,'hidden':16,'history_capacity':0,'device':device}}))
    return path


def test_upgrade_v10_db_preserves_jobs(tmp_path):
    jobs = tmp_path/'jobs'; jobs.mkdir()
    with sqlite3.connect(jobs/'jobs.sqlite3') as db:
        db.executescript(SCHEMA)
        db.execute("""INSERT INTO jobs(job_id, experiment_id,source_sha256,spec_json,
                  resume,status,created_at) VALUES (?,?,?,?,?,?,?)""",
                  ('a'*32, 'old', 'z'*64, '{}',0,'queued','now'))
    q=JobQueue(jobs,tmp_path/'runs')
    row=q.get('a'*32)
    assert row['cpu_threads']==1 and row['gpu_id'] is None
    assert row['retry_of'] is None and row['status']=='queued'


def test_resource_validation_and_exclusive_gpu_claim(tmp_path):
    q=JobQueue(tmp_path/'jobs',tmp_path/'runs')
    path=spec(tmp_path/'s.json',device='cuda')
    with pytest.raises(JobError,match='GPU slot'): q.submit(path)
    with pytest.raises(JobError,match='cpu_threads'):q.submit(path,cpu_threads=0,gpu_id=0)
    with pytest.raises(JobError,match='timeout_seconds'):q.submit(path,gpu_id=0,timeout_seconds=-1)
    j1=q.submit(path, gpu_id=0, timeout_seconds=10)
    path2=spec(tmp_path/'s2.json',name='other',device='cuda')
    j2=q.submit(path2,gpu_id=0)
    a=q._claim_one()
    assert a['job_id']==j1['job_id']
    assert q._claim_one() is None
    assert q.get(j2['job_id'])['status']=='queued'


def test_failed_job_retry_requires_checkpoint(tmp_path):
    q=JobQueue(tmp_path/'jobs',tmp_path/'runs')
    orig=q.submit(spec(tmp_path/'s.json'))
    row=q._claim_one(); assert row['job_id']==orig['job_id']
    q._set_result(orig['job_id'], 1)
    with pytest.raises(JobError,match='checkpoint'):q.retry(orig['job_id'])
    with pytest.raises(KeyError,match='unknown job'):q.retry('c'*32)


def test_retry_resumes_only_remaining_episodes(tmp_path):
    q=JobQueue(tmp_path/'jobs',tmp_path/'runs')
    source=spec(tmp_path/'s.json',episodes=3)
    original=q.submit(source)
    # Simulate an interrupted three-episode run with a valid one-episode
    # checkpoint and a stale experiment.json as found after a real crash.
    finished=spec(tmp_path/'partial.json',episodes=1)
    ExperimentRunner(tmp_path/'runs').run(finished)
    statefile=tmp_path/'runs'/'v11'/'experiment.json'
    state=json.loads(statefile.read_text())
    state.update(status='interrupted',previous_episodes=0,episodes_seen=0)
    statefile.write_text(json.dumps(state))
    q._claim_one(); q._set_result(original['job_id'],1)
    retried=q.retry(original['job_id'])
    assert retried['resume']==1 and retried['retry_of']==original['job_id']
    with q._db() as db:
        row=db.execute('SELECT * FROM jobs WHERE job_id=?',(retried['job_id'],)).fetchone()
    assert json.loads(row['spec_json'])['training']['episodes']==2
    with pytest.raises(JobError,match='active or successful retry'):q.retry(original['job_id'])
    result=q.work(once=True,poll_seconds=.05)
    assert result=={'completed':1,'failed':0}
    assert ExperimentRunner(tmp_path/'runs').status('v11')['episodes_seen']==3
    assert q.get(retried['job_id'])['status']=='succeeded'


def test_timeout_exits_and_releases_scheduler(tmp_path,monkeypatch):
    import ddz_experiments.jobs as m
    q=JobQueue(tmp_path/'jobs',tmp_path/'runs')
    source=spec(tmp_path/'s.json',episodes=1)
    job=q.submit(source, timeout_seconds=1, cpu_threads=2)
    original=m.subprocess.Popen
    def slow_popen(*args,**kwargs):
        # Safe mock worker: no training, simple isolated subprocess.
        args=([os.sys.executable,'-c','import time; time.sleep(30)'],)
        assert kwargs['env']['OMP_NUM_THREADS']=='2'
        assert kwargs['start_new_session'] is True
        return original(*args,**kwargs)
    monkeypatch.setattr(m.subprocess,'Popen',slow_popen)
    result=q.work(once=True,poll_seconds=.05)
    assert result=={'completed':1,'failed':1}
    assert q.get(job['job_id'])['status']=='failed'
    assert q.get(job['job_id'])['exit_code']==124
    assert q.get(job['job_id'])['error']=='worker timed out'


def test_crossplay_correction_is_descriptive_and_bounded():
    from ddz_models.network import NetworkSpec
    from ddz_models.policy import RoleBundle
    from ddz_eval.matrix import evaluate_matrix
    bundle=RoleBundle(NetworkSpec(hidden=16))
    opts={'x':bundle,'y':bundle,'z':bundle}
    report=evaluate_matrix(opts,landlords=['x','y','z'],farmer_ups=['x'],
                           farmer_downs=['y'],deals=2,draws=100,seed=433)
    assert len(report['paired_differences'])==3
    assert report['multiplicity']['number_of_pairs']==3
    assert not report['multiplicity']['tail_resolution_sufficient']
    assert all(pair['familywise_ci'] is None for pair in report['paired_differences'])
    assert all(not pair['eligible_for_comparison'] for pair in report['paired_differences'])
