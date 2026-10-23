"""v9 experiment runner, paired-run summaries, crossplay, authentication regressions."""
import json
from pathlib import Path
import pytest

from ddz_experiments.runner import ExperimentRunner, ExperimentError


def spec(path, *, episodes=1, hidden=16, lr=0.0003):
    path.write_text(json.dumps({'schema':'ddz-experiment-v1','id':'candidate-a',
                                'training':{'episodes':episodes,'batch_episodes':1,
                                            'eval_games':1,'eval_every':1,'hidden':hidden,
                                            'lr':lr, 'history_capacity':0}}))
    return path


def test_experiment_run_resume_and_config_guard(tmp_path):
    torch = pytest.importorskip('torch')
    run = ExperimentRunner(tmp_path/'runs')
    file = spec(tmp_path/'spec.json')
    info=run.run(file)
    assert info['status']=='finished' and info['episodes_seen']==1
    assert run.status('candidate-a')['status']=='finished'
    with pytest.raises(ExperimentError, match='experiment exists'):
        run.run(file)
    spec(file, episodes=2)
    new=run.run(file,resume=True)
    assert new['episodes_seen']==3
    spec(file, episodes=1, lr=0.01)
    with pytest.raises(ExperimentError, match='mismatch'):
        run.run(file,resume=True)


def test_reject_invalid_experiment_spec_and_paths(tmp_path):
    run=ExperimentRunner(tmp_path/'runs')
    file=spec(tmp_path/'spec.json')
    obj=json.loads(file.read_text());obj['id']='../traverse';file.write_text(json.dumps(obj))
    with pytest.raises(ExperimentError):run.run(file)
    obj['id']='fine';obj['training']['output']='/tmp/unsafe';file.write_text(json.dumps(obj))
    with pytest.raises(ExperimentError):run.run(file)
    obj['training']={'episodes':True};file.write_text(json.dumps(obj))
    with pytest.raises(ExperimentError):run.run(file)


def test_crossplay_reproducible_with_role_mapping():
    pytest.importorskip('torch')
    from ddz_models.policy import RoleBundle
    from ddz_models.network import NetworkSpec
    from ddz_eval.crossplay import run_crossplay
    model=RoleBundle(NetworkSpec(hidden=16))
    roled={'landlord':model,'farmer_up':model,'farmer_down':model}
    one=run_crossplay({'a':roled},seed=2026,deals=2)
    two=run_crossplay({'a':roled},seed=2026,deals=2)
    assert one==two
    assert len(one['lineups']['a']['outcomes'])==2
    with pytest.raises(ValueError):run_crossplay({'a':{'landlord':model}},seed=1,deals=1)


def test_runs_comparison_and_auth_flow(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from ddz_web.api import create_app
    for folder in ('runs','reports','replays','logs','registry'):
        (tmp_path/folder).mkdir()
    for name,score in [('a',0.4),('b',0.6)]:
        run=tmp_path/'runs'/name;run.mkdir()
        (run/'evaluations.jsonl').write_text(json.dumps({'episodes_seen':10,'balanced_score':score})+'\n')
    app=create_app(tmp_path/'registry',tmp_path/'replays',allow_baseline=True,
                   runs_dir=tmp_path/'runs',reports_dir=tmp_path/'reports',log_dir=tmp_path/'logs',
                   operator_token='a-long-test-secret-that-is-not-logged')
    with TestClient(app) as c:
        assert c.get('/api/health').status_code==200
        assert c.get('/api/models').status_code==401
        assert c.get('/api/auth/status').json()=={'required':True,'authenticated':False}
        assert c.post('/api/auth/login',headers={'X-Operator-Token':'invalid'}).status_code==401
        with pytest.raises(Exception):
            with c.websocket_connect('/api/games/abc/events') as ws:
                ws.receive_json()
        ok=c.post('/api/auth/login',headers={'X-Operator-Token':'a-long-test-secret-that-is-not-logged'})
        assert ok.status_code==200
        assert 'httponly' in ok.headers['set-cookie'].lower()
        assert c.get('/api/auth/status').json()['authenticated'] is True
        comparison=c.get('/api/lab/compare?left=a&right=b')
        assert comparison.status_code==200
        assert comparison.json()['balanced_score_delta']==pytest.approx(0.2)
        assert c.get('/api/lab/compare?left=../etc&right=b').status_code==404
        assert c.post('/api/games',headers={'Origin':'https://foreign.test'},json={'model_ids':['@random']*3}).status_code==403
        assert c.post('/api/auth/logout').status_code==200
        assert c.get('/api/models').status_code==401


def test_disabled_auth_keeps_local_compatibility(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('fastapi');pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from ddz_web.api import create_app
    with TestClient(create_app(tmp_path/'registry',tmp_path/'replays',allow_baseline=True)) as c:
        assert c.get('/api/auth/status').json()=={'required':False,'authenticated':True}
        assert c.get('/api/models').status_code==200
        assert c.get('/static/auth.js').status_code==200


def test_early_stopping_and_checkpoint_resume(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    import ddz_training.trainer as module
    from ddz_training.trainer import DMCTrainer, TrainConfig
    def constant_evaluation(*_args, **_kwargs):
        return {'balanced_score':0.5}
    monkeypatch.setattr(module,'evaluate_bundle', constant_evaluation)
    cfg=TrainConfig(episodes=12,batch_episodes=1,eval_every=1,eval_games=1,
                    hidden=16, output=str(tmp_path/'early'),
                    history_capacity=0,early_stop_patience_evals=2)
    runner=DMCTrainer(cfg)
    result=runner.train()
    assert runner.episodes_seen == 3
    assert result['stopped_early'] and result['evals_without_improvement']==2
    resumed=DMCTrainer(cfg)
    resumed.resume(tmp_path/'early'/'latest.pt')
    assert resumed.evals_without_improvement==2


def test_operator_cookie_expiry_and_tampering(monkeypatch):
    from ddz_web.auth import OperatorAuth, TTL_SECONDS
    import ddz_web.auth as module
    auth=OperatorAuth('secure-test-secret-that-is-long-enough')
    cookie=auth.issue_cookie()
    assert auth.valid_cookie(cookie)
    assert not auth.valid_cookie(cookie+'t')
    monkeypatch.setattr(module.time, 'time', lambda: int(cookie.split('.')[0]) + TTL_SECONDS + 1)
    assert not auth.valid_cookie(cookie)
