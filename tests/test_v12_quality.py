"""v12 diagnostics, seed-level inference, collaboration interaction, safe retention."""
from pathlib import Path
import json
import random

import pytest

from ddz_quality.analysis import QualityError, analyze_runs
from ddz_quality.cooperation import interaction_from_crossplay
from ddz_quality.retention import prune_history


def seeded_run(root: Path, name: str, scores: list[float], *, seed: int, hidden: int = 16) -> Path:
    folder = root / name
    folder.mkdir()
    settings = {'seed': seed, 'hidden': hidden, 'batch_episodes': 2, 'output': str(folder)}
    (folder / 'experiment.json').write_text(json.dumps({'settings': settings}))
    with (folder / 'evaluations.jsonl').open('w') as f:
        for i, score in enumerate(scores, 1):
            f.write(json.dumps({'episodes_seen': i*10, 'balanced_score': score,
                                'landlord_rate': score, 'farmer_rate': score})+'\n')
    with (folder / 'throughput.jsonl').open('w') as f:
        f.write(json.dumps({'episodes_per_s': 1.0, 'samples_per_s': 20,
                            'decisions_per_s': 2.0, 'episodes': 10, 'truncated': 0})+'\n')
    return folder


def test_three_seed_interval_is_reproducible_and_detects_peak_drop(tmp_path):
    runs = [seeded_run(tmp_path, f'seed-{i}', scores, seed=i)
            for i, scores in enumerate(([.8, .6], [.5, .55], [.65, .62]), 1)]
    first = analyze_runs(runs, draws=300, seed=77, regression_tolerance=.1)
    assert first == analyze_runs(runs, draws=300, seed=77, regression_tolerance=.1)
    assert first['summary']['evidence_sufficient']
    assert first['summary']['config_comparable']
    assert first['summary']['bootstrap_ci95'][0] <= first['summary']['mean_latest_score'] <= first['summary']['bootstrap_ci95'][1]
    assert 'score_drop_from_historical_peak' in first['runs'][0]['alerts']
    assert not first['runs'][1]['alerts']


def test_seed_level_ci_with_two_runs_is_deliberately_unavailable(tmp_path):
    runs = [seeded_run(tmp_path, f'r{i}', [.6], seed=i) for i in range(2)]
    result = analyze_runs(runs, draws=100)
    assert result['summary']['bootstrap_ci95'] is None
    assert not result['summary']['evidence_sufficient']
    assert 'insufficient_evaluation_history' in result['runs'][0]['alerts']


def test_mismatched_settings_are_not_aggregated_as_comparable(tmp_path):
    a = seeded_run(tmp_path, 'one', [.6, .7], seed=1)
    b = seeded_run(tmp_path, 'two', [.4, .5], seed=2, hidden=32)
    c = seeded_run(tmp_path, 'three', [.5, .6], seed=3)
    result = analyze_runs([a, b, c], draws=100)
    assert not result['summary']['config_comparable']
    assert result['summary']['bootstrap_ci95'] is None


def test_metrics_corruption_and_symlink_rejected(tmp_path):
    run = seeded_run(tmp_path, 'one', [.6], seed=1)
    with (run / 'evaluations.jsonl').open('a') as f:
        f.write('{broken\n')
    with pytest.raises(QualityError, match='corrupt JSONL'):
        analyze_runs([run])
    (run / 'evaluations.jsonl').unlink()
    outside = tmp_path / 'outside'
    outside.write_text('')
    (run / 'evaluations.jsonl').symlink_to(outside)
    with pytest.raises(QualityError, match='symlink'):
        analyze_runs([run])


def test_duplicate_and_nonfinite_metrics_rejected(tmp_path):
    run = seeded_run(tmp_path, 'one', [.5, .4], seed=1)
    path = run/'evaluations.jsonl'
    data = [json.loads(x) for x in path.read_text().splitlines()]
    data[1]['episodes_seen'] = data[0]['episodes_seen']
    path.write_text(''.join(json.dumps(row)+'\n' for row in data))
    with pytest.raises(QualityError, match='strictly increasing'):
        analyze_runs([run])
    data[1]['episodes_seen'] = 20
    data[1]['balanced_score'] = float('nan')
    path.write_text(''.join(json.dumps(row)+'\n' for row in data))
    with pytest.raises(QualityError, match='nonfinite'):
        analyze_runs([run])


def factorial(rows, *, truncated=None):
    # Synergy positive: both candidates win where neither alone helps.
    names = ('base_base','candidate_up','candidate_down','both_candidates')
    return {name: {'outcomes': [
        {'deal_seed': seed, 'outcome': ('truncated' if truncated == seed else
         ('FARMERS' if name == 'both_candidates' else 'LANDLORD'))}
        for seed in range(rows)]} for name in names}


def test_farmer_factorial_interaction_bootstrap_and_exclusion():
    result = interaction_from_crossplay(factorial(40, truncated=1), seed=9, draws=400)
    assert result['paired_deals'] == 39 and result['excluded_deals'] == 1
    assert result['interaction'] == 1.0 and result['ci95'] == [1., 1.]
    assert result['eligible']
    early = interaction_from_crossplay(factorial(2), draws=100)
    assert not early['eligible'] and early['ci95'] is None


def test_farmer_fixture_misalignment_and_duplicate_rejected():
    matrix = factorial(40)
    matrix['candidate_up']['outcomes'].pop()
    with pytest.raises(QualityError, match='same initial deals'):
        interaction_from_crossplay(matrix)
    matrix = factorial(40)
    matrix['base_base']['outcomes'].append(matrix['base_base']['outcomes'][0])
    with pytest.raises(QualityError, match='duplicate'):
        interaction_from_crossplay(matrix)


def _sample_checkpoint(folder: Path, step: int, score: float) -> Path:
    torch = pytest.importorskip('torch')
    from ddz_models.network import NetworkSpec
    from ddz_models.policy import RoleBundle
    from ddz_training.checkpoints import save_checkpoint
    torch.set_num_threads(1)
    bundle = RoleBundle(NetworkSpec(hidden=16))
    dest = folder / 'history' / f'step-{step:09d}.weights.pt'
    return save_checkpoint(dest, bundle, {}, episodes_seen=step, updates=step,
                           rng=random.Random(1), metrics={'balanced_score':score},
                           config={'seed': 1}, inference_only=True)


def test_history_retention_dry_run_and_top_score_protection(tmp_path):
    for step, score in [(1, .9), (2, .1), (3, .2), (4, .3)]:
        _sample_checkpoint(tmp_path, step, score)
    dry = prune_history(tmp_path, keep_recent=1, keep_best=1)
    assert dry['kept'] == ['step-000000001.weights.pt', 'step-000000004.weights.pt']
    assert len(dry['prunable']) == 2 and not dry['removed']
    assert len(list((tmp_path / 'history').glob('*.pt'))) == 4
    applied = prune_history(tmp_path, keep_recent=1, keep_best=1, apply=True)
    assert applied['removed'] == dry['prunable']
    assert len(list((tmp_path / 'history').glob('*.pt'))) == 2
    assert len(list((tmp_path / 'history').glob('*.json'))) == 2


def test_history_tamper_causes_fail_closed_without_any_deletion(tmp_path):
    a = _sample_checkpoint(tmp_path, 1, .1)
    b = _sample_checkpoint(tmp_path, 2, .2)
    a.with_suffix(a.suffix + '.json').write_text('{}')
    with pytest.raises(QualityError):
        prune_history(tmp_path, keep_recent=1, apply=True)
    assert a.exists() and b.exists()


def test_history_symlink_rejected(tmp_path):
    _sample_checkpoint(tmp_path, 1, .1)
    outside = tmp_path / 'outside'
    outside.write_text('unsafe')
    fake = tmp_path/'history'/'step-000000002.weights.pt'
    fake.symlink_to(outside)
    with pytest.raises(QualityError, match='symlink'):
        prune_history(tmp_path, keep_recent=1, apply=True)
    assert outside.read_text() == 'unsafe'


def test_training_keep_best_archives_and_preserves_latest(tmp_path):
    pytest.importorskip('torch')
    from ddz_training.trainer import TrainConfig, DMCTrainer
    run = tmp_path / 'train'
    cfg = TrainConfig(episodes=3, batch_episodes=1, eval_every=1, eval_games=1,
                      hidden=16, output=str(run), actor_workers=1,
                      history_capacity=1, history_keep_top=1)
    result = DMCTrainer(cfg).train()
    assert result['episodes_seen'] == 3
    assert (run/'latest.pt').exists() and (run/'best.weights.pt').exists()
    assert 1 <= len(list((run/'history').glob('*.weights.pt'))) <= 2
