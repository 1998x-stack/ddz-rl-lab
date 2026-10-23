"""Safe checkpoint import, immutable registry, audited manual promotion."""
from pathlib import Path
import json
import random
import shutil
import pytest

torch = pytest.importorskip('torch')
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from ddz_training.checkpoints import save_checkpoint
from ddz_registry import ModelRegistry


def _checkpoint(folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    model = RoleBundle(NetworkSpec(hidden=12))
    save_checkpoint(path, model, {}, episodes_seen=2, updates=1, rng=random.Random(1),
                    metrics={'balanced_score':.5}, config={}, inference_only=True)
    return path


def test_register_bootstrap_verify_and_reject_overwrite(tmp_path):
    source = tmp_path/'trusted'
    path = _checkpoint(source, 'candidate.pt')
    r = ModelRegistry(tmp_path/'registry')
    registered = r.register('baseline', path, trusted_source_dir=source)
    assert r.verify('baseline')['sha256'] == registered['sha256']
    assert r.bootstrap_active('baseline')['active'] == 'baseline'
    with pytest.raises(ValueError, match='already has active'):
        r.bootstrap_active('baseline')
    with pytest.raises(ValueError, match='non-overwritable'):
        r.register('baseline', path, trusted_source_dir=source)
    assert r.list_models()['active'] == 'baseline'
    assert r.list_models()['models']['baseline']['sha256'] == registered['sha256']


def test_reject_path_traversal_modified_artifacts_and_untrusted_source(tmp_path):
    source = tmp_path/'trusted'
    path = _checkpoint(source, 'candidate.pt')
    r = ModelRegistry(tmp_path/'registry')
    with pytest.raises(ValueError, match='invalid model id'):
        r.register('../escape', path, trusted_source_dir=source)
    with pytest.raises(ValueError, match='outside trusted'):
        r.register('outside', path, trusted_source_dir=tmp_path/'registry')
    r.register('candidate', path, trusted_source_dir=source)
    dest = tmp_path/'registry'/'bundles'/'candidate'/'weights.pt'
    with dest.open('ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError, match='hash mismatch'):
        r.verify('candidate')


def test_promotion_requires_matched_evidence_and_registered_champion(tmp_path):
    source = tmp_path/'trusted'
    incumbent = _checkpoint(source, 'incumbent.pt')
    challenger = _checkpoint(source, 'challenger.pt')
    r = ModelRegistry(tmp_path/'registry')
    a = r.register('incumbent', incumbent, trusted_source_dir=source)
    b = r.register('challenger', challenger, trusted_source_dir=source)
    r.bootstrap_active('incumbent')
    report = {'schema': 'ddz-tournament-report-v1', 'incumbent_id': 'incumbent',
              'recommended_id': 'challenger', 'min_deals': 30, 'min_delta': 0.0,
              'model_hashes': {'incumbent': a['sha256'], 'challenger': b['sha256']},
              'candidates': {'challenger': {'eligible': True,
                   'vs_incumbent': {'deal_clusters': 40, 'ci': [.04,.20]}}}}
    assert r.promote('challenger', report=report)['active'] == 'challenger'
    with pytest.raises(ValueError, match='incumbent changed'):
        r.promote('challenger', report=report)
    report['incumbent_id'] = 'challenger'
    report['model_hashes']['challenger'] = '0'*64
    with pytest.raises(ValueError, match='model hashes'):
        r.promote('challenger', report=report)


def test_promotion_rejects_inconclusive_and_missing_report(tmp_path):
    source = tmp_path/'trusted'
    source_ckpt = _checkpoint(source, 'p.pt')
    r = ModelRegistry(tmp_path/'registry')
    r.register('one', source_ckpt, trusted_source_dir=source)
    r.register('two', source_ckpt, trusted_source_dir=source)
    r.bootstrap_active('one')
    with pytest.raises(ValueError, match='does not recommend'):
        r.promote('two', report={})
    with pytest.raises(ValueError, match='evaluation gate'):
        r.promote('two', report={'schema':'ddz-tournament-report-v1',
                 'recommended_id':'two', 'min_deals':30,
                 'candidates':{'two':{'eligible':False,
                     'vs_incumbent':{'deal_clusters':4, 'ci':[-.1,.4]}}}})
