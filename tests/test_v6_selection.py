"""Tournament clustering, seed blocking and explicit promotion eligibility."""
import random
import pytest

torch = pytest.importorskip('torch')
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from ddz_eval.statistics import clustered_interval, paired_interval
from ddz_eval.selection import evaluate_candidates, seed_plan
from ddz_eval.league import compare_bundles


def _rows(values, opponent='random'):
    return [{'deal_seed': i, 'opponent': opponent, 'side': side, 'won': x}
            for i, value in enumerate(values) for side, x in [('landlord', value), ('farmers', value)]]


def test_cluster_interval_constant_and_deterministic():
    rows = _rows([1]*20)
    v = clustered_interval(rows, draws=300, seed=12)
    assert v['mean'] == 1.0 and v['ci'] == [1.0, 1.0]
    assert clustered_interval(rows, draws=300, seed=12) == v
    with pytest.raises(ValueError, match='duplicate'):
        clustered_interval(rows + [rows[0]])
    with pytest.raises(ValueError, match='incomplete'):
        clustered_interval(rows[:-1])


def test_paired_difference_preserves_deal_clusters():
    a = _rows([1]*40)
    b = _rows([0]*40)
    c = paired_interval(a, b, draws=300, seed=11)
    assert c['ci'] == [1.0, 1.0] and c['deal_clusters'] == 40
    with pytest.raises(ValueError, match='share all'):
        paired_interval(a, b[:-2])
    with pytest.raises(ValueError, match='share all'):
        paired_interval(a, _rows([0]*40, opponent='different'))


def test_seed_overlap_and_ranked_result():
    assert seed_plan([100, 200], 3) == [(100,3),(200,3)]
    with pytest.raises(ValueError, match='overlapping'):
        seed_plan([100, 102], 3)
    a = RoleBundle(NetworkSpec(hidden=12))
    b = RoleBundle(NetworkSpec(hidden=12))
    report = evaluate_candidates(candidates={'current':a,'new':b}, incumbent_id='current',
                                 frozen=[], seed_starts=[120], deals_per_seed=2,
                                 min_deals=30, draws=100)
    assert report['recommended_id'] is None
    assert report['candidates']['new']['eligible'] is False
    assert report['candidates']['new']['vs_incumbent']['deal_clusters'] == 2
    assert sorted(report['ranked_ids']) == ['current', 'new']
    again = evaluate_candidates(candidates={'current':a,'new':b}, incumbent_id='current',
                                frozen=[], seed_starts=[120], deals_per_seed=2,
                                min_deals=30, draws=100)
    assert report == again


def test_outcomes_are_per_deal_and_role():
    b = RoleBundle(NetworkSpec(hidden=12))
    result = compare_bundles(b, None, games=2, seed=91, record_outcomes=True)
    assert len(result['outcomes']) == 4
    assert {x['side'] for x in result['outcomes']} == {'landlord', 'farmers'}
