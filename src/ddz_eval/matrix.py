"""Bounded role-combination cross-play using a common deterministic deal set.

This is descriptive *within-triplet* cross-play, not a proof of absolute model
strength. Candidate promotion requires the v6 frozen-opponent tournament.
"""
from __future__ import annotations

from itertools import product, combinations
import math
import random
from typing import Mapping, Sequence

from ddz_models.policy import RoleBundle
from .crossplay import run_crossplay


def evaluate_matrix(models: Mapping[str, RoleBundle], *, landlords: Sequence[str],
                    farmer_ups: Sequence[str], farmer_downs: Sequence[str],
                    seed: int = 2026, deals: int = 20, draws: int = 200) -> dict:
    pools = (landlords, farmer_ups, farmer_downs)
    if any(not pool or len(set(pool)) != len(pool) for pool in pools):
        raise ValueError('each role pool needs distinct nonempty candidates')
    combinations_count = math.prod(len(pool) for pool in pools)
    if combinations_count > 16 or deals < 1 or deals > 1000 or deals * combinations_count > 64000:
        raise ValueError('cross-play matrix exceeds evaluation bounds')
    if not 100 <= draws <= 10000:
        raise ValueError('bootstrap draws must be in 100..10000')
    if any(x not in models for pool in pools for x in pool):
        raise ValueError('unknown candidate model')
    if len(set(models)) > 64:
        raise ValueError('too many registered models')
    lineups = {}
    lineup_keys = {}
    for i, (lord, up, down) in enumerate(product(*pools)):
        name = f'combo-{i:02d}'
        lineup_keys[name] = {'landlord': lord, 'farmer_up': up, 'farmer_down': down}
        lineups[name] = {role: models[id_] for role, id_ in lineup_keys[name].items()}
    raw = run_crossplay(lineups, seed=seed, deals=deals)
    rows = []
    for name, stats in raw['lineups'].items():
        rows.append({'id': name, 'models': lineup_keys[name],
                     'landlord_win_rate': stats['landlord_win_rate'],
                     'farmer_win_rate': stats['farmer_win_rate'],
                     'completed': stats['completed'], 'truncated': stats['truncated'],
                     'outcomes': stats['outcomes']})
    rows.sort(key=lambda r: (r['landlord_win_rate'] is None,
                             -(r['landlord_win_rate'] or 0), r['id']))
    # Paired bootstrap per entire initial deal, with the same seeds across triplets.
    rng = random.Random(seed + 5371)
    pairs = []
    # Family-wise protection when inspecting many combinations. These are
    # descriptive bootstrap intervals, NOT an automatic champion selector.
    family_size = len(rows) * (len(rows) - 1) // 2
    family_confidence = .95
    corrected_tail = (1 - family_confidence) / (2 * max(1, family_size))
    correction_resolved = draws * corrected_tail >= 2  # enough tail samples

    for a, b in combinations(rows, 2):
        ar = {x['deal_seed']: x['outcome'] for x in a['outcomes']}
        br = {x['deal_seed']: x['outcome'] for x in b['outcomes']}
        keys = sorted(k for k in ar.keys() & br.keys()
                      if ar[k] != 'truncated' and br[k] != 'truncated')
        if not keys:
            pairs.append({'first': a['id'], 'second': b['id'], 'delta': None,
                          'ci': None, 'familywise_ci': None, 'paired_deals': 0,
                          'eligible_for_comparison': False})
            continue
        diffs = [int(ar[k] == 'LANDLORD') - int(br[k] == 'LANDLORD') for k in keys]
        n = len(diffs)
        values = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                        for _ in range(draws))
        ci95 = [values[max(0, math.floor(.025 * draws))],
                values[min(draws-1, math.ceil(.975 * draws)-1)]]
        adjusted_ci = None
        if correction_resolved:
            adjusted_ci = [values[max(0, math.floor(corrected_tail * draws))],
                           values[min(draws-1, math.ceil((1-corrected_tail) * draws)-1)]]
        pairs.append({'first': a['id'], 'second': b['id'],
                      'delta': sum(diffs) / n, 'ci': ci95,
                      'familywise_ci': adjusted_ci,
                      'paired_deals': n, 'eligible_for_comparison': n >= 30})
    return {'schema': 'ddz-crossplay-matrix-v1', 'seed': seed, 'deals': deals,
            'rows': rows, 'paired_differences': pairs,
            'multiplicity': {'method': 'bonferroni_percentile_bootstrap',
                             'family_confidence': family_confidence,
                             'number_of_pairs': family_size,
                             'tail_resolution_sufficient': correction_resolved,
                             'min_independent_deals_for_comparison': 30},
            'note': 'Pairwise CIs cluster on common initial deals. Results compare role-triplets, '
                    'not isolated agent strength. Unadjusted CIs must not be used for rank selection; '
                    'some adjusted CIs are unavailable when bootstrap resolution is insufficient; '
                    'bidding uses a random baseline and this matrix never automatically promotes models.'}
