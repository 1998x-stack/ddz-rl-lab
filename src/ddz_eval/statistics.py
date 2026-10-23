"""Deal-clustered bootstrap inference for reproducible DDZ comparisons.

A *deal* (not a seat result or opponent fixture) is the sampling unit. Each
resampled deal carries both side outcomes against all frozen opponents.
Bootstrapping describes uncertainty under this fixed evaluation protocol,
not proof of universal playing strength or perfect-play optimality.
"""
from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Mapping, Sequence


def _validate_records(rows: Sequence[dict]) -> dict[int, dict[tuple[str, str], float]]:
    by_deal: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        deal, opponent, side = int(row['deal_seed']), str(row['opponent']), str(row['side'])
        if side not in ('landlord', 'farmers') or not opponent:
            raise ValueError('invalid tournament outcome row')
        key = (opponent, side)
        if key in by_deal[deal]:
            raise ValueError(f'duplicate fixture for deal {deal}: {key}')
        value = row['won']
        if value not in (0, 1, False, True):
            raise ValueError('outcomes must be binary wins')
        by_deal[deal][key] = float(value)
    if not by_deal:
        raise ValueError('no completed tournament fixtures')
    fixtures = set(next(iter(by_deal.values())))
    if not fixtures or any(set(fixtures) != set(r) for r in by_deal.values()):
        raise ValueError('incomplete or inconsistent deal fixture matrix')
    if {side for _, side in fixtures} != {'landlord', 'farmers'}:
        raise ValueError('both landlord and farmer fixtures are required')
    return dict(by_deal)


def _deal_scores(rows: Sequence[dict]) -> dict[int, float]:
    matrix = _validate_records(rows)
    return {deal: sum(x.values()) / len(x) for deal, x in matrix.items()}


def clustered_interval(rows: Sequence[dict], *, draws: int = 2000,
                       seed: int = 2026, confidence: float = 0.95) -> dict:
    if draws < 100 or not 0.5 < confidence < 1.0:
        raise ValueError('bootstrap requires >=100 draws and 0.5<confidence<1')
    scores = _deal_scores(rows)
    values = list(scores.values())
    rng = random.Random(seed)
    n = len(values)
    draws_sorted = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                          for _ in range(draws))
    alpha = (1 - confidence) / 2
    low = draws_sorted[max(0, math.floor(alpha * draws))]
    high = draws_sorted[min(draws - 1, math.ceil((1-alpha)*draws)-1)]
    return {'mean': sum(values)/n, 'ci': [low, high], 'deal_clusters': n,
            'draws': draws, 'confidence': confidence, 'method': 'deal_cluster_bootstrap'}


def paired_interval(candidate_rows: Sequence[dict], incumbent_rows: Sequence[dict],
                    *, draws: int = 2000, seed: int = 2026,
                    confidence: float = 0.95) -> dict:
    """Paired confidence interval of candidate minus incumbent score.

    Exact same deal seeds and opponent/side fixtures are required. A deal's
    two role outcomes plus all opponents are kept together during resampling.
    """
    if draws < 100 or not 0.5 < confidence < 1.0:
        raise ValueError('bootstrap requires >=100 draws and 0.5<confidence<1')
    a, b = _validate_records(candidate_rows), _validate_records(incumbent_rows)
    if set(a) != set(b) or any(a[k].keys() != b[k].keys() for k in a):
        raise ValueError('candidate and incumbent must share all deal/fixture keys')
    diffs = [sum(a[k][f] - b[k][f] for f in a[k]) / len(a[k]) for k in sorted(a)]
    n = len(diffs)
    rng = random.Random(seed)
    samples = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(draws))
    alpha = (1 - confidence) / 2
    return {'delta': sum(diffs) / n,
            'ci': [samples[max(0, math.floor(alpha * draws))],
                   samples[min(draws-1, math.ceil((1-alpha)*draws)-1)]],
            'deal_clusters': n, 'draws': draws, 'confidence': confidence,
            'method': 'paired_deal_cluster_bootstrap'}
