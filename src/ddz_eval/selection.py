"""Seeded candidate ranking and conservative incumbent promotion recommendation.

This is an *evaluation* policy. Actual registry promotion is a separate action.
Do not repeatedly tune candidates against the reported holdout seeds and then
claim those holdouts remain independent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json
import math
from ddz_models.policy import RoleBundle
from .league import run_tournament
from .statistics import clustered_interval, paired_interval


def _flatten(out: dict) -> list[dict]:
    rows = []
    for name, values in out['opponents'].items():
        rows.extend({**r, 'opponent': name} for r in values.get('outcomes', []))
    return rows


def seed_plan(starts: Sequence[int], deals_per_seed: int) -> list[tuple[int, int]]:
    if deals_per_seed < 1 or not starts or len(set(starts)) != len(starts):
        raise ValueError('invalid seed blocks')
    pairs = [(start, start + i) for start in starts for i in range(deals_per_seed)]
    if len({deal for _, deal in pairs}) != len(pairs):
        raise ValueError('overlapping tournament deal-seed blocks')
    return [(int(start), deals_per_seed) for start in starts]


def evaluate_candidates(*, candidates: dict[str, RoleBundle],
                        incumbent_id: str, frozen: Sequence[tuple[str, RoleBundle]],
                        seed_starts: Sequence[int], deals_per_seed: int,
                        min_deals: int = 30, min_delta: float = 0.0,
                        draws: int = 2000, bootstrap_seed: int = 2026,
                        max_steps: int = 1200) -> dict:
    if incumbent_id not in candidates or len(candidates) < 2 or min_deals < 1:
        raise ValueError('need incumbent and at least one challenger, min_deals >=1')
    if min_delta < 0 or not math.isfinite(min_delta):
        raise ValueError('min_delta must be finite and nonnegative')
    if 'random' in dict(frozen) or len({name for name, _ in frozen}) != len(frozen):
        raise ValueError('invalid opponent identifiers')
    blocks = seed_plan(seed_starts, deals_per_seed)
    result: dict = {}
    rows_by_id = {}
    for key, model in candidates.items():
        if not key or '/' in key or '\\' in key:
            raise ValueError('invalid candidate identifier')
        records = []
        truncations = 0
        for start, n in blocks:
            tournament = run_tournament(model, frozen, games=n, seed=start,
                                        max_steps=max_steps, record_outcomes=True)
            truncations += sum(x['truncated'] for x in tournament['opponents'].values())
            records.extend(_flatten(tournament))
        if truncations:
            raise ValueError(f'cannot rank truncated games: candidate={key}, count={truncations}')
        scores = clustered_interval(records, draws=draws, seed=bootstrap_seed)
        result[key] = {'score': scores, 'fixture_count': len(records)}
        rows_by_id[key] = records
    for key, entry in result.items():
        if key == incumbent_id:
            entry['vs_incumbent'] = None
            entry['eligible'] = False
        else:
            interval = paired_interval(rows_by_id[key], rows_by_id[incumbent_id],
                                       draws=draws, seed=bootstrap_seed)
            eligible = (interval['deal_clusters'] >= min_deals and
                        interval['ci'][0] > min_delta)
            entry['vs_incumbent'] = interval
            entry['eligible'] = eligible
    ranked = sorted(result, key=lambda k: (-result[k]['score']['mean'], k))
    qualified = [k for k in ranked if result[k]['eligible']]
    # Promote only one candidate, selected by predeclared highest point estimate
    # among challengers whose paired lower confidence bound clears threshold.
    recommendation = qualified[0] if qualified else None
    return {'schema': 'ddz-tournament-report-v1', 'incumbent_id': incumbent_id,
            'recommended_id': recommendation, 'candidates': result, 'ranked_ids': ranked,
            'seed_starts': list(seed_starts), 'deals_per_seed': deals_per_seed,
            'min_deals': min_deals, 'min_delta': min_delta,
            'bootstrap_seed': bootstrap_seed, 'draws': draws,
            'opponents': ['random', *[name for name, _ in frozen]],
            'note': 'Paired bootstrap on deal clusters; exploration and bidding fixed. '
                    'Research evidence, not proof of universal strength. '
                    'Do not reuse this holdout repeatedly for hyperparameter tuning.'}


def save_report(path: str | Path, report: dict) -> Path:
    """Save readable tournament evidence for a separate explicit promotion call."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path
