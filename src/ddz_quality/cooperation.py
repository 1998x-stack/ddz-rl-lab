"""2x2 factorial farmer-team interaction on identical initial deals.

This measures an interaction, NOT an absolute or causal measure of cooperation.
Bidding is the existing fixed seeded random baseline.
"""
from __future__ import annotations

import math
import random
from typing import Mapping

from .analysis import QualityError

ORDER = ('base_base', 'candidate_up', 'candidate_down', 'both_candidates')


def interaction_from_crossplay(lineups: Mapping[str, dict], *, seed: int = 2026,
                               draws: int = 2000, min_deals: int = 30) -> dict:
    if set(lineups) != set(ORDER) or not 100 <= draws <= 20000 or min_deals < 2:
        raise QualityError('expected four factorial lineups and valid draws/min_deals')
    outcomes = {}
    for key in ORDER:
        row = lineups[key]
        records = row.get('outcomes', [])
        if not isinstance(records, list):
            raise QualityError('missing crossplay outcomes')
        by_deal = {}
        for item in records:
            deal, outcome = item['deal_seed'], item['outcome']
            if type(deal) is not int or deal in by_deal or outcome not in ('FARMERS', 'LANDLORD', 'truncated'):
                raise QualityError('invalid or duplicate deal fixture')
            by_deal[deal] = outcome
        outcomes[key] = by_deal
    if any(set(outcomes[ORDER[0]]) != set(outcomes[k]) for k in ORDER[1:]):
        raise QualityError('factorial fixtures must use the same initial deals')
    valid = [i for i in sorted(outcomes[ORDER[0]]) if all(outcomes[k][i] != 'truncated'
            for k in ORDER)]
    dropped = len(outcomes[ORDER[0]]) - len(valid)
    if not valid:
        return {'schema': 'ddz-farmer-interaction-v1', 'paired_deals': 0,
                'excluded_deals': dropped, 'interaction': None,
                'ci95': None, 'eligible': False}
    wins = {k: [int(outcomes[k][i] == 'FARMERS') for i in valid] for k in ORDER}
    values = [wins['both_candidates'][i] - wins['candidate_up'][i] -
              wins['candidate_down'][i] + wins['base_base'][i]
              for i in range(len(valid))]
    point = sum(values) / len(values)
    ci = None
    if len(valid) >= min_deals:
        n = len(valid)
        rng = random.Random(seed)
        b = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(draws))
        ci = [b[math.floor(.025 * draws)], b[min(draws - 1, math.ceil(.975 * draws)-1)]]
    return {'schema': 'ddz-farmer-interaction-v1', 'paired_deals': len(valid),
            'excluded_deals': dropped, 'interaction': point, 'ci95': ci,
            'eligible': ci is not None,
            'farmer_win_rates': {k: sum(v)/len(v) for k,v in wins.items()},
            'interpretation': 'Positive interaction means joint replacements outperform '
                              'the sum of separate improvements on matched deals. '
                              'It is not evidence of explicit communication or superior absolute strength; '
                              'no automatic model promotion.'}
