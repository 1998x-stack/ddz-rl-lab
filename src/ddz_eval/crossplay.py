"""Cross-play of immutable role model bundles on common seeded deals.

No monetary scores, no claims of optimal policy. Bidding is a common random
baseline to isolate playing strength; fixed seeds make comparisons repeatable.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Mapping

from ddz_core.state import Phase, Team
from ddz_env.baselines import random_action_index
from ddz_env.environment import DDZEnv
from ddz_models.policy import RoleBundle, role_for_observation
from ddz_registry import ModelRegistry
from ddz_training.checkpoints import load_checkpoint
from ddz_models.network import NetworkSpec


def load_lineup(registry: ModelRegistry, mapping: Mapping[str, str]) -> dict[str, RoleBundle]:
    if set(mapping) != {'landlord', 'farmer_up', 'farmer_down'}:
        raise ValueError('lineup must specify landlord/farmer_up/farmer_down')
    bundles: dict[str, RoleBundle] = {}
    for role, model_id in mapping.items():
        record = registry.verify(model_id, load_weights=False)
        path, _ = registry._model_files(model_id)
        bundle = RoleBundle(NetworkSpec(hidden=int(record['spec']['hidden'])))
        load_checkpoint(path, bundle)
        bundles[role] = bundle
    return bundles


def run_crossplay(lineups: Mapping[str, Mapping[str, RoleBundle]], *, seed: int,
                  deals: int, max_steps: int = 1200) -> dict:
    if not lineups or len(lineups) > 16 or deals < 1 or deals > 10000 or max_steps < 1:
        raise ValueError('invalid cross-play size')
    results = {}
    for lineup_id, roles in lineups.items():
        if set(roles) != {'landlord', 'farmer_up', 'farmer_down'}:
            raise ValueError('incomplete lineup')
        rows = []
        wins = Counter()
        for i in range(deals):
            deal_seed = seed + i
            env = DDZEnv(deal_seed)
            bidding_rng = random.Random(deal_seed * 17 + 937)
            decision_rng = random.Random(deal_seed * 31 + 991)
            for _ in range(max_steps):
                if env.done:
                    break
                obs = env.current_observation()
                if obs.phase != Phase.PLAY:
                    index = random_action_index(env, bidding_rng)
                else:
                    role = role_for_observation(obs)
                    index = roles[role].act(obs, decision_rng, 0.0).local_index
                env.step(index)
            if not env.done or env.engine.state.truncated:
                rows.append({'deal_seed': deal_seed, 'outcome': 'truncated'})
                wins['truncated'] += 1
                continue
            winner = env.engine.state.winner
            assert winner is not None
            wins[winner.value] += 1
            rows.append({'deal_seed': deal_seed, 'outcome': winner.value})
        valid = wins[Team.LANDLORD.value] + wins[Team.FARMERS.value]
        results[lineup_id] = {
            'deals': deals, 'completed': valid, 'truncated': wins['truncated'],
            'landlord_win_rate': wins[Team.LANDLORD.value] / valid if valid else None,
            'farmer_win_rate': wins[Team.FARMERS.value] / valid if valid else None,
            'outcomes': rows,
        }
    return {'schema': 'ddz-crossplay-v1', 'seed': seed, 'max_steps': max_steps,
            'lineups': results,
            'limitations': 'Role triplets play against each other; these win rates are not absolute strength. '
                           'Bidding uses the same random policy. Use external frozen opponent fixtures for promotion.'}
