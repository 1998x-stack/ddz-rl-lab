"""Paired diagnostic of overriding v2 teammate-pass when landlord has <=2 cards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from loguru import logger

from ddz_core.actions import ActionKind
from ddz_core.state import Phase
from ddz_env.baselines import reasonable_action_score
from ddz_env.environment import DDZEnv
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from ddz_registry import ModelRegistry
from ddz_quality.policy_identity import policy_fingerprint
from ddz_training.checkpoints import load_checkpoint


def run(bundle: RoleBundle, seed: int, override: bool) -> tuple[str, int]:
    env = DDZEnv(seed)
    rng = random.Random(seed ^ 0xD0D5)
    interventions = 0
    for _ in range(1200):
        if env.done:
            break
        obs = env.current_observation(event_limit=16)
        choice = bundle.act(obs, rng, 0.0)
        idx = choice.local_index
        # Define the two policy variants EXPLICITLY. Older v2 always
        # passed when a farmer's teammate led, even against a critical
        # one- or two-card landlord. The new candidate allows a legal beat.
        critical = (obs.phase == Phase.PLAY and obs.seat != obs.landlord and
                    obs.last_play_seat is not None and
                    obs.last_play_seat != obs.landlord and
                    obs.remaining_counts[obs.landlord] <= 2)
        if critical:
            passing = next((i for i, action in enumerate(obs.legal_actions)
                            if action.kind == ActionKind.PASS), None)
            playable = [i for i, action in enumerate(obs.legal_actions)
                        if action.kind == ActionKind.PLAY]
            if passing is not None and playable:
                if not override:
                    idx = passing  # historical unconditional teammate-PASS rule
                elif idx == passing:
                    model = bundle.models[choice.role]
                    state = torch.tensor(bundle.encode_observation(obs), dtype=torch.float32)
                    candidates = torch.tensor([bundle.encode_action(obs, obs.legal_actions[i])
                                               for i in playable], dtype=torch.float32)
                    with torch.inference_mode():
                        scores = model.score_legal(state, candidates).tolist()
                    idx = playable[max(range(len(playable)), key=lambda j: scores[j] +
                                   .15 * reasonable_action_score(obs, obs.legal_actions[playable[j]]))]
                if override and idx != passing:
                    interventions += 1
        env.step(idx, include_digest=False)
    if not env.done or env.engine.state.truncated:
        raise RuntimeError(f'incomplete game seed={seed}')
    return env.engine.state.winner.value, interventions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--seed', type=int, default=987654321)
    p.add_argument('--registry-dir', type=Path, default=Path('runs/registry'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.games <= 10000 or a.output.exists():
        p.error('invalid games or existing output')
    logger.remove()
    torch.set_num_threads(1)
    registry = ModelRegistry(a.registry_dir)
    model_id = registry.list_models()['active']
    info = registry.verify(model_id)
    bundle = RoleBundle(NetworkSpec.from_dict(info['spec']))
    load_checkpoint(a.registry_dir / 'bundles' / model_id / 'weights.pt', bundle)
    rows = []
    for i in range(a.games):
        seed = (a.seed + i) % 2**32
        base, _ = run(bundle, seed, False)
        trial, interventions = run(bundle, seed, True)
        rows.append({'seed': seed, 'baseline': base, 'override': trial,
                     'interventions': interventions})
    switched_to_farmer = sum(r['baseline'] == 'LANDLORD' and r['override'] == 'FARMERS'
                             for r in rows)
    switched_to_landlord = sum(r['baseline'] == 'FARMERS' and r['override'] == 'LANDLORD'
                               for r in rows)
    paired = [int(r['override'] == 'FARMERS') - int(r['baseline'] == 'FARMERS')
              for r in rows]
    bootstrap_rng = random.Random(a.seed ^ 0xB007)
    draws = sorted(sum(paired[bootstrap_rng.randrange(a.games)] for _ in range(a.games)) /
                   a.games for _ in range(10000))
    report = {'schema': 'ddz-urgent-pass-ablation-v1', 'model_id': model_id,
              'model_sha256': info['sha256'],
              'policy_sha256': policy_fingerprint(info['sha256'],info['spec'])['sha256'],
              'comparison': 'historical_unconditional_pass_vs_conditional_urgent_response',
              'games': a.games,
              'intervened_games': sum(r['interventions'] > 0 for r in rows),
              'interventions': sum(r['interventions'] for r in rows),
              'switched_to_farmer': switched_to_farmer,
              'switched_to_landlord': switched_to_landlord,
              'paired_farmer_win_delta': (switched_to_farmer - switched_to_landlord)/a.games,
              'paired_delta_bootstrap_ci95': [draws[249], draws[9749]],
              'rows': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
