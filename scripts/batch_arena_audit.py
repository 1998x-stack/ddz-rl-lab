"""Run complete three-model Arena games and verify every saved replay."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import torch
from loguru import logger

from ddz_core.actions import ActionKind
from ddz_core.state import Phase
from ddz_env.environment import DDZEnv
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from ddz_quality.play_audit import audit_replay
from ddz_registry import ModelRegistry
from ddz_training.checkpoints import load_checkpoint


def _write_private_json(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=987654321)
    parser.add_argument('--registry-dir', type=Path, default=Path('runs/registry'))
    parser.add_argument('--output', type=Path, default=Path('runs/real-play-100'))
    args = parser.parse_args()
    if not 1 <= args.games <= 10000 or not 0 <= args.seed < 2**32:
        parser.error('invalid games or seed')
    if args.output.exists():
        parser.error('output already exists; choose a fresh directory')
    args.output.mkdir(parents=True, mode=0o700)
    replay_dir = args.output / 'replays'
    replay_dir.mkdir(mode=0o700)
    logger.remove()
    torch.set_num_threads(1)
    registry = ModelRegistry(args.registry_dir)
    model_id = registry.list_models()['active']
    info = registry.verify(model_id)
    weights = args.registry_dir / 'bundles' / model_id / 'weights.pt'
    bundle = RoleBundle(NetworkSpec.from_dict(info['spec']))
    load_checkpoint(weights, bundle)
    totals: Counter[str] = Counter()
    winners: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    rows = []
    started = time.monotonic()
    for game_no in range(args.games):
        seed = (args.seed + game_no) % 2**32
        env = DDZEnv(seed)
        rng = random.Random(seed ^ 0xD0D5)
        first_calls = 0
        first_bid_decisions = 0
        for _ in range(1200):
            if env.done:
                break
            obs = env.current_observation(event_limit=16)
            choice = bundle.act(obs, rng, 0.0)
            action = obs.legal_actions[choice.local_index]
            if obs.phase == Phase.CALL and obs.deal_index == 0:
                first_bid_decisions += 1
                first_calls += int(action.kind == ActionKind.CALL)
            roles[choice.role] += 1
            env.step(choice.local_index, include_digest=False)
        if not env.done:
            raise RuntimeError(f'decision budget exceeded game {game_no}')
        replay = env.export_replay()
        replay['model_ids'] = [model_id] * 3
        replay['model_sha256'] = [info['sha256']] * 3
        replay['random_bidding'] = False
        game_id = f'batch-{game_no:04d}'
        audit = audit_replay(replay, game_id=game_id, example_limit=10) if replay['terminated'] else None
        if audit is not None:
            totals.update(audit['flags'])
        winner = replay['winner'] or 'TRUNCATED'
        winners[winner] += 1
        row = {'game_id': game_id, 'seed': seed, 'steps': len(replay['actions']),
               'winner': winner, 'first_bid_decisions': first_bid_decisions,
               'first_calls': first_calls, 'flags': audit['flags'] if audit else {},
               'examples': audit['examples'] if audit else []}
        rows.append(row)
        _write_private_json(replay_dir / f'{game_id}.json', replay)
        if (game_no + 1) % 10 == 0:
            print(f'completed={game_no + 1}/{args.games} elapsed_s={time.monotonic()-started:.1f}',
                  flush=True)
    report = {'schema': 'ddz-complete-selfplay-audit-v1', 'model_id': model_id,
              'model_sha256': info['sha256'], 'games': args.games, 'seed_start': args.seed,
              'elapsed_s': time.monotonic() - started, 'winners': dict(winners),
              'decision_roles': dict(roles), 'flags': dict(totals),
              'first_deal_call_rate': sum(r['first_calls'] for r in rows) /
                                      max(1, sum(r['first_bid_decisions'] for r in rows)),
              'first_deal_all_declined': sum(r['first_calls'] == 0 for r in rows),
              'game_reports': rows}
    _write_private_json(args.output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'game_reports'},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
