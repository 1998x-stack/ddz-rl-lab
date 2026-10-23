"""CLI for offline, bounded quality evidence and human-reviewed retention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import QualityError, analyze_runs
from .cooperation import interaction_from_crossplay
from .play_audit import audit_replay_dir
from .retention import prune_history
from .auction_audit import paired_auction_audit


def _write_report(result: dict, destination: Path | None) -> None:
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if destination is not None:
        if destination.is_symlink():
            raise QualityError('refusing symlink report destination')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    print(encoded, end='')


def main() -> None:
    parser = argparse.ArgumentParser(description='DDZ v12 training quality and farmer interaction audit')
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('analyze', help='multi-run stability and diagnostic alerts')
    a.add_argument('--run', type=Path, action='append', required=True)
    a.add_argument('--draws', type=int, default=2000)
    a.add_argument('--seed', type=int, default=2026)
    a.add_argument('--min-runs', type=int, default=3)
    a.add_argument('--regression-tolerance', type=float, default=.10)
    a.add_argument('--report', type=Path)
    r = sub.add_parser('retention', help='inspect or prune verified archival history snapshots')
    r.add_argument('--run', type=Path, required=True)
    r.add_argument('--keep-recent', type=int, default=6)
    r.add_argument('--keep-best', type=int, default=0)
    r.add_argument('--apply', action='store_true', help='actually remove files (default: dry run)')
    r.add_argument('--report', type=Path)
    c = sub.add_parser('cooperation', help='2x2 paired farmer role interaction')
    c.add_argument('--registry-dir', type=Path, default=Path('runs/registry'))
    for name in ('landlord', 'up-base', 'up-candidate', 'down-base', 'down-candidate'):
        c.add_argument(f'--{name}', required=True)
    c.add_argument('--seed', type=int, default=2026)
    c.add_argument('--deals', type=int, default=40)
    c.add_argument('--draws', type=int, default=2000)
    c.add_argument('--min-deals', type=int, default=30)
    c.add_argument('--report', type=Path)
    p = sub.add_parser('play-audit', help='verify completed Arena replays and inspect play decisions')
    p.add_argument('--replay-dir', type=Path, default=Path('runs/arena-replays'))
    p.add_argument('--example-limit', type=int, default=20)
    p.add_argument('--report', type=Path)
    d = sub.add_parser('export-decisions', help='export verified Arena decisions with all legal candidates')
    d.add_argument('--replay-dir', type=Path, default=Path('runs/arena-replays'))
    d.add_argument('--output', type=Path, required=True)
    d.add_argument('--include-incomplete', action='store_true',
                   help='include verified session journals without terminal labels')
    d.add_argument('--report', type=Path)
    q = sub.add_parser('auction-audit', help='paired CALL vs DECLINE on same initial deals')
    q.add_argument('--seed', type=int, default=2026)
    q.add_argument('--deals', type=int, default=30)
    q.add_argument('--draws', type=int, default=2000)
    q.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'analyze':
            result = analyze_runs(args.run, draws=args.draws, seed=args.seed,
                                  min_independent_runs=args.min_runs,
                                  regression_tolerance=args.regression_tolerance)
        elif args.command == 'retention':
            result = prune_history(args.run, keep_recent=args.keep_recent,
                                   keep_best=args.keep_best, apply=args.apply)
        elif args.command == 'auction-audit':
            result = paired_auction_audit(args.seed, args.deals, args.draws)
        elif args.command == 'play-audit':
            if not 0 <= args.example_limit <= 100:
                raise QualityError('example-limit must be 0..100')
            result = audit_replay_dir(args.replay_dir, example_limit=args.example_limit)
        elif args.command == 'export-decisions':
            from .decision_data import export_decisions
            result = export_decisions(args.replay_dir, args.output,
                                      include_incomplete=args.include_incomplete)
        else:
            from ddz_eval.crossplay import load_lineup, run_crossplay
            from ddz_registry import ModelRegistry
            registry = ModelRegistry(args.registry_dir)
            specs = {
                'base_base': [args.landlord, args.up_base, args.down_base],
                'candidate_up': [args.landlord, args.up_candidate, args.down_base],
                'candidate_down': [args.landlord, args.up_base, args.down_candidate],
                'both_candidates': [args.landlord, args.up_candidate, args.down_candidate],
            }
            lineups = {key: load_lineup(registry, dict(zip(
                ('landlord', 'farmer_up', 'farmer_down'), model_ids)))
                for key, model_ids in specs.items()}
            raw = run_crossplay(lineups, seed=args.seed, deals=args.deals)
            result = interaction_from_crossplay(raw['lineups'], seed=args.seed,
                                                draws=args.draws, min_deals=args.min_deals)
            result['models'] = specs
        _write_report(result, args.report)
    except (QualityError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
