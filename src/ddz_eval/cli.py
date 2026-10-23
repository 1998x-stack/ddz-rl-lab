"""Operator CLI: seeded evaluation, immutable registry, evidence-gated promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from ddz_core.logging_config import configure_logging
from ddz_eval.league import load_frozen
from ddz_eval.selection import evaluate_candidates, save_report
from ddz_registry import ModelRegistry


def _pair(spec: str) -> tuple[str, Path]:
    key, sep, path = spec.partition('=')
    if not sep or not key or not path or any(c in key for c in '/\\'):
        raise argparse.ArgumentTypeError('use MODEL_ID=/absolute/trusted/path.weights.pt')
    return key, Path(path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='DDZ RL Lab offline tournament and local model registry')
    p.add_argument('--registry-dir', type=Path, default=Path('runs/registry'))
    p.add_argument('--log-dir', type=Path)
    sub = p.add_subparsers(dest='command', required=True)
    ev = sub.add_parser('evaluate', help='Compare matched checkpoints on frozen seeded deals')
    ev.add_argument('--incumbent', required=True, type=_pair)
    ev.add_argument('--candidate', required=True, action='append', type=_pair)
    ev.add_argument('--opponent', action='append', type=_pair, default=[])
    ev.add_argument('--seed-start', type=int, action='append', required=True)
    ev.add_argument('--deals-per-seed', type=int, default=40)
    ev.add_argument('--min-deals', type=int, default=30)
    ev.add_argument('--min-delta', type=float, default=0.)
    ev.add_argument('--draws', type=int, default=2000)
    ev.add_argument('--bootstrap-seed', type=int, default=2026)
    ev.add_argument('--max-steps', type=int, default=1200)
    ev.add_argument('--report', type=Path, required=True)
    reg = sub.add_parser('register', help='Vetted local inference-only checkpoint')
    reg.add_argument('--model-id', required=True)
    reg.add_argument('--checkpoint', type=Path, required=True)
    reg.add_argument('--trusted-source-dir', type=Path, required=True)
    first = sub.add_parser('bootstrap', help='Explicitly activate initial model once')
    first.add_argument('--model-id', required=True)
    promote = sub.add_parser('promote', help='Evidence-gated promotion of registered challenger')
    promote.add_argument('--model-id', required=True)
    promote.add_argument('--report', type=Path, required=True)
    verify = sub.add_parser('verify', help='Verify model checksum and runtime compatibility')
    verify.add_argument('--model-id', required=True)
    sub.add_parser('list', help='List registry and active champion')
    return p


def run(argv: list[str] | None = None) -> dict:
    args = parser().parse_args(argv)
    configure_logging(args.log_dir, 'INFO')
    registry = ModelRegistry(args.registry_dir)
    if args.command == 'evaluate':
        choices = [args.incumbent, *args.candidate]
        keys = [name for name, _ in choices]
        if len(set(keys)) != len(keys):
            raise ValueError('model identifiers must be unique')
        frozen = [(name, load_frozen(path)) for name, path in args.opponent]
        models = {name: load_frozen(path) for name, path in choices}
        result = evaluate_candidates(
            candidates=models, incumbent_id=args.incumbent[0], frozen=frozen,
            seed_starts=args.seed_start, deals_per_seed=args.deals_per_seed,
            min_deals=args.min_deals, min_delta=args.min_delta, draws=args.draws,
            bootstrap_seed=args.bootstrap_seed, max_steps=args.max_steps)
        result['model_hashes'] = {name: _sha(path) for name, path in choices}
        result['opponent_hashes'] = {name: _sha(path) for name, path in args.opponent}
        result['report_type'] = 'operator-generated-untrusted-until-reviewed'
        save_report(args.report, result)
    elif args.command == 'register':
        result = registry.register(args.model_id, args.checkpoint,
                                   trusted_source_dir=args.trusted_source_dir)
    elif args.command == 'bootstrap':
        result = registry.bootstrap_active(args.model_id)
    elif args.command == 'promote':
        report = json.loads(args.report.read_text(encoding='utf-8'))
        result = registry.promote(args.model_id, report=report)
    elif args.command == 'verify':
        result = registry.verify(args.model_id)
    else:
        result = registry.list_models()
    return result


def main() -> None:
    try:
        print(json.dumps(run(), indent=2, sort_keys=True))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == '__main__':
    main()
