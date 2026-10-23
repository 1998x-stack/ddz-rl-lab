"""Explicit foreground run/resume/status; no background training side effects."""
from __future__ import annotations
import argparse
import json
from .runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description='DDZ reproducible training experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='start an experiment from a JSON spec')
    run.add_argument('--spec', required=True)
    run.add_argument('--resume', action='store_true')
    stat = sub.add_parser('status')
    stat.add_argument('experiment_id')
    for item in (run, stat):
        item.add_argument('--runs-dir', default='runs')
    args = parser.parse_args()
    runner = ExperimentRunner(args.runs_dir)
    result = runner.run(args.spec, resume=args.resume) if args.command == 'run' else runner.status(args.experiment_id)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
