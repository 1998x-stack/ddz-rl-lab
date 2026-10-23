"""CLI cross-play: --lineup id:landlord,farmer_up,farmer_down (registered IDs)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .crossplay import load_lineup, run_crossplay
from ddz_registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description='Cross-play registered role triples on common seeded deals')
    parser.add_argument('--registry-dir', default='runs/registry')
    parser.add_argument('--lineup', action='append', required=True,
                        help='name:landlord_id,farmer_up_id,farmer_down_id (repeatable)')
    parser.add_argument('--deals', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-steps', type=int, default=1200)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    registry = ModelRegistry(args.registry_dir)
    lineups = {}
    for entry in args.lineup:
        label, colon, ids = entry.partition(':')
        values = ids.split(',')
        if not colon or not label.isidentifier() or label in lineups or len(values) != 3:
            parser.error('expected distinct name:landlord,farmer_up,farmer_down')
        lineups[label] = load_lineup(registry, dict(zip(('landlord', 'farmer_up', 'farmer_down'), values)))
    result = run_crossplay(lineups, deals=args.deals, seed=args.seed, max_steps=args.max_steps)
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.report:
        if args.report.exists() or args.report.is_symlink():
            parser.error('report path already exists; refusing overwrite')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
