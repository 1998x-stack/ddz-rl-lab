"""Registered checkpoint role-matrix entrypoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from ddz_registry import ModelRegistry
from .crossplay import load_lineup
from .matrix import evaluate_matrix


def main() -> None:
    p = argparse.ArgumentParser(description='DDZ bounded role-triplet Cross-Play matrix')
    p.add_argument('--registry-dir', default='runs/registry')
    p.add_argument('--landlords', required=True, help='comma separated model IDs')
    p.add_argument('--farmer-ups', required=True)
    p.add_argument('--farmer-downs', required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--deals', type=int, default=20)
    p.add_argument('--draws', type=int, default=200)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    pools = [x.split(',') for x in (a.landlords, a.farmer_ups, a.farmer_downs)]
    names = set().union(*(set(x) for x in pools))
    reg = ModelRegistry(a.registry_dir)
    models = {}
    for model_id in names:
        models[model_id] = load_lineup(reg, {'landlord': model_id,
                          'farmer_up': model_id, 'farmer_down': model_id})['landlord']
    result = evaluate_matrix(models, landlords=pools[0], farmer_ups=pools[1],
                             farmer_downs=pools[2], seed=a.seed, deals=a.deals, draws=a.draws)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if a.report:
        if a.report.exists() or a.report.is_symlink():
            p.error('report already exists; refusing overwrite')
        a.report.parent.mkdir(parents=True, exist_ok=True)
        with a.report.open('x', encoding='utf-8') as f:
            f.write(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
