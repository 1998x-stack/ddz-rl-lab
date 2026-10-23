"""Guarded local archival-checkpoint pruning; defaults to dry run.

Only files in the run's history/ directory named step-NNNNNNNNN.weights.pt
may be considered. latest.pt, best.weights.pt and registry models are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .analysis import QualityError

_NAME = re.compile(r'^step-([0-9]{9})\.weights\.pt$')


def prune_history(run_dir: str | Path, *, keep_recent: int,
                  keep_best: int = 0, apply: bool = False) -> dict:
    if (type(keep_recent) is not int or type(keep_best) is not int or
            keep_recent < 0 or keep_best < 0 or keep_recent + keep_best < 1):
        raise QualityError('at least one retained checkpoint is required')
    root = Path(run_dir)
    folder = root / 'history'
    if root.is_symlink() or folder.is_symlink():
        raise QualityError('unsafe history path')
    if not folder.exists():
        return {'schema': 'ddz-history-retention-v1', 'dry_run': not apply,
                'kept': [], 'prunable': [], 'removed': []}
    from ddz_training.checkpoints import _digest
    entries = []
    for path in folder.iterdir():
        match = _NAME.fullmatch(path.name)
        if not match:
            continue
        sidecar = path.with_suffix(path.suffix + '.json')
        if path.is_symlink() or sidecar.is_symlink() or not sidecar.is_file():
            raise QualityError('checkpoint contains missing/symlink metadata')
        try:
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            if metadata.get('checkpoint') != path.name or metadata.get('sha256') != _digest(path):
                raise QualityError('checkpoint metadata/hash mismatch')
            value = metadata['metrics']['balanced_score']
            if type(value) not in (float, int) or not 0 <= value <= 1:
                raise QualityError('invalid checkpoint score')
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise QualityError(f'untrusted history checkpoint: {path.name}') from exc
        entries.append((int(match.group(1)), path, float(value)))
    if len({x[0] for x in entries}) != len(entries):
        raise QualityError('duplicate checkpoint episode')
    entries.sort(key=lambda x: x[0])
    selected = {p for _,p,_ in entries[-keep_recent:]} if keep_recent else set()
    selected.update(x[1] for x in sorted(entries, key=lambda x: (x[2], x[0]), reverse=True)[:keep_best])
    prune = [p for _,p,_ in entries if p not in selected]
    if apply:
        for path in prune:
            # Verified all candidates before deleting any artifacts. Never follow symlinks.
            path.unlink()
            path.with_suffix(path.suffix + '.json').unlink()
    return {'schema': 'ddz-history-retention-v1', 'dry_run': not apply,
            'kept': [p.name for _,p,_ in entries if p in selected],
            'prunable': [p.name for p in prune],
            'removed': [p.name for p in prune] if apply else []}
