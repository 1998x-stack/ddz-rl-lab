"""Auditable multi-seed training quality analysis.

Input is existing evaluation and throughput JSONL, NOT game authority state.
The unit of bootstrap resampling is an independently seeded *training run*.
Repeated evaluation checkpoints within one run are never treated as independent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics

from loguru import logger

MAX_FILE_BYTES = 12_000_000
MAX_ROWS = 20_000


class QualityError(ValueError):
    """Invalid or incompatible experiment evidence."""


def _rows(path: Path, required: bool) -> list[dict]:
    if path.is_symlink():
        raise QualityError(f'refusing symlink evidence: {path}')
    if not path.exists():
        if required:
            raise QualityError(f'missing evaluation evidence: {path}')
        return []
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise QualityError('invalid or oversized metrics file')
    result = []
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if line_number > MAX_ROWS:
                raise QualityError('too many metric rows')
            try:
                data = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise QualityError(f'corrupt JSONL {path.name}:{line_number}') from exc
            if not isinstance(data, dict):
                raise QualityError(f'non-object JSONL row: {line_number}')
            result.append(data)
    if required and not result:
        raise QualityError('evaluation history is empty')
    return result


def _number(value: object, field: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise QualityError(f'nonfinite or missing metric: {field}')
    v = float(value)
    if (minimum is not None and v < minimum) or (maximum is not None and v > maximum):
        raise QualityError(f'out-of-range metric: {field}')
    return v


def _fingerprint(run: Path) -> tuple[str, dict]:
    """Best-effort provenance: do not pretend missing configurations match."""
    path = run / 'experiment.json'
    if not path.is_file() or path.is_symlink():
        return ('unavailable', {})
    try:
        metadata = json.loads(path.read_text(encoding='utf-8'))
        settings = metadata['settings']
        if not isinstance(settings, dict):
            raise ValueError('settings')
        frozen = {k: v for k, v in settings.items()
                  if k not in ('seed', 'output', 'episodes')}
        return ('experiment-settings', frozen)
    except (ValueError, KeyError, TypeError) as exc:
        raise QualityError(f'invalid experiment metadata: {path}') from exc


def analyze_runs(run_dirs: list[str | Path], *, draws: int = 2000, seed: int = 2026,
                 regression_tolerance: float = 0.10,
                 min_independent_runs: int = 3) -> dict:
    if not 1 <= len(run_dirs) <= 100 or not 100 <= draws <= 20000:
        raise QualityError('invalid experiment or bootstrap count')
    if type(min_independent_runs) is not int or min_independent_runs < 2:
        raise QualityError('min_independent_runs must be >=2')
    if not 0 <= regression_tolerance <= 1:
        raise QualityError('invalid regression tolerance')
    runs = []
    names: set[str] = set()
    fingerprints = []
    for item in run_dirs:
        run = Path(item)
        resolved = run.resolve(strict=True)
        if run.is_symlink() or not resolved.is_dir() or resolved.name in names:
            raise QualityError('duplicate or unsafe run')
        names.add(resolved.name)
        evaluations = _rows(resolved / 'evaluations.jsonl', required=True)
        throughput = _rows(resolved / 'throughput.jsonl', required=False)
        episodes = []
        scores = []
        for row in evaluations:
            ep = _number(row.get('episodes_seen'), 'episodes_seen', minimum=1)
            if not ep.is_integer() or (episodes and ep <= episodes[-1]):
                raise QualityError('evaluation episodes must be strictly increasing')
            episodes.append(int(ep))
            scores.append(_number(row.get('balanced_score'), 'balanced_score',
                                  minimum=0, maximum=1))
            for field in ('landlord_rate', 'farmer_rate'):
                _number(row.get(field), field, minimum=0, maximum=1)
        throughput_warnings = []
        for row in throughput:
            for field in ('episodes_per_s', 'samples_per_s', 'decisions_per_s'):
                value = _number(row.get(field), field, minimum=0)
                if value == 0:
                    throughput_warnings.append(f'zero_{field}')
            total = _number(row.get('episodes'), 'episodes', minimum=0)
            trunc = _number(row.get('truncated'), 'truncated', minimum=0)
            if trunc > total:
                raise QualityError('truncated episodes exceed total episodes')
            if total and trunc / total > 0.05:
                throughput_warnings.append('high_truncation_fraction')
        peak = max(scores)
        latest = scores[-1]
        alerts = sorted(set(throughput_warnings))
        if peak - latest > regression_tolerance:
            alerts.append('score_drop_from_historical_peak')
        if len(evaluations) < 2:
            alerts.append('insufficient_evaluation_history')
        provenance, frozen = _fingerprint(resolved)
        fingerprints.append((provenance, frozen))
        runs.append({'run_id': resolved.name, 'evaluations': len(scores),
                     'last_episodes': episodes[-1], 'latest_score': latest,
                     'peak_score': peak, 'latest_landlord_rate': evaluations[-1]['landlord_rate'],
                     'latest_farmer_rate': evaluations[-1]['farmer_rate'],
                     'throughput_records': len(throughput), 'alerts': alerts,
                     'provenance': provenance})
    comparable = (len(runs) == 1 or
                  all(fingerprints[0] == x for x in fingerprints[1:]) and
                  fingerprints[0][0] != 'unavailable')
    if len(runs) > 1 and not comparable:
        logger.warning('cross_seed_configs_not_verified')
    scores = [x['latest_score'] for x in runs]
    n = len(scores)
    interval = None
    if n >= min_independent_runs and comparable:
        rng = random.Random(seed)
        samples = sorted(sum(scores[rng.randrange(n)] for _ in range(n)) / n
                         for _ in range(draws))
        interval = [samples[math.floor(.025 * draws)],
                    samples[min(draws-1, math.ceil(.975 * draws)-1)]]
    return {'schema': 'ddz-training-quality-v1', 'runs': runs,
            'summary': {'independent_runs': n, 'mean_latest_score': statistics.mean(scores),
                        'std_latest_score': statistics.stdev(scores) if n > 1 else None,
                        'bootstrap_ci95': interval,
                        'config_comparable': comparable,
                        'evidence_sufficient': interval is not None,
                        'min_independent_runs': min_independent_runs},
            'limitations': 'Seed-level bootstrap assumes independently initialized runs. '
                           'Repeated checkpoints within a run are correlated. Peak-to-latest '
                           'regression is a diagnostic heuristic, not a statistical promotion test; '
                           'shared evaluation seeds, differing settings or selected runs can bias estimates.'}
