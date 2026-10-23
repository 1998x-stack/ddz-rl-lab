"""Versioned, bounded training specs; checkpoint-aware foreground execution.

Runs deliberately stay in the invoking process; no hidden background daemon.
State metadata is advisory; the checkpoint and its verified sidecar are authoritative.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from loguru import logger

from ddz_training.trainer import DMCTrainer, TrainConfig

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ALLOWED = {f.name for f in fields(TrainConfig)} - {'output'}


class ExperimentError(ValueError):
    pass


def _name(value: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ExperimentError('invalid experiment id')
    return value


def _safe_run(root: Path, name: str) -> Path:
    folder = root / _name(name)
    if folder.is_symlink() or (folder.exists() and not folder.resolve().is_relative_to(root.resolve())):
        raise ExperimentError('unsafe run directory')
    return folder


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix='.experiment-', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _read_spec(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 32_768:
        raise ExperimentError('spec too large')
    spec = json.loads(path.read_text('utf-8'))
    if not isinstance(spec, dict) or set(spec) != {'schema', 'id', 'training'} or spec['schema'] != 'ddz-experiment-v1':
        raise ExperimentError('expected ddz-experiment-v1 schema/id/training')
    _name(spec['id'])
    overrides = spec['training']
    if not isinstance(overrides, dict) or set(overrides) - _ALLOWED:
        raise ExperimentError('invalid training options')
    if not overrides or 'episodes' not in overrides:
        raise ExperimentError('training.episodes is required')
    # Reject booleans as integer settings and very long, resource-exhausting jobs.
    for key in ('episodes', 'batch_episodes', 'eval_games', 'eval_every', 'hidden',
                'max_steps', 'actor_workers', 'replay_capacity', 'replay_sample_size',
                'updates_per_batch', 'history_capacity', 'seed'):
        if key in overrides and type(overrides[key]) is not int:
            raise ExperimentError(f'{key} must be an integer')
    if overrides['episodes'] > 2_000_000 or overrides.get('actor_workers', 1) > 64:
        raise ExperimentError('run exceeds local safety limit')
    if overrides.get('device', 'cpu') not in {'cpu', 'cuda', 'mps'}:
        raise ExperimentError('invalid device')
    # Single directory per experiment. Never accept output paths from JSON.
    return spec


def _fingerprint(spec: dict[str, Any]) -> str:
    # Episode count is a per-invocation increment; all other settings frozen.
    frozen = {k: v for k, v in spec['training'].items() if k != 'episodes'}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentRunner:
    def __init__(self, runs_dir: str | Path):
        self.runs_dir = Path(runs_dir).resolve()

    def run(self, spec_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
        spec = _read_spec(Path(spec_path))
        name = spec['id']
        run_dir = _safe_run(self.runs_dir, name)
        config = TrainConfig(**spec['training'], output=str(run_dir))
        state_file = run_dir / 'experiment.json'
        old = None
        if state_file.exists():
            old = json.loads(state_file.read_text('utf-8'))
        if resume:
            if old is None or old.get('config_fingerprint') != _fingerprint(spec):
                raise ExperimentError('resume config mismatch or uninitialized experiment')
            if (run_dir / 'latest.pt').is_symlink() or not (run_dir / 'latest.pt').is_file():
                raise ExperimentError('no last verified checkpoint for resume')
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise ExperimentError('experiment exists; use --resume with matching config')
        run_dir.mkdir(parents=True, exist_ok=True)
        status = {'schema': 'ddz-experiment-state-v1', 'id': name,
                  'config_fingerprint': _fingerprint(spec),
                  'settings': asdict(config), 'status': 'running', 'started_at': _now(),
                  'pid': os.getpid(), 'resume': resume,
                  'previous_episodes': (old or {}).get('episodes_seen', 0)}
        _atomic_json(state_file, status)
        try:
            trainer = DMCTrainer(config)
            if resume:
                trainer.resume(run_dir / 'latest.pt')
                # A crashed run may have checkpointed after experiment.json was last
                # updated. The verified checkpoint is the authoritative progress.
                recorded = old.get('episodes_seen', old.get('previous_episodes', 0))
                if (trainer.episodes_seen < recorded or
                    (old.get('status') == 'finished' and trainer.episodes_seen != recorded)):
                    raise ExperimentError('checkpoint/status episode mismatch')
                status['previous_episodes'] = trainer.episodes_seen
                _atomic_json(state_file, status)
            result = trainer.train()
        except BaseException as exc:
            status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          finished_at=_now(), error_type=type(exc).__name__)
            _atomic_json(state_file, status)
            logger.bind(experiment=name).exception('experiment execution interrupted or failed')
            raise
        status.update(status='finished', finished_at=_now(),
                      episodes_seen=trainer.episodes_seen, updates=trainer.updates,
                      balanced_score=result.get('balanced_score'))
        _atomic_json(state_file, status)
        logger.bind(experiment=name, episodes=trainer.episodes_seen).info('experiment_finished')
        return status

    def status(self, experiment_id: str) -> dict[str, Any]:
        run = _safe_run(self.runs_dir, experiment_id)
        path = run / 'experiment.json'
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(experiment_id)
        value = json.loads(path.read_text('utf-8'))
        if value.get('schema') != 'ddz-experiment-state-v1' or value.get('id') != experiment_id:
            raise ExperimentError('invalid experiment state')
        return value
