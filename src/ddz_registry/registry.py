"""Immutable local inference registry with SHA validation and gated promotion.

The API assumes an operator-managed, trusted checkpoint source directory.
It does not expose remote uploads or accept executable arbitrary model code.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterator
from loguru import logger

from ddz_env.encoding import encoder_ids
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from ddz_training.checkpoints import SCHEMA, load_checkpoint

_REGISTRY_SCHEMA = 'ddz-model-registry-v1'
_IDENT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.index-', dir=path.parent, suffix='.json')
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


class ModelRegistry:
    """Single-host registry. Lock serializes writers on macOS and Linux."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / 'registry.json'

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        import fcntl
        with (self.root / '.registry.lock').open('a+') as lockfile:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict:
        if not self.index.exists():
            return {'schema': _REGISTRY_SCHEMA, 'active': None, 'models': {}}
        data = json.loads(self.index.read_text(encoding='utf-8'))
        if data.get('schema') != _REGISTRY_SCHEMA or not isinstance(data.get('models'), dict):
            raise ValueError('unknown registry index schema')
        return data

    def list_models(self) -> dict:
        return self._read()

    @staticmethod
    def _validate_id(model_id: str) -> None:
        if not _IDENT.fullmatch(model_id) or model_id in ('.', '..'):
            raise ValueError('invalid model id')

    def _model_files(self, model_id: str) -> tuple[Path, Path]:
        self._validate_id(model_id)
        folder = self.root / 'bundles' / model_id
        return folder / 'weights.pt', folder / 'weights.pt.json'

    def register(self, model_id: str, checkpoint: str | Path,
                 *, trusted_source_dir: str | Path, device: str = 'cpu') -> dict:
        """Copy a vetted local inference bundle after checking its schema and hash.

        Caller explicitly designates a trusted source directory. Paths that
        escape that directory (including symlinks) are rejected.
        """
        self._validate_id(model_id)
        source = Path(checkpoint).resolve(strict=True)
        trusted = Path(trusted_source_dir).resolve(strict=True)
        if not source.is_file() or not source.is_relative_to(trusted):
            raise ValueError('checkpoint outside trusted directory')
        metadata_path = source.with_suffix(source.suffix + '.json')
        if not metadata_path.is_file() or not metadata_path.resolve().is_relative_to(trusted):
            raise ValueError('checkpoint metadata outside trusted directory')
        metadata = json.loads(metadata_path.read_text('utf-8'))
        if metadata.get('inference_only') is not True or metadata.get('schema') != SCHEMA:
            raise ValueError('registry requires an inference-only checkpoint')
        if metadata.get('checkpoint') != source.name or metadata.get('sha256') != _sha(source):
            raise ValueError('source checkpoint integrity error')
        spec_data = metadata.get('spec', {})
        spec = NetworkSpec.from_dict(spec_data)
        if spec_data != {'state_dim': spec.state_dim, 'action_dim': spec.action_dim,
                         'hidden': spec.hidden, 'schema': spec.schema}:
            raise ValueError('unsupported network specification')
        if metadata.get('encoders') != encoder_ids(spec.schema):
            raise ValueError('incompatible encoders')
        # Safe weights_only load; any failure aborts registration before mutation.
        load_checkpoint(source, RoleBundle(spec, device))
        output, sidecar = self._model_files(model_id)
        with self._lock():
            idx = self._read()
            if model_id in idx['models'] or output.parent.exists():
                raise ValueError(f'non-overwritable model already exists: {model_id}')
            output.parent.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix='.staging-', dir=output.parent.parent))
            try:
                copied = temporary / output.name
                shutil.copyfile(source, copied)
                if _sha(copied) != metadata['sha256']:
                    raise ValueError('checksum changed while copying')
                packaged = dict(metadata, checkpoint='weights.pt')
                (temporary / sidecar.name).write_text(json.dumps(packaged, sort_keys=True, indent=2),
                                                      encoding='utf-8')
                os.rename(temporary, output.parent)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            idx['models'][model_id] = {'sha256': metadata['sha256'],
                                       'episodes_seen': metadata['episodes_seen'],
                                       'updates': metadata['updates'],
                                       'encoder_version': metadata['encoders']['observation'],
                                       'action_version': metadata['encoders']['action'],
                                       'spec': metadata['spec']}
            _atomic_json(self.index, idx)
            logger.bind(component='registry', model=model_id).info('registered immutable inference bundle')
            return dict(idx['models'][model_id])

    def verify(self, model_id: str, *, load_weights: bool = True) -> dict:
        self._validate_id(model_id)
        idx = self._read()
        record = idx['models'].get(model_id)
        if record is None:
            raise KeyError(f'unknown model: {model_id}')
        path, meta_path = self._model_files(model_id)
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f'missing model artifacts for {model_id}')
        if path.is_symlink() or meta_path.is_symlink():
            raise ValueError('registry artifacts cannot be symlinks')
        meta = json.loads(meta_path.read_text('utf-8'))
        if not meta.get('inference_only') or meta.get('sha256') != record['sha256'] or _sha(path) != record['sha256']:
            raise ValueError('registry checkpoint hash mismatch')
        spec = NetworkSpec.from_dict(meta['spec'])
        if meta.get('encoders') != encoder_ids(spec.schema):
            raise ValueError('registry encoder mismatch')
        if load_weights:
            load_checkpoint(path, RoleBundle(spec))
        return dict(record)

    def promote(self, model_id: str, *, report: dict) -> dict:
        """Explicit promotion; the report must recommend this exact model.

        A report alone is not cryptographically authenticated; promotion must be
        an operator-only action in a trusted workspace. It is never exposed to
        anonymous Web Arena viewers.
        """
        self._validate_id(model_id)
        if report.get('schema') != 'ddz-tournament-report-v1' or report.get('recommended_id') != model_id:
            raise ValueError('report does not recommend requested model')
        info = report.get('candidates', {}).get(model_id, {})
        if info.get('eligible') is not True or info.get('vs_incumbent', {}).get('deal_clusters', 0) < report.get('min_deals', 0):
            raise ValueError('candidate does not satisfy evaluation gate')
        if not info['vs_incumbent']['ci'][0] > report['min_delta']:
            raise ValueError('paired confidence lower bound does not clear threshold')
        with self._lock():
            idx = self._read()
            if idx.get('active') != report.get('incumbent_id'):
                raise ValueError('incumbent changed since evaluation')
            if model_id not in idx['models']:
                raise KeyError(f'model must be registered before promotion: {model_id}')
            if report.get('model_hashes', {}).get(model_id) != idx['models'][model_id]['sha256'] or \
               report.get('model_hashes', {}).get(idx['active']) != idx['models'][idx['active']]['sha256']:
                raise ValueError('evaluation evidence not bound to registered model hashes')
            self.verify(model_id)
            idx['active'] = model_id
            _atomic_json(self.index, idx)
            logger.bind(component='registry', model=model_id).info('champion promoted after evidence gate')
            return {'active': model_id, 'sha256': idx['models'][model_id]['sha256']}

    def bootstrap_active(self, model_id: str) -> dict:
        """Explicit operator action for the first ever champion (no comparison)."""
        with self._lock():
            idx = self._read()
            if idx['active'] is not None:
                raise ValueError('already has active champion; use gated promotion')
            self.verify(model_id)
            idx['active'] = model_id
            _atomic_json(self.index, idx)
            return {'active': model_id, 'bootstrap': True}
