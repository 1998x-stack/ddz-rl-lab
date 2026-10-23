"""Crash-consistent local Arena journals; NEVER serve these private files via HTTP.

The journal contains the private deal seed and RNG states. It is intentionally
stored outside the public replay listing and guarded by a verified replay path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from ddz_core.actions import Action
from ddz_env.environment import DDZEnv

SESSION_SCHEMA = 'ddz-arena-session-v1'
_ID = re.compile(r'^[0-9a-f]{24}$')
MAX_JOURNAL_BYTES = 4_000_000


def _nested_tuples(value: Any) -> Any:
    return tuple(_nested_tuples(x) for x in value) if isinstance(value, list) else value


class ArenaJournal:
    def __init__(self, replay_dir: Path):
        self.directory = Path(replay_dir).resolve() / 'sessions'
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise ValueError('unsafe journal directory')
        os.chmod(self.directory, 0o700)

    def _path(self, game_id: str) -> Path:
        if not _ID.fullmatch(game_id):
            raise ValueError('invalid game id')
        return self.directory / f'{game_id}.json'

    def save(self, game: Any) -> None:
        env = game.env
        path = self._path(game.game_id)
        if path.is_symlink():
            raise ValueError('refusing journal symlink')
        payload = {
            'schema': SESSION_SCHEMA, 'game_id': game.game_id,
            'seed': env.engine.seed, 'model_ids': list(game.policy_ids),
            'model_sha256': list(game.model_hashes),
            'delay': game.delay, 'random_bidding': game.random_bidding,
            'state_digest': env.engine.state_digest(),
            'actions': [a.to_dict() for a in env.engine.action_history],
            'rng_states': [rng.getstate() for rng in game.rngs],
            'completed': env.done,
        }
        value = json.dumps(payload, separators=(',', ':'), allow_nan=False)
        if len(value.encode('utf-8')) > MAX_JOURNAL_BYTES:
            raise ValueError('arena journal size limit exceeded')
        fd, tmp = tempfile.mkstemp(prefix='.session-', suffix='.tmp', dir=self.directory)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as out:
                out.write(value)
                out.write('\n')
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, path)
            # Durable directory entry after successful atomic replacement.
            parent_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def _read_path(self, path: Path) -> dict[str, Any] | None:
        try:
            if path.is_symlink() or not path.is_file() or not _ID.fullmatch(path.stem):
                return None
            if path.stat().st_size > MAX_JOURNAL_BYTES:
                return None
            data = json.loads(path.read_text('utf-8'))
            if (data.get('schema') != SESSION_SCHEMA or data.get('game_id') != path.stem or
                type(data.get('seed')) is not int or not 0 <= data['seed'] < 2**32 or
                not isinstance(data.get('actions'), list) or len(data['actions']) > 1200 or
                not isinstance(data.get('model_ids'), list) or len(data['model_ids']) != 3 or
                not isinstance(data.get('model_sha256'), list) or len(data['model_sha256']) != 3 or
                not isinstance(data.get('rng_states'), list) or len(data['rng_states']) != 3 or
                type(data.get('delay')) not in (int, float) or
                not 0.05 <= data['delay'] <= 3.0 or
                type(data.get('completed')) is not bool or
                type(data.get('random_bidding', False)) is not bool):
                return None
            return data
        except (ValueError, OSError, TypeError, KeyError, AttributeError):
            return None

    def read_one(self, game_id: str) -> dict[str, Any] | None:
        return self._read_path(self._path(game_id))

    def read_all(self, limit: int) -> list[dict[str, Any]]:
        records = []
        def modified(path: Path) -> float:
            try:
                return path.lstat().st_mtime
            except OSError:
                return -1.0
        for path in sorted(self.directory.glob('*.json'), key=modified,
                           reverse=True)[:max(limit * 3, 20)]:
            data = self._read_path(path)
            if data is not None:
                records.append(data)
                if len(records) == limit:
                    break
        return records

    @staticmethod
    def restore_env(record: dict[str, Any]) -> DDZEnv:
        env = DDZEnv(record['seed'])
        for encoded in record['actions']:
            env.step_action(Action.from_dict(encoded))
        if env.engine.state_digest() != record['state_digest'] or env.done != record['completed']:
            raise ValueError('journal state digest or completion mismatch')
        return env

    @staticmethod
    def restore_rng(rng: Any, state: Any) -> None:
        rng.setstate(_nested_tuples(state))
