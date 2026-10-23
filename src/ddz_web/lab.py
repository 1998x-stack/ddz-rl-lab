"""Read-only, bounded, operator-local research artefact projections.

The web layer only reads explicitly configured directories and known file types.
It never loads model weights, executes paths, or streams private training states.
Completed replay frames are regenerated from public GameEvents and verified.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ddz_core.actions import Action
from ddz_core.engine import GameEngine
from ddz_core.rules import RuleConfig
from ddz_core.state import Phase

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_MAX_FILES = 128
_MAX_JSON_BYTES = 2_000_000
_MAX_LINES = 500
_MAX_TAIL_BYTES = 512_000
_MAX_REPLAY_ACTIONS = 1200


class LabError(ValueError):
    """Malformed identifier or configured local artefact."""


def _identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value) or value in ('.', '..'):
        raise LabError('invalid resource identifier')
    return value


def _safe_file(directory: Path, filename: str) -> Path:
    root = directory.resolve()
    p = directory / filename
    if p.is_symlink() or not p.is_file() or p.resolve() != root / filename:
        raise FileNotFoundError(filename)
    return p


def _list_names(directory: Path, suffix: str, *, folders: bool = False) -> list[str]:
    if not directory.is_dir():
        return []
    entries: list[tuple[float, str]] = []
    for path in directory.iterdir():
        if path.is_symlink() or not _IDENTIFIER.fullmatch(path.stem if not folders else path.name):
            continue
        if folders and path.is_dir() or not folders and path.is_file() and path.suffix == suffix:
            try:
                entries.append((path.stat().st_mtime, path.name if folders else path.stem))
            except OSError:
                continue
    return [name for _, name in sorted(entries, reverse=True)[:_MAX_FILES]]


def _json_file(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise LabError('artefact exceeds size limit')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise LabError('expected object')
    return value


def _tail_jsonl(path: Path, *, limit: int) -> dict[str, Any]:
    """Read bounded trailing bytes; discard partial first line after seek."""
    size = path.stat().st_size
    with path.open('rb') as f:
        offset = max(0, size - _MAX_TAIL_BYTES)
        f.seek(offset)
        content = f.read(_MAX_TAIL_BYTES)
    lines = content.splitlines()
    if offset and lines:
        lines = lines[1:]
    records = []
    invalid = 0
    for line in lines[-min(limit, _MAX_LINES):]:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
            else:
                invalid += 1
        except (UnicodeDecodeError, ValueError):
            invalid += 1
    return {'records': records, 'truncated': offset > 0 or len(lines) > limit,
            'invalid_lines': invalid}


def _frame(engine: GameEngine) -> dict[str, Any]:
    state = engine.state
    return {'step': state.step_index, 'phase': state.phase.value,
            'remaining': [len(h) for h in state.hands], 'landlord': state.landlord,
            'acting_seat': state.current_seat, 'last_play_seat': state.last_play_seat,
            'last_play_counts': list(state.last_play_counts),
            'winner': state.winner.value if state.winner else None,
            'events': [e.to_dict() for e in state.events[-3:]]}


class LabStore:
    def __init__(self, runs_dir: Path, reports_dir: Path, replay_dir: Path, log_dir: Path):
        self.runs_dir = Path(runs_dir).resolve()
        self.reports_dir = Path(reports_dir).resolve()
        self.replay_dir = Path(replay_dir).resolve()
        self.log_dir = Path(log_dir).resolve()

    def runs(self) -> list[dict[str, Any]]:
        results = []
        for name in _list_names(self.runs_dir, '', folders=True):
            folder = self.runs_dir / name
            if not folder.resolve().is_relative_to(self.runs_dir):
                continue
            latest = {}
            try:
                metrics = _safe_file(folder, 'evaluations.jsonl')
                records = _tail_jsonl(metrics, limit=1)['records']
                latest = records[-1] if records else {}
            except (FileNotFoundError, OSError, ValueError):
                pass
            results.append({'run_id': name, 'episodes_seen': latest.get('episodes_seen'),
                            'balanced_score': latest.get('balanced_score'),
                            'best_score': latest.get('best_score'),
                            'has_evaluation': bool(latest),
                            'has_throughput': (folder / 'throughput.jsonl').is_file()})
        return results

    def metrics(self, run_id: str, *, limit: int = 300) -> dict[str, Any]:
        name = _identifier(run_id)
        folder = self.runs_dir / name
        if folder.is_symlink() or not folder.is_dir() or not folder.resolve().is_relative_to(self.runs_dir):
            raise FileNotFoundError(name)
        output: dict[str, Any] = {'run_id': name}
        for metric_name, filename in [('throughput', 'throughput.jsonl'), ('evaluations', 'evaluations.jsonl')]:
            try:
                output[metric_name] = _tail_jsonl(_safe_file(folder, filename), limit=limit)
            except FileNotFoundError:
                output[metric_name] = {'records': [], 'truncated': False, 'invalid_lines': 0}
        return output

    def reports(self) -> list[dict[str, Any]]:
        result = []
        for name in _list_names(self.reports_dir, '.json'):
            try:
                report = self.report(name)
                if report.get('schema') != 'ddz-tournament-report-v1':
                    continue
                result.append({'report_id': name, 'incumbent_id': report.get('incumbent_id'),
                               'recommended_id': report.get('recommended_id'),
                               'candidate_count': len(report.get('candidates', {})),
                               'deals_per_seed': report.get('deals_per_seed')})
            except (LabError, FileNotFoundError, ValueError, OSError):
                continue
        return result

    def report(self, report_id: str) -> dict[str, Any]:
        report = _json_file(_safe_file(self.reports_dir, _identifier(report_id) + '.json'))
        if report.get('schema') != 'ddz-tournament-report-v1':
            raise LabError('unsupported tournament report')
        return report

    def replays(self) -> list[dict[str, Any]]:
        result = []
        for name in _list_names(self.replay_dir, '.json'):
            try:
                record = _json_file(_safe_file(self.replay_dir, name + '.json'))
                if record.get('schema_version') == 'replay.v1' and record.get('terminated') is True:
                    result.append({'replay_id': name, 'winner': record.get('winner'),
                                   'steps': len(record.get('actions', [])),
                                   'model_ids': record.get('model_ids', [])})
            except (LabError, FileNotFoundError, ValueError, OSError):
                continue
        return result

    def replay(self, replay_id: str) -> dict[str, Any]:
        record = _json_file(_safe_file(self.replay_dir, _identifier(replay_id) + '.json'))
        actions = record.get('actions')
        if (record.get('schema_version') != 'replay.v1' or
                record.get('terminated') is not True or
                not isinstance(actions, list) or len(actions) > _MAX_REPLAY_ACTIONS):
            raise LabError('replay must be completed and bounded')
        engine = GameEngine(int(record['seed']), RuleConfig(**record['rules']))
        frames = [_frame(engine)]
        for item in actions:
            engine.step(Action.from_dict(item))
            frames.append(_frame(engine))
        if (engine.state.phase != Phase.FINISHED or engine.state.truncated or
                engine.state_digest() != record.get('state_digest')):
            raise LabError('replay integrity verification failed')
        return {'replay_id': replay_id, 'verified': True,
                'model_ids': record.get('model_ids', []), 'frames': frames,
                'winner': engine.state.winner.value if engine.state.winner else None}

    def logs(self, *, limit: int = 120) -> dict[str, Any]:
        files = _list_names(self.log_dir, '.jsonl')
        if not files:
            return {'file': None, 'lines': [], 'truncated': False}
        filename = files[0] + '.jsonl'
        path = _safe_file(self.log_dir, filename)
        size = path.stat().st_size
        with path.open('rb') as f:
            offset = max(0, size - _MAX_TAIL_BYTES)
            f.seek(offset)
            content = f.read(_MAX_TAIL_BYTES)
        lines = content.splitlines()
        if offset and lines:
            lines = lines[1:]
        # Only output compact Loguru messages, not exception locals or full traces.
        records = []
        for line in lines[-min(limit, 200):]:
            try:
                row = json.loads(line)
                info = row.get('record', {})
                records.append({'time': info.get('time', {}).get('repr'),
                                'level': info.get('level', {}).get('name'),
                                'message': str(info.get('message', ''))[:350]})
            except (ValueError, AttributeError, UnicodeDecodeError):
                continue
        return {'file': filename, 'lines': records, 'truncated': offset > 0 or len(lines) > limit}
