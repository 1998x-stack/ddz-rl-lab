"""Export verified Arena decisions with complete legal alternatives.

The resulting JSONL is a local, private research artifact. It has no hidden
opponent cards, no deal seed, and no invented behavior propensity or expert label.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re

from ddz_core.actions import Action
from ddz_core.engine import GameEngine, replay
from ddz_core.state import Team
from ddz_env.encoding import encode_observation_v2, encoder_ids
from ddz_core.roles import role_for_observation

from .analysis import QualityError

MAX_REPLAY_BYTES = 4_000_000
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def _rule_hash(rules: dict) -> str:
    return hashlib.sha256(json.dumps(rules, sort_keys=True, separators=(',', ':'))
                          .encode('utf-8')).hexdigest()


def decisions_from_replay(record: dict, *, game_id: str,
                          allow_incomplete: bool = False) -> list[dict]:
    """Verified incomplete traces may be exported, but get no outcome label."""
    try:
        verified = replay(record)
        complete = record.get('terminated') is True and record.get('truncated') is False
        if complete:
            if verified.state.winner is None or record.get('winner') != verified.state.winner.value:
                raise ValueError('winner mismatch')
        elif (not allow_incomplete or record.get('terminated') is not False or
              record.get('truncated') is not False or record.get('winner') is not None or
              verified.state.winner is not None or verified.state.truncated):
            raise ValueError('replay must be completed or an explicitly allowed incomplete trace')
        ids = record.get('model_ids')
        if not isinstance(ids, list) or len(ids) != 3 or not all(isinstance(x, str) for x in ids):
            raise ValueError('missing three source model IDs')
        hashes = record.get('model_sha256', [None] * 3)
        if (not isinstance(hashes, list) or len(hashes) != 3 or
                not all(value is None or value in ('@random', '@reasonable', '@human') or
                        isinstance(value, str) and _SHA256.fullmatch(value)
                        for value in hashes)):
            raise ValueError('invalid source model hashes')
        engine = GameEngine(record['seed'], verified.rules)
        final_deal = verified.state.deal_index
        rows = []
        rules_hash = _rule_hash(asdict(verified.rules))
        versions = encoder_ids('ddz-candidate-mlp-v2')
        random_bidding = record.get('random_bidding', False)
        if type(random_bidding) is not bool:
            raise ValueError('invalid random_bidding marker')
        for encoded in record['actions']:
            action = Action.from_dict(encoded)
            seat = engine.state.current_seat
            if seat is None:
                raise ValueError('action after terminal state')
            obs = engine.observe(seat, event_limit=16)
            try:
                selected = obs.legal_actions.index(action)
            except ValueError as exc:
                raise ValueError('chosen action is not in legal candidates') from exc
            team = Team.LANDLORD if seat == verified.state.landlord else Team.FARMERS
            outcome = (None if not complete or obs.deal_index != final_deal else
                       1 if team == verified.state.winner else -1)
            rows.append({
                'schema': 'ddz-decision-v1', 'game_id': game_id,
                'deal_index': obs.deal_index, 'step_index': obs.step_index,
                'seat': seat, 'role': role_for_observation(obs),
                'phase': obs.phase.value,
                'source_model_id': ('@random-bid' if random_bidding and
                                    obs.phase.value in ('CALL', 'ROB') else ids[seat]),
                'rule_sha256': rules_hash, 'encoders': versions,
                'model_sha256': hashes[seat],
                'observation': list(encode_observation_v2(obs)),
                'legal_actions': [candidate.to_dict() for candidate in obs.legal_actions],
                'chosen_index': selected, 'behavior_probability': None,
                'team_outcome': outcome,
                'label_provenance': ('terminal_outcome_only' if outcome is not None else
                                     'redeal' if complete and obs.deal_index != final_deal else
                                     'censored_incomplete'),
            })
            engine.step(action)
        if engine.state_digest() != verified.state_digest():
            raise ValueError('decision reconstruction digest mismatch')
        return rows
    except (KeyError, TypeError, ValueError, AssertionError, AttributeError, IndexError) as exc:
        raise QualityError(f'invalid decision replay {game_id}: {exc}') from exc


def export_decisions(replay_dir: Path, output: Path, *,
                     include_incomplete: bool = False) -> dict:
    replay_dir, output = Path(replay_dir), Path(output)
    if not replay_dir.is_dir() or replay_dir.is_symlink():
        raise QualityError('replay directory missing or unsafe')
    paths = sorted(replay_dir.glob('*.json'))
    if len(paths) > 10000:
        raise QualityError('too many replay files')
    if output.is_symlink() or output.exists():
        raise QualityError('decision output already exists or is a symlink')
    decisions = 0
    games = 0
    roles: Counter[str] = Counter()
    incomplete_games = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            for path in paths:
                if path.is_symlink() or path.stat().st_size > MAX_REPLAY_BYTES:
                    raise QualityError(f'unsafe replay file: {path.name}')
                try:
                    record = json.loads(path.read_text('utf-8'))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise QualityError(f'unreadable replay: {path.name}') from exc
                game_rows = decisions_from_replay(record, game_id=path.stem)
                for row in game_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'),
                                            allow_nan=False) + '\n')
                    roles[row['role']] += 1
                decisions += len(game_rows)
                games += 1
            if include_incomplete:
                from ddz_web.persistence import ArenaJournal
                journal = ArenaJournal(replay_dir)
                completed_ids = {path.stem for path in paths}
                for session in journal.read_all(10000):
                    if session['completed'] or session['game_id'] in completed_ids:
                        continue
                    env = ArenaJournal.restore_env(session)
                    record = env.export_replay()
                    record['model_ids'] = session['model_ids']
                    record['model_sha256'] = session['model_sha256']
                    record['random_bidding'] = session['random_bidding']
                    game_rows = decisions_from_replay(record, game_id=session['game_id'],
                                                      allow_incomplete=True)
                    for row in game_rows:
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'),
                                                allow_nan=False) + '\n')
                        roles[row['role']] += 1
                    decisions += len(game_rows)
                    incomplete_games += 1
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {'schema': 'ddz-decision-export-summary-v1', 'verified_completed_games': games,
            'verified_incomplete_games': incomplete_games,
            'decisions': decisions, 'roles': dict(roles), 'output': str(output),
            'note': 'local private dataset; no expert labels or behavior probabilities inferred'}
