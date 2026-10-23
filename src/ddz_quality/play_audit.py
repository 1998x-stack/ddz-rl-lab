"""Verify Arena replays and flag concrete play decisions for review.

The report is diagnostic. Only passing up an available immediate finish is
treated as a hard error; breaking groups can be the correct tactical move.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from ddz_core.actions import Action, ActionKind
from ddz_core.cards import RANKS
from ddz_core.engine import GameEngine, replay
from ddz_core.state import Phase

from .analysis import QualityError

MAX_REPLAY_BYTES = 4_000_000


def _cards(counts: tuple[int, ...]) -> str:
    return ' '.join(f'{RANKS[i]}×{n}' if n > 1 else RANKS[i]
                    for i, n in enumerate(counts) if n) or '空'


def audit_replay(record: dict, *, game_id: str = 'unknown', example_limit: int = 20) -> dict:
    """Recreate every decision, checking the final digest before using labels."""
    try:
        verified = replay(record)
        if (not record.get('terminated') or record.get('truncated') or
                verified.state.winner is None or record.get('winner') != verified.state.winner.value):
            raise ValueError('not a verified completed game')
        engine = GameEngine(record['seed'], verified.rules)
        counts: Counter[str] = Counter()
        examples: list[dict] = []
        play_decisions = 0
        for step, encoded in enumerate(record['actions']):
            action = Action.from_dict(encoded)
            state = engine.state
            if state.phase == Phase.PLAY:
                play_decisions += 1
                assert state.current_seat is not None
                seat = state.current_seat
                obs = engine.observe(seat, event_limit=16)
                hand = obs.own_counts
                finish = [candidate for candidate in obs.legal_actions
                          if candidate.kind == ActionKind.PLAY and candidate.counts == hand]
                flags = []
                if finish and action not in finish:
                    flags.append('missed_immediate_finish')
                if (action.kind == ActionKind.PASS and obs.landlord is not None and
                        seat != obs.landlord and obs.last_play_seat is not None and
                        obs.last_play_seat != obs.landlord and
                        obs.remaining_counts[obs.landlord] <= 2 and
                        any(candidate.kind == ActionKind.PLAY for candidate in obs.legal_actions)):
                    flags.append('teammate_pass_enemy_critical_review')
                if action.kind == ActionKind.PLAY and action.counts != hand:
                    if any(have == 4 and 0 < used < 4
                           for have, used in zip(hand, action.counts)):
                        flags.append('broken_bomb_review')
                    if (action.pattern_key and action.pattern_key[0] == 'SINGLE' and
                            any(have >= 2 and used == 1
                                for have, used in zip(hand, action.counts))):
                        if any(candidate.kind == ActionKind.PLAY and
                               candidate.pattern_key and
                               candidate.pattern_key[0] == 'SINGLE' and
                               any(have == 1 and used == 1 for have, used in
                                   zip(hand, candidate.counts))
                               for candidate in obs.legal_actions):
                            flags.append('avoidable_single_split_review')
                        else:
                            flags.append('forced_single_split_review')
                for flag in flags:
                    counts[flag] += 1
                if flags and len(examples) < example_limit:
                    examples.append({
                        'step': step, 'seat': seat, 'landlord': obs.landlord,
                        'own_hand': _cards(hand), 'chosen': _cards(action.counts),
                        'chosen_pattern': list(action.pattern_key) if action.pattern_key else action.kind.value,
                        'available_finish_patterns': [list(candidate.pattern_key)
                                                      for candidate in finish],
                        'last_play_seat': obs.last_play_seat,
                        'remaining_counts': list(obs.remaining_counts),
                        'flags': flags,
                    })
            engine.step(action)
        if engine.state_digest() != verified.state_digest():
            raise ValueError('decision reconstruction digest mismatch')
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise QualityError(f'invalid replay {game_id}: {exc}') from exc
    return {'game_id': game_id, 'steps': len(record['actions']),
            'play_decisions': play_decisions, 'winner': verified.state.winner.value,
            'flags': dict(counts), 'examples': examples}


def audit_replay_dir(directory: Path, *, example_limit: int = 20) -> dict:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise QualityError('replay directory missing or unsafe')
    paths = sorted(directory.glob('*.json'))
    if len(paths) > 10000:
        raise QualityError('too many replay files')
    games = []
    for path in paths:
        if path.is_symlink() or path.stat().st_size > MAX_REPLAY_BYTES:
            raise QualityError(f'unsafe replay file: {path.name}')
        try:
            record = json.loads(path.read_text('utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QualityError(f'unreadable replay: {path.name}') from exc
        games.append(audit_replay(record, game_id=path.stem,
                                  example_limit=example_limit))
    totals: Counter[str] = Counter()
    for game in games:
        totals.update(game['flags'])
    return {'schema': 'ddz-arena-play-audit-v1', 'verified_completed_games': len(games),
            'play_decisions': sum(game['play_decisions'] for game in games),
            'flags': dict(totals), 'games': games,
            'interpretation': 'missed_immediate_finish is a hard error; split flags require tactical review'}
