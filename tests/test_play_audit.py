"""Replay audit checks actual legal alternatives, not a reconstructed guess."""
import random

import pytest

from ddz_core.actions import ActionKind
from ddz_core.actions import Action
from ddz_core.engine import GameEngine
from ddz_core.rules import RuleConfig
from ddz_core.state import Phase
from ddz_env.baselines import reasonable_action_index
from ddz_quality.analysis import QualityError
from ddz_quality.play_audit import audit_replay
from ddz_quality.decision_data import decisions_from_replay, export_decisions


def _completed_replay_with_missed_finish():
    rng = random.Random(41)
    for seed in range(10, 30):
        engine = GameEngine(seed)
        for _ in range(300):
            if engine.state.phase == Phase.FINISHED:
                break
            seat = engine.state.current_seat
            obs = engine.observe(seat, event_limit=16)
            if obs.phase == Phase.PLAY:
                finish = [a for a in obs.legal_actions
                          if a.kind == ActionKind.PLAY and a.counts == obs.own_counts]
                bad = [a for a in obs.legal_actions if a not in finish]
                if finish and bad:
                    engine.step(bad[0])
                    for _ in range(300):
                        if engine.state.phase == Phase.FINISHED:
                            return engine.export_replay()
                        obs = engine.observe(engine.state.current_seat, event_limit=16)
                        engine.step(obs.legal_actions[reasonable_action_index(obs, rng)])
                    break
            engine.step(obs.legal_actions[reasonable_action_index(obs, rng)])
    raise AssertionError('no suitable completed fixture')


def test_play_audit_find_missed_finish_and_verify_integrity():
    record = _completed_replay_with_missed_finish()
    result = audit_replay(record, game_id='fixture')
    assert result['flags']['missed_immediate_finish'] >= 1
    assert any(example['available_finish_patterns'] for example in result['examples'])
    record['state_digest'] = '0' * 64
    with pytest.raises(QualityError, match='invalid replay'):
        audit_replay(record)


def test_verified_decision_export_has_legal_set_and_separate_redeal(tmp_path):
    rng = random.Random(9)
    engine = GameEngine(901)
    for _ in range(3):
        engine.step(Action.simple(ActionKind.DECLINE_CALL))
    assert engine.state.deal_index == 1
    for _ in range(300):
        if engine.state.phase == Phase.FINISHED:
            break
        obs = engine.observe(engine.state.current_seat, event_limit=16)
        engine.step(obs.legal_actions[reasonable_action_index(obs, rng)])
    assert engine.state.winner is not None
    record = engine.export_replay()
    record['model_ids'] = ['a', 'b', 'c']
    rows = decisions_from_replay(record, game_id='fixture')
    assert len(rows) == len(record['actions'])
    assert all(row['legal_actions'][row['chosen_index']] == record['actions'][i]
               for i, row in enumerate(rows))
    assert all(row['team_outcome'] is None and row['label_provenance'] == 'redeal'
               for row in rows[:3])
    assert {row['team_outcome'] for row in rows[3:]} == {-1, 1}
    assert all(row['behavior_probability'] is None for row in rows)
    replay_dir = tmp_path / 'replays'
    replay_dir.mkdir()
    import json
    (replay_dir / 'fixture.json').write_text(json.dumps(record), encoding='utf-8')
    output = tmp_path / 'private.jsonl'
    summary = export_decisions(replay_dir, output)
    assert summary['decisions'] == len(rows)
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(QualityError, match='already exists'):
        export_decisions(replay_dir, output)


def test_rule_config_rejects_mistyped_variants():
    for value in [dict(max_redeals=True), dict(max_redeals=1.2),
                  dict(repeated_four_two_singles=1), dict(rule_id=12)]:
        with pytest.raises(ValueError, match='invalid rule configuration'):
            RuleConfig(**value)


def test_incomplete_decision_trace_cannot_gain_terminal_label():
    engine = GameEngine(1001)
    engine.step(Action.simple(ActionKind.CALL))
    record = engine.export_replay()
    record['model_ids'] = ['a', 'b', 'c']
    with pytest.raises(QualityError, match='completed'):
        decisions_from_replay(record, game_id='partial')
    rows = decisions_from_replay(record, game_id='partial', allow_incomplete=True)
    assert len(rows) == 1
    assert rows[0]['team_outcome'] is None
    assert rows[0]['label_provenance'] == 'censored_incomplete'
