"""CLI smoke-test and no-stakes baseline game simulation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from loguru import logger

from .actions import ActionKind
from .engine import GameEngine, replay
from .logging_config import configure_logging, default_log_directory
from .state import Phase, Team


def baseline_move(engine: GameEngine, rng: random.Random):
    actions = engine.legal_actions()
    if engine.state.phase == Phase.CALL:
        selected = ActionKind.CALL if rng.random() < 0.35 else ActionKind.DECLINE_CALL
        return next(a for a in actions if a.kind == selected)
    if engine.state.phase == Phase.ROB:
        selected = ActionKind.ROB if rng.random() < 0.25 else ActionKind.DECLINE_ROB
        return next(a for a in actions if a.kind == selected)
    # Favor lead-over-pass to avoid drawing out the smoke test unnecessarily.
    plays = [a for a in actions if a.kind == ActionKind.PLAY]
    if plays and (rng.random() < 0.8 or len(plays) == len(actions)):
        return rng.choice(plays)
    return next(a for a in actions if a.kind == ActionKind.PASS)


def simulate(seed: int, limit: int = 1200) -> GameEngine:
    engine = GameEngine(seed)
    policy_rng = random.Random(seed ^ 0xDA7A)
    for _ in range(limit):
        if engine.state.phase == Phase.FINISHED:
            return engine
        engine.step(baseline_move(engine, policy_rng))
    raise RuntimeError(f"simulation exceeded {limit} decisions for seed {seed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dou Dizhu no-stakes rule-engine smoke demo")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replay-out", type=Path, help="write the final game's replay JSON")
    parser.add_argument("--log-dir", type=Path, default=default_log_directory())
    parser.add_argument("--log-level", choices=["TRACE", "DEBUG", "INFO", "WARNING"], default="INFO")
    args = parser.parse_args()
    if args.games <= 0 or args.games > 10000:
        parser.error("--games must be between 1 and 10000")
    configure_logging(args.log_dir, args.log_level)
    total = {Team.LANDLORD.value: 0, Team.FARMERS.value: 0, "truncated": 0}
    for idx in range(args.games):
        engine = simulate(args.seed + idx)
        s = engine.state
        label = s.winner.value if s.winner else "truncated"
        total[label] += 1
        print(f"game={idx + 1}, seed={args.seed + idx}, result={label}, "
              f"landlord={s.landlord}, steps={s.step_index}, deals={s.deal_index + 1}, "
              f"hash={engine.state_digest()[:12]}")
        if idx == args.games - 1 and args.replay_out:
            args.replay_out.parent.mkdir(parents=True, exist_ok=True)
            record = engine.export_replay()
            args.replay_out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
            restored = replay(record)
            assert restored.state_digest() == engine.state_digest()
            logger.info("replay_written path={} steps={}", args.replay_out, s.step_index)
    print("summary:", total)


if __name__ == "__main__":
    main()
