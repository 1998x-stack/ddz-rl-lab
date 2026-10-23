"""Reproducible no-learning baselines for environment smoke testing."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ddz_core.actions import ActionKind
from ddz_core.state import Phase
from ddz_core.logging_config import configure_logging, default_log_directory
from .environment import DDZEnv


def random_action_index(env: DDZEnv, rng: random.Random) -> int:
    actions = env.legal_actions()
    if env.engine.state.phase == Phase.CALL:
        preferred = ActionKind.CALL if rng.random() < 0.35 else ActionKind.DECLINE_CALL
        return next(i for i, a in enumerate(actions) if a.kind == preferred)
    if env.engine.state.phase == Phase.ROB:
        preferred = ActionKind.ROB if rng.random() < 0.25 else ActionKind.DECLINE_ROB
        return next(i for i, a in enumerate(actions) if a.kind == preferred)
    plays = [i for i, a in enumerate(actions) if a.kind == ActionKind.PLAY]
    if plays and (rng.random() < 0.8 or len(plays) == len(actions)):
        return rng.choice(plays)
    return next(i for i, a in enumerate(actions) if a.kind == ActionKind.PASS)


def play_episode(seed: int, decision_limit: int = 1200) -> tuple[DDZEnv, int]:
    env = DDZEnv(seed)
    rng = random.Random(seed ^ 0xDA7A)
    for step in range(decision_limit):
        if env.done:
            return env, step
        env.step(random_action_index(env, rng))
    raise RuntimeError("episode exceeded decision_limit")


def main() -> None:
    parser = argparse.ArgumentParser(description="No-stakes three-agent RL ENV baseline")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replay-out", type=Path)
    parser.add_argument("--log-dir", type=Path, default=default_log_directory())
    args = parser.parse_args()
    if not 1 <= args.episodes <= 10000:
        parser.error("episodes must be 1..10000")
    configure_logging(args.log_dir, "WARNING")
    wins = {"LANDLORD": 0, "FARMERS": 0, "TRUNCATED": 0}
    for index in range(args.episodes):
        env, decisions = play_episode(args.seed + index)
        winner = env.engine.state.winner
        label = winner.value if winner else "TRUNCATED"
        wins[label] += 1
        print(f"episode={index + 1} seed={args.seed + index} outcome={label} "
              f"actions={decisions} rewards={env.rewards()} digest={env.engine.state_digest()[:12]}")
        if args.replay_out and index == args.episodes - 1:
            args.replay_out.parent.mkdir(parents=True, exist_ok=True)
            args.replay_out.write_text(json.dumps(env.export_replay(), indent=2) + "\n", encoding="utf-8")
    print("win_counts:", wins)


if __name__ == "__main__":
    main()
