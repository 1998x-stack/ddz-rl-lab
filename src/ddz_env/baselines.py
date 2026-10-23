"""Reproducible no-learning baselines for environment smoke testing."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ddz_core.actions import Action, ActionKind
from ddz_core.state import Phase, PlayerObservation
from ddz_core.logging_config import configure_logging, default_log_directory
from .environment import DDZEnv


def random_action_index(env: DDZEnv, rng: random.Random) -> int:
    """Legacy seeded baseline retained for comparable historical evaluations."""
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


def reasonable_action_index(obs: PlayerObservation, rng: random.Random) -> int:
    """Seeded stochastic play baseline using only the acting player's observation.

    Every candidate comes from the authoritative legal list. The score is a
    deliberately simple hand-shape preference, not a claim of optimal play.
    """
    actions = obs.legal_actions
    if not actions:
        raise ValueError("no legal actions")
    if obs.phase == Phase.CALL:
        wanted = ActionKind.CALL if rng.random() < .35 else ActionKind.DECLINE_CALL
        return next(i for i, action in enumerate(actions) if action.kind == wanted)
    if obs.phase == Phase.ROB:
        wanted = ActionKind.ROB if rng.random() < .25 else ActionKind.DECLINE_ROB
        return next(i for i, action in enumerate(actions) if action.kind == wanted)
    if obs.phase != Phase.PLAY:
        raise ValueError("reasonable baseline expects an active phase")

    finish = [i for i, action in enumerate(actions)
              if action.kind == ActionKind.PLAY and action.counts == obs.own_counts]
    if finish:
        return rng.choice(finish)

    teammate_leads = (obs.landlord is not None and obs.seat != obs.landlord and
                      obs.last_play_seat is not None and
                      obs.last_play_seat != obs.landlord)
    passes = [i for i, action in enumerate(actions) if action.kind == ActionKind.PASS]
    if teammate_leads and passes:
        return passes[0]

    scores = [reasonable_action_score(obs, action) + rng.uniform(-.45, .45)
              for action in actions]
    return max(range(len(actions)), key=scores.__getitem__)


def reasonable_action_score(obs: PlayerObservation, action: Action) -> float:
    """Public-information hand-shape prior; higher means a cleaner play."""
    opponent_counts = [obs.remaining_counts[seat] for seat in range(3)
                       if seat != obs.seat and
                       (obs.seat == obs.landlord or seat == obs.landlord)]
    urgent = bool(opponent_counts and min(opponent_counts) <= 2)
    if action.kind == ActionKind.PASS:
        return -8.0 if urgent else 0.0
    if action.kind != ActionKind.PLAY:
        raise ValueError("expected a play or pass action")
    after = [have - used for have, used in zip(obs.own_counts, action.counts)]
    split = sum(1 for have, used, left in zip(obs.own_counts, action.counts, after)
                if have >= 2 and used and left == 1)
    singles_left = sum(left == 1 for left in after)
    kind = action.pattern_key[0] if action.pattern_key else ''
    bomb = kind in ('BOMB', 'ROCKET')
    rank = action.pattern_key[1] if action.pattern_key else 0
    return (sum(action.counts) * .85 - split * 1.8 - singles_left * .45
            - rank * .06 + (2.0 if urgent else 0.0)
            - (2.0 if urgent else 6.0) * bomb)


def play_episode(seed: int, decision_limit: int = 1200) -> tuple[DDZEnv, int]:
    env = DDZEnv(seed)
    rng = random.Random(seed ^ 0xDA7A)
    for step in range(decision_limit):
        if env.done:
            return env, step
        env.step(random_action_index(env, rng), include_digest=False)
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
