"""Controlled tournament against frozen, trusted checkpoints.

One candidate per fixed initial deal is evaluated as landlord and as farmer
team, with identical random bidding seeds. Results are NOT independent
estimates of perfect-play strength; Wilson intervals are descriptive.
"""
from __future__ import annotations
import math
from pathlib import Path
import random
from typing import Iterable
from ddz_core.state import Phase, Team
from ddz_env.environment import DDZEnv
from ddz_env.baselines import random_action_index
from ddz_models.policy import RoleBundle
from ddz_models.network import NetworkSpec
from ddz_training.checkpoints import load_checkpoint


def load_frozen(path: str | Path, device: str = "cpu") -> RoleBundle:
    import json
    p = Path(path)
    metadata = json.loads(p.with_suffix(p.suffix + ".json").read_text("utf-8"))
    bundle = RoleBundle(NetworkSpec(hidden=int(metadata["spec"]["hidden"])), device)
    load_checkpoint(p, bundle)
    return bundle


def wilson(wins: int, games: int, z: float = 1.95996398454005) -> tuple[float, float]:
    if games <= 0:
        return (0., 1.)
    p = wins / games
    denom = 1 + z*z/games
    center = (p + z*z/(2*games)) / denom
    rad = z * math.sqrt(p*(1-p)/games + z*z/(4*games*games)) / denom
    return (max(0., center-rad), min(1., center+rad))


def compare_bundles(candidate: RoleBundle, opponent: RoleBundle | None,
                    *, games: int, seed: int, max_steps: int = 1200) -> dict:
    if games < 1 or max_steps < 1:
        raise ValueError("games/max_steps must be positive")
    counts = {"landlord": {"wins": 0, "games": 0},
              "farmers": {"wins": 0, "games": 0}, "truncated": 0}
    rng_model = random.Random(seed ^ 0xAFFE)
    for offset in range(games):
        for side in ("landlord", "farmers"):
            env = DDZEnv(seed + offset)
            bidding_rng = random.Random((seed+offset)*17+937)
            for _ in range(max_steps):
                if env.done:
                    break
                obs = env.current_observation()
                if obs.phase != Phase.PLAY:
                    action = random_action_index(env, bidding_rng)
                else:
                    selected = ((obs.seat == obs.landlord) == (side == "landlord"))
                    if selected:
                        action = candidate.act(obs, rng_model, 0.).local_index
                    elif opponent is None:
                        action = random_action_index(env, bidding_rng)
                    else:
                        action = opponent.act(obs, rng_model, 0.).local_index
                env.step(action)
            if not env.done:
                raise RuntimeError(f"tournament step limit exceeded deal={seed+offset}")
            if env.engine.state.truncated:
                counts["truncated"] += 1
                continue
            result = counts[side]
            result["games"] += 1
            won = env.engine.state.winner == (Team.LANDLORD if side == "landlord"
                                               else Team.FARMERS)
            result["wins"] += int(won)
    values: dict = {"seed": seed, "deals": games, "truncated": counts["truncated"]}
    for side in ("landlord", "farmers"):
        item = counts[side]
        wins, n = item["wins"], item["games"]
        values[side] = {**item, "win_rate": wins/n if n else None,
                        "wilson95": list(wilson(wins, n))}
    both = [values[side]["win_rate"] for side in ("landlord", "farmers")]
    values["balanced_score"] = sum(both)/2 if all(x is not None for x in both) else None
    return values


def run_tournament(candidate: RoleBundle, frozen: Iterable[tuple[str, RoleBundle]],
                   *, games: int, seed: int, max_steps: int = 1200) -> dict:
    # Each opponent uses same deals and both roles; cannot inflate sample size
    # by aggregating paired outcomes as if they were independent.
    opponents = [("random", None), *frozen]
    names = [n for n, _ in opponents]
    if len(set(names)) != len(names):
        raise ValueError("tournament opponent identifiers must be unique")
    results = {name: compare_bundles(candidate, opponent, games=games, seed=seed,
                                     max_steps=max_steps)
               for name, opponent in opponents}
    scores = [r["balanced_score"] for r in results.values()
              if r["balanced_score"] is not None]
    return {"opponents": results,
            "mean_balanced_score": sum(scores)/len(scores) if scores else None,
            "note": "Wilson intervals per side are descriptive; paired deals require clustered inference."}
