"""Evaluate against frozen seeded legal random opponents (not self-play win rate)."""
from __future__ import annotations
import random
from loguru import logger
from ddz_core.state import Phase, Team
from ddz_env.environment import DDZEnv
from ddz_env.baselines import random_action_index
from ddz_models.policy import RoleBundle


def evaluate_bundle(bundle: RoleBundle, seed: int = 34000, games: int = 20,
                    max_steps: int = 1200) -> dict[str, float | int]:
    if games < 1 or max_steps < 1:
        raise ValueError("games/max_steps must be positive")
    scores = {"landlord_wins": 0, "farmer_wins": 0, "landlord_games": 0,
              "farmer_games": 0, "truncated": 0}
    # Each deal is evaluated in both conditions: learned LANDLORD vs random farmers,
    # and learned FARMERS vs random landlord. Bidding is always random and frozen.
    for j in range(games):
        for side in ("landlord", "farmers"):
            env = DDZEnv(seed + j)
            rng = random.Random((seed + j) * 17 + 937)
            for _ in range(max_steps):
                if env.done:
                    break
                obs = env.current_observation()
                if obs.phase != Phase.PLAY:
                    action_id = random_action_index(env, rng)
                elif (obs.seat == obs.landlord) == (side == "landlord"):
                    action_id = bundle.act(obs, rng, 0.0).local_index
                else:
                    action_id = random_action_index(env, rng)
                env.step(action_id)
            if not env.done:
                raise RuntimeError("evaluation decision limit exceeded")
            if env.engine.state.truncated:
                scores["truncated"] += 1
                continue
            if side == "landlord":
                scores["landlord_games"] += 1
                scores["landlord_wins"] += int(env.engine.state.winner == Team.LANDLORD)
            else:
                scores["farmer_games"] += 1
                scores["farmer_wins"] += int(env.engine.state.winner == Team.FARMERS)
    landlord_rate = scores["landlord_wins"] / scores["landlord_games"] if scores["landlord_games"] else 0.0
    farmer_rate = scores["farmer_wins"] / scores["farmer_games"] if scores["farmer_games"] else 0.0
    result: dict[str, float | int] = dict(scores)
    result.update(landlord_rate=landlord_rate, farmer_rate=farmer_rate,
                  balanced_score=(landlord_rate + farmer_rate) / 2)
    logger.bind(component="evaluation").info("external baseline evaluation {}", result)
    return result
