"""Evaluate against frozen seeded legal random opponents (not self-play win rate)."""
from __future__ import annotations
import random
from loguru import logger
from ddz_core.actions import ActionKind
from ddz_core.state import Phase, Team
from ddz_env.environment import DDZEnv
from ddz_env.baselines import random_action_index, reasonable_action_index
from ddz_models.policy import RoleBundle


def evaluate_bundle(bundle: RoleBundle, seed: int = 34000, games: int = 20,
                    max_steps: int = 1200, play_opponent: str = 'random') -> dict:
    if games < 1 or max_steps < 1:
        raise ValueError("games/max_steps must be positive")
    if play_opponent not in {'random', 'reasonable'}:
        raise ValueError('play_opponent must be random or reasonable')
    scores = {"landlord_wins": 0, "farmer_wins": 0, "landlord_games": 0,
              "farmer_games": 0, "truncated": 0}
    quality = {"play_decisions": 0, "played_actions": 0,
               "finish_opportunities": 0,
               "missed_immediate_wins": 0, "nonterminal_singles": 0,
               "passes_with_legal_beat": 0,
               "passes_to_enemy_with_beat": 0,
               "passes_on_teammate_lead": 0,
               "critical_enemy_passes": 0,
               "bombs_with_normal_response": 0}
    # Each deal is evaluated in both conditions: learned LANDLORD vs random farmers,
    # and learned FARMERS vs random landlord. Bidding is always random and frozen.
    for j in range(games):
        for side in ("landlord", "farmers"):
            env = DDZEnv(seed + j)
            rng = random.Random((seed + j) * 17 + 937)
            policy_rng = random.Random((seed + j) * 31 + 991)
            for _ in range(max_steps):
                if env.done:
                    break
                obs = env.current_observation(event_limit=16)
                if obs.phase != Phase.PLAY:
                    action_id = random_action_index(env, rng)
                elif (obs.seat == obs.landlord) == (side == "landlord"):
                    action_id = bundle.act(obs, policy_rng, 0.0).local_index
                    chosen = obs.legal_actions[action_id]
                    plays = [a for a in obs.legal_actions if a.kind == ActionKind.PLAY]
                    quality["play_decisions"] += 1
                    if any(a.counts == obs.own_counts for a in plays):
                        quality["finish_opportunities"] += 1
                        quality["missed_immediate_wins"] += int(chosen.counts != obs.own_counts)
                    if chosen.kind == ActionKind.PLAY:
                        quality["played_actions"] += 1
                        quality["nonterminal_singles"] += int(
                            sum(chosen.counts) == 1 and sum(obs.own_counts) > 1)
                        if chosen.pattern_key and chosen.pattern_key[0] in ("BOMB", "ROCKET"):
                            quality["bombs_with_normal_response"] += int(any(
                                a.pattern_key and a.pattern_key[0] not in ("BOMB", "ROCKET")
                                for a in plays))
                    elif chosen.kind == ActionKind.PASS:
                        quality["passes_with_legal_beat"] += int(bool(plays))
                        opponents = [seat for seat in range(3) if seat != obs.seat and
                                     (obs.seat == obs.landlord or seat == obs.landlord)]
                        enemy_leads = (obs.last_play_seat is not None and
                                       (obs.seat == obs.landlord or
                                        obs.last_play_seat == obs.landlord))
                        quality["passes_to_enemy_with_beat"] += int(bool(plays) and enemy_leads)
                        quality["passes_on_teammate_lead"] += int(bool(plays) and not enemy_leads)
                        quality["critical_enemy_passes"] += int(
                            bool(plays) and enemy_leads and
                            any(obs.remaining_counts[seat] <= 2 for seat in opponents))
                else:
                    action_id = (reasonable_action_index(obs, rng)
                                 if play_opponent == 'reasonable' else
                                 random_action_index(env, rng))
                env.step(action_id, include_digest=False)
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
    result: dict = dict(scores)
    result.update(landlord_rate=landlord_rate, farmer_rate=farmer_rate,
                  balanced_score=(landlord_rate + farmer_rate) / 2,
                  role_min_rate=min(landlord_rate, farmer_rate),
                  play_opponent=play_opponent)
    result["play_quality"] = quality
    result["auction_audit"] = audit_bidding(bundle, seed=seed, deals=games)
    logger.bind(component="evaluation").info("external baseline evaluation {}", result)
    return result


def audit_bidding(bundle: RoleBundle, *, seed: int, deals: int) -> dict[str, float | int]:
    """Probe one no-exploration auction per seed, independent of play evaluation."""
    decisions = calls = all_declined = 0
    for offset in range(deals):
        env = DDZEnv(seed + offset)
        policy_rng = random.Random((seed + offset) * 43 + 1031)
        while env.engine.state.phase == Phase.CALL and env.engine.state.deal_index == 0:
            obs = env.current_observation(event_limit=16)
            action = obs.legal_actions[bundle.act(obs, policy_rng, 0.0).local_index]
            decisions += 1
            calls += int(action.kind == ActionKind.CALL)
            env.step_action(action, include_digest=False)
        all_declined += int(env.engine.state.deal_index > 0)
    return {"first_deal_count": deals, "bid_decisions": decisions,
            "calls": calls, "call_rate": calls / decisions if decisions else 0.0,
            "all_declined": all_declined, "all_declined_rate": all_declined / deals}
