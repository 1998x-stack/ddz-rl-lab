"""Spawn-safe multiprocessing rollouts; no GPU tensors or hidden hands cross IPC.

Workers receive frozen network state once per batch via initializer, and emit
only terminal-labeled, information-safe experiences and scalar metrics.
"""
from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import random
import time

import torch
from ddz_core.state import Phase
from ddz_env.environment import DDZEnv
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from .trainer import Experience


@dataclass(frozen=True)
class ActorResult:
    seed: int
    samples: list[Experience]
    decisions: int
    duration_s: float
    truncated: bool
    opponent: str | None = None


_ACTIVE: RoleBundle | None = None
_FROZEN: list[tuple[str, RoleBundle]] = []
_EPSILON = 0.0
_BID_EPSILON = 0.0
_MAX_STEPS = 1200
_HIST_PROB = 0.0
_REDEAL_PENALTY = 1.0


def label_samples(data: list[tuple[str, tuple[float, ...], tuple[float, ...], int, int]],
                  *, final_deal: int, truncated: bool,
                  reward: tuple[float, float, float], redeal_penalty: float) -> list[Experience]:
    """Give abandoned auctions their own target, never a later deal's result."""
    return [Experience(role, state, action, seat,
                       (-redeal_penalty if deal < final_deal or truncated
                        else float(reward[seat])))
            for role, state, action, seat, deal in data]


def initialize_actor(spec: NetworkSpec, current: dict, historical: list[tuple[str, dict]],
                     epsilon: float, bid_epsilon: float, max_steps: int,
                     historical_probability: float,
                     redeal_penalty: float) -> None:
    global _ACTIVE, _FROZEN, _EPSILON, _BID_EPSILON, _MAX_STEPS, _HIST_PROB, _REDEAL_PENALTY
    torch.set_num_threads(1)
    _ACTIVE = RoleBundle(spec, "cpu")
    _ACTIVE.load_state_dict(current)
    _FROZEN = []
    for name, weights in historical:
        frozen = RoleBundle(spec, "cpu")
        frozen.load_state_dict(weights)
        _FROZEN.append((name, frozen))
    _EPSILON, _MAX_STEPS, _HIST_PROB = epsilon, max_steps, historical_probability
    _BID_EPSILON = bid_epsilon
    _REDEAL_PENALTY = redeal_penalty


def rollout_actor(seed: int) -> ActorResult:
    if _ACTIVE is None:
        raise RuntimeError("actor was not initialized")
    started = time.monotonic()
    rng = random.Random(seed ^ 0xD0D5)
    env = DDZEnv(seed=seed)
    data: list[tuple[str, tuple[float, ...], tuple[float, ...], int, int]] = []
    opponent: str | None = None
    frozen: RoleBundle | None = None
    learn_side: str | None = None
    if _FROZEN and rng.random() < _HIST_PROB:
        opponent, frozen = rng.choice(_FROZEN)
        learn_side = rng.choice(["landlord", "farmers"])
    decisions = 0
    for _ in range(_MAX_STEPS):
        if env.done:
            break
        obs = env.current_observation(event_limit=16)
        # Bid is jointly trained; historical opponents are applied only after
        # roles have been assigned to avoid mixing seat/role semantics.
        learned = (frozen is None or obs.phase != Phase.PLAY or
                   ((obs.seat == obs.landlord) == (learn_side == "landlord")))
        bundle = _ACTIVE if learned else frozen
        assert bundle is not None
        with torch.inference_mode():
            exploration = (_BID_EPSILON if obs.phase != Phase.PLAY
                           else _EPSILON) if learned else 0.0
            choice = bundle.act(obs, rng, exploration)
        if learned:
            data.append((choice.role, _ACTIVE.encode_observation(obs),
                         _ACTIVE.encode_action(obs, obs.legal_actions[choice.local_index]), obs.seat,
                         obs.deal_index))
        env.step(choice.local_index, include_digest=False)
        decisions += 1
    if not env.done:
        raise RuntimeError(f"actor decision budget exceeded seed={seed}")
    truncated = bool(env.engine.state.truncated)
    reward = env.rewards()
    final_deal = env.engine.state.deal_index
    # A declined deal is a separate auction: its actions must not inherit the
    # result of a later, unrelated shuffle. Truncated auctions still teach the
    # bidder that repeated unanimous declines have a cost.
    samples = label_samples(data, final_deal=final_deal, truncated=truncated,
                            reward=reward, redeal_penalty=_REDEAL_PENALTY)
    return ActorResult(seed, samples, decisions, time.monotonic()-started, truncated, opponent)


def collect_batch(seeds: list[int], *, spec: NetworkSpec, weights: dict,
                  historical: list[tuple[str, dict]], epsilon: float,
                  bid_epsilon: float, max_steps: int,
                  historical_probability: float, redeal_penalty: float,
                  workers: int) -> list[ActorResult]:
    """Spawn workers only when requested; preserve task ordering for determinism.

    Model snapshots are sent once per worker at process initialization. A new
    executor is used each training batch so policy staleness is bounded.
    """
    args = (spec, weights, historical, epsilon, bid_epsilon, max_steps,
            historical_probability, redeal_penalty)
    if workers == 1:
        initialize_actor(*args)
        return [rollout_actor(seed) for seed in seeds]
    if workers < 1:
        raise ValueError("workers must be positive")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=initialize_actor, initargs=args) as pool:
        return list(pool.map(rollout_actor, seeds))
