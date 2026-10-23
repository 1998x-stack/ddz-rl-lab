"""Paired initial-auction counterfactual audit.

This is an OFFLINE simulator experiment. Both continuations receive the same
original deal, but rollout policies see only their own PlayerObservation.
Paired uncertainty is computed over independent initial-deal seeds, not moves.
"""
from __future__ import annotations
import math
import random

from ddz_core.actions import ActionKind, Action
from ddz_core.engine import GameEngine
from ddz_core.state import Phase, Team
from ddz_env.baselines import reasonable_action_index
from ddz_quality.analysis import QualityError

def _hand_strength(obs) -> int:
    hand = obs.own_counts
    return (sum(hand[rank] for rank in range(10, 15))
            + sum(hand[rank] >= 3 for rank in range(13))
            + 2 * int(hand[13] and hand[14]))

def _continuation(seed: int, initial: Action, acting_seat: int) -> int | None:
    engine = GameEngine(seed)
    if engine.state.current_seat != acting_seat:
        raise AssertionError('initial seat differs')
    engine.step(initial)
    rng = random.Random(seed ^ 0x5151)
    for _ in range(1200):
        if engine.state.phase == Phase.FINISHED:
            if engine.state.truncated:
                return None
            won = ((engine.state.landlord == acting_seat and
                    engine.state.winner == Team.LANDLORD) or
                   (engine.state.landlord != acting_seat and
                    engine.state.winner == Team.FARMERS))
            return 1 if won else -1
        obs = engine.observe(engine.state.current_seat, event_limit=16)
        engine.step(obs.legal_actions[reasonable_action_index(obs, rng)])
    return None

def paired_auction_audit(seed: int = 2026, deals: int = 30, draws: int = 2000) -> dict:
    if type(seed) is not int or type(deals) is not int or type(draws) is not int or (
            seed < 0 or not 2 <= deals <= 2000 or not 100 <= draws <= 20000):
        raise QualityError('invalid auction audit limits')
    rows=[]
    for offset in range(deals):
        n = seed + offset
        engine = GameEngine(n)
        seat = engine.state.current_seat
        obs = engine.observe(seat, event_limit=16)
        strength = _hand_strength(obs)
        call = _continuation(n, Action.simple(ActionKind.CALL), seat)
        decline = _continuation(n, Action.simple(ActionKind.DECLINE_CALL), seat)
        if call is None or decline is None:
            continue
        rows.append({'deal_seed': n, 'strength': strength,
                     'bucket': 'weak' if strength <= 3 else 'medium' if strength <= 7 else 'strong',
                     'call_reward': call, 'decline_reward': decline,
                     'paired_gain': (call-decline)/2})
    if len(rows) < 2:
        raise QualityError('insufficient completed paired auctions')
    scores=[r['paired_gain'] for r in rows]
    rng=random.Random(seed^0xDCB2)
    n=len(scores)
    ordered=sorted(sum(scores[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    buckets={}
    for bucket in ('weak','medium','strong'):
        samples=[row for row in rows if row['bucket']==bucket]
        buckets[bucket]={'deals':len(samples),
                         'mean_call_gain':(sum(x['paired_gain'] for x in samples)/len(samples)
                                           if samples else None)}
    return {'schema':'ddz-auction-paired-audit-v1',
            'paired_deals':n,'excluded_deals':deals-n,
            'mean_call_gain':sum(scores)/n,
            'ci95':[ordered[int(.025*draws)],ordered[min(draws-1,math.ceil(.975*draws)-1)]],
            'by_strength':buckets,
            'records':rows,
            'interpretation':'Simulator-only paired counterfactual against frozen heuristic opponents; '
                             'not an unbiased auction policy value estimate and not a promotion gate.'}
