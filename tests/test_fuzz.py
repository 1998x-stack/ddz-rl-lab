"""Small deterministic smoke suite; does not make training-quality claims."""
import random

from ddz_core import ActionKind, GameEngine
from ddz_core.state import Phase


def test_many_random_complete_games():
    for seed in range(25):
        engine = GameEngine(seed)
        policy_rng = random.Random(80000 + seed)
        for _ in range(1500):
            if engine.state.phase == Phase.FINISHED:
                break
            options = engine.legal_actions()
            if engine.state.phase == Phase.CALL:
                chosen = next(a for a in options if a.kind == ActionKind.CALL)
            elif engine.state.phase == Phase.ROB:
                chosen = next(a for a in options if a.kind == ActionKind.DECLINE_ROB)
            else:
                chosen = policy_rng.choice(options)
            engine.step(chosen)
        assert engine.state.phase == Phase.FINISHED
        assert engine.state.winner is not None
        engine.assert_invariants()
