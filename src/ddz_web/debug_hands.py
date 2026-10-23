"""Explicit opt-in, local-only debug view. Never passed to a policy."""
from __future__ import annotations

from ddz_core.cards import card_rank


def hand_debug_payload(game) -> dict:
    state = game.env.engine.state
    return {
        'schema': 'ddz-local-debug-hands-v1',
        'game_id': game.game_id,
        'sequence': state.events[-1].sequence if state.events else 0,
        'landlord': state.landlord,
        'acting_seat': state.current_seat,
        'hands': [sorted(hand, key=lambda card: (-card_rank(card), card))
                  for hand in state.hands],
    }
