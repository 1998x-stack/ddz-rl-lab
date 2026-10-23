"""Fixed global action catalogue for standard PettingZoo Discrete action space.

Training should use compact legal-candidate matrices instead of dense 56k masks;
the PettingZoo adapter exposes a dense action mask for API interoperability.
"""
from __future__ import annotations

from functools import lru_cache

from ddz_core.actions import Action, ActionKind
from ddz_core.cards import FULL_COUNTS
from ddz_core.rules import RuleConfig, generate_plays

# The semantic action inventory is built using max-available rank counts.
# Nonstandard RuleConfigs may need a different catalogue version; see docs.
DEFAULT_RULES = RuleConfig()
BIDDING = (Action.simple(ActionKind.CALL), Action.simple(ActionKind.DECLINE_CALL),
           Action.simple(ActionKind.ROB), Action.simple(ActionKind.DECLINE_ROB),
           Action.simple(ActionKind.PASS))


@lru_cache(maxsize=1)
def action_catalogue() -> tuple[Action, ...]:
    # All patterns possible from a standard 54-card deck, including combinations
    # requiring more than a single hand. It is a semantic superset by design.
    return BIDDING + generate_plays(FULL_COUNTS, None, DEFAULT_RULES)


@lru_cache(maxsize=1)
def _reverse() -> dict[Action, int]:
    return {action: i for i, action in enumerate(action_catalogue())}


def global_action_id(action: Action) -> int:
    return _reverse()[action]


def global_legal_ids(actions: tuple[Action, ...]) -> tuple[int, ...]:
    return tuple(global_action_id(action) for action in actions)
