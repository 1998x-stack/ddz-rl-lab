"""Deterministic Dou Dizhu rules engine. This project has no wagering or payment support."""
from .actions import Action, ActionKind, Pattern, PatternKind
from .cards import RANKS, card_rank, cards_to_counts, counts_from_ranks, full_deck
from .engine import GameEngine, IllegalAction, replay
from .rules import RuleConfig, beats, generate_plays, recognize_patterns
from .state import Phase, Team

__all__ = [
    "Action", "ActionKind", "Pattern", "PatternKind", "RANKS", "card_rank",
    "cards_to_counts", "counts_from_ranks", "full_deck", "GameEngine",
    "IllegalAction", "replay", "RuleConfig", "beats", "generate_plays",
    "recognize_patterns", "Phase", "Team",
]
