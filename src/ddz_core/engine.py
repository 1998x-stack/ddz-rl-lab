"""Authoritative deterministic three-seat Dou Dizhu engine.

This implementation is for an offline no-stakes RL research simulator.
No network, accounts, virtual currency, wagers or external service integration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict
from typing import Any

from loguru import logger

from .actions import Action, ActionKind
from .cards import EMPTY_COUNTS, cards_to_counts, full_deck, take_physical_cards
from .rules import RuleConfig, generate_plays, pattern_from_action
from .state import GameEvent, GameState, Phase, PlayerObservation, Team


class IllegalAction(ValueError):
    """Rejected action. State must remain unchanged."""


class GameEngine:
    def __init__(self, seed: int, rules: RuleConfig | None = None):
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        self.rules = rules or RuleConfig()
        self.seed = seed
        self.action_history: list[Action] = []
        self.state = self._new_state(0)
        self.state.current_seat = self.state.first_bidder
        self._emit("DEALT", None, {
            "deal_index": 0, "first_bidder": self.state.first_bidder,
            "hand_sizes": [17, 17, 17], "bottom_count": 3,
        })
        logger.bind(game_seed=seed, rule_id=self.rules.rule_id).debug("game_created")
        self.assert_invariants()

    def _new_state(self, deal_index: int) -> GameState:
        entropy = hashlib.sha256(f"ddz-v1:{self.seed}:{deal_index}".encode()).digest()
        rng = random.Random(int.from_bytes(entropy, "big"))
        cards = list(full_deck())
        rng.shuffle(cards)
        hands = [sorted(cards[i * 17:(i + 1) * 17]) for i in range(3)]
        return GameState(
            seed=self.seed, rule_id=self.rules.rule_id, deal_index=deal_index,
            phase=Phase.CALL, hands=hands, bottom_cards=sorted(cards[51:]),
            first_bidder=rng.randrange(3), current_seat=None,
        )

    def _emit(self, kind: str, actor: int | None, data: dict[str, Any]) -> None:
        self.state.events.append(GameEvent(len(self.state.events) + 1, kind, actor, data))

    def legal_actions(self) -> tuple[Action, ...]:
        s = self.state
        if s.phase == Phase.FINISHED:
            return ()
        if s.phase == Phase.CALL:
            return (Action.simple(ActionKind.CALL), Action.simple(ActionKind.DECLINE_CALL))
        if s.phase == Phase.ROB:
            return (Action.simple(ActionKind.ROB), Action.simple(ActionKind.DECLINE_ROB))
        if s.phase == Phase.PLAY:
            assert s.current_seat is not None
            actions = generate_plays(cards_to_counts(s.hands[s.current_seat]), s.last_play, self.rules)
            if s.last_play is not None:
                return actions + (Action.simple(ActionKind.PASS),)
            return actions
        raise AssertionError("unknown game phase")

    def _redeal(self) -> None:
        previous = self.state
        deal_index = previous.deal_index + 1
        if deal_index > self.rules.max_redeals:
            previous.phase = Phase.FINISHED
            previous.current_seat = None
            previous.truncated = True
            self._emit("TRUNCATED", None, {"reason": "MAX_REDEALS", "deal_index": previous.deal_index})
            return
        replacement = self._new_state(deal_index)
        replacement.current_seat = replacement.first_bidder
        replacement.events = previous.events
        replacement.step_index = previous.step_index
        self.state = replacement
        self._emit("DEALT", None, {
            "deal_index": deal_index, "first_bidder": replacement.first_bidder,
            "hand_sizes": [17, 17, 17], "bottom_count": 3,
        })

    def _next_rob_seat_or_finalize(self) -> None:
        s = self.state
        while s.rob_cursor < len(s.rob_turns):
            seat = s.rob_turns[s.rob_cursor]
            if s.call_passed[seat] or s.claimant == seat:
                s.rob_cursor += 1
            else:
                s.current_seat = seat
                return
        assert s.claimant is not None
        s.landlord = s.claimant
        s.phase = Phase.PLAY
        s.current_seat = s.landlord
        s.hands[s.landlord].extend(s.bottom_cards)
        s.hands[s.landlord].sort()
        self._emit("LANDLORD_GRANTED", s.landlord, {
            "landlord_seat": s.landlord,
            "public_bottom_cards": list(s.bottom_cards),
            "rob_count": s.rob_count,
        })

    def step(self, action: Action) -> GameEvent:
        s = self.state
        if not isinstance(action, Action) or action not in self.legal_actions():
            raise IllegalAction(f"illegal action in phase {s.phase.value}: {action!r}")
        assert s.current_seat is not None
        seat = s.current_seat
        s.step_index += 1
        self.action_history.append(action)
        if s.phase == Phase.CALL:
            if action.kind == ActionKind.DECLINE_CALL:
                s.call_passed[seat] = True
                s.call_count += 1
                self._emit("DECLINE_CALL", seat, {})
                if s.call_count == 3:
                    self._emit("ALL_DECLINED", None, {"deal_index": s.deal_index})
                    self._redeal()
                else:
                    s.current_seat = (seat + 1) % 3
            else:
                s.caller = seat
                s.claimant = seat
                s.phase = Phase.ROB
                s.rob_turns = [(seat + 1) % 3, (seat + 2) % 3, seat]
                self._emit("CALLED", seat, {})
                self._next_rob_seat_or_finalize()
        elif s.phase == Phase.ROB:
            if action.kind == ActionKind.ROB:
                s.claimant = seat
                s.rob_count += 1
                self._emit("ROBBED", seat, {"rob_count": s.rob_count})
            else:
                self._emit("DECLINE_ROB", seat, {})
            s.rob_cursor += 1
            self._next_rob_seat_or_finalize()
        elif s.phase == Phase.PLAY:
            if action.kind == ActionKind.PASS:
                self._emit("PASS", seat, {})
                s.passes += 1
                if s.passes == 2:
                    assert s.last_play_seat is not None
                    leader = s.last_play_seat
                    s.last_play = None
                    s.last_play_counts = EMPTY_COUNTS
                    s.last_play_seat = None
                    s.passes = 0
                    s.current_seat = leader
                    self._emit("TRICK_RESET", None, {"leader": leader})
                else:
                    s.current_seat = (seat + 1) % 3
            else:
                cards = take_physical_cards(s.hands[seat], action.counts)
                for card in cards:
                    s.hands[seat].remove(card)
                s.played_by[seat].extend(cards)
                s.last_play = pattern_from_action(action)
                s.last_play_counts = action.counts
                s.last_play_seat = seat
                s.passes = 0
                self._emit("PLAYED", seat, {
                    "physical_cards": list(cards),
                    "counts": list(action.counts),
                    "pattern": list(action.pattern_key or ()),
                    "remaining": len(s.hands[seat]),
                })
                if not s.hands[seat]:
                    s.phase = Phase.FINISHED
                    s.winner = Team.LANDLORD if seat == s.landlord else Team.FARMERS
                    s.winner_seat = seat
                    s.current_seat = None
                    self._emit("FINISHED", seat, {"winner_team": s.winner.value, "winning_seat": seat})
                    logger.bind(game_seed=self.seed, steps=s.step_index,
                                winning_team=s.winner.value).info("game_finished")
                else:
                    s.current_seat = (seat + 1) % 3
        self.assert_invariants()
        return self.state.events[-1]

    def observe(self, seat: int) -> PlayerObservation:
        if seat not in (0, 1, 2):
            raise ValueError("seat must be 0, 1, or 2")
        s = self.state
        return PlayerObservation(
            seat=seat, phase=s.phase, current_seat=s.current_seat,
            own_physical_cards=tuple(s.hands[seat]),
            own_counts=cards_to_counts(s.hands[seat]),
            remaining_counts=tuple(len(hand) for hand in s.hands),  # type: ignore[arg-type]
            played_counts_by_seat=tuple(cards_to_counts(x) for x in s.played_by),
            public_bottom_counts=(cards_to_counts(s.bottom_cards) if s.landlord is not None
                                  else EMPTY_COUNTS),
            landlord=s.landlord, caller=s.caller, claimant=s.claimant,
            first_bidder=s.first_bidder, last_play=s.last_play,
            last_play_counts=s.last_play_counts, last_play_seat=s.last_play_seat,
            public_events=tuple(
                GameEvent(ev.sequence, ev.kind, ev.actor, copy.deepcopy(ev.public))
                for ev in s.events
            ),
            legal_actions=self.legal_actions() if s.current_seat == seat else (),
            deal_index=s.deal_index, step_index=s.step_index,
            winner=s.winner,
        )

    def assert_invariants(self) -> None:
        s = self.state
        all_cards = [c for hand in s.hands for c in hand]
        all_cards += [c for played in s.played_by for c in played]
        if s.landlord is None:
            all_cards += s.bottom_cards
        if len(all_cards) != 54 or set(all_cards) != set(full_deck()):
            raise AssertionError("card conservation failed")
        if s.landlord is None and (s.phase not in (Phase.CALL, Phase.ROB, Phase.FINISHED)):
            raise AssertionError("landlord missing during play")
        if s.phase == Phase.FINISHED:
            if s.current_seat is not None or (not s.truncated and s.winner is None):
                raise AssertionError("inconsistent finished state")
        elif s.current_seat not in (0, 1, 2):
            raise AssertionError("missing acting seat")
        if not (0 <= s.passes <= 1) and s.phase == Phase.PLAY:
            raise AssertionError("invalid consecutive pass count")
        if s.last_play is None and s.last_play_seat is not None:
            raise AssertionError("orphan last play actor")
        if s.landlord is not None and s.landlord not in (0, 1, 2):
            raise AssertionError("invalid landlord seat")

    def clone(self) -> "GameEngine":
        return copy.deepcopy(self)

    def _digest_payload(self) -> dict[str, Any]:
        s = self.state
        return {
            "seed": self.seed, "rule": asdict(self.rules), "deal_index": s.deal_index,
            "phase": s.phase.value, "hands": s.hands, "bottom": s.bottom_cards,
            "played_by": s.played_by, "first_bidder": s.first_bidder,
            "current": s.current_seat, "passed_call": s.call_passed,
            "call_count": s.call_count, "caller": s.caller, "claimant": s.claimant,
            "rob_turns": s.rob_turns, "rob_cursor": s.rob_cursor, "rob_count": s.rob_count,
            "landlord": s.landlord, "last_play": s.last_play.key if s.last_play else None,
            "last_play_counts": s.last_play_counts, "last_play_seat": s.last_play_seat,
            "passes": s.passes, "winner": s.winner.value if s.winner else None,
            "winner_seat": s.winner_seat, "truncated": s.truncated, "step": s.step_index,
            "events": [event.to_dict() for event in s.events],
        }

    def state_digest(self) -> str:
        payload = json.dumps(self._digest_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def export_replay(self) -> dict[str, Any]:
        return {
            "schema_version": "replay.v1", "seed": self.seed,
            "rules": asdict(self.rules),
            "actions": [action.to_dict() for action in self.action_history],
            "state_digest": self.state_digest(),
            "terminated": self.state.phase == Phase.FINISHED and not self.state.truncated,
            "truncated": self.state.truncated,
            "winner": self.state.winner.value if self.state.winner else None,
        }


def replay(record: dict[str, Any]) -> GameEngine:
    if record.get("schema_version") != "replay.v1":
        raise ValueError("unsupported replay schema")
    engine = GameEngine(int(record["seed"]), RuleConfig(**record["rules"]))
    for serialized in record["actions"]:
        engine.step(Action.from_dict(serialized))
    if engine.state_digest() != record["state_digest"]:
        raise ValueError("replay state digest mismatch")
    return engine
