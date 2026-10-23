"""Async, event-driven three-policy no-stakes Arena.

The UI only receives PUBLIC game state; policies obtain individually redacted
observations. Registered checkpoints are pinned and verified before a game starts.
An explicitly enabled baseline may be used for development, never relabeled RL.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import secrets
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ddz_core.cards import cards_to_counts
from ddz_core.state import Phase
from ddz_env.baselines import random_action_index
from ddz_env.environment import DDZEnv
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from ddz_registry import ModelRegistry
from ddz_training.checkpoints import load_checkpoint


MAX_TURNS = 1200
MIN_DELAY = 0.05
MAX_DELAY = 3.0


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".replay-", dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def public_snapshot(env: DDZEnv, game_id: str, policies: tuple[str, str, str],
                    *, running: bool, delay: float, status: str,
                    error: str | None = None) -> dict[str, Any]:
    """Never serialize GameState.hands, unrevealed bottom, or policy inputs."""
    s = env.engine.state
    return {
        "type": "snapshot", "game_id": game_id, "phase": s.phase.value,
        "status": status, "running": running, "delay": delay,
        "model_ids": list(policies), "seed": env.engine.seed if env.done else None,
        "sequence": s.events[-1].sequence if s.events else 0,
        "step_index": s.step_index, "acting_seat": s.current_seat,
        "landlord": s.landlord, "remaining": [len(hand) for hand in s.hands],
        "last_play_seat": s.last_play_seat, "last_play_counts": list(s.last_play_counts),
        "bottom_counts": list(cards_to_counts(s.bottom_cards)) if s.landlord is not None else None,
        "winner": s.winner.value if s.winner else None,
        "truncated": s.truncated, "error": error,
        "events": [event.to_dict() for event in s.events[-12:]],
    }


@dataclass
class ArenaGame:
    game_id: str
    env: DDZEnv
    policy_ids: tuple[str, str, str]
    policies: tuple[RoleBundle | None, RoleBundle | None, RoleBundle | None]
    rngs: tuple[random.Random, random.Random, random.Random]
    replay_dir: Path
    delay: float = 0.7
    running: bool = False
    status: str = "paused"
    error: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    task: asyncio.Task | None = None

    def snapshot(self) -> dict:
        return public_snapshot(self.env, self.game_id, self.policy_ids,
                               running=self.running, delay=self.delay,
                               status=self.status, error=self.error)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)

    def _publish(self) -> None:
        packet = self.snapshot()
        for queue in list(self.subscribers):
            if queue.full():
                try:
                    queue.get_nowait()  # next snapshot always supersedes older snapshots
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(packet)

    async def step_once(self) -> dict:
        async with self.lock:
            if self.env.done or self.status == "error":
                return self.snapshot()
            if self.env.engine.state.step_index >= MAX_TURNS:
                self.running = False
                self.status = "error"
                self.error = "step limit exceeded; replay retained for diagnosis"
                self._publish()
                return self.snapshot()
            seat = self.env.acting_seat
            assert seat is not None
            policy = self.policies[seat]
            obs = self.env.current_observation()
            try:
                if policy is None:
                    idx = random_action_index(self.env, self.rngs[seat])
                    role = "baseline"
                else:
                    # Offload CPU-heavy candidate scoring to avoid blocking the WS event loop.
                    decision = await asyncio.to_thread(policy.act, obs, self.rngs[seat], 0.0)
                    idx = decision.local_index
                    role = decision.role
                event = self.env.step(idx)
            except Exception as exc:
                self.running = False
                self.status = "error"
                self.error = f"inference or game error: {type(exc).__name__}"
                logger.bind(game_id=self.game_id, seat=seat).exception("arena_step_failed")
                self._publish()
                return self.snapshot()
            logger.bind(game_id=self.game_id, seat=seat, role=role,
                        step_index=self.env.engine.state.step_index).debug("arena_action")
            if event.terminated or event.truncated:
                self.running = False
                self.status = "finished" if event.terminated else "truncated"
                replay = self.env.export_replay()
                replay["model_ids"] = list(self.policy_ids)
                _atomic_json(self.replay_dir / f"{self.game_id}.json", replay)
                logger.bind(game_id=self.game_id, outcome=replay["winner"]).info("arena_game_ended")
            self._publish()
            return self.snapshot()

    async def _autoplay(self) -> None:
        try:
            while self.running and not self.env.done:
                await self.step_once()
                if not self.running:
                    break
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            raise
        finally:
            if self.task is asyncio.current_task():
                self.task = None

    async def control(self, action: str, delay: float | None = None) -> dict:
        if action not in {"play", "pause", "step", "speed"}:
            raise ValueError("unknown control command")
        if action == "speed":
            if delay is None or not MIN_DELAY <= delay <= MAX_DELAY:
                raise ValueError("delay must be between 0.05 and 3 seconds")
            self.delay = delay
        elif action == "pause":
            self.running = False
            if not self.env.done and self.status != "error":
                self.status = "paused"
        elif action == "step":
            if self.running:
                raise ValueError("pause before stepping")
            return await self.step_once()
        elif action == "play" and not self.env.done and self.status != "error":
            self.running = True
            self.status = "running"
            if self.task is None or self.task.done():
                self.task = asyncio.create_task(self._autoplay(), name=f"arena:{self.game_id}")
        self._publish()
        return self.snapshot()

    async def close(self) -> None:
        self.running = False
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


class ArenaManager:
    def __init__(self, registry_dir: Path, replay_dir: Path,
                 *, allow_baseline: bool = False, max_sessions: int = 12):
        self.registry = ModelRegistry(registry_dir)
        self.replay_dir = Path(replay_dir)
        self.allow_baseline = allow_baseline
        self.max_sessions = max_sessions
        self.games: dict[str, ArenaGame] = {}
        self._models: dict[tuple[str, str], RoleBundle] = {}

    def available_models(self) -> dict:
        index = self.registry.list_models()
        return {"active": index.get("active"), "models": index["models"],
                "baseline_enabled": self.allow_baseline}

    def _load_model(self, model_id: str) -> RoleBundle | None:
        if model_id == "baseline":
            if not self.allow_baseline:
                raise ValueError("baseline mode is disabled; register a trained inference bundle")
            return None
        record = self.registry.verify(model_id, load_weights=False)
        key = model_id, record["sha256"]
        if key not in self._models:
            path, _ = self.registry._model_files(model_id)
            bundle = RoleBundle(NetworkSpec(hidden=int(record["spec"]["hidden"])))
            load_checkpoint(path, bundle)
            self._models[key] = bundle
        return self._models[key]

    def create(self, ids: tuple[str, str, str], *, seed: int | None = None,
               delay: float = .7) -> ArenaGame:
        if len(self.games) >= self.max_sessions:
            raise ValueError("arena is full; restart server to clear completed sessions")
        if len(ids) != 3 or not all(isinstance(x, str) for x in ids):
            raise ValueError("exactly three model IDs required")
        if not MIN_DELAY <= delay <= MAX_DELAY:
            raise ValueError("invalid playback speed")
        if seed is None:
            seed = secrets.randbits(31)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be unsigned 32-bit integer")
        policies = tuple(self._load_model(x) for x in ids)
        game_id = secrets.token_hex(12)
        game = ArenaGame(game_id, DDZEnv(seed), ids, policies,
                         tuple(random.Random(seed + i * 1009) for i in range(3)),
                         self.replay_dir, delay)
        self.games[game_id] = game
        logger.bind(game_id=game_id, models=ids, seed=seed).info("arena_created")
        return game

    def get(self, game_id: str) -> ArenaGame:
        try:
            return self.games[game_id]
        except KeyError:
            raise KeyError("unknown arena game") from None

    async def close(self) -> None:
        for game in list(self.games.values()):
            await game.close()
