"""Capacity-bounded, reproducible replay of public-observation DMC decisions.

Only immutable encoded visible observations, encoded actions, role and team
returns are allowed in the buffer. No authoritative GameState is stored.
"""
from __future__ import annotations
from collections import deque
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Experience


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buffer: deque[Experience] = deque(maxlen=capacity)
        self.seen = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, samples: list[Experience]) -> None:
        self._buffer.extend(samples)
        self.seen += len(samples)

    def sample(self, size: int, rng: random.Random) -> list[Experience]:
        if size < 1:
            raise ValueError("sample size must be positive")
        if not self._buffer:
            return []
        return rng.sample(list(self._buffer), k=min(size, len(self._buffer)))

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen,
                "entries": [(x.role, x.observation, x.action, x.actor_seat,
                             x.return_value, x.preferred_action, x.alternative_action,
                             x.deal_id) for x in self._buffer]}

    def load_state_dict(self, data: dict) -> None:
        from .trainer import Experience
        if data["capacity"] != self.capacity:
            raise ValueError("replay buffer capacity changed on resume")
        entries = data["entries"]
        if len(entries) > self.capacity:
            raise ValueError("oversized replay snapshot")
        if not all(len(x) in (5, 8) for x in entries):
            raise ValueError("malformed replay entry")
        self._buffer.clear()
        self._buffer.extend(Experience(*x) for x in entries)
        self.seen = int(data["seen"])
