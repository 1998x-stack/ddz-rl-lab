"""Wall-clock throughput and structured, append-only experiment metrics."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import time


@dataclass
class ThroughputMeter:
    started: float = 0.0
    episodes: int = 0
    decisions: int = 0
    samples: int = 0
    truncated: int = 0
    actor_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.started:
            self.started = time.monotonic()

    def record(self, results: list) -> None:
        for r in results:
            self.episodes += 1
            self.decisions += r.decisions
            self.samples += len(r.samples)
            self.truncated += int(r.truncated)
            self.actor_seconds += r.duration_s

    def snapshot(self, elapsed_s: float | None = None) -> dict:
        elapsed = max(1e-9, time.monotonic() - self.started if elapsed_s is None else elapsed_s)
        return {"episodes": self.episodes, "decisions": self.decisions,
                "samples": self.samples, "truncated": self.truncated,
                "wall_seconds": elapsed, "actor_seconds": self.actor_seconds,
                "episodes_per_s": self.episodes / elapsed,
                "decisions_per_s": self.decisions / elapsed,
                "samples_per_s": self.samples / elapsed}


def append_metrics(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(values, allow_nan=False, sort_keys=True) + "\n")
