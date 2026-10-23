"""Version 5: reproducible role DMC baseline with bounded replay and actor pool.

It uses offline Monte Carlo regression (not PPO, not an exact DouZero
implementation). Multi-process actors synchronize weights between batches;
workers never receive private information outside their game ENV.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from loguru import logger
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle, ROLES
from ddz_eval.tournament import evaluate_bundle
from .checkpoints import load_checkpoint, save_checkpoint
from .replay_buffer import ReplayBuffer
from .metrics import ThroughputMeter, append_metrics


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 100
    batch_episodes: int = 8
    eval_games: int = 16
    eval_every: int = 40
    seed: int = 2026
    hidden: int = 128
    lr: float = .0003
    epsilon: float = .15
    max_steps: int = 1200
    device: str = "cpu"
    output: str = "runs/dmc-v5"
    actor_workers: int = 1
    replay_capacity: int = 50000
    replay_sample_size: int = 2048
    updates_per_batch: int = 1
    history_capacity: int = 6
    history_probability: float = .35
    early_stop_patience_evals: int = 0
    min_score_improvement: float = 0.0

    def __post_init__(self) -> None:
        positives = ("episodes", "batch_episodes", "eval_games", "eval_every", "hidden",
                     "max_steps", "actor_workers", "replay_capacity", "replay_sample_size",
                     "updates_per_batch")
        if any(getattr(self, x) < 1 for x in positives):
            raise ValueError("positive training counts required")
        if self.lr <= 0 or not 0 <= self.epsilon <= 1 or self.history_capacity < 0:
            raise ValueError("invalid learning or history configuration")
        if (type(self.early_stop_patience_evals) is not int or
                self.early_stop_patience_evals < 0 or self.min_score_improvement < 0):
            raise ValueError('invalid early stopping configuration')
        if not 0 <= self.history_probability <= 1:
            raise ValueError("history_probability must be in [0,1]")


@dataclass(frozen=True)
class Experience:
    role: str
    observation: tuple[float, ...]
    action: tuple[float, ...]
    actor_seat: int
    return_value: float


class DMCTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        self.rng = random.Random(config.seed)
        self.bundle = RoleBundle(NetworkSpec(hidden=config.hidden), config.device)
        self.optimizers = {role: torch.optim.Adam(net.parameters(), lr=config.lr)
                           for role, net in self.bundle.models.items()}
        self.replay = ReplayBuffer(config.replay_capacity)
        self.episodes_seen = 0
        self.updates = 0
        self.best_score = float("-inf")
        self.evals_without_improvement = 0
        self.stopped_early = False
        self.latest_metrics: dict = {}
        self.output = Path(config.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.meter = ThroughputMeter()

    def _frozen_history(self) -> list[tuple[str, dict]]:
        paths = sorted((self.output / "history").glob("step-*.weights.pt"))[-self.config.history_capacity:] if self.config.history_capacity else []
        result = []
        for path in paths:
            frozen = RoleBundle(NetworkSpec(hidden=self.config.hidden), "cpu")
            load_checkpoint(path, frozen)
            result.append((path.stem, frozen.state_dict()))
        return result

    def rollout(self, seed: int) -> list[Experience]:
        """Single-episode compatibility API; reuses the v5 actor path."""
        return self._collect([seed])[0].samples

    def _collect(self, seeds: list[int]):
        from .actors import collect_batch
        # Snapshots are CPU tensors even if the learner is on CUDA.
        weights = {role: {name: value.detach().cpu().clone()
                          for name, value in net.state_dict().items()}
                   for role, net in self.bundle.models.items()}
        historical = self._frozen_history()
        return collect_batch(seeds, spec=self.bundle.spec, weights=weights,
                             historical=historical, epsilon=self.config.epsilon,
                             max_steps=self.config.max_steps,
                             historical_probability=self.config.history_probability,
                             workers=self.config.actor_workers)

    def learn(self, samples: list[Experience]) -> dict[str, float]:
        losses: dict[str, float] = {}
        for role in ROLES:
            batch = [x for x in samples if x.role == role]
            if not batch:
                continue
            model = self.bundle.models[role]
            model.train()
            observations = torch.tensor([x.observation for x in batch], dtype=torch.float32,
                                        device=self.bundle.device)
            actions = torch.tensor([x.action for x in batch], dtype=torch.float32,
                                   device=self.bundle.device)
            targets = torch.tensor([x.return_value for x in batch], dtype=torch.float32,
                                   device=self.bundle.device)
            opt = self.optimizers[role]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(observations, actions), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss for {role}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
            losses[role] = float(loss.item())
        self.updates += 1
        return losses

    def resume(self, path: str | Path) -> None:
        payload = load_checkpoint(Path(path), self.bundle, self.optimizers)
        saved = payload["config"]
        for field in ("seed", "hidden", "lr", "epsilon", "batch_episodes", "max_steps",
                      "actor_workers", "replay_capacity", "replay_sample_size",
                      "updates_per_batch", "history_capacity", "history_probability",
                      "early_stop_patience_evals", "min_score_improvement"):
            if saved.get(field, getattr(self.config, field)) != getattr(self.config, field):
                raise ValueError(f"cannot resume with changed training config: {field}")
        self.episodes_seen = int(payload["episodes_seen"])
        self.updates = int(payload["updates"])
        self.best_score = float(payload["metrics"].get("best_score", float("-inf")))
        self.evals_without_improvement = int(payload.get('trainer_state', {}).get('evals_without_improvement', 0))
        self.stopped_early = False
        self.latest_metrics = dict(payload["metrics"])
        self.rng.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("trainer_state", {}).get("replay"):
            self.replay.load_state_dict(payload["trainer_state"]["replay"])

    def train(self) -> dict:
        evaluated_at: int | None = None
        target = self.episodes_seen + self.config.episodes
        while self.episodes_seen < target:
            remaining = target - self.episodes_seen
            until_update = self.config.batch_episodes - self.episodes_seen % self.config.batch_episodes
            until_eval = self.config.eval_every - self.episodes_seen % self.config.eval_every
            count = min(remaining, until_update, until_eval)
            seeds = [self.config.seed + 100000 + i
                     for i in range(self.episodes_seen, self.episodes_seen + count)]
            before = time.monotonic()
            results = self._collect(seeds)
            self.meter.record(results)
            self.replay.add([sample for result in results for sample in result.samples])
            self.episodes_seen += count
            stats = self.meter.snapshot()
            stats.update(episodes_seen=self.episodes_seen, replay_size=len(self.replay),
                         actor_workers=self.config.actor_workers,
                         collection_wall_s=time.monotonic()-before,
                         historical_episodes=sum(x.opponent is not None for x in results))
            append_metrics(self.output / "throughput.jsonl", stats)
            logger.bind(component="actors", episodes=self.episodes_seen).info(
                "collect episodes={} samples={} ep_s={:.2f} decisions_s={:.2f}",
                count, sum(len(r.samples) for r in results), stats["episodes_per_s"],
                stats["decisions_per_s"])
            if self.episodes_seen % self.config.batch_episodes == 0:
                self._learn_from_replay()
            if self.episodes_seen % self.config.eval_every == 0:
                self._evaluate_and_save()
                evaluated_at = self.episodes_seen
                if self.stopped_early:
                    logger.bind(component='learner', episodes=self.episodes_seen).info('early_stopping_triggered')
                    break
        if not self.stopped_early and self.episodes_seen % self.config.batch_episodes:
            self._learn_from_replay()
            # The last evaluation happened before this final partial update.
            evaluated_at = None
        if evaluated_at != self.episodes_seen and not self.stopped_early:
            self._evaluate_and_save()
        return self.latest_metrics

    def _learn_from_replay(self) -> None:
        for _ in range(self.config.updates_per_batch):
            samples = self.replay.sample(self.config.replay_sample_size, self.rng)
            if samples:
                losses = self.learn(samples)
                logger.bind(component="learner", episodes=self.episodes_seen).info(
                    "update={} samples={} loss={}", self.updates, len(samples), losses)

    def _evaluate_and_save(self) -> None:
        result = evaluate_bundle(self.bundle, self.config.seed + 200000,
                                 self.config.eval_games, self.config.max_steps)
        score = float(result["balanced_score"])
        improved = score > self.best_score + self.config.min_score_improvement
        if improved:
            self.best_score = score
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
        self.stopped_early = (self.config.early_stop_patience_evals > 0 and
                              self.evals_without_improvement >= self.config.early_stop_patience_evals)
        self.latest_metrics = {**result, "best_score": self.best_score,
                               "episodes_seen": self.episodes_seen, "updates": self.updates,
                               "replay_size": len(self.replay),
                               "throughput": self.meter.snapshot(),
                               "evals_without_improvement": self.evals_without_improvement,
                               "stopped_early": self.stopped_early}
        params = dict(asdict(self.config))
        save_checkpoint(self.output / "latest.pt", self.bundle, self.optimizers,
                        episodes_seen=self.episodes_seen, updates=self.updates,
                        rng=self.rng, metrics=self.latest_metrics, config=params,
                        extra_state={"replay": self.replay.state_dict(),
                                     "evals_without_improvement": self.evals_without_improvement})
        if improved:
            save_checkpoint(self.output / "best.weights.pt", self.bundle, {},
                            episodes_seen=self.episodes_seen, updates=self.updates,
                            rng=self.rng, metrics=self.latest_metrics, config=params,
                            inference_only=True)
        if self.config.history_capacity:
            dest = self.output / "history" / f"step-{self.episodes_seen:09d}.weights.pt"
            save_checkpoint(dest, self.bundle, {}, episodes_seen=self.episodes_seen,
                            updates=self.updates, rng=self.rng, metrics=self.latest_metrics,
                            config=params, inference_only=True)
            checkpoints = sorted(dest.parent.glob("step-*.weights.pt"))
            for old in checkpoints[:-self.config.history_capacity]:
                old.unlink()
                old.with_suffix(old.suffix + ".json").unlink(missing_ok=True)
        append_metrics(self.output / "evaluations.jsonl", self.latest_metrics)
        logger.bind(component="evaluation").info(
            "checkpoint episodes={} score={:.4f} best={:.4f}",
            self.episodes_seen, score, self.best_score)
