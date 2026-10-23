"""First usable DMC baseline: sequential actors, four role policies, terminal MC targets.

Not a claim to reproduce the official DouZero training algorithm exactly;
this is a small, inspectable PyTorch first pass with one synchronous learner.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from loguru import logger
from ddz_core.state import Phase
from ddz_env.encoding import encode_action, encode_observation
from ddz_env.environment import DDZEnv
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle, ROLES, role_for_observation
from ddz_eval.tournament import evaluate_bundle
from .checkpoints import load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 100
    batch_episodes: int = 8
    eval_games: int = 16
    eval_every: int = 40
    seed: int = 2026
    hidden: int = 128
    lr: float = 0.0003
    epsilon: float = 0.15
    max_steps: int = 1200
    device: str = "cpu"
    output: str = "runs/dmc-v4"

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.batch_episodes < 1 or self.eval_games < 1 or self.eval_every < 1:
            raise ValueError("episodes/batch_episodes/eval_games/eval_every must be positive")
        if self.max_steps < 1 or self.hidden < 1 or self.lr <= 0 or not 0 <= self.epsilon <= 1:
            raise ValueError("invalid optimization or rollout settings")


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
        self.rng = random.Random(config.seed)
        self.bundle = RoleBundle(NetworkSpec(hidden=config.hidden), config.device)
        self.optimizers = {
            role: torch.optim.Adam(net.parameters(), lr=config.lr)
            for role, net in self.bundle.models.items()
        }
        self.episodes_seen = 0
        self.updates = 0
        self.best_score = float("-inf")
        self.latest_metrics: dict = {}
        self.output = Path(config.output)
        self.output.mkdir(parents=True, exist_ok=True)

    def rollout(self, seed: int) -> list[Experience]:
        env = DDZEnv(seed=seed)
        samples: list[tuple[str, tuple[float, ...], tuple[float, ...], int]] = []
        for _ in range(self.config.max_steps):
            if env.done:
                break
            obs = env.current_observation()
            decision = self.bundle.act(obs, self.rng, self.config.epsilon)
            samples.append((decision.role, encode_observation(obs),
                            encode_action(obs.legal_actions[decision.local_index]), obs.seat))
            env.step(decision.local_index)
        if not env.done:
            raise RuntimeError(f"actor step budget exceeded for seed={seed}")
        if env.engine.state.truncated:
            logger.warning("discard truncated self-play episode seed={}", seed)
            return []
        rewards = env.rewards()
        return [Experience(role, state, action, seat, float(rewards[seat]))
                for role, state, action, seat in samples]

    def learn(self, experience: list[Experience]) -> dict[str, float]:
        result: dict[str, float] = {}
        for role in ROLES:
            batch = [sample for sample in experience if sample.role == role]
            if not batch:
                continue
            model = self.bundle.models[role]
            model.train()
            observations = torch.tensor([x.observation for x in batch], device=self.bundle.device)
            actions = torch.tensor([x.action for x in batch], device=self.bundle.device)
            targets = torch.tensor([x.return_value for x in batch], device=self.bundle.device)
            optimizer = self.optimizers[role]
            optimizer.zero_grad(set_to_none=True)
            pred = model(observations, actions)
            loss = F.mse_loss(pred, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss for {role}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            result[role] = float(loss.item())
        self.updates += 1
        return result

    def resume(self, path: str | Path) -> None:
        payload = load_checkpoint(Path(path), self.bundle, self.optimizers)
        previous = payload["config"]
        for field in ("seed", "hidden", "lr", "epsilon", "batch_episodes", "max_steps"):
            if previous[field] != getattr(self.config, field):
                raise ValueError(f"cannot resume with changed training config: {field}")
        self.episodes_seen = int(payload["episodes_seen"])
        self.updates = int(payload["updates"])
        self.best_score = float(payload["metrics"].get("best_score", float("-inf")))
        self.latest_metrics = dict(payload["metrics"])
        self.rng.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])

    def train(self) -> dict:
        buffer: list[Experience] = []
        evaluated_at: int | None = None
        for _ in range(self.config.episodes):
            # Seed each episode by global index to make resumed experience traceable.
            index = self.episodes_seen
            samples = self.rollout(self.config.seed + 100000 + index)
            buffer.extend(samples)
            self.episodes_seen += 1
            if self.episodes_seen % self.config.batch_episodes == 0:
                if buffer:
                    losses = self.learn(buffer)
                    logger.bind(run_dir=str(self.output)).info(
                        "learner step episodes={} samples={} losses={}",
                        self.episodes_seen, len(buffer), losses)
                buffer.clear()
            if self.episodes_seen % self.config.eval_every == 0:
                self._evaluate_and_save()
                evaluated_at = self.episodes_seen
        if buffer:
            self.learn(buffer)
        if buffer or evaluated_at != self.episodes_seen:
            self._evaluate_and_save()
        return self.latest_metrics

    def _evaluate_and_save(self) -> None:
        metrics = evaluate_bundle(self.bundle, self.config.seed + 200000,
                                  self.config.eval_games, self.config.max_steps)
        score = float(metrics["balanced_score"])
        improved = score > self.best_score
        if improved:
            self.best_score = score
        self.latest_metrics = {**metrics, "best_score": self.best_score,
                               "episodes_seen": self.episodes_seen, "updates": self.updates}
        config = asdict(self.config)
        save_checkpoint(self.output / "latest.pt", self.bundle, self.optimizers,
                        episodes_seen=self.episodes_seen, updates=self.updates,
                        rng=self.rng, metrics=self.latest_metrics, config=config)
        if improved:
            save_checkpoint(self.output / "best.weights.pt", self.bundle, {},
                            episodes_seen=self.episodes_seen, updates=self.updates,
                            rng=self.rng, metrics=self.latest_metrics, config=config,
                            inference_only=True)
        logger.bind(run_dir=str(self.output)).info(
            "checkpoint step={} score={:.4f} best={:.4f}",
            self.episodes_seen, score, self.best_score)
