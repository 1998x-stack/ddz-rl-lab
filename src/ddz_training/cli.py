"""Command-line train / evaluate; optional PyTorch dependency via project extra."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import torch
from ddz_core.logging_config import configure_logging
from ddz_eval.tournament import evaluate_bundle
from ddz_eval.league import load_frozen, run_tournament
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from .checkpoints import load_checkpoint
from .trainer import DMCTrainer, TrainConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-stakes DDZ Deep Monte Carlo RL baseline")
    parser.add_argument("command", choices=["train", "evaluate", "tournament"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/dmc-v4")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--actor-workers", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--replay-sample-size", type=int, default=2048)
    parser.add_argument("--updates-per-batch", type=int, default=1)
    parser.add_argument("--history-capacity", type=int, default=6)
    parser.add_argument("--history-probability", type=float, default=.35)
    parser.add_argument("--opponent-checkpoint", action="append", type=Path, default=[])
    return parser


def main() -> None:
    args = _parser().parse_args()
    configure_logging(args.log_dir, "INFO")
    if args.command == "train":
        config = TrainConfig(episodes=args.episodes, batch_episodes=args.batch_episodes,
                             eval_games=args.eval_games, eval_every=args.eval_every,
                             seed=args.seed, hidden=args.hidden, lr=args.lr,
                             epsilon=args.epsilon, max_steps=args.max_steps,
                             device=args.device, output=args.output,
                             actor_workers=args.actor_workers,
                             replay_capacity=args.replay_capacity,
                             replay_sample_size=args.replay_sample_size,
                             updates_per_batch=args.updates_per_batch,
                             history_capacity=args.history_capacity,
                             history_probability=args.history_probability)
        trainer = DMCTrainer(config)
        if args.checkpoint:
            trainer.resume(args.checkpoint)
        result = trainer.train()
    else:
        if args.checkpoint is None:
            raise SystemExit("evaluate/tournament requires --checkpoint")
        # Load stored metadata before instantiating; do not execute stored code.
        meta = json.loads(args.checkpoint.with_suffix(args.checkpoint.suffix + ".json").read_text())
        bundle = RoleBundle(NetworkSpec(hidden=int(meta["spec"]["hidden"])), args.device)
        load_checkpoint(args.checkpoint, bundle)
        if args.command == "evaluate":
            result = evaluate_bundle(bundle, seed=args.seed, games=args.eval_games,
                                     max_steps=args.max_steps)
        else:
            frozen = [(str(path), load_frozen(path, args.device))
                      for path in args.opponent_checkpoint]
            result = run_tournament(bundle, frozen, games=args.eval_games,
                                    seed=args.seed, max_steps=args.max_steps)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
