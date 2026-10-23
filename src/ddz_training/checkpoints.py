"""Atomic local checkpoints. Only load artifacts from a trusted directory.

Training payloads may include optimizer and RNG state; inference payload contains
only the four role networks. Never accept arbitrary user-supplied .pt files.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import torch
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle, ROLES
from ddz_env.encoding import ENCODER_VERSION, ACTION_ENCODER_VERSION

SCHEMA = "ddz-training-checkpoint-v1"


def _write_atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    os.close(fd)
    try:
        write(Path(name))
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(path: Path, bundle: RoleBundle, optimizers: dict,
                    *, episodes_seen: int, updates: int, rng: random.Random,
                    metrics: dict, config: dict, inference_only: bool = False) -> Path:
    if set(optimizers) != set(ROLES) and not inference_only:
        raise ValueError("missing optimizer role")
    payload = {
        "schema": SCHEMA,
        "spec": {"state_dim": bundle.spec.state_dim, "action_dim": bundle.spec.action_dim,
                 "hidden": bundle.spec.hidden, "schema": bundle.spec.schema},
        "encoders": {"observation": ENCODER_VERSION, "action": ACTION_ENCODER_VERSION},
        "model": bundle.state_dict(), "episodes_seen": episodes_seen,
        "updates": updates, "metrics": metrics, "config": config,
    }
    if not inference_only:
        payload.update(optimizer={role: optimizer.state_dict() for role, optimizer in optimizers.items()},
                       python_rng=rng.getstate(), torch_rng=torch.get_rng_state())
    path = Path(path)
    _write_atomic(path, lambda target: torch.save(payload, target))
    metadata = {"checkpoint": path.name, "sha256": _digest(path), "schema": SCHEMA,
                "inference_only": inference_only, "episodes_seen": episodes_seen,
                "updates": updates, "metrics": metrics, "encoders": payload["encoders"],
                "spec": payload["spec"]}
    _write_atomic(path.with_suffix(path.suffix + ".json"), lambda target: target.write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"))
    return path


def load_checkpoint(path: Path, bundle: RoleBundle, optimizers: dict | None = None,
                    *, verify_hash: bool = True) -> dict:
    path = Path(path)
    meta = path.with_suffix(path.suffix + ".json")
    if verify_hash:
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        if metadata["checkpoint"] != path.name or _digest(path) != metadata["sha256"]:
            raise ValueError("checkpoint hash mismatch")
    # weights_only explicitly blocks arbitrary object unpickling; trust boundary
    # still requires only loading local, vetted training artifacts.
    payload = torch.load(path, map_location=bundle.device, weights_only=True)
    if payload["schema"] != SCHEMA or payload["spec"] != {
        "state_dim": bundle.spec.state_dim, "action_dim": bundle.spec.action_dim,
        "hidden": bundle.spec.hidden, "schema": bundle.spec.schema,
    } or payload["encoders"] != {
        "observation": ENCODER_VERSION, "action": ACTION_ENCODER_VERSION,
    }:
        raise ValueError("checkpoint incompatible with runtime encoders or network")
    bundle.load_state_dict(payload["model"])
    if optimizers is not None:
        if "optimizer" not in payload or set(optimizers) != set(payload["optimizer"]):
            raise ValueError("not a resumable training checkpoint")
        for role, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizer"][role])
    return payload
