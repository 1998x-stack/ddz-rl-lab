"""Train/serve parity, legal inference, data isolation, restart and integrity gates."""
import json
import random
from dataclasses import replace
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
from ddz_env import DDZEnv, encode_observation
from ddz_core.state import Phase
from ddz_models.network import CandidateQNetwork, NetworkSpec
from ddz_models.policy import RoleBundle, ROLES, role_for_observation
from ddz_training.trainer import DMCTrainer, TrainConfig
from ddz_training.checkpoints import load_checkpoint
from ddz_eval.tournament import evaluate_bundle


def test_network_shape_gradient_and_candidate_parity():
    model = CandidateQNetwork(NetworkSpec(hidden=24))
    state = torch.rand(1, model.spec.state_dim)
    actions = torch.rand(7, model.spec.action_dim)
    a = model(state.expand(7, -1), actions)
    b = model.score_legal(state[0], actions)
    assert a.shape == b.shape == (7,)
    torch.testing.assert_close(a, b)
    a.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_role_assignment_and_redacted_policy():
    env = DDZEnv(seed=2026)
    bundle = RoleBundle(NetworkSpec(hidden=24))
    rng = random.Random(2026)
    for _ in range(800):
        if env.done:
            break
        obs = env.current_observation()
        assert len(encode_observation(obs)) == bundle.spec.state_dim
        decision = bundle.act(obs, rng, epsilon=0.0)
        assert decision.role in ROLES
        assert 0 <= decision.local_index < len(obs.legal_actions)
        if obs.phase == Phase.PLAY:
            expected = ("landlord" if obs.seat == obs.landlord else
                        "farmer_down" if obs.seat == (obs.landlord + 1) % 3
                        else "farmer_up")
            assert decision.role == expected
        env.step(decision.local_index)
    assert env.done
    env.engine.assert_invariants()


def test_training_checkpoint_evaluate_and_resume(tmp_path):
    out = str(tmp_path / "experiment")
    cfg = TrainConfig(episodes=2, batch_episodes=1, eval_every=2, eval_games=2,
                      hidden=24, output=out)
    trainer = DMCTrainer(cfg)
    result = trainer.train()
    assert result["episodes_seen"] == 2
    assert result["updates"] == 2
    assert 0 <= result["balanced_score"] <= 1
    latest = Path(out) / "latest.pt"
    best = Path(out) / "best.weights.pt"
    assert latest.exists() and best.exists()
    rebuilt = RoleBundle(NetworkSpec(hidden=24))
    data = load_checkpoint(best, rebuilt)
    assert "optimizer" not in data
    eval_metrics = evaluate_bundle(rebuilt, games=2, seed=250)
    assert eval_metrics["landlord_games"] <= 2
    assert eval_metrics["farmer_games"] <= 2
    resumed = DMCTrainer(replace(cfg, episodes=1))
    resumed.resume(latest)
    assert resumed.episodes_seen == 2
    result2 = resumed.train()
    assert result2["episodes_seen"] == 3
    with pytest.raises(ValueError, match="cannot resume"):
        incompatible = DMCTrainer(replace(cfg, lr=0.01))
        incompatible.resume(latest)


def test_checkpoint_hash_rejection(tmp_path):
    cfg = TrainConfig(episodes=1, batch_episodes=1, eval_every=2, eval_games=1,
                      hidden=16, output=str(tmp_path / "run"))
    DMCTrainer(cfg).train()
    path = Path(cfg.output) / "latest.pt"
    meta = path.with_suffix(path.suffix + ".json")
    payload = json.loads(meta.read_text())
    payload["sha256"] = "0" * 64
    meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path, RoleBundle(NetworkSpec(hidden=16)))


def test_invalid_config_and_empty_candidates():
    with pytest.raises(ValueError):
        TrainConfig(episodes=0)
    with pytest.raises(ValueError):
        NetworkSpec(hidden=-1)
    env = DDZEnv(3)
    bundle = RoleBundle(NetworkSpec(hidden=16))
    with pytest.raises(ValueError):
        bundle.act(env.current_observation(), random.Random(3), epsilon=1.2)
