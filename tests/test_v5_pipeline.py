"""v5 reproducibility, bounded memory, workers, pool and tournament contracts."""
import json
import random
from dataclasses import replace
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
from ddz_models.network import NetworkSpec
from ddz_models.policy import RoleBundle
from ddz_eval.league import wilson, compare_bundles, run_tournament, load_frozen
from ddz_training.replay_buffer import ReplayBuffer
from ddz_training.metrics import ThroughputMeter
from ddz_training.trainer import DMCTrainer, TrainConfig, Experience
from ddz_training.checkpoints import load_checkpoint


def test_replay_capacity_sampling_resume():
    buffer = ReplayBuffer(3)
    values = [Experience("bid", (float(i),), (1.,), 0, float(i)) for i in range(7)]
    buffer.add(values)
    assert len(buffer) == 3 and buffer.seen == 7
    assert [x.return_value for x in buffer.sample(10, random.Random(1))] == [4., 6., 5.]
    restored = ReplayBuffer(3)
    restored.load_state_dict(buffer.state_dict())
    assert [x.return_value for x in restored.sample(3, random.Random(1))] == [4., 6., 5.]
    with pytest.raises(ValueError):
        ReplayBuffer(2).load_state_dict(buffer.state_dict())


def test_throughput_metrics():
    from ddz_training.actors import ActorResult
    from ddz_training.metrics import append_metrics
    meter = ThroughputMeter(started=1.)
    meter.record([ActorResult(12, [], 11, .5, False)])
    metrics = meter.snapshot(elapsed_s=2.)
    assert metrics['episodes_per_s'] == .5
    assert metrics['decisions_per_s'] == 5.5
    assert metrics['truncated'] == 0


def test_redeal_bid_targets_do_not_inherit_later_deal_result():
    from ddz_training.actors import label_samples
    data = [('bid', (0.,), (1.,), 0, 0),
            ('bid', (1.,), (1.,), 0, 1),
            ('landlord', (1.,), (1.,), 0, 1)]
    values = label_samples(data, final_deal=1, truncated=False,
                           reward=(2., -1., -1.), redeal_penalty=1.)
    assert [x.return_value for x in values] == [-1., 2., 2.]
    aborted = label_samples(data[:2], final_deal=1, truncated=True,
                            reward=(0., 0., 0.), redeal_penalty=1.)
    assert [x.return_value for x in aborted] == [-1., -1.]


def test_wilson_and_tournament_deterministic():
    assert wilson(0, 0) == (0., 1.)
    assert wilson(5, 10)[0] < .5 < wilson(5, 10)[1]
    a = RoleBundle(NetworkSpec(hidden=12))
    b = RoleBundle(NetworkSpec(hidden=12))
    one = compare_bundles(a, b, games=2, seed=33)
    two = compare_bundles(a, b, games=2, seed=33)
    assert one == two
    assert 0 <= one['balanced_score'] <= 1
    league = run_tournament(a, [('historic-1', b)], games=1, seed=42)
    assert set(league['opponents']) == {'random', 'historic-1'}
    with pytest.raises(ValueError):
        run_tournament(a, [('random', b)], games=1, seed=42)


def test_v5_training_history_replay_and_resume(tmp_path):
    config = TrainConfig(episodes=3, batch_episodes=1, eval_every=1, eval_games=1,
                         hidden=12, replay_capacity=35, replay_sample_size=8,
                         history_capacity=2, history_probability=1., output=str(tmp_path/'exp'))
    trainer = DMCTrainer(config)
    result = trainer.train()
    assert result['episodes_seen'] == 3 and result['updates'] == 3
    assert 0 < len(trainer.replay) <= 35
    assert trainer.replay.seen >= len(trainer.replay)
    assert len(list((tmp_path/'exp'/'history').glob('*.weights.pt'))) == 2
    assert (tmp_path/'exp'/'throughput.jsonl').exists()
    assert len((tmp_path/'exp'/'throughput.jsonl').read_text().splitlines()) == 3
    assert (tmp_path/'exp'/'evaluations.jsonl').exists()
    frozen = load_frozen(tmp_path/'exp'/'best.weights.pt')
    assert isinstance(frozen, RoleBundle)
    trainer2 = DMCTrainer(replace(config, episodes=1))
    trainer2.resume(tmp_path/'exp'/'latest.pt')
    assert len(trainer2.replay) == len(trainer.replay)
    assert trainer2.train()['episodes_seen'] == 4


def test_spawn_multiprocess_actor(tmp_path):
    cfg = TrainConfig(episodes=2, batch_episodes=2, eval_every=2, eval_games=1,
                      hidden=12, actor_workers=2, history_capacity=0,
                      output=str(tmp_path/'multi'))
    train = DMCTrainer(cfg)
    metrics = train.train()
    assert metrics['episodes_seen'] == 2
    assert metrics['updates'] == 1
    assert metrics['throughput']['decisions'] > 0
    assert len(train.replay) > 0


def test_persistent_actor_pool_refresh_matches_serial_collection():
    from ddz_training.actors import ActorPool, collect_batch
    spec = NetworkSpec(hidden=12)
    first = RoleBundle(spec)
    second = RoleBundle(spec)
    options = dict(spec=spec, historical=[], epsilon=.15, bid_epsilon=.7,
                   max_steps=1200, historical_probability=0., redeal_penalty=1.)
    seeds = [719, 720, 721]
    with ActorPool(2) as pool:
        before = pool.collect(seeds, weights=first.state_dict(), **options)
        after = pool.collect(seeds, weights=second.state_dict(), **options)
    expected_before = collect_batch(seeds, weights=first.state_dict(), workers=1, **options)
    expected_after = collect_batch(seeds, weights=second.state_dict(), workers=1, **options)
    for actual, expected in ((before, expected_before), (after, expected_after)):
        assert [item.seed for item in actual] == seeds
        assert [(item.samples, item.decisions, item.truncated) for item in actual] == [
            (item.samples, item.decisions, item.truncated) for item in expected]


def test_validation():
    for arguments in [dict(actor_workers=0), dict(replay_capacity=0),
                      dict(replay_sample_size=0), dict(history_probability=1.1)]:
        with pytest.raises(ValueError):
            TrainConfig(**arguments)
