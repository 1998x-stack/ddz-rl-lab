import random

import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from ddz_env.pettingzoo_env import PettingZooDDZEnv


def test_pettingzoo_spaces_and_seeded_play():
    env = PettingZooDDZEnv(seed=2026)
    for agent in env.possible_agents:
        assert env.observation_space(agent).contains(env.observe(agent))
        assert env.action_space(agent).n > 50000
    rng = random.Random(45)
    for _ in range(1200):
        if not env.agents:
            break
        selected = env.agent_selection
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
        else:
            ids = info["legal_action_ids"]
            assert len(ids) == int(obs["action_mask"].sum())
            env.step(rng.choice(ids))
    assert not env.agents


def test_pettingzoo_rejects_illegal_global_action():
    env = PettingZooDDZEnv(seed=99)
    mask = env.observe(env.agent_selection)["action_mask"]
    bad_id = next(i for i, bit in enumerate(mask) if bit == 0)
    with pytest.raises(ValueError):
        env.step(bad_id)


def test_pettingzoo_api_contract():
    from pettingzoo.test import api_test
    env = PettingZooDDZEnv(seed=123)
    api_test(env, num_cycles=1200, verbose_progress=False)
