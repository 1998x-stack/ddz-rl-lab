# DDZ RL Lab — v3.0

[![CI](https://github.com/1998x-stack/ddz-rl-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/1998x-stack/ddz-rl-lab/actions/workflows/ci.yml)

**Reproducible, no-stakes three-player Dou Dizhu research environment** for multi-agent reinforcement learning.

- `v1.0`: deterministic authoritative game core, legal rank-multiset action generation, call/rob rules, Loguru, canonical replays and tests.
- `v2.0`: P2 environment specification and safe-forward promotion workflow.
- **`v3.0`: runnable P2 multi-agent environment, candidate encodings, terminal team rewards, fixed global PettingZoo action catalogue, AEC adapter, CLI rollouts, tests and documentation.**
- **Not included:** DMC / PPO neural network training, checkpoints, model server, Web UI, money, wagers, payments, or Tencent integration.

> This is an independent rules-based educational simulator, not Tencent's proprietary code or a live mini-program replica. Rule divergences and versioned house rules are described in [`docs/RULE_SPEC.md`](docs/RULE_SPEC.md).

## Requirements

- Python **3.10+**, Linux/macOS (Windows should work but is not in CI).
- GPU is **not** required for v3.0. Training neural networks is a later stage.
- `loguru` is a base dependency; PettingZoo, Gymnasium and NumPy are optional RL extras.

## Quick start

```bash
git clone https://github.com/1998x-stack/ddz-rl-lab.git
cd ddz-rl-lab
git switch v3.0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,rl]'
python -m pytest -q

# Three seeded non-learning agents play 5 complete games.
ddz-env-demo --episodes 5 --seed 2026 --replay-out replays/example.json

# Original v1 engine CLI remains supported.
ddz-demo --games 3 --seed 2026
```

**Ubuntu 20.04:** use a Python 3.10+ environment; do not overwrite system Python. **Ubuntu 22.04/24.04:** use a supported interpreter. With offline environments without RL extras, install `-e '.[dev]'` and use `DDZEnv` directly; the PettingZoo-specific test file will skip.

## Code map

| Area | Import path | Responsibility |
|---|---|---|
| Rules | `ddz_core.cards` | 54 physical IDs, rank counts (15), selection |
| Patterns | `ddz_core.rules` | Canonical patterns, full legal candidate enumeration |
| Authoritative engine | `ddz_core.engine` | Call/rob/play phases, validity, public events, replay and invariants |
| Multi-agent environment | `ddz_env.environment` | Seat observations, sequential step, per-seat rewards, termination |
| Feature encoding | `ddz_env.encoding` | Normalized fixed-size observation and candidate-action vectors |
| Global catalogue | `ddz_env.catalogue` | Stable global IDs for PettingZoo Discrete action space |
| PettingZoo | `ddz_env.pettingzoo_env` | Optional `AECEnv` adapter with action mask |
| Baseline | `ddz_env.baselines` | Seeded random legal-action rollouts and CLI |
| Diagnostics | `ddz_core.logging_config` | Loguru rotating logs |
| Tests | `tests/` | Unit, random rollout, replay and PettingZoo API tests |

## Using the compact environment

The core `DDZEnv` intentionally has **no PettingZoo, Gymnasium or NumPy dependency**. At each decision it exposes the currently acting seat's redacted `PlayerObservation` and an exhaustive local candidate list. `step(index)` takes a **local, zero-based** index in that list.

```python
from ddz_env import DDZEnv, encode_observation, encode_action

env = DDZEnv(seed=2026)
while not env.done:
    actor = env.acting_seat
    obs = env.current_observation()
    features = encode_observation(obs)
    candidates = [encode_action(a) for a in obs.legal_actions]
    action_index = 0  # deterministic baseline, NOT a learned policy
    transition = env.step(action_index)
print(env.rewards(), env.engine.state.winner)
```

For realistic baseline behavior, use `ddz_env.baselines.play_episode` (non-learning randomized choices). Do **not** expose `env.engine.state.hands` to a policy — the engine contains hidden state for validation only. Every `PlayerObservation` deliberately omits opponent unrevealed hands and hidden bottom cards.

The reward convention is **landlord win = (+2, -1, -1)**, **farmer team win = (-2, +1, +1)**, assigned according to the actual landlord seat. Intermediate and truncated rewards are zero. This is non-monetary team utility. Normal completion sets `terminated=True`; a configured redeal safety cap sets `truncated=True`, with no fictitious winner.

## PettingZoo AEC integration

```python
from ddz_env.pettingzoo_env import env

arena = env(seed=2026)
for agent in arena.agent_iter(max_iter=1200):
    observation, reward, terminated, truncated, info = arena.last()
    if terminated or truncated:
        arena.step(None)
    else:
        # In actual training, use the policy to score candidates.
        # This demo selects the first legal global action id.
        arena.step(info["legal_action_ids"][0])
arena.close()
```

**Important action IDs:** compact `DDZEnv.step(local_id)` and PettingZoo `env.step(global_id)` use **different IDs**. The PettingZoo observation contains a standard fixed-size `action_mask`. `infos[agent]['legal_action_ids']` exposes sparse legal global IDs for efficient training; avoid allocating a dense mask in high-throughput actor workers.

The global catalogue has **56,251** semantic actions in the v1 classical ruleset (5 non-PLAY actions + 56,246 rank-multiset PLAY candidates). It is a *superset* of any single 20-card hand and thus has many never-live options. The catalogue is deliberately stable: changes to the action generator or house rules require new catalogue/encoder versions. This is not intended as a claim that all 56k actions are available in any one position.

## Encoding contracts

- **Cards:** 54 distinct physical IDs in the engine, 15 integer rank counts as model-facing semantics.
- **Observation:** normalized fixed-length vector, *only* the observing player's own hand and public data. Includes public per-seat played counts, public bottom cards after landlord assignment, lead, remaining counts, role/phase flags, bid markers, and current trick pass marker. No unexposed future cards.
- **Action:** action kind + count vector + pattern kind + main rank + chain length.
- **Candidate actions:** `obs.legal_actions` is an ordered immutable tuple; candidate order remains deterministic within a rules version.
- **Episodes:** `seed`, rule config, original actions, state digest, finished/truncated flag can be exported and checked with `ddz_core.engine.replay`.

Contracts and known limitations are documented in [`docs/V3_ENV_CONTRACT.md`](docs/V3_ENV_CONTRACT.md).

## Tests and CI

```bash
python -m pytest -q
python -m pytest -q tests/test_env.py
python -m pytest -q tests/test_pettingzoo_adapter.py  # requires .[rl]
```

GitHub Actions runs Python 3.10 / 3.12 / 3.13 with `[dev,rl]`. It runs the test suite and a 3-episode `ddz-env-demo` smoke test. Tests cover invariants, replay, malformed actions, observation secrecy, semantic fixed global IDs, terminal rewards, and AEC semantics. Do not claim RL training is implemented: these are **environment tests**, not learning-performance results.

## Logging and reproducibility

The baseline CLI accepts `--log-dir`; Loguru is configured through `ddz_core.logging_config`. Training in later stages should log `run_id`, `episode_id`, `rule_hash`, `policy_version` and latency without logging hidden opponent cards to normal application logs. Full event replay is a separate object and should not be reconstructed from rotated logs.

```bash
ddz-env-demo --episodes 10 --seed 42 --log-dir logs --replay-out replays/last.json
```

**Replay security:** exported offline replays include seed and action history and reveal the final complete game if reconstructed. Treat game seeds as internal state during live competition; do not send them to spectators or policies while a game is in progress.

## Roadmap / branch policy

`v1.0` is immutable P0/P1 reference, `v2.0` preserves P2 design, `v3.0` provides executable P2 env. Next: `v4.0` neural candidate-action scorer + DMC trainer + evaluation baseline; later: checkpoint selection and Web Arena. `main` should only be fast-forward advanced to the **latest tested release branch**. The promotion workflow is included under `.github/workflows/` and must have write permission to work automatically. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: pettingzoo` | Install `python -m pip install -e '.[rl]'` or use dependency-light `DDZEnv` |
| `IllegalAction` on step | Use the current legal candidate's **local** index, not its global catalogue ID |
| AEC invalid global action | Check `info['legal_action_ids']` or `observation['action_mask']` |
| No GPU | Expected; v3 is an ENV, not a neural trainer |
| Different replay digest | Match exact RuleSpec / version / seed and action sequence |
| CI checks Pending | Allow CI to complete and inspect Actions before promoting a release |

Licensed under MIT. Contributions should maintain deterministic legal action ordering and include regression tests for rules changes.
