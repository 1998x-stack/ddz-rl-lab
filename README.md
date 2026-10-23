# DDZ RL Lab — v6.0

[![CI](https://github.com/1998x-stack/ddz-rl-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/1998x-stack/ddz-rl-lab/actions/workflows/ci.yml)

**Reproducible, no-stakes three-player Dou Dizhu research environment** for multi-agent reinforcement learning.

- `v1.0`: deterministic authoritative game core, legal rank-multiset action generation, call/rob rules, Loguru, canonical replays and tests.
- `v2.0`: P2 environment specification and safe-forward promotion workflow.
- `v3.0`: runnable multi-agent environment, candidate encodings, terminal team rewards, PettingZoo AEC, and test baselines.
- `v4.0`: runnable PyTorch role-aware candidate-action MLP, synchronous Deep Monte Carlo self-play, Loguru metrics, checkpoint save/resume, deterministic external-random-opponent evaluation, and CI smoke tests.
- `v5.0`: local multi-process actor sampling, bounded replay with restorable snapshots, rolling frozen historical opponents, throughput JSONL metrics, and two-sided tournament against historical checkpoints.**
- **`v6.0`: multi-seed paired tournaments with deal-cluster bootstrap, immutable inference model registry, and evidence-gated manual champion promotion.**
- **Not included:** multi-host distributed actor–learner cluster, production-ready strong models, PPO, model-serving API, Web UI, money, wagers, payments, or Tencent integration.

> This is an independent rules-based educational simulator, not Tencent's proprietary code or a live mini-program replica. Rule divergences and versioned house rules are described in [`docs/RULE_SPEC.md`](docs/RULE_SPEC.md).

## Requirements

- Python **3.10+**, Linux/macOS (Windows should work but is not in CI).
- CPU is sufficient for smoke training; CUDA may accelerate the learner but is not required.
- `loguru` is a base dependency. Install `.[train]` for PyTorch training and `.[rl]` for PettingZoo/Gymnasium.

## Quick start

```bash
git clone https://github.com/1998x-stack/ddz-rl-lab.git
cd ddz-rl-lab
git switch v6.0
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
| Training models | `ddz_models` | Role Q(s,a) network, private-safe candidate inference |
| Learner / Checkpoint | `ddz_training` | Spawn actor pool, bounded replay, MC learner, frozen pool, SHA-256 model bundles, resume |
| Evaluation | `ddz_eval` | Frozen random-opponent trials, seeded cross-checkpoint bilateral tournament, paired deal-cluster statistics and selection |
| Model registry | `ddz_registry` | SHA-verified immutable inference models and explicit champion pointer |
| Diagnostics | `ddz_core.logging_config` | Loguru rotating logs |
| Tests | `tests/` | Unit, random rollout, replay and PettingZoo API tests |

## Run real RL training (v5.0)

```bash
# Local multi-process actor-learner baseline (CPU compatible).
OMP_NUM_THREADS=1 ddz-train train --episodes 200 --batch-episodes 8 --eval-games 32 \
  --eval-every 40 --seed 2026 --actor-workers 2 --replay-capacity 20000 \
  --replay-sample-size 1024 --output runs/dmc-v5 --log-dir logs

# Load the best integrity-checked inference bundle.
ddz-train evaluate --checkpoint runs/dmc-v5/best.weights.pt \
  --seed 7800 --eval-games 200

# Continue from saved optimizer and RNG states.
ddz-train train --checkpoint runs/dmc-v5/latest.pt --episodes 100 \
  --batch-episodes 8 --eval-every 40 --eval-games 32 --seed 2026 \
  --actor-workers 2 --replay-capacity 20000 --replay-sample-size 1024 \
  --output runs/dmc-v5
```

Run a comparison against frozen checkpoints:

```bash
OMP_NUM_THREADS=1 ddz-train tournament --checkpoint runs/dmc-v5/best.weights.pt \
  --opponent-checkpoint runs/dmc-v5/history/step-000000040.weights.pt \
  --eval-games 500 --seed 80000
```

Read [v5 training, pool, replay and tournament design](docs/V5_TRAINING_AND_TOURNAMENT.md) before comparing models. Training loss is **not** game strength. A 2-episode CI run only proves functionality.

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

GitHub Actions runs Python 3.10 / 3.12 / 3.13 with `[dev,rl]`. It runs the test suite and a 3-episode `ddz-env-demo` smoke test. Tests cover invariants, replay, malformed actions, observation secrecy, semantic global IDs, terminal rewards, AEC semantics, training, checkpoint compatibility, and registry selection gates. CI smoke runs do not prove convergence or superior playing strength.

## Logging and reproducibility

The baseline CLI accepts `--log-dir`; Loguru is configured through `ddz_core.logging_config`. Training in later stages should log `run_id`, `episode_id`, `rule_hash`, `policy_version` and latency without logging hidden opponent cards to normal application logs. Full event replay is a separate object and should not be reconstructed from rotated logs.

```bash
ddz-env-demo --episodes 10 --seed 42 --log-dir logs --replay-out replays/last.json
```

**Replay security:** exported offline replays include seed and action history and reveal the final complete game if reconstructed. Treat game seeds as internal state during live competition; do not send them to spectators or policies while a game is in progress.

## Roadmap / branch policy

`v1.0` preserves P0/P1, `v2.0` preserves P2 design, `v3.0` provides the ENV, `v4.0` adds baseline neural DMC, `v5.0` adds local actor parallelism and replay, and **`v6.0` adds multi-seed model selection and a local registry**. Next: multi-host throughput and Web Arena. `main` should only be fast-forward advanced to the **latest tested release branch**. The promotion workflow is included under `.github/workflows/` and must have write permission to work automatically. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

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

## v5.0 performance and history files

- `runs/dmc-v5/throughput.jsonl`: per-collection wall-clock throughput, truncated episodes, replay size, actor count.
- `runs/dmc-v5/evaluations.jsonl`: deterministic random-baseline evaluation history.
- `runs/dmc-v5/history/step-XXXXXXXXX.weights.pt`: rolling policy snapshots used as optional frozen opponents.
- `runs/dmc-v5/latest.pt`: restorable optimizer, model, RNG and bounded replay state.
- `runs/dmc-v5/best.weights.pt`: compact best-on-random-baseline inference policy, **not** a claimed globally strongest agent.

The opponent pool is restricted to play decisions (bidding stays learner self-play). Workers refresh snapshots between batches. See [technical details](docs/V5_TRAINING_AND_TOURNAMENT.md).

## v6.0 — Multi-seed tournament, model registry and promotion

Install training extras: `python -m pip install -e '.[dev,rl,train]'`. Tournament comparisons use **complete initial deals** as the bootstrap sampling units; both roles and all frozen opponents share a deal cluster. `--seed-start` may be repeated for disjoint ranges. Only completed, non-truncated fixtures enter selection. More seeded games reduce uncertainty but cannot prove universal strength. The deterministic seeded bidding policy remains random, so the v6 tournament **evaluates only the play policies**, not learned bidding quality.

```bash
# Candidate and champion must be TRUSTED inference-only .pt checkpoint bundles
# accompanied by their .pt.json sidecars. Run separate training runs to obtain them.
ddz-league evaluate \
  --incumbent champion=runs/runA/best.weights.pt \
  --candidate challenger=runs/runB/best.weights.pt \
  --seed-start 810000 --seed-start 820000 --deals-per-seed 40 \
  --min-deals 50 --draws 2000 --report runs/selection/report.json

# Optional stronger frozen opponent: --opponent history=runs/runA/history/step-000000040.weights.pt
# Registration is an explicit operator action against a trusted source directory.
ddz-league --registry-dir runs/registry register \
  --model-id champion --checkpoint runs/runA/best.weights.pt \
  --trusted-source-dir runs/runA
ddz-league --registry-dir runs/registry register \
  --model-id challenger --checkpoint runs/runB/best.weights.pt \
  --trusted-source-dir runs/runB

# Only the first model can be activated without tournament evidence.
ddz-league --registry-dir runs/registry bootstrap --model-id champion
ddz-league --registry-dir runs/registry list

# Manually review the report. If and only if recommended_id == challenger:
ddz-league --registry-dir runs/registry promote \
  --model-id challenger --report runs/selection/report.json
```

Promotion is **not automatic**. The registry requires an immutable known model, verifies SHA-256 and encoder/schema compatibility, ensures the incumbent has not changed and checks a positive lower paired-confidence bound over `--min-deals`. Otherwise the incumbent remains active. Model ID = versioned registry ID, not Git branch. The evaluation JSON report is **operator-managed evidence**, not cryptographically signed; never expose registration or promotion actions to anonymous clients. No end-user model upload/serving is implemented.

The bootstrap is a 95% percentile interval by default. Initial deals, not seat outcomes, are resampled; when several candidates are selected using one holdout, the reported confidence intervals are exploratory, not multiple-comparison-adjusted. Use a new, untouched confirmation set before making high-stakes claims. Explicitly separated holdout seeds are the operator's responsibility.

See [`docs/V6_MODEL_SELECTION.md`](docs/V6_MODEL_SELECTION.md) for statistical limitations, registry layout, threat boundaries and test gates.
