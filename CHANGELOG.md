# Changelog

## v7.0 — three-model Web Arena

- FastAPI local spectator HTTP API + WebSocket current-state streaming.
- Three SHA-verified immutable registered inference bundles pinned per game, one policy per seat; no arbitrary weight uploads.
- Operator-only opt-in non-learning baseline mode, explicitly labelled.
- Browser UI for starting a new game, play/pause/step/speed, public event feed, and replay download.
- Public snapshots exclude hidden hands and shuffle seed until game completion.
- Deterministic action replay persisted atomically after terminal/truncated games.
- Loguru action context, session cap, local-host/origin guards, backend and browser packaging tests.
- No production authentication, public multiuser infrastructure, game client integration or claim of strong AI ability.


## v6.0 — multi-seed evaluation and local model registry

- Deal-clustered bootstrap confidence intervals and paired candidate–incumbent differences.
- Deterministic multi-seed tournament with complete fixture validation, model ranking and conservative recommendation gate.
- Immutable local registry for trusted inference-only bundles; SHA-256, encoder, schema and safe `weights_only` validation.
- Explicit first champion bootstrap; subsequent promotions require operator-reviewed, checkpoint-hash-bound tournament evidence.
- `ddz-league` CLI for evaluate/register/bootstrap/list/verify/promote; dedicated tests and CI smoke.
- No automatic model serving, learned-bidding comparison, multiple-testing correction, Web UI or new trained strong model.

## v5.0 — local multi-process DMC and tournament

- Spawn-safe parallel Actor workers and frozen snapshots per collection batch.
- Seeded, capacity-bounded experience replay persisted in latest training checkpoints.
- Rolling historical policy pool for play-only frozen opponents.
- Structured throughput and evaluation metrics, plus Loguru trace context.
- Two-sided seeded checkpoint tournament with Wilson descriptive intervals.
- CLI flags, extended documentation, regression tests and CI parallel smoke.
- Does not implement multi-host distributed training or a Web UI.


## v4.0 — P3 / P4 trainable DMC baseline

- Four role-specific PyTorch candidate-action MLP networks (bid, landlord, farmer_up, farmer_down).
- Synchronous complete-episode legal self-play, terminal Monte Carlo targets, Adam optimization.
- Atomic learner checkpoints and inference bundles with integrity metadata, strict encoder compatibility and resume.
- Seeded bilateral model-versus-random evaluation and validation-based best bundle selection.
- Loguru training/evaluation observability, new `ddz-train` CLI, tests and CPU-only CI training smoke.
- Explicitly *not* distributed actor-learner, PPO, converged policy, web server or GUI.


## v3.0 — P2 runnable environment

- `ddz_env.DDZEnv`: deterministic sequential multi-agent environment, local candidate action indexing, per-seat observations, terminal team rewards, and replay export.
- Versioned normalized observation and action features, public/private boundary and fixed feature dimensions.
- Global fixed semantic action catalogue and optional PettingZoo AEC adapter with action masks / sparse legal IDs.
- `ddz-env-demo` seeded three-agent baseline rollouts; extended tests and CI smoke run.
- Detailed environment documentation, constraints and stage roadmap.
- No model training, GPU requirement, cash/equivalent currency, or remote Tencent integration.

## v2.0 — Environment design

- P2 ENV specification and guarded main promotion workflow.

## v1.0 — Rules foundation

- P0/P1 deterministic three-seat game rules, call/rob, legal patterns, Loguru, CLI, replay and tests.
