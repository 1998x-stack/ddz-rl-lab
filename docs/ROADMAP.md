## v9.0 implemented

Foreground experiment runner (JSON specs and checkpoints), configurable Early Stopping, run comparison,
seeded three-role Cross-Play, optional single-operator authenticated Web (including WebSocket).
See [V9_EXPERIMENTS_AND_SECURITY.md](V9_EXPERIMENTS_AND_SECURITY.md).

## v10.0 candidates (not implemented)

Persist and securely recover unfinished Arena sessions; bounded background job scheduler;
checkpoint lineup heatmaps and stronger evaluation robustness; experiment lineage.

# 分阶段 Code Category / Package Roadmap

目标是保证规则内核长期稳定，后续训练和可视化系统单独演进。目录代表计划中的模块名，尚未存在的目录不视为已实现。

| Stage | Code Category | Package/Folder | Deliverables | Acceptance gates |
|---|---|---|---|---|
| P0 | `specs` | `docs/` | RuleSpec, contracts, ADR, stage roadmap | Explicit variant decisions; versioned semantics |
| P1 | `game_core` | `src/ddz_core/` | cards, patterns, state, bidding, legality, event trace, replay | Unit tests, integrity invariants, deterministic replay |
| P2 | `game_env` | `src/ddz_env/` | AEC API, reward, observation encoder, action codecs, vector runner | PettingZoo API tests, no private info leakage, truncation semantics |
| P3 | `rl_models` | `src/ddz_models/` | role-aware networks, candidate-action scoring, inference API | Shape/gradient tests; train-serving encoder parity |
| P4 | `rl_training` | `src/ddz_training/` | self-play actors, learner, trajectory buffers, config | Baseline learning vs random, restart/resume, throughput metrics |
| P5 | `rl_evaluation` | `src/ddz_eval/` | frozen baseline pool, tournaments, checkpoint promotion | Paired-seed evaluations and cross-play robustness |
| P6 | `inference_server` | `apps/api/` | FastAPI session API, model registry, WS event stream | Illegal actions rejected; stable version pinning, reconnection |
| P7 | `web_console` | `apps/web/` | Next.js Arena, training dashboard, replay inspector | Three models finish a game and replay accurately |
| P8 | `observability` | `src/ddz_ops/`, `deploy/` | Loguru aggregation, metrics, packaging, security checks | Multi-process logging, resource limits, deployment smoke test |

## Planned dependency graph

```text
cards/patterns ──> rules/state ──> game_core API
                                    │
                  ┌─────────────────┤
                  ▼                 ▼
              game_env          replay/eval
                  │                 ▲
                  ▼                 │
               rl_models ──> rl_training
                  │               │
                  └────> model_registry <──── eval
                                │
                                ▼
                     inference_server
                                │
                                ▼
                           web_console
```

**Core must never import PyTorch, FastAPI or browser UI.** `ddz_env` uses observation projections, not privileged state serialization. Every RuleSpec schema change requires a fresh `rule_hash` / encoder version and explicit compatibility review.

## Delivery convention

- `P{n}` milestone branch/tag; independent `CHANGELOG.md` for major API changes.
- Tests for each stage must run without requiring the later stages.
- Distinguish `TrainingCheckpoint` from immutable `InferenceModelBundle`.
- Public game events and authorized internal traces have different access controls.
- Never market a future module as implemented until code plus tests exist.

## Non-goals

- No integration with official game clients or automation of third-party matches.
- No wager/virtual-currency/payment/odds or gambling mechanics.
- No unlicensed commercial artwork or assets.
- No unsupported claim that all current proprietary app variants are reproduced.

## Versioned implementation progress

- `v1.0` (P0/P1) **done**: core rules, game state, replay and tests.
- `v2.0` **done as specification**: ENV interfaces and branch promotion design (not executable P2).
- `v3.0` (P2) **done**: compact multi-agent ENV, candidate-action features, fixed global catalogue, PettingZoo AEC bridge, reward rules, deterministic baselines, verification.
- `v4.0` (P3/P4) **done (baseline scope)**: role-aware candidate-scoring MLP, synchronous DMC self-play, trajectory collection, resumable checkpoint, seeded random-opponent evaluation. Distributed actors and full model tournament remain future work.
- `v5.0` delivered: multi-process local actors, bounded replay, history pool and bilateral tournament.
- `v6.0` delivered: deal-cluster paired tournaments, operator-gated immutable model registry.
- `v7.0` delivered: local FastAPI/WebSocket real-time registered three-agent spectator Arena, controls, redacted public snapshots, replay download and tests.
- `v8.0` delivered: local read-only experiment dashboard for training JSONL, model registry, tournament reports, validated historical replays and bounded Loguru summaries.
- Later **not done**: production authentication, continuous streaming training updates, persistent multiuser Arena and multi-host cluster.

All new branches must be created from the prior release and promoted to main only after tests pass; prefer fast-forward updates and do not force overwrite history.


## v5.0 delivered

Local spawn-based parallel actor sampling, replay buffer, history snapshots, throughput metrics, and bilateral checkpoint tournaments. v6+ should consider clustered CI for comparative evaluation, CUDA learner scaling, model registry and a Web Arena.

## v6.0 delivered

- `ddz_eval.statistics` — deal-level clustered confidence intervals and paired differences, not independence-violating per-seat intervals.
- `ddz_eval.selection` — multi-seed cross-play runner, ranking, promotion recommendation with min sample threshold.
- `ddz_registry` — operator-owned immutable inference model store and active pointer; checksum, compatible network/encoder and explicit review gate.
- `ddz_eval.cli` — `ddz-league` CLI to evaluate/register/promote.

Remaining: version-pinned WebSocket serving, graphical Arena, operational dashboard, better learned bidding, multi-candidate adjusted statistical selection, and nontrivial model convergence benchmarks.

## v7.0 delivered

- `ddz_web.arena`: per-game actor scheduling and immutable per-seat registered policy selection.
- `ddz_web.api`: local FastAPI and WebSocket public snapshots, control API and replay downloads.
- `ddz_web.static`: responsive accessible spectator UI; no Tencent assets.
- `ddz_web.cli`: `ddz-arena` loopback launch; Loguru persistence.

Next: more robust on-disk session lifecycle, server auth for optional LAN deployments, read-only dashboards with training metrics, and larger tournament benchmarks.

## v8.0 delivered

- `ddz_web.lab`: bounded read-only artefact index, structured metrics, reports, replay verification and restricted logs.
- `ddz_web.static/lab.js`: training charts, registry, tournament comparison, verified public replay scrubbing and diagnostic summaries.
- New REST GET APIs and CLI folder flags. No background learner control or internet-facing authentication.
