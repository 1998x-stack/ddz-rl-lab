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
- Later **not done**: tournament/robust selection, FastAPI/WebSocket real-time triplet arena, Loguru training dashboard.

All new branches must be created from the prior release and promoted to main only after tests pass; prefer fast-forward updates and do not force overwrite history.
