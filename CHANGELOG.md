# Changelog

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
