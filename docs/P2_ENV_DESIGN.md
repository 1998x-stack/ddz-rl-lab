# P2 Implementation Contract — Multi-Agent RL Environment

**Branch:** `v2.0` (created from `v1.0`).
**Status:** design specification only; PettingZoo adapter, reward code, and tensor encoders are **not implemented in this commit**.

## Objectives

Expose the already tested `ddz_core.GameEngine` through a PettingZoo **AEC** environment while preserving information-set integrity and identical game semantics in CLI, training, evaluation, and online inference.

1. AEC environment routes turns by `GameState.current_seat` and never allows simultaneous play.
2. Actions use dynamic legal candidates with stable deterministic ordering.
3. Observation built from `GameEngine.observe(seat)` — NEVER privileged `GameState`.
4. Bidding (CALL, DECLINE_CALL, ROB, DECLINE_ROB) and play (PLAY, PASS) share versioned action schema.
5. Game terminates only after LANDLORD/FARMERS winner. Max-redeal is `truncated`, not a loss.
6. Terminal rewards are non-monetary: landlord win gives (2,-1,-1); farmer win gives (-2,1,1); truncated sessions have no tournament result.

## Package layout (proposed)

```text
src/ddz_env/
  __init__.py
  config.py               # EnvConfig, rule and encoder versions
  aec.py                  # AEC adapter
  observations.py         # only information sets, no private state
  action_codec.py         # canonical structured action <-> candidate ID
  rewards.py              # terminal team rewards
  registration.py         # PettingZoo/Gymnasium integration
  wrappers.py             # optional training wrappers
  vector_runner.py        # efficient CPU batched simulation
tests/
  test_env_aec.py
  test_env_observations.py
  test_env_rewards.py
  test_env_actions.py
  test_env_replay_equivalence.py
  test_env_performance.py
```

## Core contracts

### Reset

`reset(seed, options)` generates a new deterministic 3-seat game; optional fixed deal and snapshot support require fully validated trusted inputs. P2 must not silently change P1 shuffle, bidding, or play rules.

### Agent order

The acting agent is the current physical seat, not the landlord-relative role. After bidding finalizes, map seat to `LANDLORD`, `LANDLORD_UP`, `LANDLORD_DOWN` for future policies. Verify the mapping against turn direction.

### Observation boundary

- Private: self hand only.
- Public: player remaining counts, prior public plays, public bottom after reveal, bidding history, last leading pattern.
- Legal actions: only nonempty for the acting seat.
- Never include `seed`, `bottom_cards` before reveal, opponent hand IDs, or raw replay digest in a policy observation.
- Encode rank counts as stable 15-vector with normalized role and phase features; persist `observation_schema_version`.
- Keep candidate enumeration complete; candidate ID is LOCAL to the current observation and never a globally stable game move ID.

### Action semantics

Out-of-range or stale candidate ID is invalid and MUST NOT be coerced into PASS. Every legal action maps back to P1 `Action` with exact `pattern_key`. If the last successful play is cleared after two PASS actions, leading player cannot PASS.

### Reward & Done semantics

`terminated=True` only after ordinary winner determination. `truncated=True` for external limit / redeal safety cutoff. A replay must recover the same winner, actions and hash as P1's engine.

### Performance

Keep all game rules CPU-local to actors; candidate encoder may cache deterministic states. No remote request per ENV step. Use batched tensor conversion with masks, and limit debug logs under multi-actor training.

## P2 acceptance criteria

- PettingZoo `api_test` and (where applicable) seed/replay validation pass.
- Run ≥1000 seeded random full games, with zero illegal executed moves and zero card-conservation failures.
- Observation leakage tests compare two hidden worlds with the same public information and acting-player hand; observations must be equal.
- Output candidate actions exactly match authoritative `GameEngine.legal_actions()`, preserving their `pattern_key`.
- Stable `observation_schema_version` and `action_schema_version` guard incompatible models.
- Training and core replay agree on terminal rewards and complete-state digest.
- Unit tests continue to pass on Python 3.10, 3.12 and 3.13.

## Out of scope for P2

No RL network weights, distributed learner, Web UI, online accounts, wagers, payment flows or integration with external game clients. Those belong to later independently tested stages.

## Publishing gate

Do not promote any future P2 implementation to `main` until local tests and GitHub CI pass. `main` may track a documentation-only `v2.0` planning commit while P2 implementation is still pending; documentation must clearly identify that status.
