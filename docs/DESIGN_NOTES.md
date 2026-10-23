# Architecture decisions and risk review

## ADR-001: Python standard-library game core

Stage 1 uses Python dataclasses, enum, functools caching, and deterministic `random.Random`. No torch, NumPy, FastAPI, game loop timer or network socket is imported into `ddz_core`.

**Rationale:** Strictly isolate rule correctness from GPU/runtime concerns; tiny CPU-only local test loop.

## ADR-002: Two card representations

Unique physical IDs are used for deck conservation, audit and animation. Fifteen-rank vectors are used to generate suit-invariant actions. Suit choice is canonicalized to the lowest physical IDs in the current actor's hand.

**Rationale:** Prevent identical physical-suit choices from exploding the RL action space. Preserve enough physical identity for replay.

## ADR-003: Action identity includes pattern type

A 15D card multiset can occasionally admit more than one legal interpretation. Deduplication uses `(counts,pattern_key)` rather than only counts.

**Rationale:** Avoid silently dropping strategic actions from a future RL policy.

## ADR-004: Public event vs privileged replay

No unrevealed hands are ever placed in `GameEvent.public` or `PlayerObservation` for another seat. Full seed + action replay is privileged research data because it can reconstruct the entire initial shuffle.

**Rationale:** Hidden information is part of the game's learning objective.

## ADR-005: Deterministic seeded redeals

Each deal uses SHA-256 of `(seed, deal_index)` to derive the RNG seed. All-pass is recorded as an event and then a new deal is generated, while history and step counters remain continuous.

**Rationale:** Ensures reproducible all-pass histories with no hidden mutable RNG dependency.

## ADR-006: Stage 1 no resource contention assumptions

Legal-action generation uses a bounded LRU cache for rank counts; duplicate pattern semantic choices remain separate. Before adopting native code, benchmark candidate generation and scoring separately.

**Rationale:** Avoid premature C++/Rust extensions when profile results are unknown.

## Risks and limitations

1. The exact live product rule variant is not public/stable; freeze and test a documented classic reference instead.
2. Airplane repeated single wings and four-with-two-single variants differ by table rules; exposed as RuleConfig flags, not hidden assumptions.
3. GameEvent is in memory only; later add persistence, indexed snapshots and spectator permissions.
4. Python `random.Random` is seeded for reproducibility, not cryptographic fairness for adversarial online environments; online sessions will require a dedicated randomness design.
5. `GameEngine.state` intentionally exposes privileged state to trusted local developers; future API must not serialize it for Agent or spectators.
6. No guaranteed cross-Python-build bitwise determinism contract; record runtime image and code commit in future.
7. The random demo is not a reinforcement-learning baseline with measured competitive strength; it is a functional smoke test.
8. No real-time deadline policy, network reconnect or model-serving system in P1.
