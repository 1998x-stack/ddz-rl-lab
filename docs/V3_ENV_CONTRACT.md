# V3.0 RL ENV Contract (P2 implementation)

## Scope and non-goals

An independently implemented deterministic three-player classic Dou Dizhu environment. P2 is a research simulator **without wagers or currency** and without neural training. Rules are frozen by `docs/RULE_SPEC.md` and `ddz_core.rules.RuleConfig`. This release is not an assertion about every 2026 commercial client's rules.

## Data ownership

- `GameEngine` contains all three hidden hands. Only orchestrators may access it.
- `PlayerObservation` contains only own cards, public information and current legal actions (for active seat). `encode_observation(PlayerObservation)` is the policy input; never pass raw `GameState` to a model.
- `DDZEnv.step(index)` chooses the *local* rank-multiset semantic action from the current legal candidate tuple.
- PettingZoo adapter accepts *global* IDs into a fixed action catalogue and maps into the local candidate list. Distinguish these explicitly in logs and wire contracts.
- Rule variant changes may require a new action catalogue and observation-schema version, not just a different user-facing version label.

## Normalization

All numeric observation entries are within `[0,1]`. Rank count divided by 4, remaining cards divided by 20, 3-seat index one-hots and null slots, phase one-hots, pass marker, and bounded redeal count. Candidate feature entries include kind, rank counts, pattern type, main rank, and chain length. Model-facing inputs intentionally do not encode physical suit because suit is semantically irrelevant in these classic rules.

**v3 limitation:** the fixed observation vector captures the current public aggregates, but not an explicit ordered temporal representation of all previous plays. Sequence-aware models must read and encode `public_events` from each seat's observation without using privileged state. This should use an explicitly versioned `HistoryEncoder` in v4+.

## Rewards

Terminal-only zero-sum team utility: landlord ±2, farmer ±1 each, with sign according to the winning team. Truncated games yield zero rewards. Because roles are assigned only after bidding, rewards are assigned by final seat-to-role mapping, not hard-coded seat 0/1/2.

## Determinism and replay

The engine derives deterministic shuffles from seed and deal index. Replay includes seeds, rules, every submitted action, canonical digest and end reason. A replay is full-information audit data, not a live spectator payload.

## Compliance checks

1. 54 physical card conservation after all legal state transitions.
2. Forced termination vs truncated must never be confused.
3. All emitted local/global actions belong to the current legal set.
4. Invalid action is rejected before any state mutation.
5. Observation for seat X never contains the unrevealed cards of seats Y and Z.
6. AEC spaces and masks have fixed shape, with only active agent mask populated.
7. Debug logs must not publish full internal hands or private seed to external viewers.

## Performance considerations

Global semantic catalogue is currently 56,251 actions. Allocating a 56,251-element dense `action_mask` per state is acceptable for PettingZoo interoperability but expensive at self-play scale. The high-throughput learner should consume a *variable-length candidate matrix*, sharing the state embedding across candidates. Avoid expanding all actions during GPU inference. Sampling/training algorithm should treat illegal actions as strictly impossible, not as negative-reward executable moves.

## Multi-role self-play boundary

V3 includes a randomly acting three-seat baseline, not a policy network. The future learner should route by *assigned role*, not fixed seat, and support bidding before the landlord role is known. Keep the initial PlayOnly experiment distinct from the FullGame experiment for valid metrics comparisons.
