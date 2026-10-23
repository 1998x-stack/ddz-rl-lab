# Data contracts — Stage 1

## Physical card IDs

- IDs 0–51: `rank=card_id % 13`, `suit=card_id // 13`.
- ID 52: small joker; ID 53: big joker.
- Rank vector length 15, order `[3,4,5,6,7,8,9,10,J,Q,K,A,2,SJ,BJ]`.
- Fixed deck capacity per rank `[4]*13 + [1,1]`.

## GameState (privileged)

`hands[3]`, `bottom_cards`, `played_by[3]`, `phase`, `first_bidder`, `current_seat`, bidding history/state, `landlord`, current play, pass counter, terminal result, events, seed/deal_index, step_index. Only a trusted rule engine may access this structure.

When `landlord is None`, reserved bottom is accounted outside hands; once landlord is fixed, the bottom cards remain as a public reference but their physical IDs are included inside landlord's hand (never double count).

## PlayerObservation (restricted)

`seat`, `own_physical_cards`, `own_counts`, `remaining_counts`, `public_bottom_counts` (zeros until revealed), public actions/played counts, current turn, public role/calling state, previous lead and the current seat's legal actions. No other hands. **A user-facing API must authenticate the requesting seat before serving this DTO.**

Observation is a Python in-process projection, not yet a secure network boundary. A future `PublicSpectatorView` must not send any hand data (including your own) unless explicitly authorized.

## Action

Non-play actions: `CALL`, `DECLINE_CALL`, `ROB`, `DECLINE_ROB`, `PASS`.

PLAY: `kind=PLAY`, `counts[15]` and `pattern_key=(pattern_type, main_rank, chain_length)`. The action is **suit-invariant**, and the rule engine deterministically selects physical IDs from the actor's hand. If identical rank counts admit multiple pattern interpretations, those are separate actions.

## GameEvent

Each event contains `sequence`, `kind`, `actor`, and `public` only. GameEvent never includes undealt deck order, hidden player hands or unrevealed bottom cards. Physical IDs of played cards and revealed bottom cards are public after the respective transition.

## Replay (privileged)

`schema_version`, `seed`, complete rules settings, all action objects, final state digest and terminal outcome. This is meant for local validation and later authorized analysis, not a player/spectator public HTTP endpoint.

Reconstruction: `replay(record)` creates the engine with seed/rules, applies actions, verifies SHA-256 digest. A game with the same seed but a different action sequence is not identical.

## ABI/API compatibility

- Stage 1: Python dataclasses with deterministic JSON action format. No promise of wire-format stability beyond `replay.v1`.
- Stage 2: Create dedicated DTO/encoder versions and explicit `terminated` vs `truncated` rewards for PettingZoo.
