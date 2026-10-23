# Classic Dou Dizhu Research RuleSpec v1

**Status:** Stage-1 implementable baseline, not a statement of all rules currently live in Tencent's mini program. Historical Tencent classic-game public rules (2016) motivated the call/rob structure; several variant-dependent details are explicitly chosen below.

## 1. Deck and turn order

- 54 unique physical cards: 52 suited cards (13 rank values × 4 suits) plus two unique jokers.
- Rank order: 3,4,5,6,7,8,9,10,J,Q,K,A,2,SJ,BJ.
- Deal 17 cards to each of 3 seats and reserve 3 bottom cards.
- Seat numbers advance `0 → 1 → 2 → 0`, defining the circular turn order (the UI can render it counterclockwise).
- Shuffle and first bidder are derived deterministically from a seed and deal index.
- Bottom cards are hidden until the landlord is determined, then publicly revealed and added to landlord's 17 cards.

## 2. Call and rob sequence

1. Starting from a seeded first bidder, each player can `CALL` or `DECLINE_CALL` in seat order.
2. The first `CALL` ends the calling phase; if all three decline, a new deterministic deal is drawn.
3. The first caller is initially the landlord claimant.
4. Rob opportunity order after calling is the two next seats, then the original caller; players who previously declined calling are ineligible. If the opportunity seat is already the claimant, the decision is skipped.
5. `ROB` transfers the claim to that actor; `DECLINE_ROB` does not. After the opportunity list is exhausted, the claimant becomes landlord.
6. We track `rob_count` for analysis but **do not compute money, virtual currency or stake multipliers**.
7. Every actual decision creates a public event. Automatic stage transitions also create public events.
8. A configured `max_redeals` safety bound ends with `truncated=True, winner=None`, never a fictitious winner.

**Variant note:** Tencent's historical text refers to each player's single rob opportunity but leaves some UI-specific priority/timing details implicit; this repository freezes the deterministic ordering above as a house-rule choice. It is not an assertion of the exact 2026 client flow.

## 3. Play and pattern semantics

The landlord leads. Each player can follow with the same pattern and total size at a strictly higher main rank, or beat with a bomb/rocket. The last winning play is cleared when the next two players consecutively PASS; the same winning player leads a fresh trick. Leading player cannot PASS.

Supported: SINGLE, PAIR, TRIPLE, TRIPLE_SINGLE, TRIPLE_PAIR, STRAIGHT (≥5), PAIR_CHAIN (≥3 pairs), TRIPLE_CHAIN (≥2 triples), AIRPLANE_SINGLE, AIRPLANE_PAIR, FOUR_TWO_SINGLE, FOUR_TWO_PAIR, BOMB, ROCKET.

- Straight, pair chain and triple chain main ranks use only 3–A.
- Triple / quadruple groups can include 2 but not jokers.
- 3+single/3+pair kickers may not share the main triple's rank.
- Airplane uses a consecutive sequence of ≥2 triples; exactly one non-main single or pair wing per triple.
- Pairs used as wings have distinct ranks; jokers are not eligible to form pairs.
- `repeated_airplane_single_wings=True`: repeated single wing ranks are permitted as long as the available hand contains the copies, but no wing may share any main triple rank. Alternate variant is explicitly configurable.
- `repeated_four_two_singles=True`: e.g., 5555+88 is supported. Alternate variant is explicitly configurable.
- `forbid_double_joker_as_wings=True`: cannot use both jokers as the two single wings in one combination.
- FOUR_TWO_SINGLE/PAIR is **not** a bomb; a BOMB is exactly 4 equal cards on its own.
- ROCKET beats everything; BOMB beats any ordinary pattern and weaker BOMB.
- Pattern ambiguity is retained as separate actions using `(type, main_rank, chain_length)` keys; the action includes the complete 15D multiset.

## 4. Winner and ending

If landlord empties hand first, winner = LANDLORD; if either farmer empties hand first, winner = FARMERS. There is no draw for a normally finished game. `truncated` means a technical/safety bound, not a team win. No stake, settlement or coin logic.

## 5. Out of scope

- LaiZi / wildcards, no-shuffle decks, revealing one player's entire hand (明牌), timeouts, spectator auth, social interactions.
- Monetary or quasi-monetary scoring, doubling for stake, in-app purchases or accounts.
- Call-rob priority subrules not expressly frozen above.

## 6. Rule changes

Any change to wing restrictions, bidding order, legal actions or observations must use a new RuleSpec version and invalidate incompatible model/trajectory bundles. Runtime `RuleConfig.rule_id` is a descriptive version field; future phase P2 adds canonical SHA-256 config and code build IDs.
