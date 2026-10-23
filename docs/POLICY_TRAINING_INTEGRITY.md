# P0–P2 integrity iteration: implementation and evidence

## P0 correctness
- Core/ENV role assignment moved to `ddz_core.roles`: torch-free imports.
- Web-only tests explicitly skip when the optional FastAPI/Torch extras are absent.
- Golden tests cover a pair of Q failing to beat 2, pair 2 beating Q,
  bombs and rocket, illegal move immutability, and complete replay digest.
- Registry pins `policy_fingerprint` containing SHA-256 for weights, network
  schema, encoder names, RuleConfig, and installed rule/decision source bytes.
  A fingerprint mismatch blocks registry inference verification until the policy
  is re-evaluated. Legacy records without this field are not silently upgraded;
  treat their provenance as unverified.

## P1 policy and training
- Danger PASS override is conditional on landlord's remaining hand count being
  greater than 2; this is a diagnostic candidate, not an established stronger policy.
- New `ddz-team-normalized-pairwise-v1` targets map final team outcomes to ±1,
  keep abandoned auctions' configurable explicit opportunity cost, and use Huber
  return fitting plus rule-certified immediate-finish pairwise loss.
- Existing training checkpoints cannot resume under the new target contract.
  Inference-only historic weights still load under their declared network schema.
- `ddz-quality auction-audit --deals 100 --draws 2000` runs matched CALL vs
  DECLINE branches of the same initial deal and groups their differences by a
  transparent visible-hand strength heuristic; uncertainty is over seed clusters.
  This is a *counterfactual diagnostic against heuristic opponents*, not unbiased
  offline policy evaluation, and has no automatic promotion.

## P2 feedback and performance
- `POST /api/games/{id}/feedback` and `ddz-quality feedback` validate a
  historic decision by replaying authoritative actions. They accept a suspected
  issue or a legal alternative, and write an immutable private mode-0600 report.
  Every report is `unreviewed`; no submitted suggestion becomes an expert label
  or model input by default. A human adjudication process is a separate gate.
- `python scripts/benchmark_actor_pool.py --episodes 20 --workers 1 2 --seed 2026
  --output runs/reports/actor-benchmark.json` compares seeded actors under fixed
  weights; wall time, decisions/s, samples/s and approximate process RSS are
  reported. The command rejects nonmatching decision signatures.
- Review CI jobs and tournament results before moving main. Statistical gains
  have NOT been established by implementing these changes.

## Optional paired-auction teacher (P1 experimental)

`ddz-train train --model-schema ddz-candidate-mlp-v2 --bid-counterfactual-deals 2
--episodes 200 --batch-episodes 8 --output runs/bid-counterfactual-v2`

For each actor batch, the trainer uses up to two disjoint initial-deal seeds.
The first bidder's CALL and DECLINE outcomes are estimated by two complete
rollouts against **the same seeded frozen heuristic opponents**. The learner
receives only the initial bidder's own observation and the two encoded legal
actions. The hidden deck is used only to produce simulator teacher outcomes.
This applies to initial CALL decisions; ROB requires a separate future
counterfactual coverage expansion. Models must be measured on held-out seeds
because heuristic-teacher bias can make the performance worse.

The old DMC target contract uses per-role magnitudes; the new contract
normalizes terminal team outcomes to ±1 and adds narrow, rule-certified
immediate-finish ranking. Abandoned auctions still incur a configurable cost:
`--redeal-penalty`. Call-rate calibration should be compared by visible
hand-strength bins and on matched complete games. It is **not established** by
merely running the new training command.

## Release policy

- Core-only Python 3.10/3.12/3.13 and full torch/web Python 3.12 CI must pass.
- No action outside authoritative legal candidates may enter a training rollout.
- No previous unreviewed human suggestion is treated as a training preference.
- Never promote a changed policy hash without new, independent seeded evaluation.
- Passing legal replay tests and smoke training does NOT imply competitive
  improvement; archive all paired reports with their confidence intervals.
