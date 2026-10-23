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
