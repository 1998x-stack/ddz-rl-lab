# v6.0 Model Selection and Registry — Contract

## Scope

P5 research evaluation and immutable operator-managed inference registry. Does not
implement model serving, GUI Arena, true learned-bidding tournaments, or multi-host
training. All experiments are no-stakes and completely independent of Tencent.

## Input evidence

- Inference-only `*.pt` and sidecar `*.pt.json` emitted by `save_checkpoint`.
- Hash-verified and loaded with `torch.load(weights_only=True)` through the established
  checkpoint loader; four role-specific networks and matching v1 encoders.
- Arbitrary model code and network architecture plugins are not loaded.
- Models and opponents are pre-existing trusted local files, not remote uploads.
- All candidate and incumbent models must see exactly the same deals, in both roles,
  against each declared frozen opponent (plus deterministic legal random baseline).
- Bidding in the v5 tournament is seeded random for all sides. **This v6 release
  does not measure learned bidding-policy quality**.

## Score and statistical model

Each (deal seed, opponent, side) produces binary `won`. Define the per-deal
score as the mean across both sides and all opponents. Deal seeds are independent
sampling clusters by construction; the same seed is shared by all candidates.
Within-cluster role/opponent results are NOT treated as independent. For a
candidate versus incumbent, subtract their deal-cluster scores and bootstrap
these paired differences with `random.Random(bootstrap_seed)` for reproducible
95% percentile intervals. This is a description of uncertainty under the fixed
sampling protocol, not proof of calibrated playing strength.

- Reject duplicate deal/fixture rows, missing sides/opponents, overlapping seed
  ranges and truncated matches.
- Report ranked point estimates, confidence interval, number of independent deal
  clusters and paired delta vs incumbent.
- Default selection rule: at least 30 deal clusters and the lower bound of the
  paired candidate minus incumbent interval must be **strictly greater** than
  `min_delta` (default 0). No candidate passing = `recommended_id: null`.
- Selection among several challengers and repeated analysis reuses a holdout and
  risks selection bias; intervals are not adjusted for multiple testing. Reserve
  separate untouched confirmation seeds; do not present this as a guarantee.
- Entire training history/seed selection should be recorded by experiment owner.

## Registry invariants

- Model identifier may contain only safe ASCII alphanumeric, underscore, dash,
  and period (no traversal); registry entry is immutable after registration.
- Source weight and sidecar must reside inside an operator-specified trusted
  source directory, including after symlink resolution.
- SHA-256 checked before and after copying. Model schema, full network shape,
  observation and action encoder version must match the running code.
- A local file lock serializes modifying registry actions on macOS/Linux, and
  writes use atomic file replacement. A registry is **single-host**, not a
  distributed database.
- `active` is initially null; operator must explicitly `bootstrap` the first
  model. Every later `promote` requires `recommended_id` and a matching
  incumbent, model digests, minimum deal count and confidence gate.
- The report JSON is operator-managed evidence; it is NOT signed or tamperproof.
  Do not expose register, bootstrap or promote endpoints to untrusted users.
- Existing running game sessions must pin their exact bundle/model digest when
  serving is built later; a promotion must not hot-swap a mid-game policy.

## Output layout

```
runs/registry/
  registry.json                 # active name + immutable model metadata
  .registry.lock               # writer lock
  bundles/<model-id>/
    weights.pt                 # inference-only checkpoint
    weights.pt.json            # compatibility metadata + digest
```

`ddz-league evaluate` writes a JSON report with `candidates`, `ranked_ids`,
`recommended_id`, `model_hashes`, fixed `seed_starts`, `deals_per_seed`,
`opponents`, bootstrap configuration and interval method.

## CI gates

- Statistical deterministic/noisy/constant examples, duplicate & incomplete
  matrix detection, candidate–incumbent pairing, seed-range overlap.
- Real short tournament integration using v5 learner output.
- Checkpoint import and hash verification, untrusted directory and traversal
  rejection, immutability, changed incumbent and inconclusive evidence rejection.
- Whole prior regression suite, Python 3.10/3.12/3.13; CPU-only smoke.
