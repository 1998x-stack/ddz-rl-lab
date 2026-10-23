# v12.0 — Long-running training quality, safe retention, farmer interaction

This release is an **offline research diagnostic layer**, not a claim of stronger DDZ play, nor an automatic checkpoint promotion mechanism. All evaluations use the existing classic ruleset; bidding during role-comparison cross-play remains a seeded random baseline.

## Architecture and code categories

| Category | Path | Contract |
|---|---|---|
| Experiment quality | `src/ddz_quality/analysis.py` | Read-only existing `evaluations.jsonl`, `throughput.jsonl`, `experiment.json`. Validates input, emits per-run diagnostics and seed-level confidence interval only if configurations can be compared and enough independent runs exist. |
| Farmer cooperation | `src/ddz_quality/cooperation.py` | 2x2 matched-deal factorial study of two farmer role replacements. Excludes all four arms of any truncated deal. Bootstrap resamples entire shared initial deals. |
| Archival retention | `src/ddz_quality/retention.py` | Only affects `history/step-NNNNNNNNN.weights.pt` + metadata files; sha256-verified; dry run by default. Protects newest N plus highest-scoring M. Never touches best, latest, registry. |
| CLI | `src/ddz_quality/cli.py` | `ddz-quality analyze`, `retention`, `cooperation`; reports saved with exclusive-create to prevent accidental overwrites. |
| Trainer integration | `src/ddz_training/trainer.py` | `history_keep_top=0` preserves v11 behavior. Optional best-M archive protection. No change to observation, action, reward, network or model encoders. |

## Independent training seeds

Train **separately initialized** runs, modifying only the `seed`, output and per-invocation episode budget. Keep all learning, model and evaluation settings identical; use a pre-registered holdout plan for final claims. Each run produces a final balanced score from the same frozen evaluator protocol. The report bootstraps whole runs, not repeated checkpoints, to avoid pseudoreplication.

```bash
# After independent runs have written evaluations.jsonl:
ddz-quality analyze --run runs/seed1 --run runs/seed2 --run runs/seed3 \
  --draws 2000 --min-runs 3 --regression-tolerance 0.10 \
  --report runs/reports/quality-v12.json
```

Validation: reject duplicated/out-of-order evaluation steps, malformed/nonfinite metrics, scores outside [0,1], bad throughput ratios, oversized evidence, unsafe symlinks. Alerts are diagnostic, **not proof of performance degradation**. A score drop from best to latest can occur through noise; repeat on untouched comparison sets before deployment decisions. If settings are unavailable/different or fewer than three independent runs exist, no seed-level confidence interval is reported.

## Archival checkpoint policy

```bash
# Read-only preview; verifies SHA-256 metadata for every candidate:
ddz-quality retention --run runs/seed1 --keep-recent 6 --keep-best 3
# Operator review first. This will actually delete only verified history snapshots:
ddz-quality retention --run runs/seed1 --keep-recent 6 --keep-best 3 --apply

# Optional automatic archive pruning after each training evaluation:
ddz-train train --episodes 100 --history-capacity 6 --history-keep-top 3 \
  --output runs/seed1
```

`history_keep_top=0` is the v11-compatible default. `best.weights.pt`, `latest.pt`, any file in the registry, and any non-matching archival file are **never** considered for pruning. A malformed or hash-mismatched checkpoint blocks deletion. Run archival pruning only while the experiment is stopped; concurrent actors may be using older snapshots. When resuming training, all immutable training parameters including `history_keep_top` must match the checkpoint.

The protected top-M snapshots are based on the **existing single-evaluator balanced score**. Selecting an archive for preservation is not the same as promoting a model. For the latter use the v6 independent tournament selection gate.

## Farmer cooperation interaction

```
                    down baseline       down candidate
up baseline         base_base           candidate_down
up candidate        candidate_up        both_candidates
```

All four fixtures have the same **frozen landlord** policy and initial seeded deals, the same bidding protocol, and the same deterministic inference setting. For deal `d` with win indicator `w`, compute

`interaction(d) = w(both) - w(up_only) - w(down_only) + w(neither)`.

The mean interaction across non-truncated matched deals is bootstrapped by deal, with a 95% percentile CI when at least 30 fully matched deals are available. A positive interaction suggests complementarity under this fixed setup. It **does not prove explicit cooperation, generalize to arbitrary opponents, or isolate bidding skill**. With several planned comparisons, use multiplicity protection and holdout confirmation before drawing strong conclusions.

```bash
ddz-quality cooperation --registry-dir runs/registry \
  --landlord landlord-v1 \
  --up-base farmer-up-v1 --up-candidate farmer-up-v2 \
  --down-base farmer-down-v1 --down-candidate farmer-down-v2 \
  --deals 200 --draws 4000 --min-deals 30 \
  --report runs/reports/farmer-interaction.json
```

All named models must already be registered/trusted. No arbitrary user-uploaded PyTorch model is loaded. This report is exploratory, never triggers an automatic promotion, and should not be interpreted as a real-money or monetary score.

## Test and release gates

`pytest -q tests/test_v12_quality.py`; full `pytest -q`; `python -m compileall -q src`; `ddz-quality --help` and CLI smoke; Python 3.10/3.12/3.13 and CPU PyTorch CI. Tests cover input corruption, determinism, sufficient seed count, unequal configs, interdependent farmer fixtures, bad checkpoint hash, symlinks, dry-run and guarded deletion, training archive integration. Maintains previous versions and moves `main` only via fast-forward after green CI.

## Known limitations / future work

- No automatic hyperparameter tuning or model-quality guarantee.
- Quality reports are offline, not a continuously running health watcher.
- Scores retain the frozen random-bidding baseline rather than evaluating learned bidding quality end-to-end.
- Percentile bootstrap is exploratory with only three seeds; more independent seeds and confirmation on new deals are recommended.
- Retention operations are local and not transactional across a crash between deleting a weight and its sidecar; a subsequent audit will fail closed.
- No multi-host job scheduler, no system-wide GPU memory quota, and no multi-user role-based access control.
