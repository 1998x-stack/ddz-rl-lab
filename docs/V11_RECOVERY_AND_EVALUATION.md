# v11.0 — Local training recovery and statistical guardrails

## Source modules

- `src/ddz_experiments/jobs.py`: v10/v11 SQLite migrations; job queue and manual resume; worker runtime bounds and GPU slot assignment.
- `src/ddz_experiments/jobs_cli.py`: resource flags and explicit `retry` command.
- `src/ddz_experiments/runner.py`: interrupted experiment status reconciliation.
- `src/ddz_eval/matrix.py`: role-triplet matrix, paired initial-deal bootstrap and multiple-comparison diagnostics.
- `tests/test_v11_reliability.py`: schema upgrade, no-checkpoint refusal, exact remaining episodes, CPU/GPU resource settings, process-group timeout and CI calculations.

## Job lifecycle

`queued → running → succeeded|failed`, with separate `interrupted` on supervisor loss. Only one local supervisor holds the OS file lock; each job is claimed in a SQLite IMMEDIATE transaction. Processes run in dedicated sessions. On an explicit timeout, the supervisor sends SIGTERM to the process group, escalates after grace period, and records exit code 124. The supervisor log is per job and privately created.

If a supervisor crashes while jobs are running, orphan processes may continue running even without their original parent. On restart, the supervisor **refuses to start** while it can still observe any recorded worker PID; the operator must reconcile the process before proceeding. Once the old PID is absent, its job becomes `interrupted`, *not requeued*. No automatic retry of an ambiguous job.

`ddz-jobs retry JOB_ID` is operator-initiated and allowed for a failed or interrupted job only. It reads the original validated spec, checks experiment metadata and the trusted `latest.pt` plus SHA-256 sidecar, and computes `remaining=original_requested_episodes - (checkpoint_episodes - invocation_baseline)`. The new job is queued with `resume=1` and `retry_of` pointing to the original job. This is **at-least-once work since the last checkpoint**, not exactly-once execution of all samples. Jobs without a valid saved checkpoint are refused.

Since `experiment.json` is not atomically updated at each learner evaluation, a crashed experiment may report an older episode count. The learner checkpoint's verified `episodes_seen` is authoritative on resume; if it is less than stored known progress, or a finished experiment mismatches the checkpoint, resume fails.

## Isolation boundaries

`--cpu-threads N` (1–32) configures worker numerical thread environment variables. `--gpu-id N` (0–15) is mandatory when the training spec selects CUDA and mutually exclusive among jobs launched by this supervisor. The subprocess receives `CUDA_VISIBLE_DEVICES=N`. `--timeout-seconds S` allows 0 (unlimited) to 86400. Hardware availability is validated by the underlying PyTorch runtime, not assumed by the scheduler. The limits do not constitute cgroup CPU/memory isolation, GPU VRAM partitioning, hardware tenancy security, or inter-host coordination. Secure multi-tenant deployment is out of scope.

## Matrix inference policy

Every role-triplet uses the same initial deal seeds. For each pair, the model differences are bootstrapped by initial deal (not by individual seat decision), preserving within-deal dependence. `ci` is an **unadjusted** 95% percentile bootstrap interval; `familywise_ci` uses Bonferroni-tail bounds for the total number of pairwise comparisons, conditional on adequate draw resolution. If fewer than two bootstrap samples fall in the corrected tail, it is suppressed (`null`) and the report indicates insufficient resolution. If fewer than 30 common completed deals exist, `eligible_for_comparison=false`.

These are descriptive approximations, not proofs of strong play or guaranteed frequentist coverage. Outcome truncation and the random bidding baseline remain limitations. The matrix runner supports **at most 16 triplets**, matching `run_crossplay`'s explicit bound; attempting more is rejected. The model registry and manual v6 champion promotion are not changed.

## Operational acceptance

1. Run the full test suite on supported Python versions (the PettingZoo-specific tests require the RL extra).
2. Run the one-job queue smoke with CPU threads and timeout, then verify its `latest.pt` and job state.
3. Exercise crash recovery on a **copy** of experimental data; never edit training metadata on a production run.
4. Verify that two queued GPU jobs sharing a slot are not concurrently claimed.
5. Use sufficient independent deals and bootstrap draws before comparing role combinations.

No money, account payment, wagers or Tencent-service integration is included.
