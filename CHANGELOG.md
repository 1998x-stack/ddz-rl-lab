## v12.0 — Training quality and reproducibility

- Add `ddz-quality analyze`: bounded, validated evaluation/throughput JSONL parsing, per-run anomaly flags, independent-seed aggregation and seed-cluster bootstrap only for verified-compatible runs.
- Add matched-deal farmer role 2x2 interaction analysis with complete-case filtering, paired bootstrap CI, and explicitly exploratory interpretation.
- Add history archive SHA-verified guarded pruning, safe dry-run default and `history_keep_top` retention for best-scoring archival snapshots; protect latest, best and registry weights.
- Extend training CLI with `--history-keep-top`, preserve backwards compatible default and validate changed config on resume.
- Add CLI, documentation, v12 regression tests and CI smoke checks.
- No learned bidding change, no automatic checkpoint champion promotion, no new Web auth or multi-host scheduling.

## v11.0 — explicit crash recovery, local resource control and safer matrix statistics

- Migrate existing v10 SQLite job queues in place; preserve earlier jobs and expose CPU, GPU, timeout and retry lineage metadata.
- Explicit `ddz-jobs retry`: verify `latest.pt` SHA-256 and resume only the uncompleted episodes. Never retry missing or corrupt checkpoints automatically.
- Fix stale `experiment.json` episode counters after a crash: the verified learner checkpoint is authoritative.
- Refuse concurrent supervisors when an orphan worker PID is still present; terminate timed-out process groups.
- Per-job thread environment limits, CUDA visibility selection and GPU-slot exclusivity among jobs in a single scheduler.
- Matrix evaluation supports at most 16 lineups (consistent with the underlying runner); includes paired-deal diagnostics and Bonferroni-aware bootstrap comparisons, suppressing adjusted intervals when draw resolution is inadequate.
- New docs and six targeted tests; rules, ENV, model registry, Arena and manual promotion unchanged.
- No cgroup memory enforcement, multi-host scheduling, automatic retry, guaranteed model convergence or online wagering.

## v10.0 — durable Arena + bounded training jobs + role Cross-Play matrix

- Atomic private Arena journal on creation and every successful action; model SHA pin,
  RNG-state continuation, deterministic digest-verified restart; unfinished games recover paused.
- SQLite bounded local subprocess supervisor (`ddz-jobs`) with duplicate guard,
  immutable spec checksum, parent experiment lineage, safe interrupted-job handling.
- Role Checkpoint Cross-Play Cartesian matrix (`ddz-crossplay-matrix`), common deal seeds,
  pairwise clustered intervals; no auto model promotion.
- Read-only operator-protected Training Jobs and Cross-Play Matrix Web tabs/APIs.
- v10 test coverage and CI smoke; release wheel contains updated static assets.
- Not included: multi-host scheduler, distributed Arena transactions, automatic retries,
  public gaming accounts or guaranteed strong RL models.

## v9.0 — Experiment management, role combination research, optional operator gate

- Add `ddz_experiments` with JSON RunSpec validation, foreground execution, conservative resume and persisted status.
- Add configurable evaluation-based early stopping with checkpointed patience counter.
- Add three-role seeded Cross-Play CLI for registered, verified bundles.
- Add read-only two-run comparison endpoint and Web Training tab comparison panel.
- Add optional operator token login, expiring MAC cookie, HTTP/WS guard and trusted Host configuration.
- Add regression tests and updated CI/wheel smoke for v9 components.
- Keep historical features and no-stakes research scope unchanged. Not a claim of stronger RL performance.

# Changelog

## v8.0 — read-only RL Research Console

- Six-tab web UI combining Arena, training metrics, model registry, Tournament results, completed replays, and Loguru summaries.
- Bounded read-only REST API over explicitly configured local directories; identifier validation and symlink rejection.
- Integrity-verified historical replay reconstruction: no unrevealed hands, initial seed or raw private state emitted.
- Training/evaluation JSONL score and throughput charts, tournament ranking and paired confidence intervals.
- New CLI flags (`--runs-dir`, `--reports-dir`) and runtime documentation; v7 Arena controls remain compatible.
- Comprehensive regression and security tests, CI integration and frontend wheel static resources.
- No internet authentication, training orchestration, automated promotion, financial wagering, or claimed model convergence.


## v7.0 — three-model Web Arena

- FastAPI local spectator HTTP API + WebSocket current-state streaming.
- Three SHA-verified immutable registered inference bundles pinned per game, one policy per seat; no arbitrary weight uploads.
- Operator-only opt-in non-learning baseline mode, explicitly labelled.
- Browser UI for starting a new game, play/pause/step/speed, public event feed, and replay download.
- Public snapshots exclude hidden hands and shuffle seed until game completion.
- Deterministic action replay persisted atomically after terminal/truncated games.
- Loguru action context, session cap, local-host/origin guards, backend and browser packaging tests.
- No production authentication, public multiuser infrastructure, game client integration or claim of strong AI ability.


## v6.0 — multi-seed evaluation and local model registry

- Deal-clustered bootstrap confidence intervals and paired candidate–incumbent differences.
- Deterministic multi-seed tournament with complete fixture validation, model ranking and conservative recommendation gate.
- Immutable local registry for trusted inference-only bundles; SHA-256, encoder, schema and safe `weights_only` validation.
- Explicit first champion bootstrap; subsequent promotions require operator-reviewed, checkpoint-hash-bound tournament evidence.
- `ddz-league` CLI for evaluate/register/bootstrap/list/verify/promote; dedicated tests and CI smoke.
- No automatic model serving, learned-bidding comparison, multiple-testing correction, Web UI or new trained strong model.

## v5.0 — local multi-process DMC and tournament

- Spawn-safe parallel Actor workers and frozen snapshots per collection batch.
- Seeded, capacity-bounded experience replay persisted in latest training checkpoints.
- Rolling historical policy pool for play-only frozen opponents.
- Structured throughput and evaluation metrics, plus Loguru trace context.
- Two-sided seeded checkpoint tournament with Wilson descriptive intervals.
- CLI flags, extended documentation, regression tests and CI parallel smoke.
- Does not implement multi-host distributed training or a Web UI.


## v4.0 — P3 / P4 trainable DMC baseline

- Four role-specific PyTorch candidate-action MLP networks (bid, landlord, farmer_up, farmer_down).
- Synchronous complete-episode legal self-play, terminal Monte Carlo targets, Adam optimization.
- Atomic learner checkpoints and inference bundles with integrity metadata, strict encoder compatibility and resume.
- Seeded bilateral model-versus-random evaluation and validation-based best bundle selection.
- Loguru training/evaluation observability, new `ddz-train` CLI, tests and CPU-only CI training smoke.
- Explicitly *not* distributed actor-learner, PPO, converged policy, web server or GUI.


## v3.0 — P2 runnable environment

- `ddz_env.DDZEnv`: deterministic sequential multi-agent environment, local candidate action indexing, per-seat observations, terminal team rewards, and replay export.
- Versioned normalized observation and action features, public/private boundary and fixed feature dimensions.
- Global fixed semantic action catalogue and optional PettingZoo AEC adapter with action masks / sparse legal IDs.
- `ddz-env-demo` seeded three-agent baseline rollouts; extended tests and CI smoke run.
- Detailed environment documentation, constraints and stage roadmap.
- No model training, GPU requirement, cash/equivalent currency, or remote Tencent integration.

## v2.0 — Environment design

- P2 ENV specification and guarded main promotion workflow.

## v1.0 — Rules foundation

- P0/P1 deterministic three-seat game rules, call/rob, legal patterns, Loguru, CLI, replay and tests.
