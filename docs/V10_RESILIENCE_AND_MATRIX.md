# v10.0 — Durable Arena, bounded training jobs, model role-matrix

## What is actually implemented

`v10.0` builds on `v9.0` without replacing the rule engine, hidden-information observation API, DMC learner, immutable inference registry, or protected Web Arena. The incremental additions are:

| Code area | Feature | Persistence / protocol |
|---|---|---|
| `ddz_web.persistence` | Journal unfinished and finished Arena decisions with seed, pinned model hashes, RNG states, full legal `Action` and state digest | Atomic, fsynced private JSON in `runs/arena-replays/sessions/` |
| `ddz_web.arena` | Verify and replay session journal after restart; recover **paused**, never resume autoplay without explicit command | Existing `/api/games/{id}` works after restart |
| `ddz_experiments.jobs` | SQLite queued local training, duplicate experiment guard, bounded workers, immutable job spec fingerprint, conservative crash handling | `runs/jobs/jobs.sqlite3` + per-job private logs |
| `ddz_experiments.job_worker` | Child process validates SHA-256 and the existing strict JSON experiment schema before training | `ddz-experiment`'s existing run/resume logic |
| `ddz_eval.matrix` | Cartesian product of role-specific model candidates on shared deal seeds, pairwise deal-bootstrap summaries | `ddz-crossplay-matrix` JSON report |
| `ddz_web.lab` | Read-only metadata for jobs and role matrices; no remote arbitrary command runner | `/api/lab/jobs`, `/api/lab/matrices`, `/api/lab/matrices/{name}` |

## Install / release check

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,rl,train,web,webtest]'
python -m pytest -q
node --check src/ddz_web/static/lab.js
```

Python 3.10+ on Ubuntu and macOS; `fcntl` ensures the scheduler is limited to Unix-like OS. No wagers, real money, external game automation, Tencent assets, or official API integration.

## Arena restart correctness

```bash
ddz-arena --host 127.0.0.1 --port 8765 --replay-dir runs/arena-replays --allow-baseline
```

1. A session's *creation* is journaled before it becomes available via the API. At every successful action, its exact `Action`, canonical engine state digest and three actor RNG states are atomically fsynced to disk **before** publishing the next WebSocket snapshot.
2. Each game journal is `0600` inside a `0700` folder. It is **not** in the public completed-replay listing. Hidden hands and seed remain hidden in HTTP and WebSocket snapshots until the game is complete.
3. On process restart, each session is reconstructed by replaying its saved actions through the rule engine; invalid action, mismatch in digest/completion flag, missing model ID, or mismatched immutable bundle hash rejects restoration.
4. In-progress sessions always return in `paused` state. The operator must explicitly press Play; background autoplay is **never silently restarted**.
5. A completed session is eligible for verified canonical replay export. `MAX_TURNS`, fixed maximum journal size and maximum session count still apply.

The journal is not a general distributed event database. Concurrent API server workers using the same replay directory are **not supported**; start **one** Uvicorn worker for a given sessions directory. If the process crashes during an action, the last durable committed action is authoritative. Corrupted journals are skipped and logged; do not auto-repair or discard originals without backup. For production, add lifecycle retention, database transactions, replication and cross-process locks.

## Persistent training job queue (explicit supervisor)

Example spec: `examples/v9_smoke.json` (uses the `ddz-experiment-v1` schema). Submit jobs:

```bash
ddz-jobs --runs-dir runs --jobs-dir runs/jobs submit --spec examples/v9_smoke.json
# The command returns a job_id.
ddz-jobs --runs-dir runs --jobs-dir runs/jobs list
# Run in a dedicated operator terminal or a manually configured systemd service.
ddz-jobs --runs-dir runs --jobs-dir runs/jobs work --parallel 2 --once
# Without --once, the foreground supervisor continuously polls the queue.
ddz-jobs --runs-dir runs --jobs-dir runs/jobs status <job_id>
```

- **Do not use the example spec ID twice** unless explicitly continuing training using `--resume` and compatible settings. The queue prevents two active jobs for the same experiment ID.
- `--parallel` is constrained to 1–16. Jobs spawn separate Python subprocesses; they are not distributed across machines. `--once` drains existing queued work and exits; the default `work` command continuously polls.
- Each spec is strictly validated at enqueue time and checksummed again in the child. Arbitrary shell commands are never accepted. Output is always restricted to the configured runs directory by the v9 experiment runner.
- New supervisor takes an exclusive OS lock. Any old `running` entries become `interrupted` rather than being silently requeued. If a supervisor is killed unexpectedly, orphaned workers may remain alive, so check OS processes/checkpoint status before manual reconciliation. This is conservative **at-most-once automatic dequeue**, not distributed exactly-once execution.
- A job may refer to `--parent-experiment completed-experiment`, which must exist and have `finished` status. The queue records `parent_experiment`, immutable spec SHA-256, run ID, start/end and exit status. Read with `ddz-jobs ... lineage completed-experiment`.
- Don't assume each job has a new trained strong model. The worker only runs the requested, bounded training experiment. The existing registry and promotion gate remain separate.

## Cross-Play role matrix

After registering trusted inference models:

```bash
ddz-crossplay-matrix --registry-dir runs/registry \
  --landlords lord-a,lord-b \
  --farmer-ups up-a,up-b \
  --farmer-downs down-a \
  --seed 2026 --deals 50 --draws 500 \
  --report runs/reports/v10-matrix.json
```

This evaluates 2×2×1=4 combinations using common initial-deal seeds. Bidding is the v9 fixed-seed random baseline, and each registered model supplies its role-conditioned **playing** policy. The output contains each lineup's landlord/farmer win rates and paired-bootstrap intervals for the **landlord-outcome difference** between complete triplets. **Do not treat this as a causal comparison of individual players:** changing a farmer model affects its teammate and opponent. There is no automatic promotion; use the frozen-opponent `ddz-league` tournament and independently held-out fixtures for that.

Bounds: ≤64 combinations, ≤1,000 deals per combination, ≤64,000 total lineup-games, 100–10,000 bootstrap draws. Run larger studies in partitions. Matrix report can be viewed read-only in the Web Console's Cross-Play Matrix tab.

## Web access and security

Two new read-only tabs, `Training Jobs` and `Cross-Play Matrix`, supplement the existing six. They receive only bounded status/results, never database queries, private session journal, training config, arbitrary paths or raw secrets. The v9 operator cookie gate covers these endpoints. An exposed Internet server still requires trusted HTTPS reverse proxy, access controls and operational hardening; keep localhost as default.

## Verification gates

- 97+ existing and new tests: restart state/RNG reproducibility, digest tamper rejection, hidden-seed isolation, queue idempotency, worker smoke, interrupted-job safety, metadata authorization, deterministic matrix and input bounds.
- CI: Python 3.10/3.12/3.13 core tests, PyTorch integration, new v10 smoke; wheel includes the updated frontend.
- Performance caveat: fsync every action prioritizes crash consistency over throughput. The Arena is for spectating, not high-frequency distributed self-play.

## Not implemented / follow-ons

Multi-node worker scheduling, job cancellation/retry with idempotent checkpoints, complete web-based training control, multi-operator RBAC, distributed Arena session ownership, automatic promotion, statistically adjusted multi-candidate model selection, and a converged expert-level model are **not** included.
