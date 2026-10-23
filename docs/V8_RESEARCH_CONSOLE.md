# v8.0 — Read-only Research Console (P7)

## Scope

Version 8 adds the *operator-local* research dashboard on top of the verified v7 Arena without changing the authoritative game rules, learned policies, optimization algorithm, champion selection, or the existing Arena REST/WebSocket contract. The browser does **not** trigger training, load arbitrary checkpoints, mutate the champion, or connect to Tencent.

| Tab | Backing store | Capability |
|---|---|---|
| Arena | in-memory `ArenaGame` + snapshots | Existing three-agent game creation, play/pause/step/speed |
| Training | `runs/<run_id>/throughput.jsonl`, `evaluations.jsonl` | Balanced score and throughput charts, last evaluation summary |
| Models | immutable `registry.json` | Show registered models and active identifier; no UI promotion |
| Tournament | operator-generated `*.json` | Ranked checkpoints, confidence intervals and promotion evidence |
| Replays | `replay.v1` action traces in replay directory | Validate final state digest and reconstruct public frames |
| Loguru | `app-*.jsonl` under log directory | Bounded, read-only log summaries; no raw exception traceback |

## Local startup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,rl,train,web,webtest]'
python -m pytest -q

ddz-arena --allow-baseline --host 127.0.0.1 --port 8765 \
  --runs-dir runs --reports-dir runs/reports \
  --registry-dir runs/registry --replay-dir runs/arena-replays --log-dir logs
```

Open `http://127.0.0.1:8765/` and switch among six tabs. The random baseline is *not* an RL model. Use `ddz-train train --output runs/my-run ...` for new training samples. Produce tournament reports with `ddz-league evaluate ... --report runs/reports/compare.json`. Training can occur separately: clicking the dashboard does not start a learner.

## API contract

All endpoints are **read-only GET**, and return JSON unless specified:

- `GET /api/lab/overview` — runs, reports, finished replays, registered models.
- `GET /api/lab/runs` — shallow direct-subfolder scan of run directories.
- `GET /api/lab/runs/{run_id}/metrics?limit=300` — tail of evaluation/throughput JSONL, at most 500 rows per stream.
- `GET /api/lab/reports` and `GET /api/lab/reports/{report_id}` — named validated tournament reports.
- `GET /api/lab/replays` and `GET /api/lab/replays/{replay_id}` — finished replays only; detailed response contains reconstructed *public* frames and verified state digest.
- `GET /api/lab/logs?limit=120` — compact Loguru level/time/message, max 200 messages.

## Data and security invariants

1. The server accepts only tight resource identifiers; filesystem paths cannot be supplied as user-specified API arguments.
2. Reads are restricted to explicit local directories. Symlinked files/subdirectories are not followed. An individual report/replay is limited to 2 MB; JSONL/log tails are limited to 512 KB, with bounded row counts.
3. Before returning replay frames, re-execute every recorded action with the same versioned rules, then compare the terminal digest. No original hidden hands, full raw state, seed or checkpoint payload are emitted in the *new v8 replay API*.
4. Tournament confidence intervals are displayed as computed in v6; this dashboard does **not** make new statistical claims or promote models automatically.
5. This remains a **local-operator-only service**. Read endpoints are not gated by origin and are NOT protected by authentication; origin checks alone are not authorization. Never bind a public interface without placing a proper authenticated gateway in front and reviewing data exposure.
6. Malformed or missing metrics and reports degrade to empty or explicit error states. The dashboard never executes a checkpoint or parses untrusted Python pickle bytes.

## Development gates

`tests/test_v8_lab.py` covers JSONL parsing, path traversal, symlinks, corrupted replay digests, final-state visibility, REST contracts and cross-origin write protection. Legacy `v1.0`–`v7.0` tests remain in CI. The frontend uses DOM `textContent` for displayed external data (no untrusted `innerHTML`). The `lab.js` asset is included in the wheel package.

## Known limitations

- Charts are periodic manual refreshes, not continuous event streams.
- State and metrics are not backed by a production database; the server scans local artifact directories.
- Old v7 replay download API remains available to the local operator. The v8 historical replay tab returns only public projections.
- Only `replay.v1` traces from completed games are supported. Large/broken files are rejected.
- No authentication, multi-user persistence, remote training orchestration, or proof of high-skill AI play is provided.
