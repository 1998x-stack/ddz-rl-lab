# V9 — Repeatable experiments, role Cross-Play, and operator authentication

## Scope and non-goals

V9 extends the existing standalone game engine and v8 Web research console without rewriting the RL model. It provides foreground experiment orchestration, optional early stopping, an audit-friendly two-run comparison, an experiment with three independently chosen role bundles, and an optional one-operator authentication gate. It **does not** provide a persistent distributed scheduler, user accounts, or crash-safe resumption of an unfinished Arena game. It does **not** make the DMC baseline a proven strong opponent.

## Installing

Python >=3.10; PyTorch requires the `train` extra. `pettingzoo` is optional except for AEC adapter tests.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,rl,train,web,webtest]'
pytest -q
```

## Foreground training experiments

`ddz-experiment run --spec examples/v9_experiment.json --runs-dir runs`

The spec is **JSON**, not Python or YAML with executable directives:

```json
{
  "schema": "ddz-experiment-v1",
  "id": "demo-v9",
  "training": {
    "episodes": 20,
    "seed": 2026,
    "hidden": 32,
    "batch_episodes": 2,
    "actor_workers": 2,
    "eval_every": 5,
    "eval_games": 4,
    "early_stop_patience_evals": 3,
    "min_score_improvement": 0.01
  }
}
```

Runs in the foreground and writes `runs/demo-v9/experiment.json`, `latest.pt`, `best.weights.pt`, training/evaluation JSONL and Loguru messages. A new run refuses nonempty output directories. `ddz-experiment status demo-v9 --runs-dir runs` reads the last persisted status.

To continue from the validated `latest.pt`, adjust `training.episodes` to the **additional** count and execute:

`ddz-experiment run --spec examples/v9_experiment.json --runs-dir runs --resume`

Resume requires exactly the same non-episode training overrides, and the original Trainer additionally validates the effective configuration. A crash can occur between checkpoint save and status write; such a stale manifest is **rejected** rather than silently guessing how much training completed. During active execution the JSON status says `running`; after an unclean system crash it may remain stale. No auto-respawn daemon runs in the background.

Early stopping is disabled when patience=0. When enabled, the counter increments for an evaluation not beating the current best by more than `min_score_improvement`; it resets upon improvement. Its state is checkpointed and restored. It is an experimental convenience, **not** proof of convergence. Use a held-out Tournament for selection.

## Two-run comparison

In Web > Training select two distinct runs and press **比较两个实验**, or call:

`GET /api/lab/compare?left=demo-v9&right=another-run`

Comparison reports last available evaluation and training throughput per run, plus the descriptive difference of final observed balanced scores. It does **not** calculate statistical significance from two partially comparable training logs; use the v6 deal-clustered Tournament for that. Missing metrics remain missing, never synthesized.

## Three-role Cross-Play

First register independently verified inference-only bundles using `ddz-league`. Then:

```bash
ddz-crossplay --registry-dir runs/registry \
  --lineup trial:lord-a,up-a,down-a \
  --lineup alternative:lord-b,up-b,down-b \
  --seed 2026 --deals 100 --report runs/reports/crossplay-v9.json
```

Each model supplies only its corresponding playing-role network; bidding uses the same seeded random policy, isolating play from auction policy changes. Shared seed integers are used across all lineups, and final outcomes are retained per deal. Results are descriptive and depend on the particular opponents inside each trio. They are **not** suited for champion promotion without the independent fixed opponents and significance gates of v6.

## Arena bidding mode

Arena offers a checked **随机叫牌，所选模型负责出牌** option when creating a game. It sends `random_bidding: true` to `POST /api/games`: the seeded non-learning baseline handles CALL and ROB, while the selected registered models still choose every PLAY action. The public snapshot and completed replay record this flag. Uncheck it to test the model's own bid network; the API default is `false` for existing clients. This option is useful when a weak bid network always declines and the game otherwise reaches the redeal safety cap. It does not improve the model's bidding ability or its tournament score.

## Optional operator login

By default the web app remains loopback-only and requires no login. For single-operator access, set a long random secret **in the server environment**, never in command arguments, URLs, front-end assets or Git:

```bash
export DDZ_OPERATOR_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(36))')"
ddz-arena --host 127.0.0.1 --port 8765 \
  --operator-token-env DDZ_OPERATOR_TOKEN --allow-baseline
```

The browser displays a password prompt. The raw token is sent once in an HTTPS (or local loopback) header; the server sets an 8-hour, HttpOnly, SameSite=Strict, signed cookie. API routes and WebSocket streams use the same gate. Logout clears the browser cookie; rotating the environment token invalidates previously issued cookies. Login rejects invalid tokens and Loguru does not log their values. `GET /api/health` and `/api/auth/status` remain public by design.

For **trusted HTTPS reverse proxy** setups, use `--secure-auth-cookie --allowed-host your.domain.example`. If binding to 0.0.0.0, the CLI requires the token, but that requirement alone is not a production-ready security architecture: use TLS, reverse-proxy rate limits, access rules, CSRF/origin restrictions, and trusted host allowlists. Avoid exposing this prototype directly to the public internet. Token authentication does not provide per-user permissions, multi-user auditing or password reset.

## Verification and data contracts

- Full rule/ENV/Replay compatibility regression tests.
- Deterministic Cross-Play evaluation for fixed seeds and identical weights.
- Read-only, path-confined experiment comparison.
- Forbidden output path, duplicate run and resume-config mismatch tests.
- Auth tests for unauthorized HTTP/WebSocket, invalid token, cookie tampering and expiry.
- Loguru logging is retained; engine truth remains independent of logs.

## Follow-on

V10 candidates: append-only, resumable Arena sessions with durable action sequence and model hash pinning; durable queued scheduler with crash recovery, strict concurrency limits and authenticated control endpoints; richer tournament heatmap and grouped confidence intervals; experiment lineage graph.
