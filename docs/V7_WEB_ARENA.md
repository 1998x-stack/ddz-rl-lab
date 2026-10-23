# v7.0 — Registered Three-Agent Web Arena

## Goal & boundaries

Local, no-stakes, spectator-first display of three RL-controlled seats in the same `DDZEnv` used for training. A seat picks an **operator-registered, inference-only** `RoleBundle`. For each move, the model receives `PlayerObservation` (seat-specific private hand + public state only), not `GameState`. The browser never receives hidden physical cards before the game ends. Selecting `baseline` requires explicit `--allow-baseline` and does **not** constitute RL inference.

The web app is not a WeChat/Tencent integration. No bets, payments, real/virtual chips, unlicensed game assets, online matchmaking, user accounts or external automation.

## Layer diagram

```
Browser [HTTP / WebSocket, public state only]
  |  /api/models /api/games /control /replay
FastAPI (Host + Origin protection; no model upload)
  |-- ArenaManager (max sessions; registry model cache)
  |-- ArenaGame (one asyncio task per autoplay game, async lock)
  |       -> per-seat observe() -> RoleBundle.act() -> env.step(index)
  |                                            \-> GameEvent public data
  |       -> subscriber bounded queues (snapshot coalescing)
  |       -> atomic replay JSON at completion
  |-- ModelRegistry (SHA-256, schema and encoder validation)
  +-- Loguru (action and error correlation)
```

Sessions are process-local in memory; replay persists to disk. Use **one uvicorn worker**. Multiworker backends require shared sessions and cross-worker event delivery and are not supported in v7.0.

## REST contract

- `GET /` — static app page (HTML, CSS, JS bundled with wheel).
- `GET /api/health` — health/schema.
- `GET /api/models` — active model, registered metadata, whether random baseline explicitly enabled.
- `POST /api/games` — `{ "model_ids": ["id0", "id1", "id2"], "seed": 123, "delay": 0.7 }`. Seed may be null; `delay` in `[0.05,3.0]`. Returns redacted spectator snapshot. `model_ids` are immutable for that session.
- `GET /api/games/{id}` — public snapshot.
- `POST /api/games/{id}/control` — `{ "action": "play"|"pause"|"step"|"speed", "delay": 0.7 }`. `step` requires playback paused.
- `GET /api/games/{id}/replay` — after completion only; returns `replay.v1` JSON + pinned model names. Seed never appears in in-progress public snapshots. The completed replay can be fed to `ddz_core.engine.replay` for state-digest verification.
- `WS /api/games/{id}/events` — current snapshot followed by snapshots after every state transition. `sequence` is the latest public `GameEvent.sequence`; clients reconnect via `GET /api/games/{id}`. Queue capacity is bounded and stale snapshots can be dropped, therefore `sequence` may skip: snapshots are authoritative latest state, not guaranteed every-event delivery.

Public snapshot includes phase, acting seat, landlord when known, remaining counts, last played rank counts, published bottom ranks after grant, public event suffix, winner and status. It explicitly excludes private hands, private observations, logits, and the shuffle seed while ongoing. Completed replays contain the seed and therefore reveal all cards to an authorized observer after the game; do not distribute pre-completion replays.

## Models and selection

1. Run `ddz-train train ... --output runs/research1` and confirm the inference-only output `best.weights.pt` and `.json` sidecar.
2. `ddz-league --registry-dir runs/registry register --model-id research1 --checkpoint runs/research1/best.weights.pt --trusted-source-dir runs/research1`
3. `ddz-league --registry-dir runs/registry bootstrap --model-id research1` (only for the first champion; later `promote` requires evidence).
4. `ddz-arena --registry-dir runs/registry --host 127.0.0.1 --port 8765 --log-dir logs`
5. Browse to `http://127.0.0.1:8765/`, choose three model IDs, create game, play or step. Multiple seats may use the same registered bundle; this is still three independently acting policies with separate observations.

Only SHA-verified compatible registry entries can load. Model files must be under the operator's trusted registry; weights are loaded using `weights_only=True`. The registry is never mutated via public HTTP.

## UI semantics

- `Play` schedules one legal decision at a time; model inference is offloaded to a thread.
- `Pause` stops scheduling further steps; an in-flight inference may finish.
- `Step` advances one legal action when paused (one step can produce multiple public events).
- `Speed` controls playback delay, not learned decisions.
- Seed input is deterministic if provided, though Torch inference may be platform-dependent when using certain nondeterministic algorithms.
- Q values are deliberately not shown as probabilities. This UI never claims the model is strong until evaluated on frozen independent benchmarks.

## Security / limits

- Only loopback is the supported deployment in v7.0. `TrustedHostMiddleware` defaults to localhost and 127.0.0.1; Origin matching rejects ordinary cross-site browser writes and WebSocket openings. Not a substitute for authentication. Never expose this listener publicly without a separately engineered authentication/authorization layer, rate limits, HTTPS and a hardened reverse proxy.
- Every game has a 1200-decision guard; 12 sessions max per server by default. This is an operational guard, not authoritative terminal scoring.
- Session cache is in memory, intentionally cleared on restart. Completed replay JSON is atomic and durable; log events are not themselves sufficient for deterministic replay.
- Registry updates do not mutate a running game; each game pins role-bundle object and metadata verified when created. This process has one worker; multiple server workers would violate the session model.
- Loguru logs record game id, seat and role. Debug logs do not stream hidden hands to clients. JSONL Loguru sinks rotate and retain according to existing central logging configuration.

## Testing / release gate

Run `python -m pip install -e '.[dev,rl,train,web,webtest]'` then `pytest -q`.

Included tests assert no shuffle seed or hidden hands in active snapshots; baseline opt-in; unknown-model rejection; registered checkpoint scoring; Host/Origin rejection; WebSocket state updates; step/pause logic; bounded sessions; and complete seeded replay hash verification. GitHub CI also runs registered-model HTTP smoke after the v6 model registry smoke. Python 3.10/3.12/3.13 lightweight jobs skip missing optional web/torch tests; the fully equipped 3.12 job must run them.
