# v5 Training, replay, population and tournament

## Scope and honest baseline

`v5.0` remains a **small, no-stakes, CPU-viable Deep Monte Carlo (DMC)-style baseline**, not an exact DouZero reproduction and not a production-grade multi-node distributed RL stack. The authoritative DDZ rules and player-visible observation encoder from `v3.0` are unchanged. No actual real-world wagers, tokens, Tencent integration or live external user games.

## Architecture

* Training driver owns a `RoleBundle` with four PyTorch candidate-Q models: bid, landlord, farmer-up and farmer-down.
* `actors.py` launches local `spawn` workers via `ProcessPoolExecutor` (`--actor-workers N`). Workers each initialize **CPU snapshots** of the current networks and frozen historical snapshots once per collection batch. Worker policy snapshots refresh each batch to bound policy staleness; no inference RPC or multi-host orchestration is implemented.
* Workers run `DDZEnv` locally, and only return `Experience(role, encoded_visible_observation, encoded_legal_action, seat, terminal_team_reward)`. They do not return the opponent's private hand or full internal GameState.
* `ReplayBuffer` evicts oldest decisions at `--replay-capacity` and samples up to `--replay-sample-size` experiences using a seeded trainer RNG. A checkpoint includes the bounded replay and RNG state; restoring from `latest.pt` also restores optimizer states.
* Learner performs role-wise Monte Carlo regression with MSE and clipped gradients on replay samples. This is **off-policy historical replay of terminal DMC targets**, not PPO, V-trace, prioritized replay, importance sampling or a provably convergent multi-agent algorithm.
* `history/step-NNNNNNNNN.weights.pt` snapshots provide a rolling frozen pool. When chosen, the learner controls only one randomly selected side after bidding and the historical opponent controls the other side. **All bidding remains learner self-play**. Only learner-controlled decisions enter replay. Pool selection is seeded and recorded via sampled episode metrics.
* `throughput.jsonl` records episodes/s, decisions/s, samples/s, worker count, replay size, truncated episodes and collection wall time. `evaluations.jsonl` records each fixed-opponent baseline score. Loguru app logs are separate from replay/game events.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,rl,train]'
OMP_NUM_THREADS=1 python -m pytest -q
```

## Training (2 CPU workers)

```bash
OMP_NUM_THREADS=1 ddz-train train \
  --episodes 200 --batch-episodes 8 --actor-workers 2 \
  --replay-capacity 20000 --replay-sample-size 1024 \
  --updates-per-batch 1 --history-capacity 6 \
  --history-probability 0.35 \
  --eval-every 40 --eval-games 32 --seed 2026 \
  --output runs/v5-experiment --log-dir logs
```

`--episodes` means **additional** episodes after a resume. Preserve seed, network shape, optimizer settings, replay parameters, worker count and history settings when resuming.

```bash
OMP_NUM_THREADS=1 ddz-train train --checkpoint runs/v5-experiment/latest.pt \
  --episodes 100 --batch-episodes 8 --actor-workers 2 \
  --replay-capacity 20000 --replay-sample-size 1024 \
  --updates-per-batch 1 --history-capacity 6 \
  --history-probability 0.35 --eval-every 40 --eval-games 32 \
  --seed 2026 --output runs/v5-experiment
```

## Tournament with external opponents

```bash
OMP_NUM_THREADS=1 ddz-train tournament \
  --checkpoint runs/v5-experiment/best.weights.pt \
  --opponent-checkpoint runs/v5-experiment/history/step-000000040.weights.pt \
  --eval-games 500 --seed 80000
```

Every opponent uses the **same seeded initial deals** for two candidate roles: candidate-landlord vs frozen-farmers and candidate-farmers vs frozen-landlord. Calling is performed by a fixed seeded random baseline, which means the tournament **measures play-phase strength conditioned on random bidding**, not joint learned call-and-play skill. Opponents include a fixed random baseline automatically and any explicitly added trusted model bundles. All model artifacts are loaded with `weights_only=True`, SHA-256 sidecar checks and encoder/schema guards; do not upload or load untrusted third-party PyTorch files.

Each role records games, wins, win rate, and an illustrative Wilson interval. Because both role trials share initial deals, outcomes are correlated; do not interpret independent-binomial Wilson intervals as a rigorous paired confidence interval. Formal promotion should later add clustered bootstrap over deals, score stability across opponent versions and independent holdout seeds.

## Determinism and caveats

* A fixed seed and a fixed CPU build gives deterministic starting deals, policy selection and replay sampling. Across CPU backends, threads, CUDA or differing PyTorch versions, exact floating-point and process timing equality are not guaranteed.
* Workers communicate frozen tensors and encoded decisions, **not models with GPU contexts or live ENV objects**.
* `spawn` is supported on Linux/macOS but can incur overhead for small experiments. For `--actor-workers 1`, inline sampling avoids process launch.
* Worker crashes fail the training batch rather than being silently treated as episode loss. Invalid/truncated episodes are excluded from replay.
* Frozen history retention uses a rolling count of checkpoint files; do not mistake it for Elo, TrueSkill or an automatically curated strong-opponent pool.
* Replay snapshots can be large. For large runs, use higher-capacity storage and consider tensor-backed replay and sharded checkpoint storage.
* No Ray, multi-host deployment, distributed parameter server, learned bidding benchmark, Web UI, PPO, or production serving API exists in this stage.
