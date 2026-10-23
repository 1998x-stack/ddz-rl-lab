# v4.0 — Role-aware Deep Monte Carlo training baseline

## Scope

This is the first **actually runnable** neural reinforcement learning stage. It
includes four role-specific PyTorch networks, candidate scoring, seeded self-play,
terminal Monte Carlo targets, one synchronous learner, frozen random-opponent
paired-seed evaluation, integrity-checked local checkpoints, Loguru tracing and
basic restart support. It is **not** a replica of the full DouZero algorithm or
its distributed asynchronous actor-learners. No pretrained skilled model is
included, and a passing smoke test does not imply the policies are strong.

## Role routing

`bid` is shared for both CALL and ROB before landlord is locked; `landlord` is the
landlord once PLAY begins. With clockwise seats `0 -> 1 -> 2 -> 0`, the first
farmer after landlord is `farmer_down` and the other `farmer_up`.

The model only receives `PlayerObservation` and one of its own exact
`legal_actions`. It never sees `GameState.hands` or hidden bottom cards. A state
encoder produces a single state embedding; an action encoder scores a batch of
legal candidates. No dense 56,251-action softmax is needed. Scores are uncalibrated
estimated returns, **not win probabilities**.

## Training objective

- Four independent role networks, one shared schema (`ddz-candidate-mlp-v1`).
- Each sampled decision stores only `(observation, selected action, role, seat)`.
- At normal episode end all decisions by that seat receive terminal reward from
  `DDZEnv.rewards()`: landlord +2 / -2; each farmer -1 / +1.
- Optimize MSE(Q(s,a), terminal_return) for all decisions accumulated during a
  small group of complete self-play episodes; Adam with gradient clipping.
- `epsilon` controls legal random exploration, using a seeded Python RNG.
- Truncated episodes do not enter gradient updates. Invalid actions are never
  silently repaired and bubble up as errors.

This is *on-policy-ish Monte Carlo regression with exploration*, not PPO or
Q-learning with bootstrapped TD targets. The initial implementation deliberately
uses sequential actor rollouts and one synchronous learner. Future releases can
add worker processes, policy version lag management, opponent populations, richer
sequence encoders and statistically powered tournaments.

## Commands

```bash
pip install -e '.[dev,rl,train]'
pytest -q

ddz-train train --episodes 200 --batch-episodes 8 --eval-every 40 \
  --eval-games 32 --seed 2026 --hidden 128 --output runs/dmc-v4 --log-dir logs

ddz-train evaluate --checkpoint runs/dmc-v4/best.weights.pt \
  --seed 7800 --eval-games 200

ddz-train train --checkpoint runs/dmc-v4/latest.pt \
  --episodes 100 --batch-episodes 8 --eval-every 40 --eval-games 32 \
  --seed 2026 --hidden 128 --output runs/dmc-v4
```

`episodes` on resumed training means **additional episodes**. The optimizer,
model state, Python exploration RNG, PyTorch CPU RNG, and episode index are
restored. A resume must retain core training hyperparameters; unlike fully
preemptible distributed training, partial minibatches at the interruption
boundary are not saved. Model architecture and both v3 encoding versions are
strictly checked. The model file has a SHA-256 sidecar; `torch.load` explicitly
uses `weights_only=True`. Load only vetted, trusted model artifacts.

## Evaluation and checkpoint selection

At each `eval_every` interval, the candidate is evaluated on the *same seeded*
deals as (1) learned landlord against frozen random farmers and (2) learned
farmers against frozen random landlord. Bidding in both cases uses seeded random
legal actions. Report both win rates separately and their balanced mean. The
`best.weights.pt` bundle is updated only when this **validation** score improves;
`latest.pt` always stores a resumable training checkpoint. This tiny fixed
validation set can overfit through repeated selection: report a separate
unseen test seed window before claiming any real improvement. For large-scale
study use paired-seed confidence intervals, multiple baselines and cross-play.

The best checkpoint contains all four policies from the same training iteration,
not an independently optimized trio. No production deployment yet.

## Operational observability

Use `--log-dir logs` to enable rotated compressed Loguru JSONL; training logs
learner losses, evaluations and checkpoint creation. Game events remain the
source of truth for replay. Do not expose private hand information in public
logs or WebSocket payloads. All stage-4 outputs under `runs/` are gitignored.

## Current limitations

- Synchronous single-process actors; no distributed sampling or GPU cluster.
- MLP, no LSTM/Transformer history model or recurrent experience state.
- Bid network trained from full-game outcome; no bidding-specific intermediate
  value estimation or role-conditioned lookahead.
- Short smoke runs establish correctness, not converged game strength.
- Checkpoint eval vs random opponents, not a multi-model tournament.
- No website or API; reserved for later phases.
