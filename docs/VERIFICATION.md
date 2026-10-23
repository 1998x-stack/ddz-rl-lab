# Verification report — Stage P0+P1, v0.1.0

Validation carried out in the build environment on Python 3.13, Loguru 0.7.3, pytest 9.0.2.

## Test gates

- `python -m pytest -q` : **43 tests passed** (see actual run before publishing archive).
- Deterministic replay: `replay(export_replay())` final SHA-256 digest equality.
- Full three-seat simulation with `ddz-demo --games 3 --seed 2026`.
- Error paths: illegal action rejected, all-decline redeal, max_redeals truncation, observer event mutation isolation.

## Build caveats

- This environment has restricted internet, so PEP 517 build isolation initially attempted to download build dependencies and failed. `pip install --no-build-isolation --no-deps -e .` succeeded with installed setuptools/wheel; ordinary networked environments can use documented `pip install -e '.[dev]'`.
- Python 3.10/3.12 CI workflow is included but not executed in this environment.
- This report validates implemented game mechanics against documented house rules, not byte-for-byte parity with any proprietary online game.
- No RL algorithms, checkpoint artifacts, online service or browser code have been implemented in this stage.
