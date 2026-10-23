#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e '.[dev]'
python -m pytest -q
ddz-demo --games 3 --seed 2026 --replay-out replays/demo.json --log-dir logs
