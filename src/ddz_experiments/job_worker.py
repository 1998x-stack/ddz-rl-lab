"""Dedicated subprocess entrypoint for a validated job; no arbitrary commands."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from .runner import ExperimentRunner, _read_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--runs-dir', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    path = Path(args.spec)
    if path.is_symlink() or path.stat().st_size > 32768:
        parser.error('invalid worker spec')
    spec = _read_spec(path)
    import json
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != args.sha256:
        parser.error('job spec checksum mismatch')
    ExperimentRunner(args.runs_dir).run(path, resume=args.resume)


if __name__ == '__main__':
    main()
