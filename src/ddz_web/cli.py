"""Operator CLI for local Arena. Use a protected reverse proxy for any remote access."""
from __future__ import annotations

import argparse
import uvicorn
from ddz_core.logging_config import configure_logging
from .api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description='Local three-agent DDZ RL spectator Arena')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--registry-dir', default='runs/registry')
    parser.add_argument('--replay-dir', default='runs/arena-replays')
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--runs-dir', default='runs', help='read-only training artifacts root')
    parser.add_argument('--reports-dir', default='runs/reports', help='read-only tournament reports root')
    parser.add_argument('--allow-baseline', action='store_true',
                        help='enable NON-learning random policies for development only')
    args = parser.parse_args()
    configure_logging(args.log_dir)
    uvicorn.run(create_app(args.registry_dir, args.replay_dir,
                           allow_baseline=args.allow_baseline,
                           runs_dir=args.runs_dir, reports_dir=args.reports_dir,
                           log_dir=args.log_dir), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
