"""Operator CLI for local Arena. Use a protected reverse proxy for any remote access."""
from __future__ import annotations

import argparse
import os
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
    parser.add_argument('--jobs-dir', default='runs/jobs', help='read-only job queue metadata root')
    parser.add_argument('--reports-dir', default='runs/reports', help='read-only tournament reports root')
    parser.add_argument('--allow-baseline', action='store_true',
                        help='enable NON-learning random policies for development only')
    parser.add_argument('--allowed-host', action='append', help='trusted HTTP Host header value; repeatable')
    parser.add_argument('--operator-token-env', help='environment variable holding secret (>=24 chars)')
    parser.add_argument('--secure-auth-cookie', action='store_true', help='set when serving via HTTPS')
    args = parser.parse_args()
    token = os.environ.get(args.operator_token_env) if args.operator_token_env else None
    if args.operator_token_env and not token:
        parser.error('operator token env variable is missing/empty')
    if args.host not in {'localhost', '127.0.0.1', '::1'} and not token:
        parser.error('non-loopback bind requires --operator-token-env and HTTPS reverse proxy')
    configure_logging(args.log_dir)
    uvicorn.run(create_app(args.registry_dir, args.replay_dir,
                           allow_baseline=args.allow_baseline,
                           runs_dir=args.runs_dir, reports_dir=args.reports_dir,
                           log_dir=args.log_dir, jobs_dir=args.jobs_dir, operator_token=token,
                           secure_auth_cookie=args.secure_auth_cookie,
                           allowed_hosts=['localhost', '127.0.0.1', 'testserver',
                                          *(args.allowed_host or [])]),
                host=args.host, port=args.port)


if __name__ == '__main__':
    main()
