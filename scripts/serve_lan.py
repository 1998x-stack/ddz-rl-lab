"""Serve the local operator Arena on a trusted LAN with HTTPS and HTTP redirect."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import uvicorn

from ddz_core.logging_config import configure_logging
from ddz_web.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description='HTTPS LAN Arena with an HTTP entry redirect')
    parser.add_argument('--host', required=True)
    parser.add_argument('--http-port', type=int, default=8765)
    parser.add_argument('--https-port', type=int, default=8766)
    parser.add_argument('--cert', type=Path, required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    args = parser.parse_args()
    token = args.token_file.read_text(encoding='utf-8').strip()
    if len(token) < 24:
        parser.error('operator token must have at least 24 characters')
    for path in (args.cert, args.key):
        if not path.is_file():
            parser.error(f'missing TLS file: {path}')

    target = f'https://{args.host}:{args.https_port}/'

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header('Location', target)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()

        def do_HEAD(self) -> None:
            self.do_GET()

    redirect = ThreadingHTTPServer((args.host, args.http_port), Redirect)
    threading.Thread(target=redirect.serve_forever, daemon=True).start()
    configure_logging('logs')
    app = create_app('runs/registry', 'runs/arena-replays', runs_dir='runs',
                     reports_dir='runs/reports', log_dir='logs',
                     operator_token=token, secure_auth_cookie=True,
                     allowed_hosts=['localhost', '127.0.0.1', args.host])
    try:
        uvicorn.run(app, host=args.host, port=args.https_port,
                    ssl_certfile=str(args.cert), ssl_keyfile=str(args.key),
                    timeout_graceful_shutdown=5)
    finally:
        redirect.shutdown()
        redirect.server_close()


if __name__ == '__main__':
    main()
