"""Optional single-operator token authentication for the self-hosted RL lab.

Cookies contain a timestamp, random nonce and HMAC (never the raw API token).
This is for a single operator behind a trusted HTTPS proxy, not full IAM.
"""
from __future__ import annotations
import hashlib
import hmac
import secrets
import time

COOKIE = 'ddz_operator'
TTL_SECONDS = 8 * 3600


class OperatorAuth:
    def __init__(self, token: str | None, *, secure_cookie: bool = False):
        if token is not None and (not isinstance(token, str) or len(token) < 24):
            raise ValueError('operator token must contain at least 24 characters')
        self.enabled = token is not None
        self.secure_cookie = secure_cookie
        self._token = token or ''

    def _signature(self, timestamp: str, nonce: str) -> str:
        message = f'ddz-operator-session-v1:{timestamp}:{nonce}'.encode()
        return hmac.new(self._token.encode(), message, hashlib.sha256).hexdigest()

    def issue_cookie(self) -> str:
        if not self.enabled:
            raise ValueError('no operator authentication configured')
        timestamp, nonce = str(int(time.time())), secrets.token_hex(16)
        return f'{timestamp}.{nonce}.{self._signature(timestamp, nonce)}'

    def valid_cookie(self, value: str | None) -> bool:
        if not self.enabled:
            return True
        if not isinstance(value, str) or len(value) > 180:
            return False
        try:
            timestamp, nonce, sig = value.split('.')
            sent = int(timestamp)
            age = int(time.time()) - sent
            return (0 <= age <= TTL_SECONDS and len(nonce) == 32 and len(sig) == 64
                    and secrets.compare_digest(sig, self._signature(timestamp, nonce)))
        except (ValueError, AttributeError):
            return False

    def login(self, candidate: str | None) -> bool:
        return self.enabled and bool(candidate) and secrets.compare_digest(candidate, self._token)
