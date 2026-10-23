"""Optional FastAPI facade; importing ddz_web.persistence must not require web extras."""
from __future__ import annotations

def create_app(*args, **kwargs):
    from .api import create_app as build_app
    return build_app(*args, **kwargs)

__all__ = ["create_app"]
