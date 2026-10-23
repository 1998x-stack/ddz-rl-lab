"""Reproducible inference identity: weights + deployed rules + encoder + policy sources.

Changing a hard-coded PASS rule changes the identity even when .pt stays identical.
The manifest is local, deterministic and non-executable; no private cards or seeds.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import re

from ddz_core.rules import RuleConfig
from ddz_env.encoding import encoder_ids

_HEX = re.compile(r'^[0-9a-f]{64}$')
_ROOT = Path(__file__).resolve().parent.parent
_FILES = (
    'ddz_core/actions.py', 'ddz_core/cards.py', 'ddz_core/engine.py',
    'ddz_core/rules.py', 'ddz_core/roles.py',
    'ddz_env/encoding.py', 'ddz_env/baselines.py',
    'ddz_models/network.py', 'ddz_models/policy.py',
)

def policy_fingerprint(weights_sha256: str, spec: dict,
                       *, rules: RuleConfig | None = None) -> dict:
    if not isinstance(weights_sha256, str) or not _HEX.fullmatch(weights_sha256):
        raise ValueError('invalid model weight hash')
    if not isinstance(spec, dict) or spec.get('schema') not in (
            'ddz-candidate-mlp-v1', 'ddz-candidate-mlp-v2'):
        raise ValueError('unknown model schema')
    files = {name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest() for name in _FILES}
    manifest = {'identity_schema': 'ddz-complete-policy-v1',
                'weights_sha256': weights_sha256,
                'network_spec': dict(spec),
                'encoders': encoder_ids(spec['schema']),
                'rules': asdict(rules or RuleConfig()),
                'source_files': files}
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return {'schema': 'ddz-complete-policy-v1',
            'sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
            'manifest': manifest}
