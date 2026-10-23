"""Changes to decision code or rule settings must change policy identity."""
from dataclasses import replace
import pytest
from ddz_core.rules import RuleConfig
from ddz_quality.policy_identity import policy_fingerprint

SPEC={'state_dim': 125, 'action_dim': 51, 'hidden': 32, 'schema': 'ddz-candidate-mlp-v1'}
def test_weights_and_rules_are_in_identity():
    a=policy_fingerprint('a'*64,SPEC)
    assert a==policy_fingerprint('a'*64,dict(SPEC))
    assert a['sha256']!=policy_fingerprint('b'*64,SPEC)['sha256']
    assert a['sha256']!=policy_fingerprint('a'*64,SPEC,rules=RuleConfig(max_redeals=2))['sha256']
    assert a['sha256']!=policy_fingerprint('a'*64,{**SPEC,'hidden':64})['sha256']
    for name in ('ddz_models/policy.py', 'ddz_env/encoding.py','ddz_core/rules.py'):
        assert len(a['manifest']['source_files'][name])==64
    with pytest.raises(ValueError):
        policy_fingerprint('bad', SPEC)
