"""Paired bid evidence is seed-reproducible and grouped by initial deal."""
import pytest
from ddz_quality.auction_audit import paired_auction_audit
from ddz_quality.analysis import QualityError

def test_auction_paired_audit_deterministic():
    a=paired_auction_audit(seed=15100,deals=3,draws=100)
    b=paired_auction_audit(seed=15100,deals=3,draws=100)
    assert a==b
    assert a['paired_deals']+a['excluded_deals']==3
    assert len(a['ci95'])==2
    assert sum(x['deals'] for x in a['by_strength'].values())==a['paired_deals']
    with pytest.raises(QualityError):
        paired_auction_audit(3,0,100)
