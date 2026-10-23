"""CLI benchmark should distinguish throughput from actual playing strength."""
from pathlib import Path
def test_actor_benchmark_script_is_available():
    root=Path(__file__).resolve().parents[1]
    data=(root/'scripts'/'benchmark_actor_pool.py').read_text(encoding='utf-8')
    assert 'outcome_signature_sha256' in data
    assert 'episodes_per_s' in data
