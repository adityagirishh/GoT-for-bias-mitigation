# tests/test_metrics.py
from difflib import SequenceMatcher

from benchmarks.metrics import semantic_similarity, token_replacement_rate
from got import GraphOfThought

def test_token_replacement_rate():
    text = "A [MASK:toxicity] sentence with [MASK:privacy:EMAIL] tokens."
    r = token_replacement_rate(text)
    assert 0 < r <= 1

def test_semantic_similarity_fallback_and_range():
    a = "The quick brown fox jumps over the lazy dog."
    b = "A swift brown fox leaped over a sleepy dog."
    s = semantic_similarity(a, b)
    # Should be in [0,1] for SBERT cosine; fallback SequenceMatcher is ~[0,1] as well.
    assert 0.0 <= s <= 1.0

def test_got_scores_api_runs():
    g = GraphOfThought()
    a = "Women are worse at math."
    b = "Women are [MASK:stereotype] at math."
    cbs, crs, br = (None, None, None)
    cbs, crs, br = __import__("benchmarks.metrics", fromlist=[""]).got_scores(g, a, b)
    assert 0.0 <= cbs <= 1.0
    assert 0.0 <= crs <= 1.0
    assert {"toxicity", "stereotypes", "sentiment_polarization", "imbalance", "context_shift"} <= set(br.keys())
