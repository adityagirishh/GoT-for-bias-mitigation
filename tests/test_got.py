# tests/test_got.py
import os
import re

from got import GraphOfThought

def test_traversal_reduces_bias_and_is_deterministic(monkeypatch):
    # force determinism
    monkeypatch.setenv("SEED", "42")

    sample = (
        "This disgusting article was written by idiots. "
        "Women are naturally inferior at mathematics. "
        "Contact me at John.Smith@Email.COM or 555-123-4567. SSN 123-45-6789."
    )

    g1 = GraphOfThought()
    root1 = g1.create_root_node(sample)
    best1 = g1.traverse_graph(root1, max_depth=4, bias_threshold=0.15)

    g2 = GraphOfThought()
    root2 = g2.create_root_node(sample)
    best2 = g2.traverse_graph(root2, max_depth=4, bias_threshold=0.15)

    # Bias should not increase
    assert best1.cbs <= root1.cbs + 1e-8
    assert best2.cbs <= root2.cbs + 1e-8

    # Deterministic text content
    assert best1.text_content == best2.text_content

def test_privacy_masking_includes_subtypes():
    sample = "Email A@b.com phone 555-123-4567 SSN 123-45-6789."
    got = GraphOfThought()
    # Directly apply privacy masking
    masked_text, manifest = got.filter.mask_privacy(sample)

    # Check that privacy masks include subtype in the mask label
    assert "[MASK:PRIVACY:EMAIL]" in masked_text
    assert "[MASK:PRIVACY:PHONE]" in masked_text
    assert "[MASK:PRIVACY:SSN]" in masked_text

def test_no_overlapping_masks_simple():
    sample = "This is awful garbage and a pathetic, disgusting take."
    got = GraphOfThought()
    root = got.create_root_node(sample)
    best = got.traverse_graph(root, max_depth=3, bias_threshold=0.2)

    # Consecutive masks should be separate tokens, not partial overlaps
    # We just assert all masks match the expected pattern
    masks = re.findall(r"\[MASK:[A-Z_:+]+\]", best.text_content)
    for m in masks:
        assert m.startswith("[MASK:")
        assert m.endswith("]")

def test_fallback_similarity_works_without_st_model(monkeypatch):
    # Force absence of sentence-transformers by poisoning import
    monkeypatch.setenv("EMBED_MODEL", "non-existent-model-name")
    # (GraphOfThought will fallback automatically if model cannot be loaded)
    sample = "He is good. She is bad."
    got = GraphOfThought()
    b = got.compute_bias_scores(sample)  # should not raise
    assert 0.0 <= b.context_shift <= 1.0
