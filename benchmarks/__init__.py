# benchmarks/__init__.py
"""
Benchmark suite for: "GoT for Bias Mitigation"

Run:
    python -m benchmarks.runner --dataset synthetic
    python -m benchmarks.runner --dataset crows --samples 200
"""
from .metrics import semantic_similarity, token_replacement_rate

__all__ = [
    "datasets",
    "baselines",
    "metrics",
    "semantic_similarity",
    "token_replacement_rate",
]
