# benchmarks/__init__.py
"""
Benchmark suite for: "GoT for Bias Mitigation"

Run:
    python -m benchmarks.runner --dataset synthetic
    python -m benchmarks.runner --dataset crows --samples 200
"""
from .got import GraphOfThought, BiasScores

__all__ = [
    "GraphOfThought",
    "BiasScores",
    "datasets",
    "baselines",
    "metrics",
]
