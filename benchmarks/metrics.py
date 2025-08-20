# benchmarks/metrics.py
import json
import math
import os
import re
from difflib import SequenceMatcher
from typing import Dict, Tuple

import numpy as np

from got import GraphOfThought, BiasScores

_MASK_RX = re.compile(r"\[MASK:[^\]]+\]")

# --- Similarity (SBERT if available, else SequenceMatcher) ---

try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    _ST = SentenceTransformer(_ST_MODEL_NAME)
except Exception:
    _ST = None

def semantic_similarity(a: str, b: str) -> float:
    if _ST is None:
        return SequenceMatcher(None, a or "", b or "").ratio()
    va = _ST.encode(a or "")
    vb = _ST.encode(b or "")
    va = np.asarray(va, dtype=np.float32); vb = np.asarray(vb, dtype=np.float32)
    va = va / (np.linalg.norm(va) + 1e-9)
    vb = vb / (np.linalg.norm(vb) + 1e-9)
    return float(np.dot(va, vb))

def token_replacement_rate(text: str) -> float:
    """Share of masked tokens among all tokens (approx)."""
    if not text:
        return 0.0
    masked = len(_MASK_RX.findall(text))
    # crude token count (words only)
    total = max(1, len(re.findall(r"\b[\w']+\b", text)))
    return masked / total

def got_scores(got: GraphOfThought, original: str, variant: str) -> Tuple[float, float, Dict]:
    """
    Returns: (CBS, CRS, breakdown_dict)
    CBS is computed via got.compute_cbs on bias scores for 'variant'
    CRS is SequenceMatcher retention vs original (consistent with got.py)
    """
    b: BiasScores = got.compute_bias_scores(variant, original_for_shift=original)
    cbs = got.compute_cbs(b)
    crs = SequenceMatcher(None, original or "", variant or "").ratio()
    return float(cbs), float(crs), {k: float(v) for k, v in b.to_dict().items()}

def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def aggregate_means(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = {k for r in rows for k in r.keys() if isinstance(r.get(k), (int, float))}
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            out[f"mean_{k}"] = float(sum(vals) / len(vals))
    return out
