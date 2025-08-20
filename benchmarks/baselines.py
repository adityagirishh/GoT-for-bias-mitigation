# benchmarks/baselines.py
import os
import random
import re
from typing import Dict, Tuple

from got import GraphOfThought, BiasFiltering, BiasDetector

RNG = random.Random(int(os.getenv("SEED", "42")))

MASK = "[MASKED]"

def identity(text: str) -> Tuple[str, Dict]:
    return text, {"name": "identity", "info": {}}

def random_mask(text: str, frac: float = 0.12) -> Tuple[str, Dict]:
    words = re.findall(r"\S+|\s+", text)
    n_tokens = len([w for w in words if not w.isspace()])
    n_to_mask = max(1, int(n_tokens * frac))
    idxs = [i for i, w in enumerate(words) if not w.isspace()]
    RNG.shuffle(idxs)
    masked = 0
    for i in idxs:
        if masked >= n_to_mask:
            break
        if not words[i].startswith("[MASK:"):
            words[i] = MASK
            masked += 1
    return "".join(words), {"name": "random_mask", "info": {"frac": frac, "masked": masked, "tokens": n_tokens}}

def deterministic_single_pass(text: str, detector: BiasDetector | None = None) -> Tuple[str, Dict]:
    """
    A simple, fixed-order masking baseline: toxicity -> stereotypes -> privacy
    No graph search, no duplicate checks. This is the key baseline.
    """
    det = detector or BiasDetector()
    bf = BiasFiltering(det)
    masked, manifest = bf.apply_sequence(text, ["toxicity", "stereotypes", "privacy"])
    return masked, {"name": "fixed_order_mask", "info": {"order": ["toxicity", "stereotypes", "privacy"], "manifest_len": len(manifest)}}

def stereotypes_then_privacy(text: str, detector: BiasDetector | None = None) -> Tuple[str, Dict]:
    det = detector or BiasDetector()
    bf = BiasFiltering(det)
    masked, manifest = bf.apply_sequence(text, ["stereotypes", "privacy"])
    return masked, {"name": "stereotypes_then_privacy", "info": {"order": ["stereotypes", "privacy"], "manifest_len": len(manifest)}}

def privacy_only(text: str, detector: BiasDetector | None = None) -> Tuple[str, Dict]:
    det = detector or BiasDetector()
    bf = BiasFiltering(det)
    masked, manifest = bf.apply_sequence(text, ["privacy"])
    return masked, {"name": "privacy_only", "info": {"order": ["privacy"], "manifest_len": len(manifest)}}

BASELINES = {
    "identity": identity,
    "random_mask": random_mask,
    "fixed_order_mask": deterministic_single_pass,
    "stereotypes_then_privacy": stereotypes_then_privacy,
    "privacy_only": privacy_only,
}
