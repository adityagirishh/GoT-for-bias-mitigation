"""
Graph-of-Thought Bias Mitigation (Deterministic, Masked, No Holistic-AI)
Integrated with Quick Wins Patch

Patched drop-in with the following improvements:
- True depth-limited best-first traversal (max_depth applies to path length)
- Deterministic iteration (sorted sets), seeded layout
- Span merging to avoid overlapping/adjacent multi-masks
- Privacy masking keeps subtype in manifest and mask label; case-insensitive email
- Windowed duplicate check; configurable similarity threshold
- Minor clarity cleanups (explicit origin_text for children)
- Integrated Quick Wins: Privacy masking with unique tokens, CRS constraints, enhanced duplicate logging

Requirements:
    pip install numpy networkx matplotlib scipy sentence-transformers

Notes:
- If the SentenceTransformer model cannot load (e.g., no internet), the code falls back to
  SequenceMatcher for context shift and simple string equality for duplicates.
"""

import os
import re
import json
import math
import heapq
import random
import logging
import datetime
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from scipy.spatial.distance import cosine

# Try to load sentence-transformers (offline safe fallback)
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    _ST_AVAILABLE = False

# -------------------------
# Config & Reproducibility
# -------------------------
SEED = int(os.getenv("SEED", "42"))
np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GoT-Bias-FIXED")

MASK_TOKEN_CANON = "[MASKED]"              # canonical single-token mask if needed
MASK_TYPED = lambda t: f"[MASK:{t.upper()}]"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DEFAULT_EMBED_SIM_THRESHOLD = float(os.getenv("EMBED_SIM_THRESHOLD", "0.85"))
REWARD_EPS = 0.02                          # epsilon for reward shaping term

# -------------------------
# Enhanced Quick Wins Patch
# -------------------------

def apply_privacy_mask_enhanced(text, additional_spans=None):
    """
    MANDATORY privacy masking with comprehensive patterns + additional spans.
    """
    # Comprehensive privacy patterns
    PRIVACY_PATTERNS_ENHANCED = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE), "EMAIL"),
        (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "PHONE"),
        (re.compile(r"\b\d{1,5}\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|blvd|boulevard)\b", re.IGNORECASE), "ADDRESS"),
        (re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"), "FULL_NAME"),  # Basic name pattern
        (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "ZIP_CODE"),
    ]
    
    spans = []
    labels = []
    
    # Find all privacy patterns
    for rx, label in PRIVACY_PATTERNS_ENHANCED:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))
            labels.append(label)
    
    # Add any additional spans passed in
    if additional_spans:
        for span_info in additional_spans:
            spans.append((span_info["start"], span_info["end"]))
            labels.append(span_info.get("type", "PRIVACY"))
    
    if not spans:
        logger.info("  [PRIVACY] No privacy spans found to mask")
        return text, []
    
    # Sort and apply masks
    masked_text = text
    manifest = []
    
    for (s, e), label in sorted(zip(spans, labels), key=lambda x: x[0][0], reverse=True):
        token = f"<MASK{uuid.uuid4().hex[:8]}>"
        original = masked_text[s:e]
        masked_text = masked_text[:s] + token + masked_text[e:]
        manifest.append({
            "token": token,
            "span": (s, e),
            "original": original,
            "type": label
        })
    
    manifest.reverse()
    logger.info(f"  [PRIVACY] Masked {len(spans)} privacy spans: {sorted(set(labels))}")
    return masked_text, manifest

def should_skip_duplicate(score, similarity_threshold=0.9):
    """Log similarity score and decide whether to skip."""
    logger.info(f"[DuplicateCheck] Similarity={score:.3f}, threshold={similarity_threshold}")
    return score >= similarity_threshold

def accept_node(crs, min_crs=0.6):
    """Accept node only if CRS ≥ min_crs."""
    if crs < min_crs:
        logger.warning(f"[NodeReject] CRS={crs:.3f} < min_crs={min_crs}")
        return False
    logger.info(f"[NodeAccept] CRS={crs:.3f} (≥ {min_crs})")
    return True

def finalize_node_output_enhanced(node_output, original_spans, crs, similarity_score=None, 
                                 similarity_threshold=0.9, min_crs=0.6):
    """
    Enhanced unified entrypoint: MANDATORY privacy + constraints.
    """
    # 1. MANDATORY privacy mask (always applied)
    masked_text, privacy_manifest = apply_privacy_mask_enhanced(node_output, original_spans)
    
    # 2. Enforce CRS constraint
    if not accept_node(crs, min_crs=min_crs):
        return None
    
    # 3. Duplicate check (optional)
    if similarity_score is not None and should_skip_duplicate(similarity_score, similarity_threshold):
        return None
    
    return {
        "masked_output": masked_text,
        "privacy_manifest": privacy_manifest,
        "crs": crs
    }

# -------------------------
# Utilities
# -------------------------

_WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)

def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower()) if text else []


def whole_word_count(text: str, word: str) -> int:
    if not text or not word:
        return 0
    return len(re.findall(rf"\b{re.escape(word)}\b", text.lower()))


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

# -------------------------
# Bias scores (no external libs)
# -------------------------

@dataclass
class BiasScores:
    toxicity: float                # [0,1] fraction of toxic tokens
    sentiment_polarization: float  # [0,1] |pos-neg| / (pos+neg)
    stereotypes: float             # [0,1] normalized stereotype pattern hits
    imbalance: float               # [0,1] max of gender/race imbalance
    context_shift: float           # [0,1] 1 - cosine_sim (unit-norm) or fallback

    def to_dict(self) -> Dict[str, float]:
        return {
            "toxicity": self.toxicity,
            "sentiment_polarization": self.sentiment_polarization,
            "stereotypes": self.stereotypes,
            "imbalance": self.imbalance,
            "context_shift": self.context_shift,
        }

# -------------------------
# Graph node
# -------------------------

@dataclass
class GraphNode:
    node_id: str
    text_content: str
    bias_scores: BiasScores
    cbs: float
    crs: float
    transformation_history: List[str]
    masked_manifest: List[Dict[str, str]]  # list of {"type": ..., "text": ..., ...}
    privacy_manifest: List[Dict[str, str]]  # Separate privacy tracking

    def __hash__(self):
        return hash(self.node_id)

# -------------------------
# Enhanced Detector
# -------------------------

class BiasDetectorEnhanced:
    def __init__(self):
        # EXPANDED toxic words list
        self.toxic_words = {
            "hate", "stupid", "idiot", "terrible", "awful", "disgusting",
            "loser", "pathetic", "worthless", "trash", "garbage", "moron", "morons",
            "dumb", "useless", "incompetent", "failure", "evil", "sick",
            "vile", "nasty", "foul", "repulsive", "horrible", "malignant",
            "ignorant", "brainless", "unintelligent", "backward", "primitive",
            # ADDED MISSING WORDS:
            "idiots", "lazy", "parasites", "psychopaths", "greedy", "hypocritical",
            "fanatics", "terrorists", "inferior", "superior", "drains", "burden", "burdens"
        }

        self.sentiment_words = {
            "positive": {"good", "great", "excellent", "amazing", "wonderful", "fantastic", "outstanding",
                         "brilliant", "superb", "positive", "favorable", "optimistic", "joyful", "happy"},
            "negative": {"bad", "terrible", "awful", "horrible", "disgusting", "pathetic", "useless",
                         "inferior", "worse", "negative", "unfavorable", "pessimistic", "sad", "angry",
                         "idiotic", "morons", "lazy", "evil", "greedy"},
        }

        # ENHANCED stereotype patterns with more comprehensive coverage
        self.stereotype_patterns = {
            "gender": [
                r"\bwomen\s+are\s+(?:naturally\s+)?inferior\b",
                r"\bmen\s+are\s+(?:obviously\s+)?superior\b",
                r"\bwomen\s+are\b.*?\b(?:inferior|worse|bad|terrible)\b",
                r"\bmen\s+are\b.*?\b(?:superior|better|logical)\b",
                r"\b(?:women|girls|females)\s+(?:should\s+)?stick\s+to\s+(?:cooking|cleaning|children)\b",
                r"\b(?:women|girls|females)\s+(?:are\s+)?(?:naturally\s+)?(?:bad|worse|terrible)\s+at\s+(?:math|science|logic)\b",
                r"\bshe\s+is\s+just\s+a\s+housewife\b",
            ],
            "race": [
                r"\b(?:black|african)\s+people\s+are\s+(?:lazy|violent|inferior|criminals?)\b",
                r"\b(?:asian)\s+people\s+are\s+(?:robots?|soulless|terrible)\b",
                r"\b(?:white)\s+people\s+are\s+(?:superior|better)\b",
                r"\bblack\s+people\b",  # Often used in stereotypical contexts
                r"\basian\s+people\b",
                r"\bwhite\s+people\b",
                r"\bgenetically\s+predisposed\s+to\s+criminal\b",
            ],
            "age": [
                r"\b(?:old|elderly)\s+people\s+are\s+(?:useless|drains?|burdens?)\b",
                r"\b(?:young)\s+people\s+are\s+(?:irresponsible|stupid)\b",
                r"\bold\s+people\b.*?\b(?:useless|drains?|die)\b",
                r"\byoung\s+people\b.*?\b(?:irresponsible|stupid|don\'t\s+deserve)\b",
            ],
            "religion": [
                r"\b(?:muslims?)\s+are\s+(?:all\s+)?terrorists?\b",
                r"\b(?:christians?)\s+are\s+(?:hypocritical\s+)?fanatics?\b",
                r"\bjews?\s+control\s+(?:all\s+the\s+)?(?:banks?|media)\b",
                r"\bmuslims\s+are\b",
                r"\bchristians\s+are\b",
                r"\bjews\s+(?:are|control)\b",
            ],
            "socioeconomic": [
                r"\bpoor\s+people\s+are\s+(?:lazy\s+)?parasites?\b",
                r"\brich\s+people\s+are\s+(?:evil|greedy)\s+psychopaths?\b",
            ],
            "disability": [
                r"\bdisabled\s+people\s+are\s+burdens?\b",
                r"\bdisabled.*?contribute\s+nothing\b",
            ]
        }

        self.gender_pronouns = {"he", "she", "him", "her", "male", "female", "man", "woman", "boy", "girl"}
        self.race_terms = {"black", "white", "asian", "hispanic", "african", "caucasian", "oriental"}
        self.age_terms = {"young", "old", "elderly", "senior", "teenager", "kid", "child"}
        self.religion_terms = {"muslim", "christian", "jew", "hindu", "buddhist", "atheist"}

    def compute_toxicity(self, text: str) -> float:
        toks = tokenize_words(text)
        toxic_count = sum(1 for t in toks if t in self.toxic_words)
        score = clamp01(toxic_count / len(toks)) if toks else 0.0
        logger.info(f"  Toxicity: {score:.4f} (words: {toxic_count})")
        return score

    def compute_sentiment_polarization(self, text: str) -> float:
        toks = tokenize_words(text)
        pos = sum(1 for t in toks if t in self.sentiment_words["positive"])
        neg = sum(1 for t in toks if t in self.sentiment_words["negative"])
        total = pos + neg
        score = clamp01(abs(pos - neg) / total) if total else 0.0
        logger.info(f"  Sentiment Polarization: {score:.4f} (pos: {pos}, neg: {neg})")
        return score

    def compute_stereotypes(self, text: str) -> float:
        tlow = text.lower()
        hits = 0
        for _cat, patterns in sorted(self.stereotype_patterns.items(), key=lambda x: x[0]):
            for pat in patterns:
                matches = list(re.finditer(pat, tlow, flags=re.IGNORECASE))
                hits += len(matches)
                if matches:
                    logger.info(f"    Stereotype match ({_cat}): {pat} -> {len(matches)} hits")
        
        CAP = 15  # Increased cap due to more patterns
        score = clamp01(hits / CAP)
        logger.info(f"  Stereotype Hits: {hits}, Score: {score:.4f}")
        return score

    def compute_imbalance(self, text: str) -> float:
        he_cnt = whole_word_count(text, "he")
        she_cnt = whole_word_count(text, "she")
        gender_imb = abs(he_cnt - she_cnt) / max(1, max(he_cnt, she_cnt)) if (he_cnt + she_cnt) > 0 else 0.0

        race_counts = [whole_word_count(text, r) for r in self.race_terms]
        if sum(race_counts) > 0:
            mx = max(race_counts); mn = min(race_counts)
            racial_imb = (mx - mn) / max(1, mx)
        else:
            racial_imb = 0.0

        score = clamp01(max(gender_imb, racial_imb))
        logger.info(f"  Imbalance: {score:.4f} (gender: {gender_imb:.4f}, racial: {racial_imb:.4f})")
        return score

    def compute_context_shift_fallback(self, original: str, filtered: str) -> float:
        score = clamp01(1.0 - SequenceMatcher(None, original or "", filtered or "").ratio())
        logger.info(f"  Context Shift (fallback SequenceMatcher): {score:.4f}")
        return score

# -------------------------
# Enhanced Deterministic Masking
# -------------------------

class BiasFilteringEnhanced:
    """Enhanced masking with comprehensive coverage."""

    def __init__(self, detector: BiasDetectorEnhanced):
        self.detector = detector

    @staticmethod
    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not spans:
            return []
        spans = sorted(spans)
        merged: List[List[int]] = []
        for s, e in spans:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        return [(s, e) for s, e in merged]

    def _apply_spans(self, text: str, spans: List[Tuple[int, int]], mask_type: str) -> Tuple[str, List[Dict[str, str]]]:
        if not spans:
            logger.info(f"  No spans to mask for type {mask_type}.")
            return text, []
        masked = text
        manifest: List[Dict[str, str]] = []
        for s, e in sorted(spans, key=lambda x: x[0], reverse=True):
            original = masked[s:e]
            masked = masked[:s] + MASK_TYPED(mask_type) + masked[e:]
            manifest.append({"type": mask_type.split(":")[0], "text": original})
        manifest.reverse()
        logger.info(f"  Masked {len(spans)} spans for type {mask_type}. Manifest size: {len(manifest)}")
        return masked, manifest

    def mask_toxicity(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        tlow = text.lower()
        spans: List[Tuple[int, int]] = []
        for w in sorted(self.detector.toxic_words):
            for m in re.finditer(rf"\b{re.escape(w)}\b", tlow):
                spans.append(m.span())
        spans = self._merge_spans(spans)
        return self._apply_spans(text, spans, "toxicity")

    def mask_stereotypes(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        spans: List[Tuple[int, int]] = []
        for _cat, patterns in sorted(self.detector.stereotype_patterns.items(), key=lambda x: x[0]):
            for pat in patterns:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    spans.append(m.span())
        spans = self._merge_spans(spans)
        return self._apply_spans(text, spans, "stereotype")

    def apply_sequence(self, text: str, seq: List[str]) -> Tuple[str, List[Dict[str, str]]]:
        logger.info(f"Applying enhanced masking sequence: {seq}")
        masked = text
        manifest_all: List[Dict[str, str]] = []
        
        for step in seq:
            if step == "toxicity":
                masked, man = self.mask_toxicity(masked)
            elif step == "stereotypes":
                masked, man = self.mask_stereotypes(masked)
            else:
                logger.warning(f"Unknown masking step: {step}")
                continue
            manifest_all.extend(man)
        
        logger.info(f"Sequence {seq} applied. Final masked text length: {len(masked)}, Total manifest: {len(manifest_all)}")
        return masked, manifest_all

# -------------------------
# Embedding Cache & Context Shift
# -------------------------

class Embedder:
    def __init__(self, model_name: str, detector: BiasDetectorEnhanced):
        self.model_name = model_name
        self.detector = detector
        self.model = None
        self.cache: Dict[str, np.ndarray] = {}

        if _ST_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence-transformer: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not load embedding model ({self.model_name}): {e}")
                self.model = None
        else:
            logger.warning("sentence-transformers not available; using fallback similarity.")

    def _encode(self, text: str) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        if text in self.cache:
            return self.cache[text]
        vec = self.model.encode(text or "")
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v)) + 1e-9
        v = v / n
        self.cache[text] = v
        return v

    def similarity(self, a: str, b: str) -> Optional[float]:
        va = self._encode(a)
        vb = self._encode(b)
        if va is None or vb is None:
            return None
        sim = float(np.dot(va, vb))
        return clamp01(sim)

    def context_shift(self, original: str, filtered: str) -> float:
        sim = self.similarity(original, filtered)
        if sim is None:
            return self.detector.compute_context_shift_fallback(original, filtered)
        return clamp01(1.0 - sim)

# -------------------------
# Enhanced GoT Orchestrator
# -------------------------

class GraphOfThoughtEnhanced:
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 similarity_threshold: float = DEFAULT_EMBED_SIM_THRESHOLD,
                 duplicate_window: int = 50,
                 min_crs: float = 0.5):  # Lowered for more flexibility
        self.graph = nx.DiGraph()
        self.detector = BiasDetectorEnhanced()
        self.filter = BiasFilteringEnhanced(self.detector)
        self.embedder = Embedder(EMBED_MODEL_NAME, self.detector)

        base_weights = weights or {
            "toxicity": 0.25,
            "sentiment_polarization": 0.10,
            "stereotypes": 0.25,
            "imbalance": 0.10,
            "context_shift": 0.30,
        }
        s = sum(base_weights.values())
        if s <= 0:
            raise ValueError("Weights must sum to a positive value.")
        self.weights = {k: v / s for k, v in base_weights.items()}

        self.counter = 0
        self.traversal_history: List[Dict[str, Any]] = []
        self.content_cache: List[str] = []
        self.similarity_threshold = similarity_threshold
        self.duplicate_window = duplicate_window
        self.min_crs = min_crs

    def compute_cbs(self, b: BiasScores) -> float:
        d = b.to_dict()
        return float(sum(self.weights[k] * d[k] for k in self.weights))

    def compute_crs(self, original: str, filtered: str) -> float:
        return float(SequenceMatcher(None, original or "", filtered or "").ratio())

    def compute_bias_scores(self, text: str, original_for_shift: Optional[str] = None) -> BiasScores:
        if original_for_shift is None:
            original_for_shift = text
        logger.info(f"Computing bias scores for text: {text[:50]}...")
        return BiasScores(
            toxicity=self.detector.compute_toxicity(text),
            sentiment_polarization=self.detector.compute_sentiment_polarization(text),
            stereotypes=self.detector.compute_stereotypes(text),
            imbalance=self.detector.compute_imbalance(text),
            context_shift=self.embedder.context_shift(original_for_shift, text),
        )

    def _is_duplicate(self, new_text: str) -> Tuple[bool, Optional[float]]:
        if not self.content_cache:
            return False, None
        cache = self.content_cache[-self.duplicate_window:] if self.duplicate_window > 0 else self.content_cache
        if self.embedder.model is not None:
            for cached in cache:
                sim = self.embedder.similarity(cached, new_text)
                if sim is not None:
                    if should_skip_duplicate(sim, self.similarity_threshold):
                        return True, sim
            return False, None
        else:
            is_dup = new_text in cache
            return is_dup, 1.0 if is_dup else 0.0

    def _new_node(self, text: str, origin_text: str,
                  history: List[str], manifest: List[Dict[str, str]]) -> Optional[GraphNode]:
        
        # MANDATORY: Apply privacy masking to ALL text before processing
        privacy_masked_text, privacy_manifest = apply_privacy_mask_enhanced(text)
        
        # Compute CRS based on privacy-masked text
        crs = self.compute_crs(origin_text, privacy_masked_text)
        
        # Check duplicates
        is_dup, similarity_score = self._is_duplicate(privacy_masked_text)
        
        # Apply enhanced finalization (redundant privacy check, but safe)
        finalized = finalize_node_output_enhanced(
            privacy_masked_text, [], crs, similarity_score,
            self.similarity_threshold, self.min_crs
        )
        
        if finalized is None:
            return None

        node_id = f"node_{self.counter}"
        self.counter += 1

        # Compute bias scores on the privacy-protected text
        b = self.compute_bias_scores(finalized["masked_output"], original_for_shift=origin_text)
        cbs = self.compute_cbs(b)

        node = GraphNode(
            node_id=node_id,
            text_content=finalized["masked_output"],
            bias_scores=b,
            cbs=cbs,
            crs=finalized["crs"],
            transformation_history=list(history),
            masked_manifest=list(manifest),
            privacy_manifest=privacy_manifest + finalized.get("privacy_manifest", []),
        )
        self.graph.add_node(node_id, data=node)
        self.content_cache.append(finalized["masked_output"])
        return node

    def create_root_node(self, text: str) -> GraphNode:
        # Apply mandatory privacy masking to root
        privacy_masked_text, privacy_manifest = apply_privacy_mask_enhanced(text)
        
        node_id = f"node_{self.counter}"
        self.counter += 1
        
        b = self.compute_bias_scores(privacy_masked_text)
        cbs = self.compute_cbs(b)
        crs = self.compute_crs(text, privacy_masked_text)
        
        node = GraphNode(
            node_id=node_id,
            text_content=privacy_masked_text,
            bias_scores=b,
            cbs=cbs,
            crs=crs,
            transformation_history=[],
            masked_manifest=[],
            privacy_manifest=privacy_manifest,
        )
        self.graph.add_node(node_id, data=node)
        self.content_cache.append(privacy_masked_text)
        return node

    def generate_child_nodes(self, parent: GraphNode, max_children: int = 9) -> List[GraphNode]:
        logger.info(f"Generating child nodes for parent {parent.node_id}. Current CBS: {parent.cbs:.4f}")
        
        # Enhanced transformation sequences (no standalone privacy - it\'s always applied)
        orders = [
            ["toxicity"],
            ["stereotypes"],
            ["toxicity", "stereotypes"],
            ["stereotypes", "toxicity"],
        ]

        children: List[GraphNode] = []
        for seq in orders:
            if len(children) >= max_children:
                break
                
            masked_text, manifest = self.filter.apply_sequence(parent.text_content, seq)

            if masked_text == parent.text_content:
                logger.info(f"  Skipping sequence {seq}: text content unchanged after masking.")
                continue

            child = self._new_node(
                text=masked_text,
                origin_text=parent.text_content,
                history=parent.transformation_history + ["+".join(seq)],
                manifest=parent.masked_manifest + manifest,
            )
            
            if child is None:
                logger.info(f"  Skipping sequence {seq}: child node rejected by constraints.")
                continue
                
            logger.info(f"  Created child {child.node_id} with CBS {child.cbs:.4f}, CRS {child.crs:.4f}. Transformation: {'+'.join(seq)}")
            self.graph.add_edge(parent.node_id, child.node_id)
            children.append(child)
        return children

    def _reward(self, node: GraphNode, parent: Optional[GraphNode]) -> float:
        base = -node.cbs
        if parent is None:
            return base
        delta_cbs = parent.cbs - node.cbs
        delta_crs = parent.crs - node.crs
        shaped = delta_cbs * (delta_cbs / (abs(delta_crs) + REWARD_EPS))
        return base + shaped

    def traverse_graph(self, root: GraphNode, max_depth: int = 4, bias_threshold: float = 0.2) -> GraphNode:
        """Depth-limited best-first search."""
        pq: List[Tuple[float, int, int, str, Optional[str]]] = []
        tie = 0

        best_node = root
        best_reward = self._reward(root, None)
        heapq.heappush(pq, (-best_reward, tie, 0, root.node_id, None))
        tie += 1

        visited = set()

        while pq:
            neg_r, _, depth, cur_id, parent_id = heapq.heappop(pq)
            if cur_id in visited:
                continue
            visited.add(cur_id)

            cur_node: GraphNode = self.graph.nodes[cur_id]["data"]
            reward_val = -neg_r

            self.traversal_history.append({
                "depth": depth,
                "node_id": cur_id,
                "cbs": cur_node.cbs,
                "crs": cur_node.crs,
                "reward": reward_val,
                "transformations": cur_node.transformation_history,
            })

            if reward_val > best_reward:
                best_reward = reward_val
                best_node = cur_node

            if cur_node.cbs < bias_threshold or depth >= max_depth:
                continue

            for child in self.generate_child_nodes(cur_node):
                child_r = self._reward(child, cur_node)
                heapq.heappush(pq, (-child_r, tie, depth + 1, child.node_id, cur_id))
                tie += 1
        logger.info(f"Traversal done. Best CBS={best_node.cbs:.4f}, CRS={best_node.crs:.4f}")
        return best_node

    # ---- Reporting / Export ----

    def generate_report(self, node: GraphNode) -> Dict[str, Any]:
        rep = {
            "node_id": node.node_id,
            "composite_bias_score": round(node.cbs, 4),
            "content_retention_score": round(node.crs, 4),
            "bias_breakdown": {k: round(v, 4) for k, v in node.bias_scores.to_dict().items()},
            "transformations": node.transformation_history,
            "masked_manifest_len": len(node.masked_manifest),
            "masked_manifest_preview": node.masked_manifest[:5],
            "privacy_manifest_len": len(node.privacy_manifest),
            "privacy_manifest_preview": node.privacy_manifest[:5],
            "text_preview": (node.text_content[:200] + "...") if len(node.text_content) > 200 else node.text_content,
            "graph_stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges()
            }
        }
        return rep

    def visualize_graph(self, save_path: Optional[str] = None, show: bool = False):
        if not self.graph.nodes():
            logger.warning("No nodes to visualize.")
            return
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=SEED)

        node_colors = [self.graph.nodes[n]["data"].cbs for n in self.graph.nodes()]
        nx.draw(self.graph, pos,
                node_color=node_colors, cmap=plt.cm.RdYlBu_r,
                with_labels=True, node_size=450,
                font_size=8, edge_color="gray", alpha=0.85)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r,
                                   norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label="Composite Bias Score (lower=better)")
        plt.title("Graph of Thought — Bias Mitigation")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            logger.info(f"Graph image saved to {save_path}")
        if show:
            plt.show()
        plt.close()

    def export_results(self, node: GraphNode, base_filename: str = "bias_mitigation_results"):
        with open(f"{base_filename}.txt", "w", encoding="utf-8") as f:
            f.write(node.text_content)

        with open(f"{base_filename}_report.json", "w", encoding="utf-8") as f:
            json.dump(self.generate_report(node), f, indent=2)

        with open(f"{base_filename}_traversal.json", "w", encoding="utf-8") as f:
            json.dump(self.traversal_history, f, indent=2)

        with open(f"{base_filename}_privacy_manifest.json", "w", encoding="utf-8") as f:
            json.dump(node.privacy_manifest, f, indent=2)

        logger.info(f"Results exported with base '{base_filename}'")

    def export_masked_for_evaluation(self, node: GraphNode) -> Tuple[str, List[Dict[str, str]]]:
        context = re.sub(r"\[MASK:[A-Z_:+]+\]", MASK_TOKEN_CANON, node.text_content)
        return context, list(node.masked_manifest), list(node.privacy_manifest)

# -------------------------
# Demo main (no explicit hateful content printed)
# -------------------------

def main():
    logger.info("Initializing Graph-of-Thought Bias Mitigation (deterministic masking with Quick Wins)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bias_mitigation_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    sample_text = """
    This disgusting article was written by complete idiots and morons who don\'t understand anything.
        Women are naturally inferior at mathematics, science, and logical reasoning because their brains
        are smaller and less developed. They should stick to cooking, cleaning, and taking care of children
        where they belong. Men are obviously superior in every intellectual capacity and should be the
        ones making all important decisions in society and business.

        Black people are lazy, violent, and intellectually inferior to white people. They commit more crimes
        because they are genetically predisposed to criminal behavior. Asian people are nothing but
        soulless robots who can only copy and cheat, never innovate or create anything original.
        They have no personality and are terrible at leadership positions.

        Old people are completely useless drains on society who should just die already and stop
        wasting healthcare resources. Young people are irresponsible, stupid, and don\'t deserve
        any respect or opportunities until they prove themselves worthy.

        Muslims are all terrorists who want to destroy Western civilization. Christians are
        hypocritical fanatics who use religion to justify their hatred and bigotry. Jews control
        all the banks and media to manipulate society for their own benefit.

        Poor people are lazy parasites who deserve their poverty because they refuse to work hard.
        Rich people are evil, greedy psychopaths who exploit everyone else. Disabled people are
        burdens on society who contribute nothing of value.

        Contact John Smith at john.smith@email.com or call 555-123-4567 for more information.
        He lives at 123 Main Street, Anytown, USA 12345. His social security number is 123-45-6789.
        This pathetic loser makes $50,000 per year and has a credit score of 580.
        """

    got = GraphOfThoughtEnhanced(min_crs=0.6, similarity_threshold=0.9)

    root = got.create_root_node(sample_text)
    print("=" * 60)
    print("ORIGINAL ANALYSIS")
    print("=" * 60)
    print("Preview:", sample_text[:120] + ("..." if len(sample_text) > 120 else ""))
    print("CBS:", round(root.cbs, 4))
    print("CRS:", round(root.crs, 4))
    print("Bias breakdown:", {k: round(v, 4) for k, v in root.bias_scores.to_dict().items()})

    best = got.traverse_graph(root, max_depth=4, bias_threshold=0.2)
    report = got.generate_report(best)

    print("\n" + "=" * 60)
    print("BEST NODE SUMMARY")
    print("=" * 60)
    print("CBS:", report["composite_bias_score"])
    print("CRS:", report["content_retention_score"])
    print("Transformations:", " \u2192 ".join(report["transformations"]) if report["transformations"] else "[none]")
    print("Masked manifest (first 5):", report["masked_manifest_preview"])
    print("Privacy manifest (first 5):", report["privacy_manifest_preview"])

    print("\n" + "=" * 60)
    print("ORIGINAL vs FILTERED (preview)")
    print("=" * 60)
    print("ORIGINAL:", sample_text)
    print("FILTERED:", best.text_content)

    full_base_filename = os.path.join(output_dir, "results")
    got.export_results(best, full_base_filename)

    graph_save_path = os.path.join(output_dir, "bias_mitigation_graph.png")
    got.visualize_graph(graph_save_path, show=False)

    masked_context, main_manifest, privacy_manifest = got.export_masked_for_evaluation(best)
    print("\n" + "=" * 60)
    print("EVAL BRIDGE")
    print("=" * 60)
    print("Masked context (canonical):", masked_context)
    print("Main Manifest size:", len(main_manifest))
    print("Privacy Manifest size:", len(privacy_manifest))

    return root, best, report


if __name__ == "__main__":
    try:
        root_node, best_node, report = main()
        print("\nPROCESSING COMPLETE")
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")