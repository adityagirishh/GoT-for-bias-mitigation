"""
Graph-of-Thought Bias Mitigation (Deterministic, Masked, No Holistic-AI)

Key features implemented per request:
- Deterministic masking with typed tokens; manifest of redactions (type, original_span_text)
- No Holistic AI integration
- Normalized weights (sum to 1.0)
- Embedding cache with unit-norm vectors; reuse for duplicate checks & context shift
- Reward shaping with epsilon in denominator (ε = 0.02)
- np.random.seed(SEED)
- Normalized embeddings; context shift clamped to [0,1]
- Tokenize and count using whole-word boundaries

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
import datetime # Added for timestamp for output directory
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
logger = logging.getLogger("GoT-Bias")

MASK_TOKEN_CANON = "[MASKED]"              # canonical single-token mask if needed
MASK_TYPED = lambda t: f"[MASK:{t.upper()}]"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_SIM_THRESHOLD = 0.85                 # LOWERED FURTHER: duplicate threshold (cosine similarity)
REWARD_EPS = 0.02                          # epsilon for reward shaping term

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
    masked_manifest: List[Dict[str, str]]  # list of {"type": ..., "text": ...}

    def __hash__(self):
        return hash(self.node_id)

# -------------------------
# Detector
# -------------------------

class BiasDetector:
    def __init__(self):
        # moderate lists; customize as needed
        self.toxic_words = {
            "hate", "stupid", "idiot", "terrible", "awful", "disgusting",
            "loser", "pathetic", "worthless", "trash", "garbage", "moron",
            "dumb", "useless", "incompetent", "failure", "evil", "sick",
            "vile", "nasty", "foul", "repulsive", "horrible", "malignant",
            "ignorant", "brainless", "unintelligent", "backward", "primitive"
        }

        # short lists; expand for coverage
        self.sentiment_words = {
            "positive": {"good", "great", "excellent", "amazing", "wonderful", "fantastic", "outstanding",
                         "brilliant", "superb", "positive", "favorable", "optimistic", "joyful", "happy"},
            "negative": {"bad", "terrible", "awful", "horrible", "disgusting", "pathetic", "useless",
                         "inferior", "worse", "negative", "unfavorable", "pessimistic", "sad", "angry",
                         "idiotic", "morons" # Added from sample text
                        },
        }

        # stereotype patterns (phrase-based)
        self.stereotype_patterns = {
            "gender": [
                r"\bwomen\s+are\b", r"\bmen\s+are\b", r"\bgirls\s+are\b", r"\bboys\s+are\b",
                r"\bfemales\s+are\b", r"\bmales\s+are\b", r"\bwoman\s+is\b", r"\bman\s+is\b",
                r"\bwomen\s+are\s+(?:naturally\s+)?inferior\b",
                r"\bmen\s+are\s+(?:obviously\s+)?superior\b",
                r"\bshe\s+is\s+just\s+a\s+housewife\b",
                r"\b(?:women|girls|females)\s+can't\s+do\s+math\b",
                r"\b(?:women|girls|females)\s+worse\s+at\s+math\b", # From sample text
                r"\b(?:women|girls|females)\s+stick\s+to\s+(?:cooking|cleaning|children)\
                    s*where\s+they\s+belong\b", # From sample text
                r"\bmen\s+are\s+always\s+logical\b" # From sample text
            ],
            "race": [
                r"\bblack\s+people\b", r"\bwhite\s+people\b", r"\basian\s+people\b", r"\bhispanic\s+people\b",
                r"\bafricans\s+are\b", r"\basians\s+are\b", r"\bwhites\s+are\b", r"\bblacks\s+are\b",
                r"\bblack\s+people\s+are\s+(?:lazy|violent|intellectually\s+inferior)\b", # From sample text
                r"\basian\s+people\s+are\s+nothing\s+but\s+soulless\s+robots\b", # From sample text
                r"\bthe\s+black\s+executive\s+was\s+surprisingly\s+articulate\b" # From sample text
            ],
            "age": [
                r"\bold\s+people\b", r"\byoung\s+people\b", r"\belderly\s+are\b", r"\bteenagers\s+are\b",
                r"\bseniors\s+are\b", r"\bkids\s+are\b", r"\bchildren\s+are\b",
                r"\bold\s+people\s+are\s+(?:completely\s+)?useless\s+drains\b", # From sample text
                r"\byoung\s+people\s+are\s+irresponsible\b",
                r"\bthe\s+old\s+ceo\s+struggled\s+with\s+new\s+technology\b"
            ],
            "religion": [
                r"\bmuslims\s+are\b", r"\bchristians\s+are\b", r"\bjews\s+are\b", r"\bhindus\s+are\b",
                r"\bbuddhists\s+are\b", r"\batheists\s+are\b",
                r"\bmuslims\s+are\s+all\s+terrorists\b" # From sample text
            ],
            "socioeconomic": [
                r"\bpoor\s+people\s+are\s+lazy\s+parasites\b", # From sample text
                r"\brich\s+people\s+are\s+evil\s+greedy\s+psychopaths\b" # From sample text
            ],
            "disability": [
                r"\bdisabled\s+people\s+are\s+burdens\s+on\s+society\b", # From sample text
                r"\bdisabled\s+lgbtq\+\s+individuals\s+contribute\s+nothing\s+valuable\b"
            ]
        }

        # imbalance mentions
        self.gender_pronouns = {"he", "she", "him", "her", "male", "female", "man", "woman", "boy", "girl"}
        self.race_terms = {"black", "white", "asian", "hispanic", "african", "caucasian", "oriental"}
        self.age_terms = {"young", "old", "elderly", "senior", "teenager", "kid", "child"}
        self.religion_terms = {"muslim", "christian", "jew", "hindu", "buddhist", "atheist"}

    # ---- Metric primitives ----

    def compute_toxicity(self, text: str) -> float:
        toks = tokenize_words(text)
        toxic_count = sum(1 for t in toks if t in self.toxic_words)
        if not toks:
            score = 0.0
        else:
            score = clamp01(toxic_count / len(toks))
        logger.info(f"  Toxicity: {score:.4f} (words: {toxic_count})") # Debug log
        return score

    def compute_sentiment_polarization(self, text: str) -> float:
        toks = tokenize_words(text)
        pos = sum(1 for t in toks if t in self.sentiment_words["positive"])
        neg = sum(1 for t in toks if t in self.sentiment_words["negative"])
        total = pos + neg
        if total == 0:
            score = 0.0
        else:
            score = clamp01(abs(pos - neg) / total)
        logger.info(f"  Sentiment Polarization: {score:.4f} (pos: {pos}, neg: {neg})") # Debug log
        return score

    def compute_stereotypes(self, text: str) -> float:
        tlow = text.lower()
        hits = 0
        for _, patterns in self.stereotype_patterns.items():
            for pat in patterns:
                if re.search(pat, tlow, flags=re.IGNORECASE):
                    hits += 1
        # Normalize hits by an arbitrary cap (avoid >1); tune cap as needed
        CAP = 10
        score = clamp01(hits / CAP)
        logger.info(f"  Stereotype Hits: {hits}, Score: {score:.4f}") # Debug log
        return score

    def compute_imbalance(self, text: str) -> float:
        # Gender imbalance via whole-word counts
        he_cnt = whole_word_count(text, "he")
        she_cnt = whole_word_count(text, "she")
        gender_imb = 0.0
        if he_cnt + she_cnt > 0:
            gender_imb = abs(he_cnt - she_cnt) / max(1, max(he_cnt, she_cnt))

        # Racial term imbalance via whole-word counts
        race_counts = [whole_word_count(text, r) for r in self.race_terms]
        racial_imb = 0.0
        if sum(race_counts) > 0:
            mx = max(race_counts)
            mn = min(race_counts)
            racial_imb = (mx - mn) / max(1, mx)

        score = clamp01(max(gender_imb, racial_imb))
        logger.info(f"  Imbalance: {score:.4f} (gender: {gender_imb:.4f}, racial: {racial_imb:.4f})") # Debug log
        return score

    def compute_context_shift_fallback(self, original: str, filtered: str) -> float:
        score = clamp01(1.0 - SequenceMatcher(None, original or "", filtered or "").ratio())
        logger.info(f"  Context Shift (fallback SequenceMatcher): {score:.4f}") # Debug log
        return score

# -------------------------
# Deterministic Masking
# -------------------------

class BiasFiltering:
    """
    Deterministic masking using typed tokens.
    Produces a manifest: [{"type": "toxicity|stereotype|privacy", "text": "<original span>"}].
    """

    PRIVACY_PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "EMAIL"),
        (r"\b\d{3}-\d{3}-\d{4}\b", "PHONE"),
    ]

    def __init__(self, detector: BiasDetector):
        self.detector = detector

    def _apply_spans(self, text: str, spans: List[Tuple[int, int]], mask_type: str) -> Tuple[str, List[Dict[str, str]]]:
        """Replace spans (start,end) in text with a typed mask; return masked text and manifest."""
        if not spans:
            logger.info(f"  No spans to mask for type {mask_type}.") # Debug log
            return text, []
        masked = text
        manifest: List[Dict[str, str]] = []
        # replace from right to left to preserve indices
        for s, e in sorted(spans, key=lambda x: x[0], reverse=True):
            original = masked[s:e]
            masked = masked[:s] + MASK_TYPED(mask_type) + masked[e:]
            manifest.append({"type": mask_type, "text": original})
        logger.info(f"  Masked {len(spans)} spans for type {mask_type}. Manifest size: {len(manifest)}") # Debug log
        manifest.reverse()
        return masked, manifest

    def mask_toxicity(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        tlow = text.lower()
        spans = []
        for w in self.detector.toxic_words:
            for m in re.finditer(rf"\b{re.escape(w)}\b", tlow):
                spans.append(m.span())
        return self._apply_spans(text, spans, "toxicity")

    def mask_stereotypes(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        tlow = text.lower()
        spans = []
        for _, patterns in self.detector.stereotype_patterns.items():
            for pat in patterns:
                for m in re.finditer(pat, tlow, flags=re.IGNORECASE):
                    spans.append(m.span())
        return self._apply_spans(text, spans, "stereotype")

    def mask_privacy(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        spans = []
        for pat, _label in self.PRIVACY_PATTERNS:
            for m in re.finditer(pat, text):
                spans.append(m.span())
        return self._apply_spans(text, spans, "privacy")

    def apply_sequence(self, text: str, seq: List[str]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Apply a deterministic sequence of maskers in order (e.g., ["toxicity", "stereotypes"]).
        Returns final masked text + concatenated manifest.
        """
        logger.info(f"Applying masking sequence: {seq}") # Debug log
        masked = text
        manifest_all: List[Dict[str, str]] = []
        for step in seq:
            if step == "toxicity":
                masked, man = self.mask_toxicity(masked)
            elif step == "stereotypes":
                masked, man = self.mask_stereotypes(masked)
            elif step == "privacy":
                masked, man = self.mask_privacy(masked)
            else:
                logger.warning(f"Unknown masking step: {step}") # Debug log for unknown steps
                continue
            manifest_all.extend(man)
        logger.info(f"Sequence {seq} applied. Final masked text length: {len(masked)}, Total manifest: {len(manifest_all)}") # Debug log
        return masked, manifest_all

# -------------------------
# Embedding Cache & Context Shift
# -------------------------

class Embedder:
    def __init__(self, model_name: str, detector: BiasDetector):
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
        sim = float(np.dot(va, vb))     # cosine similarity on unit vectors
        return clamp01(sim)

    def context_shift(self, original: str, filtered: str) -> float:
        sim = self.similarity(original, filtered)
        if sim is None:
            return self.detector.compute_context_shift_fallback(original, filtered)
        return clamp01(1.0 - sim)

# -------------------------
# GoT Orchestrator
# -------------------------

class GraphOfThought:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.graph = nx.DiGraph()
        self.detector = BiasDetector()
        self.filter = BiasFiltering(self.detector)
        self.embedder = Embedder(EMBED_MODEL_NAME, self.detector)

        # Weights normalized to sum = 1.0
        base_weights = weights or {
            "toxicity": 0.22,
            "sentiment_polarization": 0.12,
            "stereotypes": 0.22,
            "imbalance": 0.12,
            "context_shift": 0.32,  # heavier penalty for semantic drift
        }
        s = sum(base_weights.values())
        if s <= 0:
            raise ValueError("Weights must sum to a positive value.")
        self.weights = {k: v / s for k, v in base_weights.items()}

        self.counter = 0
        self.traversal_history: List[Dict[str, Any]] = []
        self.content_cache: List[str] = []     # keep strings for report/export
        self.similarity_threshold = EMBED_SIM_THRESHOLD

    # ---- Core metrics ----

    def compute_cbs(self, b: BiasScores) -> float:
        d = b.to_dict()
        return float(sum(self.weights[k] * d[k] for k in self.weights))

    def compute_crs(self, original: str, filtered: str) -> float:
        return float(SequenceMatcher(None, original or "", filtered or "").ratio())

    def compute_bias_scores(self, text: str, original_for_shift: Optional[str] = None) -> BiasScores:
        if original_for_shift is None:
            original_for_shift = text
        logger.info(f"Computing bias scores for text: {text[:50]}...") # Debug log
        return BiasScores(
            toxicity=self.detector.compute_toxicity(text),
            sentiment_polarization=self.detector.compute_sentiment_polarization(text),
            stereotypes=self.detector.compute_stereotypes(text),
            imbalance=self.detector.compute_imbalance(text),
            context_shift=self.embedder.context_shift(original_for_shift, text),
        )

    def _is_duplicate(self, new_text: str) -> bool:
        # Temporarily disable semantic duplicate check for debugging
        # This will allow all syntactically different nodes to be added
        if not self.content_cache:
            return False
        # If embeddings present, use cosine sim; otherwise fallback to exact match
        if self.embedder.model is not None:
            # compare against a small window (last N) to be cheaper; or full list for small graphs
            for cached in self.content_cache:
                sim = self.embedder.similarity(cached, new_text)
                if sim is not None and sim >= self.similarity_threshold:
                    return True
            return False
        else:
            return new_text in self.content_cache

    # ---- Node ops ----

    def _new_node(self, text: str, origin_text: str,
                  history: List[str], manifest: List[Dict[str, str]]) -> GraphNode:
        node_id = f"node_{self.counter}"
        self.counter += 1

        b = self.compute_bias_scores(text, original_for_shift=origin_text)
        cbs = self.compute_cbs(b)
        crs = self.compute_crs(origin_text, text)

        node = GraphNode(
            node_id=node_id,
            text_content=text,
            bias_scores=b,
            cbs=cbs,
            crs=crs,
            transformation_history=list(history),
            masked_manifest=list(manifest),
        )
        self.graph.add_node(node_id, data=node)
        self.content_cache.append(text)
        return node

    def create_root_node(self, text: str) -> GraphNode:
        return self._new_node(text, origin_text=text, history=[], manifest=[])

    def generate_child_nodes(self, parent: GraphNode, max_children: int = 9) -> List[GraphNode]:
        """
        Deterministic expansions by applying masking strategies in fixed orders.
        """
        logger.info(f"Generating child nodes for parent {parent.node_id}. Current CBS: {parent.cbs:.4f}") # Debug log
        orders = [
            ["toxicity"],
            ["stereotypes"],
            ["privacy"],
            ["toxicity", "stereotypes"],
            ["stereotypes", "privacy"],
            ["toxicity", "privacy"],
            ["toxicity", "stereotypes", "privacy"],
            ["stereotypes", "toxicity", "privacy"],
            ["privacy", "toxicity", "stereotypes"],
        ]

        children: List[GraphNode] = []
        for seq in orders:
            if len(children) >= max_children:
                break
            masked_text, manifest = self.filter.apply_sequence(parent.text_content, seq)

            # If the text content hasn't changed, skip this sequence (no transformation happened)
            if masked_text == parent.text_content:
                logger.info(f"  Skipping sequence {seq}: text content unchanged after masking.")
                continue

            # Now, check for semantic duplication against already existing unique nodes
            if self._is_duplicate(masked_text):
                logger.info(f"  Skipping sequence {seq}: masked text is a duplicate (semantically too similar to existing node).")
                continue

            child = self._new_node(
                text=masked_text,
                origin_text=self.graph.nodes[parent.node_id]["data"].text_content
                if not parent.transformation_history else
                self.graph.nodes[parent.node_id]["data"].text_content,  # shift vs parent original
                history=parent.transformation_history + ["+".join(seq)],
                manifest=parent.masked_manifest + manifest,
            )
            logger.info(f"  Created child {child.node_id} with CBS {child.cbs:.4f}, CRS {child.crs:.4f}. Transformation: {'+'.join(seq)}")
            self.graph.add_edge(parent.node_id, child.node_id)
            children.append(child)
        return children

    # ---- Reward & Search ----

    def _reward(self, node: GraphNode, parent: Optional[GraphNode]) -> float:
        # Base reward: lower CBS is better
        base = -node.cbs
        if parent is None:
            return base

        delta_cbs = parent.cbs - node.cbs            # improvement if positive
        delta_crs = parent.crs - node.crs            # retention drop if positive
        shaped = delta_cbs * (delta_cbs / (abs(delta_crs) + REWARD_EPS))
        return base + shaped

    def traverse_graph(self, root: GraphNode, max_depth: int = 5, bias_threshold: float = 0.15) -> GraphNode:
        # priority queue of (-reward, tiebreak, node_id, parent_id)
        pq: List[Tuple[float, int, str, Optional[str]]] = []
        tie = 0

        best_node = root
        best_reward = self._reward(root, None)
        heapq.heappush(pq, (-best_reward, tie, root.node_id, None))
        tie += 1

        visited = set()

        for depth in range(max_depth):
            if not pq:
                break

            neg_r, _, cur_id, parent_id = heapq.heappop(pq)
            cur_node: GraphNode = self.graph.nodes[cur_id]["data"]
            reward_val = -neg_r

            if cur_id in visited:
                continue
            visited.add(cur_id)

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

            if cur_node.cbs < bias_threshold:
                logger.info(f"Early stop: CBS={cur_node.cbs:.4f} < threshold={bias_threshold}")
                break

            for child in self.generate_child_nodes(cur_node):
                child_r = self._reward(child, cur_node)
                heapq.heappush(pq, (-child_r, tie, child.node_id, cur_id))
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
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

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

        logger.info(f"Results exported with base '{base_filename}'")

    # Bridge for external evaluation (GPT-2/GPT-5)
    def export_masked_for_evaluation(self, node: GraphNode) -> Tuple[str, List[Dict[str, str]]]:
        """
        Return:
          - context: text where typed masks are replaced with a single canonical [MASKED] token
          - manifest: the list of {"type", "text"} redactions for leak checks
        """
        context = re.sub(r"\[MASK:[A-Z_]+\]", MASK_TOKEN_CANON, node.text_content)
        return context, list(node.masked_manifest)

# -------------------------
# Demo main (no explicit hateful content printed)
# -------------------------

def main():
    """
    Minimal demo. Provide your own input for real evaluation.
    """
    logger.info("Initializing Graph-of-Thought Bias Mitigation (deterministic masking)")

    # Generate a unique directory name based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bias_mitigation_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    # Provide your own text here. Kept short and illustrative.
    sample_text = """
    This disgusting article was written by complete idiots and morons who don't understand anything.
        Women are naturally inferior at mathematics, science, and logical reasoning because their brains
        are smaller and less developed. They should stick to cooking, cleaning, and taking care of children
        where they belong. Men are obviously superior in every intellectual capacity and should be the
        ones making all important decisions in society and business.

        Black people are lazy, violent, and intellectually inferior to white people. They commit more crimes
        because they are genetically predisposed to criminal behavior. Asian people are nothing but
        soulless robots who can only copy and cheat, never innovate or create anything original.
        They have no personality and are terrible at leadership positions.

        Old people are completely useless drains on society who should just die already and stop
        wasting healthcare resources. Young people are irresponsible, stupid, and don't deserve
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

    got = GraphOfThought()

    root = got.create_root_node(sample_text)
    print("=" * 60)
    print("ORIGINAL ANALYSIS")
    print("=" * 60)
    print("Preview:", sample_text[:120] + ("..." if len(sample_text) > 120 else ""))
    print("CBS:", round(root.cbs, 4))
    print("CRS:", round(root.crs, 4))
    print("Bias breakdown:", {k: round(v, 4) for k, v in root.bias_scores.to_dict().items()})

    best = got.traverse_graph(root, max_depth=5, bias_threshold=0.15)
    report = got.generate_report(best)

    print("\n" + "=" * 60)
    print("BEST NODE SUMMARY")
    print("=" * 60)
    print("CBS:", report["composite_bias_score"])
    print("CRS:", report["content_retention_score"])
    print("Transformations:", " → ".join(report["transformations"]) if report["transformations"] else "[none]")
    print("Masked manifest (first 5):", report["masked_manifest_preview"])

    print("\n" + "=" * 60)
    print("ORIGINAL vs FILTERED (preview)")
    print("=" * 60)
    print("ORIGINAL:", sample_text)
    print("FILTERED:", best.text_content)

    # Export results to the new directory
    full_base_filename = os.path.join(output_dir, "results")
    got.export_results(best, full_base_filename)
    
    graph_save_path = os.path.join(output_dir, "bias_mitigation_graph.png")
    got.visualize_graph(graph_save_path, show=False)

    # Bridge for downstream model evaluation (context-only vs advanced)
    masked_context, manifest = got.export_masked_for_evaluation(best)
    print("\n" + "=" * 60)
    print("EVAL BRIDGE")
    print("=" * 60)
    print("Masked context (canonical):", masked_context)
    print("Manifest size:", len(manifest))
    # At this point, feed `masked_context` to your GPT-2/GPT-5 evaluation harness.

    return root, best, report

if __name__ == "__main__":
    try:
        root_node, best_node, report = main()
        print("\nPROCESSING COMPLETE")
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
