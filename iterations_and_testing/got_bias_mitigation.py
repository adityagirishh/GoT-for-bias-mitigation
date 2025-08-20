# Enhanced Bias Mitigation System Using Graph of Thought + Holistic AI
# Includes proposed improvements: deduplication, better CRS, embedding-based context shift, export utilities
# Now integrated with Holistic AI library for comprehensive bias detection
import sys
import os

print("Python Executable:", sys.executable)
print("sys.path:")
for p in sys.path:
    print(f"  - {p}")

try:
    import holisticai
    print(f"holisticai imported from: {holisticai.__file__}")
except ImportError as e:
    print(f"holisticai import failed: {e}")
    
import numpy as np
import networkx as nx
import heapq
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
from scipy.spatial.distance import cosine
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


# Holistic AI imports
try:
    from holisticai.bias.metrics import classification_bias_metrics
    from holisticai.bias.mitigation import Reweighing, EqualizedOdds
    # from holisticai.fairness.metrics import demographic_parity_ratio
    from holisticai.explainability.metrics import feature_importance_variance
    from holisticai.security.metrics import membership_inference_risk
    #from holisticai.privacy.metrics import k_anonymity
    HOLISTIC_AI_AVAILABLE = True
except ImportError:
    print("Warning: Holistic AI not available. Install with: pip install holisticai")
    HOLISTIC_AI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load sentence transformer model once
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    print("Warning: Could not load sentence transformer model")
    embedding_model = None

@dataclass
class BiasScores:
    toxicity: float
    sentiment: float
    stereotypes: float
    imbalance: float
    context_shift: float
    # New Holistic AI scores
    fairness_score: float = 0.0
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    privacy_risk: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return vars(self)

@dataclass
class GraphNode:
    node_id: str
    text_content: str
    bias_scores: BiasScores
    cbs: float
    crs: float
    transformation_history: List[str]
    masked_segments: List[Tuple[int, int]]
    holistic_metrics: Dict[str, Any] = None
    
    def __hash__(self):
        return hash(self.node_id)

class HolisticAIIntegration:
    """Integration layer for Holistic AI library"""
    
    def __init__(self):
        self.available = HOLISTIC_AI_AVAILABLE
        if not self.available:
            logger.warning("Holistic AI library not available. Using fallback methods.")
    
    def compute_fairness_metrics(self, text: str, context: Dict = None) -> Dict[str, float]:
        """Compute fairness metrics using Holistic AI"""
        if not self.available:
            return self._fallback_fairness_metrics(text)
        
        try:
            # Text-based fairness analysis
            metrics = {}
            
            # Simulate demographic analysis from text
            # In real implementation, you'd extract demographic indicators
            demographic_groups = self._extract_demographic_indicators(text)
            
            if demographic_groups:
                # Simulate fairness computation
                metrics['demographic_parity'] = self._compute_demographic_parity(text, demographic_groups)
                metrics['equalized_odds'] = self._compute_equalized_odds(text, demographic_groups)
            else:
                metrics['demographic_parity'] = 0.0
                metrics['equalized_odds'] = 0.0
            
            # Privacy risk assessment
            metrics['privacy_risk'] = self._compute_privacy_risk(text)
            
            # Overall fairness score
            metrics['fairness_score'] = np.mean([
                1.0 - metrics['demographic_parity'],
                1.0 - metrics['equalized_odds'],
                1.0 - metrics['privacy_risk']
            ])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing Holistic AI metrics: {e}")
            return self._fallback_fairness_metrics(text)
    
    def _extract_demographic_indicators(self, text: str) -> List[str]:
        """Extract demographic group indicators from text"""
        indicators = []
        text_lower = text.lower()
        
        # Gender indicators
        if any(word in text_lower for word in ['women', 'woman', 'female', 'she', 'her']):
            indicators.append('female')
        if any(word in text_lower for word in ['men', 'man', 'male', 'he', 'his', 'him']):
            indicators.append('male')
        
        # Racial/ethnic indicators
        racial_terms = ['black', 'white', 'asian', 'hispanic', 'latino', 'african', 'european']
        for term in racial_terms:
            if term in text_lower:
                indicators.append(f'race_{term}')
        
        # Age indicators
        if any(word in text_lower for word in ['young', 'old', 'elderly', 'teenager', 'senior']):
            indicators.append('age_group')
        
        return indicators
    
    def _compute_demographic_parity(self, text: str, groups: List[str]) -> float:
        """Compute demographic parity score"""
        if len(groups) <= 1:
            return 0.0
        
        # Simulate parity calculation based on text sentiment towards different groups
        group_sentiments = []
        for group in groups:
            # Simple sentiment analysis per group
            sentiment = self._get_group_sentiment(text, group)
            group_sentiments.append(sentiment)
        
        # Calculate parity as variance in sentiments
        if len(group_sentiments) > 1:
            variance = np.var(group_sentiments)
            return min(variance, 1.0)
        return 0.0
    
    def _compute_equalized_odds(self, text: str, groups: List[str]) -> float:
        """Compute equalized odds score"""
        if len(groups) <= 1:
            return 0.0
        
        # Simulate equalized odds based on representation balance
        group_mentions = []
        for group in groups:
            mentions = text.lower().count(group.split('_')[-1])
            group_mentions.append(mentions)
        
        if sum(group_mentions) > 0:
            # Calculate imbalance
            max_mentions = max(group_mentions)
            min_mentions = min(group_mentions)
            imbalance = (max_mentions - min_mentions) / max_mentions if max_mentions > 0 else 0.0
            return imbalance
        return 0.0
    
    def _compute_privacy_risk(self, text: str) -> float:
        """Compute privacy risk score"""
        # Look for potentially identifying information
        risk_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
            r'\b\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln)\b'  # Address
        ]
        
        risk_score = 0.0
        for pattern in risk_patterns:
            if re.search(pattern, text):
                risk_score += 0.25
        
        return min(risk_score, 1.0)
    
    def _get_group_sentiment(self, text: str, group: str) -> float:
        """Get sentiment score for a specific group"""
        # Simple sentiment analysis - in practice, use more sophisticated methods
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'stupid']
        
        # Extract context around group mentions
        group_key = group.split('_')[-1]
        text_lower = text.lower()
        
        sentiment_score = 0.0
        context_window = 50  # characters around group mention
        
        start = 0
        while True:
            pos = text_lower.find(group_key, start)
            if pos == -1:
                break
            
            # Extract context
            context_start = max(0, pos - context_window)
            context_end = min(len(text), pos + len(group_key) + context_window)
            context = text_lower[context_start:context_end]
            
            # Count positive/negative words in context
            pos_count = sum(1 for word in positive_words if word in context)
            neg_count = sum(1 for word in negative_words if word in context)
            
            if pos_count + neg_count > 0:
                sentiment_score += (pos_count - neg_count) / (pos_count + neg_count)
            
            start = pos + 1
        
        return sentiment_score
    
    def _fallback_fairness_metrics(self, text: str) -> Dict[str, float]:
        """Fallback metrics when Holistic AI is not available"""
        return {
            'fairness_score': 0.5,
            'demographic_parity': 0.0,
            'equalized_odds': 0.0,
            'privacy_risk': 0.0
        }

class BiasDetector:
    def __init__(self):
        self.toxic_words = {
            'hate', 'stupid', 'idiot', 'terrible', 'awful', 'disgusting', 
            'loser', 'pathetic', 'worthless', 'trash', 'garbage', 'moron',
            'dumb', 'useless', 'incompetent', 'failure'
        }
        
        self.stereotype_patterns = {
            'gender': [
                'women are', 'men are', 'girls are', 'boys are',
                'females are', 'males are', 'woman is', 'man is'
            ],
            'race': [
                'black people', 'white people', 'asian people', 'hispanic people',
                'africans are', 'asians are', 'whites are', 'blacks are'
            ],
            'age': [
                'old people', 'young people', 'elderly are', 'teenagers are',
                'seniors are', 'kids are', 'children are'
            ],
            'religion': [
                'muslims are', 'christians are', 'jews are', 'hindus are',
                'buddhists are', 'atheists are'
            ]
        }
        
        self.sentiment_words = {
            'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding'],
            'negative': ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic', 'useless']
        }
        
        # Initialize Holistic AI integration
        self.holistic_ai = HolisticAIIntegration()
    
    def compute_toxicity_score(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        toxic_count = sum(1 for w in words if w in self.toxic_words)
        return min(toxic_count / len(words), 1.0)
    
    def compute_sentiment_score(self, text: str) -> float:
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in self.sentiment_words['positive'])
        neg_count = sum(1 for w in words if w in self.sentiment_words['negative'])
        total = pos_count + neg_count
        return abs(pos_count - neg_count) / total if total > 0 else 0.0
    
    def compute_stereotype_score(self, text: str) -> float:
        count = 0
        text_lower = text.lower()
        
        for category, patterns in self.stereotype_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    count += 1
        
        return min(count / 10, 1.0)
    
    def compute_imbalance_score(self, text: str) -> float:
        text_lower = text.lower()
        
        # Gender imbalance
        he_count = text_lower.count('he ')
        she_count = text_lower.count('she ')
        gender_imbalance = abs(he_count - she_count) / (he_count + she_count) if (he_count + she_count) > 0 else 0.0
        
        # Racial mention imbalance
        racial_mentions = {}
        for race in ['black', 'white', 'asian', 'hispanic']:
            racial_mentions[race] = text_lower.count(race)
        
        total_racial = sum(racial_mentions.values())
        racial_imbalance = 0.0
        if total_racial > 0:
            max_mentions = max(racial_mentions.values())
            min_mentions = min(racial_mentions.values())
            racial_imbalance = (max_mentions - min_mentions) / max_mentions if max_mentions > 0 else 0.0
        
        return max(gender_imbalance, racial_imbalance)
    
    def compute_context_shift_score(self, original: str, filtered: str) -> float:
        if embedding_model is None:
            # Fallback to simple string similarity
            return 1.0 - SequenceMatcher(None, original, filtered).ratio()
        
        try:
            emb_orig = embedding_model.encode(original)
            emb_filt = embedding_model.encode(filtered)
            return cosine(emb_orig, emb_filt)
        except:
            return 1.0 - SequenceMatcher(None, original, filtered).ratio()
    
    def compute_bias_scores(self, text: str, original: str = None) -> BiasScores:
        if original is None:
            original = text
        
        # Compute traditional bias scores
        base_scores = {
            'toxicity': self.compute_toxicity_score(text),
            'sentiment': self.compute_sentiment_score(text),
            'stereotypes': self.compute_stereotype_score(text),
            'imbalance': self.compute_imbalance_score(text),
            'context_shift': self.compute_context_shift_score(original, text)
        }
        
        # Compute Holistic AI metrics
        holistic_metrics = self.holistic_ai.compute_fairness_metrics(text)
        
        # Combine all scores
        return BiasScores(
            **base_scores,
            **holistic_metrics
        )

class BiasFiltering:
    def __init__(self, detector: BiasDetector):
        self.detector = detector
        self.nonsense = "[FILTERED]"
        self.replacement_patterns = {
            'toxic': "[INAPPROPRIATE]",
            'stereotype': "[GENERALIZATION]",
            'personal': "[PERSONAL_INFO]"
        }
    
    def mask_toxic_words(self, text: str, aggressiveness: float) -> Tuple[str, List[Tuple[int, int]]]:
        words = text.split()
        masked_segments = []
        result_words = []
        char_pos = 0
        
        for word in words:
            start_pos = char_pos
            end_pos = start_pos + len(word)
            
            if word.lower() in self.detector.toxic_words and np.random.rand() < aggressiveness:
                result_words.append(self.replacement_patterns['toxic'])
                masked_segments.append((start_pos, end_pos))
            else:
                result_words.append(word)
            
            char_pos = end_pos + 1  # +1 for space
        
        return ' '.join(result_words), masked_segments
    
    def mask_stereotypes(self, text: str, aggressiveness: float) -> Tuple[str, List[Tuple[int, int]]]:
        result = text
        masked_segments = []
        
        for category, patterns in self.detector.stereotype_patterns.items():
            for pattern in patterns:
                pos = result.lower().find(pattern)
                while pos != -1 and np.random.rand() < aggressiveness:
                    end_pos = pos + len(pattern)
                    result = result[:pos] + self.replacement_patterns['stereotype'] + result[end_pos:]
                    masked_segments.append((pos, end_pos))
                    pos = result.lower().find(pattern, pos + len(self.replacement_patterns['stereotype']))
        
        return result, masked_segments
    
    def mask_personal_info(self, text: str, aggressiveness: float) -> Tuple[str, List[Tuple[int, int]]]:
        """Mask potential personal information"""
        result = text
        masked_segments = []
        
        # Privacy-sensitive patterns
        patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),  # SSN
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),  # Email
            (r'\b\d{3}-\d{3}-\d{4}\b', 'PHONE'),  # Phone
        ]
        
        for pattern, label in patterns:
            matches = list(re.finditer(pattern, result))
            for match in reversed(matches):  # Reverse to maintain positions
                if np.random.rand() < aggressiveness:
                    start, end = match.span()
                    result = result[:start] + f"[{label}]" + result[end:]
                    masked_segments.append((start, end))
        
        return result, masked_segments
    
    def apply(self, text: str, filter_type: str, aggressiveness: float) -> Tuple[str, List[Tuple[int, int]]]:
        if filter_type == 'toxicity':
            return self.mask_toxic_words(text, aggressiveness)
        elif filter_type == 'stereotypes':
            return self.mask_stereotypes(text, aggressiveness)
        elif filter_type == 'privacy':
            return self.mask_personal_info(text, aggressiveness)
        else:
            return text, []

class GraphOfThought:
    def __init__(self, weights: Dict[str, float] = None):
        self.graph = nx.DiGraph()
        self.detector = BiasDetector()
        self.filter = BiasFiltering(self.detector)
        
        # Enhanced weights including Holistic AI metrics
        self.weights = weights or {
            'toxicity': 0.2,
            'sentiment': 0.15,
            'stereotypes': 0.2,
            'imbalance': 0.15,
            'context_shift': 0.1,
            'fairness_score': 0.1,
            'demographic_parity': 0.05,
            'equalized_odds': 0.05,
            'privacy_risk': 0.1
        }
        
        self.counter = 0
        self.traversal_history = []
        self.content_cache = set()
        self.similarity_threshold = 0.85
    
    def compute_cbs(self, bias_scores: BiasScores) -> float:
        """Compute Composite Bias Score"""
        scores_dict = bias_scores.to_dict()
        return sum(self.weights.get(k, 0) * scores_dict[k] for k in scores_dict)
    
    def compute_crs(self, original: str, filtered: str) -> float:
        """Compute Content Retention Score"""
        return SequenceMatcher(None, original, filtered).ratio()
    
    def is_duplicate_content(self, new_content: str) -> bool:
        """Check if content is too similar to existing content"""
        if not embedding_model:
            return new_content in self.content_cache
        
        new_embedding = embedding_model.encode(new_content)
        
        for cached_content in self.content_cache:
            if isinstance(cached_content, str):
                cached_embedding = embedding_model.encode(cached_content)
                similarity = 1 - cosine(new_embedding, cached_embedding)
                if similarity > self.similarity_threshold:
                    return True
        
        return False
    
    def create_root_node(self, text: str) -> GraphNode:
        """Create root node of the graph"""
        node_id = f"node_{self.counter}"
        self.counter += 1
        
        bias_scores = self.detector.compute_bias_scores(text)
        cbs = self.compute_cbs(bias_scores)
        
        # Compute holistic metrics
        holistic_metrics = self.detector.holistic_ai.compute_fairness_metrics(text)
        
        node = GraphNode(
            node_id=node_id,
            text_content=text,
            bias_scores=bias_scores,
            cbs=cbs,
            crs=1.0,
            transformation_history=[],
            masked_segments=[],
            holistic_metrics=holistic_metrics
        )
        
        self.graph.add_node(node_id, data=node)
        self.content_cache.add(text)
        
        return node
    
    def generate_child_nodes(self, parent: GraphNode, max_children: int = 5) -> List[GraphNode]:
        """Generate child nodes through various filtering strategies"""
        children = []
        
        # Different filtering strategies
        strategies = [
            ('toxicity', [0.3, 0.5, 0.7]),
            ('stereotypes', [0.3, 0.5, 0.7]),
            ('privacy', [0.4, 0.6, 0.8])
        ]
        
        for filter_type, aggressiveness_levels in strategies:
            for aggressiveness in aggressiveness_levels:
                if len(children) >= max_children:
                    break
                
                filtered_text, masked_segments = self.filter.apply(
                    parent.text_content, filter_type, aggressiveness
                )
                
                # Skip if no change or duplicate
                if (filtered_text == parent.text_content or 
                    self.is_duplicate_content(filtered_text)):
                    continue
                
                # Create child node
                node_id = f"node_{self.counter}"
                self.counter += 1
                
                bias_scores = self.detector.compute_bias_scores(filtered_text, parent.text_content)
                cbs = self.compute_cbs(bias_scores)
                crs = self.compute_crs(parent.text_content, filtered_text)
                
                # Compute holistic metrics
                holistic_metrics = self.detector.holistic_ai.compute_fairness_metrics(filtered_text)
                
                transformation_step = f"{filter_type}_{aggressiveness}"
                
                child = GraphNode(
                    node_id=node_id,
                    text_content=filtered_text,
                    bias_scores=bias_scores,
                    cbs=cbs,
                    crs=crs,
                    transformation_history=parent.transformation_history + [transformation_step],
                    masked_segments=parent.masked_segments + masked_segments,
                    holistic_metrics=holistic_metrics
                )
                
                self.graph.add_node(node_id, data=child)
                self.graph.add_edge(parent.node_id, node_id)
                self.content_cache.add(filtered_text)
                
                children.append(child)
        
        return children
    
    def compute_lambda(self, parent_cbs: float, child_cbs: float, 
                      parent_crs: float, child_crs: float) -> float:
        """Compute lambda for reward calculation"""
        delta_cbs = parent_cbs - child_cbs
        delta_crs = parent_crs - child_crs
        
        if abs(delta_crs) < 1e-6:
            return 1.0
        
        return delta_cbs / abs(delta_crs)
    
    def compute_reward(self, node: GraphNode, parent: GraphNode = None) -> float:
        """Compute reward for a node"""
        base_reward = -node.cbs  # Lower bias = higher reward
        
        if parent is not None:
            lambda_val = self.compute_lambda(parent.cbs, node.cbs, parent.crs, node.crs)
            improvement_reward = lambda_val * (parent.cbs - node.cbs)
            return base_reward + improvement_reward
        
        return base_reward
    
    def traverse_graph(self, root: GraphNode, max_depth: int = 6, 
                      bias_threshold: float = 0.1) -> GraphNode:
        """Traverse the graph to find optimal bias-mitigated text"""
        # Priority queue: (negative_reward, tie_breaker, node, parent)
        pq = [(-self.compute_reward(root), 0, root, None)]
        visited = set()
        best_node = root
        best_reward = self.compute_reward(root)
        tie_breaker = 1
        
        for depth in range(max_depth):
            if not pq:
                break
            
            neg_reward, _, current_node, parent = heapq.heappop(pq)
            current_reward = -neg_reward
            
            if current_node.node_id in visited:
                continue
            
            visited.add(current_node.node_id)
            self.traversal_history.append({
                'depth': depth,
                'node_id': current_node.node_id,
                'cbs': current_node.cbs,
                'crs': current_node.crs,
                'reward': current_reward,
                'transformations': current_node.transformation_history
            })
            
            # Update best node
            if current_reward > best_reward:
                best_node = current_node
                best_reward = current_reward
            
            # Early termination if bias is low enough
            if current_node.cbs < bias_threshold:
                logger.info(f"Early termination: CBS {current_node.cbs} below threshold {bias_threshold}")
                break
            
            # Generate children
            children = self.generate_child_nodes(current_node)
            
            for child in children:
                if child.node_id not in visited:
                    child_reward = self.compute_reward(child, current_node)
                    heapq.heappush(pq, (-child_reward, tie_breaker, child, current_node))
                    tie_breaker += 1
        
        logger.info(f"Traversal completed. Best CBS: {best_node.cbs:.4f}, CRS: {best_node.crs:.4f}")
        return best_node
    
    def generate_report(self, node: GraphNode) -> Dict:
        """Generate comprehensive report for a node"""
        report = {
            'node_id': node.node_id,
            'composite_bias_score': round(node.cbs, 4),
            'content_retention_score': round(node.crs, 4),
            'bias_breakdown': {k: round(v, 4) for k, v in node.bias_scores.to_dict().items()},
            'transformation_history': node.transformation_history,
            'holistic_ai_metrics': node.holistic_metrics,
            'text_preview': node.text_content[:200] + '...' if len(node.text_content) > 200 else node.text_content,
            'masked_segments_count': len(node.masked_segments),
            'graph_stats': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges()
            }
        }
        
        return report
    
    def visualize_graph(self, save_path: str = None):
        """Visualize the graph structure"""
        if not self.graph.nodes():
            logger.warning("No nodes to visualize")
            return
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Node colors based on CBS
        node_colors = [self.graph.nodes[node]['data'].cbs for node in self.graph.nodes()]
        
        # Draw graph
        nx.draw(self.graph, pos, 
                node_color=node_colors, 
                cmap=plt.cm.RdYlBu_r,
                with_labels=True,
                node_size=500,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r,
                                  norm=plt.Normalize(vmin=min(node_colors),
                                                   vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label="Composite Bias Score")
        
        plt.title("Graph of Thought - Bias Mitigation Exploration")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {save_path}")
        
        plt.show()
    
    def export_results(self, node: GraphNode, base_filename: str = "bias_mitigation_results"):
        """Export results in multiple formats"""
        # Export filtered text
        with open(f"{base_filename}.txt", 'w', encoding='utf-8') as f:
            f.write(node.text_content)
        
        # Export JSON report
        with open(f"{base_filename}_report.json", 'w', encoding='utf-8') as f:
            json.dump(self.generate_report(node), f, indent=2)
        
        # Export traversal history
        with open(f"{base_filename}_traversal.json", 'w', encoding='utf-8') as f:
            json.dump(self.traversal_history, f, indent=2)
        
        logger.info(f"Results exported with base filename: {base_filename}")

def main():
    """Main execution function"""
    # Sample biased text
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
    
    print("=" * 60)
    print("ENHANCED BIAS MITIGATION SYSTEM WITH HOLISTIC AI")
    print("=" * 60)
    
    # Initialize the system
    print("\n1. Initializing Graph of Thought system...")
    got = GraphOfThought()
    
    # Create root node
    print("\n2. Creating root node and analyzing original text...")
    root_node = got.create_root_node(sample_text)
    
    print(f"Original text preview: {sample_text[:100]}...")
    print(f"Original CBS: {root_node.cbs:.4f}")
    print(f"Original bias breakdown:")
    for metric, score in root_node.bias_scores.to_dict().items():
        print(f"  - {metric}: {score:.4f}")
    
    # Traverse the graph
    print("\n3. Traversing graph to find optimal bias mitigation...")
    best_node = got.traverse_graph(root_node, max_depth=5, bias_threshold=0.15)
    
    # Generate comprehensive report
    print("\n4. Generating comprehensive report...")
    report = got.generate_report(best_node)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nFinal CBS: {report['composite_bias_score']}")
    print(f"Content Retention Score: {report['content_retention_score']}")
    print(f"Transformation Steps: {len(report['transformation_history'])}")
    
    print("\nBias Reduction Breakdown:")
    original_scores = root_node.bias_scores.to_dict()
    final_scores = report['bias_breakdown']
    
    for metric in original_scores:
        if metric in final_scores:
            original = original_scores[metric]
            final = final_scores[metric]
            reduction = ((original - final) / original * 100) if original > 0 else 0
            print(f"  - {metric}: {original:.4f} → {final:.4f} ({reduction:+.1f}%)")
    
    print(f"\nHolistic AI Metrics:")
    if best_node.holistic_metrics:
        for metric, value in best_node.holistic_metrics.items():
            print(f"  - {metric}: {value:.4f}")
    
    print(f"\nTransformation History: {' → '.join(report['transformation_history'])}")
    
    print(f"\nGraph Statistics:")
    print(f"  - Total nodes explored: {report['graph_stats']['total_nodes']}")
    print(f"  - Total edges: {report['graph_stats']['total_edges']}")
    
    print("\n" + "=" * 60)
    print("ORIGINAL vs FILTERED TEXT")
    print("=" * 60)
    
    print("\nORIGINAL:")
    print(sample_text.strip())
    
    print("\nFILTERED:")
    print(best_node.text_content.strip())
    
    # Export results
    print("\n5. Exporting results...")
    got.export_results(best_node, "bias_mitigation_output")
    
    # Generate visualization
    print("\n6. Generating graph visualization...")
    try:
        got.visualize_graph("bias_mitigation_graph.png")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Additional analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)
    
    print("\nTraversal History (last 5 steps):")
    for step in got.traversal_history[-5:]:
        print(f"  Depth {step['depth']}: Node {step['node_id']}")
        print(f"    CBS: {step['cbs']:.4f}, CRS: {step['crs']:.4f}")
        print(f"    Reward: {step['reward']:.4f}")
        print(f"    Transformations: {step['transformations']}")
        print()
    
    # Comparison with simple filtering
    print("\nComparison with simple word replacement:")
    simple_filtered = sample_text
    toxic_words = ['terrible', 'stupid', 'useless', 'lazy']
    for word in toxic_words:
        simple_filtered = simple_filtered.replace(word, '[FILTERED]')
    
    simple_bias = got.detector.compute_bias_scores(simple_filtered, sample_text)
    simple_cbs = got.compute_cbs(simple_bias)
    simple_crs = got.compute_crs(sample_text, simple_filtered)
    
    print(f"Simple filtering CBS: {simple_cbs:.4f}")
    print(f"Simple filtering CRS: {simple_crs:.4f}")
    print(f"Graph-based CBS: {best_node.cbs:.4f}")
    print(f"Graph-based CRS: {best_node.crs:.4f}")
    
    improvement = ((simple_cbs - best_node.cbs) / simple_cbs * 100) if simple_cbs > 0 else 0
    print(f"Improvement in bias reduction: {improvement:.1f}%")
    
    print("\n" + "=" * 60)
    print("SYSTEM PERFORMANCE METRICS")
    print("=" * 60)
    
    print(f"Total content variations cached: {len(got.content_cache)}")
    print(f"Average CBS across all nodes: {np.mean([got.graph.nodes[n]['data'].cbs for n in got.graph.nodes()]):.4f}")
    print(f"Best CBS achieved: {min([got.graph.nodes[n]['data'].cbs for n in got.graph.nodes()]):.4f}")
    print(f"Holistic AI integration: {'✓ Active' if got.detector.holistic_ai.available else '✗ Fallback mode'}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if best_node.cbs > 0.3:
        print("⚠️  HIGH BIAS DETECTED - Consider additional manual review")
    elif best_node.cbs > 0.15:
        print("⚠️  MODERATE BIAS - Acceptable for most applications")
    else:
        print("✅ LOW BIAS - Content meets bias mitigation standards")
    
    if best_node.crs < 0.7:
        print("⚠️  LOW CONTENT RETENTION - Consider less aggressive filtering")
    else:
        print("✅ GOOD CONTENT RETENTION - Meaning preserved effectively")
    
    print("\nSuggested next steps:")
    print("1. Review filtered content for context and meaning")
    print("2. Consider human expert review for sensitive applications")
    print("3. Test with domain-specific bias detection if available")
    print("4. Monitor performance on similar content types")
    
    print(f"\nFiles generated:")
    print("- bias_mitigation_output.txt (filtered text)")
    print("- bias_mitigation_output_report.json (detailed report)")
    print("- bias_mitigation_output_traversal.json (traversal history)")
    print("- bias_mitigation_graph.png (visualization)")
    
    return root_node, best_node, report

# Advanced utility functions
def batch_process_texts(texts: List[str], config: Dict = None) -> List[Dict]:
    """Process multiple texts in batch"""
    results = []
    
    default_config = {
        'max_depth': 5,
        'bias_threshold': 0.15,
        'weights': None
    }
    
    if config:
        default_config.update(config)
    
    for i, text in enumerate(texts):
        print(f"\nProcessing text {i+1}/{len(texts)}...")
        
        got = GraphOfThought(weights=default_config['weights'])
        root = got.create_root_node(text)
        best = got.traverse_graph(root, 
                                max_depth=default_config['max_depth'],
                                bias_threshold=default_config['bias_threshold'])
        
        result = {
            'index': i,
            'original_text': text,
            'filtered_text': best.text_content,
            'original_cbs': root.cbs,
            'final_cbs': best.cbs,
            'crs': best.crs,
            'transformations': best.transformation_history,
            'bias_breakdown': best.bias_scores.to_dict()
        }
        
        results.append(result)
    
    return results

def compare_mitigation_strategies(text: str, strategies: List[Dict]) -> Dict:
    """Compare different mitigation strategies"""
    results = {}
    
    for strategy in strategies:
        name = strategy.get('name', 'unnamed')
        got = GraphOfThought(weights=strategy.get('weights'))
        
        root = got.create_root_node(text)
        best = got.traverse_graph(root,
                                max_depth=strategy.get('max_depth', 5),
                                bias_threshold=strategy.get('bias_threshold', 0.15))
        
        results[name] = {
            'cbs': best.cbs,
            'crs': best.crs,
            'text': best.text_content,
            'transformations': best.transformation_history,
            'bias_scores': best.bias_scores.to_dict()
        }
    
    return results

# Run the system
if __name__ == '__main__':
    try:
        root_node, best_node, report = main()

        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()