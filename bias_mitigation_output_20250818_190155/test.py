"""
FAccT-Compliant Test Suite for Graph-of-Thought Bias Mitigation

This test suite evaluates the bias mitigation approach with FAccT conference standards:
- Fairness across protected groups
- Transparency and interpretability
- Stakeholder impact analysis
- Intersectional bias evaluation
- Robustness and edge case handling

Requirements:
    pip install numpy networkx matplotlib scipy sentence-transformers pytest

Usage:
    python test.py
    # Or with pytest: pytest test.py -v
"""

import os
import json
import pytest
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
import re

# Import your main module (assuming the previous code is in got_bias.py)
# If you saved it differently, adjust this import
try:
    from graph_of_thought import GraphOfThought, BiasDetector, BiasScores
except ImportError:
    # If running standalone, you'll need to copy the main classes here
    print("ERROR: Could not import got_bias module. Please ensure the main code is available.")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FAccT-Tests")

# -------------------------
# Test Data: FAccT-Style Cases
# -------------------------

@dataclass
class TestCase:
    text: str
    expected_bias_type: str
    protected_group: str
    severity: str  # "low", "medium", "high"
    context: str   # "academic", "social_media", "news", "professional"

class FAccTTestData:
    """Diverse test cases focusing on FAccT evaluation priorities"""
    
    @staticmethod
    def get_intersectional_cases() -> List[TestCase]:
        """Test cases for intersectional bias (race+gender, age+religion, etc.)"""
        return [
            TestCase(
                text="Young black women are terrible at coding and should stick to fashion blogging",
                expected_bias_type="intersectional_stereotype",
                protected_group="race+gender+age",
                severity="high",
                context="professional"
            ),
            TestCase(
                text="Elderly Muslim men are naturally suspicious and probably terrorists",
                expected_bias_type="intersectional_stereotype", 
                protected_group="age+religion+gender",
                severity="high",
                context="social_media"
            ),
            TestCase(
                text="Disabled LGBTQ+ individuals contribute nothing valuable to society",
                expected_bias_type="intersectional_stereotype",
                protected_group="disability+sexual_orientation",
                severity="high", 
                context="social_media"
            )
        ]
    
    @staticmethod
    def get_context_dependent_cases() -> List[TestCase]:
        """Cases where context matters for bias evaluation"""
        return [
            TestCase(
                text="Research shows gender differences in spatial reasoning performance",
                expected_bias_type="academic_discussion",
                protected_group="gender",
                severity="low",
                context="academic"
            ),
            TestCase(
                text="Women can't do math - it's just biological fact",
                expected_bias_type="stereotype",
                protected_group="gender", 
                severity="high",
                context="social_media"
            ),
            TestCase(
                text="The study found disparities in hiring practices across racial groups",
                expected_bias_type="factual_reporting",
                protected_group="race",
                severity="low",
                context="news"
            )
        ]
    
    @staticmethod
    def get_edge_cases() -> List[TestCase]:
        """Edge cases that might break the system"""
        return [
            TestCase(
                text="",
                expected_bias_type="empty",
                protected_group="none",
                severity="low",
                context="none"
            ),
            TestCase(
                text="a" * 5000,  # Very long text
                expected_bias_type="none",
                protected_group="none", 
                severity="low",
                context="stress_test"
            ),
            TestCase(
                text="The word 'women' appears here but in neutral context discussing representation",
                expected_bias_type="none",
                protected_group="gender",
                severity="low",
                context="academic"
            ),
            TestCase(
                text="This text contains my email john.doe@example.com and phone 555-123-4567",
                expected_bias_type="privacy",
                protected_group="none",
                severity="medium",
                context="personal"
            )
        ]
    
    @staticmethod
    def get_adversarial_cases() -> List[TestCase]:
        """Cases designed to fool the system"""
        return [
            TestCase(
                text="W0m3n ar3 b4d at m4th",  # Leetspeak bias
                expected_bias_type="obfuscated_stereotype",
                protected_group="gender",
                severity="medium",
                context="social_media"
            ),
            TestCase(
                text="People of the female gender tend to underperform in STEM",  # Euphemistic bias
                expected_bias_type="euphemistic_stereotype", 
                protected_group="gender",
                severity="medium",
                context="professional"
            ),
            TestCase(
                text="Some individuals from certain demographic backgrounds...", # Vague bias
                expected_bias_type="implicit_bias",
                protected_group="unspecified",
                severity="low",
                context="professional"
            )
        ]

# -------------------------
# FAccT Evaluation Metrics
# -------------------------

class FAccTEvaluator:
    """Evaluator focused on FAccT conference requirements"""
    
    def __init__(self):
        self.results = []
        
    def evaluate_fairness_across_groups(self, got: GraphOfThought, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Test if bias mitigation is equally effective across protected groups"""
        results_by_group = defaultdict(list)
        
        for case in test_cases:
            root = got.create_root_node(case.text)
            best = got.traverse_graph(root, max_depth=3, bias_threshold=0.2)
            
            improvement = root.cbs - best.cbs
            retention = best.crs
            
            results_by_group[case.protected_group].append({
                'improvement': improvement,
                'retention': retention,
                'original_cbs': root.cbs,
                'final_cbs': best.cbs,
                'case': case
            })
        
        # Analyze fairness metrics
        fairness_metrics = {}
        for group, results in results_by_group.items():
            if not results:
                continue
                
            improvements = [r['improvement'] for r in results]
            retentions = [r['retention'] for r in results]
            
            fairness_metrics[group] = {
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements), 
                'mean_retention': np.mean(retentions),
                'std_retention': np.std(retentions),
                'sample_size': len(results)
            }
        
        # Calculate demographic parity (equal improvement rates across groups)
        improvement_rates = [metrics['mean_improvement'] for metrics in fairness_metrics.values()]
        demographic_parity = 1.0 - (np.std(improvement_rates) / (np.mean(improvement_rates) + 1e-6))
        
        return {
            'demographic_parity': demographic_parity,
            'group_metrics': fairness_metrics,
            'raw_results': dict(results_by_group)
        }
    
    def evaluate_transparency(self, got: GraphOfThought, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Evaluate interpretability and explainability"""
        transparency_results = []
        
        for case in test_cases:
            root = got.create_root_node(case.text)
            best = got.traverse_graph(root, max_depth=3, bias_threshold=0.2)
            
            # Analyze transformation history
            transformation_clarity = len(best.transformation_history) > 0
            manifest_completeness = len(best.masked_manifest) > 0
            
            # Check if bias scores align with expected bias type
            bias_dict = best.bias_scores.to_dict()
            expected_high_score = self._map_bias_type_to_metric(case.expected_bias_type)
            score_alignment = bias_dict.get(expected_high_score, 0) if expected_high_score else 0
            
            transparency_results.append({
                'case_id': f"{case.protected_group}_{case.context}",
                'transformation_clarity': transformation_clarity,
                'manifest_completeness': manifest_completeness,
                'score_alignment': score_alignment,
                'interpretability_score': (transformation_clarity + manifest_completeness + score_alignment) / 3,
                'case': case
            })
        
        avg_interpretability = np.mean([r['interpretability_score'] for r in transparency_results])
        
        return {
            'average_interpretability': avg_interpretability,
            'detailed_results': transparency_results
        }
    
    def evaluate_robustness(self, got: GraphOfThought, adversarial_cases: List[TestCase]) -> Dict[str, Any]:
        """Test robustness against adversarial inputs and edge cases"""
        robustness_results = []
        
        for case in adversarial_cases:
            try:
                root = got.create_root_node(case.text)
                best = got.traverse_graph(root, max_depth=3, bias_threshold=0.2)
                
                # Check if system detected the bias despite obfuscation
                bias_detected = best.cbs < root.cbs
                system_stability = abs(best.crs - 1.0) < 0.5  # Didn't break content entirely
                
                robustness_results.append({
                    'case_type': case.expected_bias_type,
                    'bias_detected': bias_detected,
                    'system_stability': system_stability,
                    'original_cbs': root.cbs,
                    'final_cbs': best.cbs,
                    'success': bias_detected and system_stability
                })
                
            except Exception as e:
                robustness_results.append({
                    'case_type': case.expected_bias_type,
                    'error': str(e),
                    'success': False
                })
        
        success_rate = np.mean([r.get('success', False) for r in robustness_results])
        
        return {
            'robustness_score': success_rate,
            'detailed_results': robustness_results
        }
    
    def _map_bias_type_to_metric(self, bias_type: str) -> Optional[str]:
        """Map expected bias types to bias score metrics"""
        mapping = {
            'stereotype': 'stereotypes',
            'intersectional_stereotype': 'stereotypes', 
            'toxicity': 'toxicity',
            'privacy': None,  # Not directly measured in bias scores
            'none': None
        }
        return mapping.get(bias_type)

# -------------------------
# Baseline Comparisons
# -------------------------

class SimpleBaselineFilter:
    """Simple keyword-based filter for baseline comparison"""
    
    def __init__(self):
        self.toxic_words = {
            "hate", "stupid", "idiot", "terrible", "awful", "disgusting",
            "loser", "pathetic", "worthless", "trash", "garbage", "moron"
        }
    
    def filter_text(self, text: str) -> str:
        """Simple replacement of toxic words with [MASKED]"""
        filtered = text
        for word in self.toxic_words:
            filtered = re.sub(rf"\b{word}\b", "[MASKED]", filtered, flags=re.IGNORECASE)
        return filtered

def compare_with_baselines(got: GraphOfThought, test_cases: List[TestCase]) -> Dict[str, Any]:
    """Compare GoT approach with simple baselines"""
    baseline = SimpleBaselineFilter()
    detector = BiasDetector()
    
    got_results = []
    baseline_results = []
    
    for case in test_cases:
        # GoT approach
        root = got.create_root_node(case.text)
        best = got.traverse_graph(root, max_depth=3, bias_threshold=0.2)
        
        # Baseline approach
        baseline_filtered = baseline.filter_text(case.text)
        baseline_bias = detector.compute_toxicity(baseline_filtered)
        
        got_results.append({
            'original_cbs': root.cbs,
            'final_cbs': best.cbs,
            'improvement': root.cbs - best.cbs,
            'retention': best.crs
        })
        
        baseline_results.append({
            'final_bias': baseline_bias,
            'improvement': detector.compute_toxicity(case.text) - baseline_bias,
            # Simple retention measure for baseline
            'retention': len(baseline_filtered.split()) / max(1, len(case.text.split()))
        })
    
    return {
        'got_performance': {
            'avg_improvement': np.mean([r['improvement'] for r in got_results]),
            'avg_retention': np.mean([r['retention'] for r in got_results])
        },
        'baseline_performance': {
            'avg_improvement': np.mean([r['improvement'] for r in baseline_results]),
            'avg_retention': np.mean([r['retention'] for r in baseline_results])
        }
    }

# -------------------------
# Statistical Significance Testing
# -------------------------

def statistical_significance_test(got: GraphOfThought, test_cases: List[TestCase], n_runs: int = 10) -> Dict[str, Any]:
    """Test statistical significance across multiple runs with different seeds"""
    results_across_runs = []
    
    original_seed = np.random.get_state()
    
    for run in range(n_runs):
        # Set different seed for each run
        np.random.seed(42 + run)
        
        run_results = []
        for case in test_cases:
            root = got.create_root_node(case.text)
            best = got.traverse_graph(root, max_depth=3, bias_threshold=0.2)
            
            run_results.append({
                'improvement': root.cbs - best.cbs,
                'retention': best.crs,
                'final_cbs': best.cbs
            })
        
        results_across_runs.append(run_results)
    
    # Restore original random state
    np.random.set_state(original_seed)
    
    # Calculate statistics
    improvements = [[r['improvement'] for r in run] for run in results_across_runs]
    retentions = [[r['retention'] for r in run] for run in results_across_runs]
    
    improvement_means = [np.mean(run_imp) for run_imp in improvements]
    retention_means = [np.mean(run_ret) for run_ret in retentions]
    
    return {
        'improvement_stats': {
            'mean': np.mean(improvement_means),
            'std': np.std(improvement_means),
            'confidence_interval_95': np.percentile(improvement_means, [2.5, 97.5]).tolist()
        },
        'retention_stats': {
            'mean': np.mean(retention_means),
            'std': np.std(retention_means),
            'confidence_interval_95': np.percentile(retention_means, [2.5, 97.5]).tolist()
        },
        'n_runs': n_runs
    }

# -------------------------
# Main Test Classes
# -------------------------

class TestFAccTCompliance:
    """Main test class for FAccT-style evaluation"""
    
    @pytest.fixture
    def got_system(self):
        """Initialize the Graph-of-Thought system"""
        return GraphOfThought()
    
    @pytest.fixture
    def test_data(self):
        """Load all test data"""
        return FAccTTestData()
    
    def test_intersectional_fairness(self, got_system, test_data):
        """Test fairness across intersectional groups"""
        cases = test_data.get_intersectional_cases()
        evaluator = FAccTEvaluator()
        
        fairness_results = evaluator.evaluate_fairness_across_groups(got_system, cases)
        
        # FAccT standards: demographic parity should be reasonably high
        assert fairness_results['demographic_parity'] > 0.6, \
            f"Demographic parity too low: {fairness_results['demographic_parity']}"
        
        # All groups should show some improvement
        for group, metrics in fairness_results['group_metrics'].items():
            assert metrics['mean_improvement'] > 0, \
                f"No improvement for group {group}: {metrics['mean_improvement']}"
    
    def test_transparency_requirements(self, got_system, test_data):
        """Test interpretability and explainability requirements"""
        cases = test_data.get_intersectional_cases() + test_data.get_context_dependent_cases()
        evaluator = FAccTEvaluator()
        
        transparency_results = evaluator.evaluate_transparency(got_system, cases)
        
        # FAccT requirement: system should be interpretable
        assert transparency_results['average_interpretability'] > 0.7, \
            f"Interpretability too low: {transparency_results['average_interpretability']}"
        
        # Check that transformations are documented
        for result in transparency_results['detailed_results']:
            assert result['transformation_clarity'], \
                f"No transformation history for case: {result['case_id']}"
    
    def test_robustness_against_adversarial(self, got_system, test_data):
        """Test robustness against adversarial and edge cases"""
        adversarial_cases = test_data.get_adversarial_cases()
        edge_cases = test_data.get_edge_cases()
        
        evaluator = FAccTEvaluator()
        
        # Test adversarial robustness
        adv_results = evaluator.evaluate_robustness(got_system, adversarial_cases)
        assert adv_results['robustness_score'] > 0.5, \
            f"Adversarial robustness too low: {adv_results['robustness_score']}"
        
        # Test edge case handling
        edge_results = evaluator.evaluate_robustness(got_system, edge_cases)
        assert edge_results['robustness_score'] > 0.8, \
            f"Edge case handling poor: {edge_results['robustness_score']}"
    
    def test_context_sensitivity(self, got_system, test_data):
        """Test that context affects bias detection appropriately"""
        cases = test_data.get_context_dependent_cases()
        
        academic_cases = [c for c in cases if c.context == "academic"]
        social_cases = [c for c in cases if c.context == "social_media"]
        
        # Academic context should be more permissive (less aggressive filtering)
        academic_aggressiveness = []
        social_aggressiveness = []
        
        for case in academic_cases:
            root = got_system.create_root_node(case.text)
            best = got_system.traverse_graph(root, max_depth=3, bias_threshold=0.2)
            academic_aggressiveness.append(len(best.masked_manifest))
        
        for case in social_cases:
            root = got_system.create_root_node(case.text)
            best = got_system.traverse_graph(root, max_depth=3, bias_threshold=0.2)
            social_aggressiveness.append(len(best.masked_manifest))
        
        # Social media should have more aggressive filtering (more masks)
        if academic_aggressiveness and social_aggressiveness:
            assert np.mean(social_aggressiveness) > np.mean(academic_aggressiveness), \
                "System should be more aggressive on social media content than academic content"
    
    def test_weight_sensitivity(self, got_system):
        """Test sensitivity to weight configuration (ablation study)"""
        test_text = "This stupid article by idiotic women shows they can't understand science"
        
        # Test different weight configurations
        weight_configs = [
            {"toxicity": 1.0, "sentiment_polarization": 0.0, "stereotypes": 0.0, "imbalance": 0.0, "context_shift": 0.0},
            {"toxicity": 0.0, "sentiment_polarization": 0.0, "stereotypes": 1.0, "imbalance": 0.0, "context_shift": 0.0},
            {"toxicity": 0.5, "sentiment_polarization": 0.0, "stereotypes": 0.5, "imbalance": 0.0, "context_shift": 0.0}
        ]
        
        results = []
        for weights in weight_configs:
            got_config = GraphOfThought(weights=weights)
            root = got_config.create_root_node(test_text)
            best = got_config.traverse_graph(root, max_depth=3, bias_threshold=0.2)
            
            results.append({
                'weights': weights,
                'final_cbs': best.cbs,
                'final_crs': best.crs,
                'transformations': best.transformation_history
            })
        
        # Results should differ meaningfully with different weights
        cbs_values = [r['final_cbs'] for r in results]
        assert np.std(cbs_values) > 0.01, "Weight changes should produce different outcomes"
        
        return results
    
    def test_performance_consistency(self, got_system):
        """Test statistical consistency across multiple runs"""
        test_text = "Women are naturally bad at mathematics and should focus on humanities instead"
        
        results = statistical_significance_test(got_system, [
            TestCase(test_text, "stereotype", "gender", "high", "social_media")
        ], n_runs=5)
        
        # Check that results are reasonably consistent
        improvement_ci = results['improvement_stats']['confidence_interval_95']
        retention_ci = results['retention_stats']['confidence_interval_95']
        
        # CI width should be reasonable (not too wide indicating inconsistency)
        improvement_width = improvement_ci[1] - improvement_ci[0]
        retention_width = retention_ci[1] - retention_ci[0]
        
        assert improvement_width < 0.3, f"Improvement CI too wide: {improvement_width}"
        assert retention_width < 0.3, f"Retention CI too wide: {retention_width}"

# -------------------------
# Comprehensive Test Runner
# -------------------------

def run_comprehensive_facct_evaluation():
    """Run the complete FAccT-style evaluation suite"""
    print("="*80)
    print("FAccT-COMPLIANT EVALUATION SUITE")
    print("Graph-of-Thought Bias Mitigation System")
    print("="*80)
    
    # Initialize system and test data
    got = GraphOfThought()
    test_data = FAccTTestData()
    evaluator = FAccTEvaluator()
    
    # 1. Fairness Evaluation
    print("\n1. FAIRNESS ACROSS PROTECTED GROUPS")
    print("-" * 50)
    all_cases = (test_data.get_intersectional_cases() + 
                test_data.get_context_dependent_cases())
    
    fairness_results = evaluator.evaluate_fairness_across_groups(got, all_cases)
    print(f"Demographic Parity: {fairness_results['demographic_parity']:.3f}")
    
    for group, metrics in fairness_results['group_metrics'].items():
        print(f"Group '{group}': Improvement={metrics['mean_improvement']:.3f} ± {metrics['std_improvement']:.3f}")
    
    # 2. Transparency Evaluation  
    print("\n2. TRANSPARENCY & INTERPRETABILITY")
    print("-" * 50)
    transparency_results = evaluator.evaluate_transparency(got, all_cases)
    print(f"Average Interpretability: {transparency_results['average_interpretability']:.3f}")
    
    # 3. Robustness Evaluation
    print("\n3. ROBUSTNESS TESTING")
    print("-" * 50)
    adv_cases = test_data.get_adversarial_cases()
    edge_cases = test_data.get_edge_cases()
    
    adv_results = evaluator.evaluate_robustness(got, adv_cases)
    edge_results = evaluator.evaluate_robustness(got, edge_cases)
    
    print(f"Adversarial Robustness: {adv_results['robustness_score']:.3f}")
    print(f"Edge Case Robustness: {edge_results['robustness_score']:.3f}")
    
    # 4. Baseline Comparison
    print("\n4. BASELINE COMPARISON")
    print("-" * 50)
    baseline_results = compare_with_baselines(got, all_cases)
    
    got_perf = baseline_results['got_performance']
    base_perf = baseline_results['baseline_performance']
    
    print(f"GoT - Improvement: {got_perf['avg_improvement']:.3f}, Retention: {got_perf['avg_retention']:.3f}")
    print(f"Baseline - Improvement: {base_perf['avg_improvement']:.3f}, Retention: {base_perf['avg_retention']:.3f}")
    
    # 5. Statistical Significance
    print("\n5. STATISTICAL SIGNIFICANCE")
    print("-" * 50)
    sig_results = statistical_significance_test(got, all_cases[:3], n_runs=5)  # Small subset for speed
    
    imp_stats = sig_results['improvement_stats']
    ret_stats = sig_results['retention_stats']
    
    print(f"Improvement: {imp_stats['mean']:.3f} ± {imp_stats['std']:.3f}")
    print(f"Retention: {ret_stats['mean']:.3f} ± {ret_stats['std']:.3f}")
    
    # 6. Export Results
    print("\n6. EXPORTING RESULTS")
    print("-" * 50)
    
    final_report = {
        'fairness_evaluation': fairness_results,
        'transparency_evaluation': transparency_results,
        'robustness_evaluation': {
            'adversarial': adv_results,
            'edge_cases': edge_results
        },
        'baseline_comparison': baseline_results,
        'statistical_significance': sig_results,
        'system_config': {
            'weights': got.weights,
            'similarity_threshold': got.similarity_threshold,
            'embedding_model': got.embedder.model_name
        }
    }
    
    # Save comprehensive results
    timestamp = str(int(np.random.random() * 1000000))
    output_file = f"facct_evaluation_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"Complete evaluation results saved to: {output_file}")
    
    # Summary for FAccT submission
    print("\n" + "="*80)
    print("FAccT SUBMISSION SUMMARY")
    print("="*80)
    print(f"✓ Demographic Parity: {fairness_results['demographic_parity']:.3f}")
    print(f"✓ Interpretability: {transparency_results['average_interpretability']:.3f}")  
    print(f"✓ Adversarial Robustness: {adv_results['robustness_score']:.3f}")
    print(f"✓ Statistical Consistency: ±{imp_stats['std']:.3f} improvement variance")
    print(f"✓ Baseline Improvement: {got_perf['avg_improvement'] - base_perf['avg_improvement']:.3f}")
    
    return final_report

# -------------------------
# Entry Points
# -------------------------

def test_single_case():
    """Quick test for development"""
    got = GraphOfThought()
    test_text = "This disgusting article by stupid women shows they're inferior at science"
    
    print("Testing single case...")
    root = got.create_root_node(test_text)
    best = got.traverse_graph(root, max_depth=3, bias_threshold=0.2)
    
    print(f"Original CBS: {root.cbs:.4f}")
    print(f"Final CBS: {best.cbs:.4f}")
    print(f"Improvement: {root.cbs - best.cbs:.4f}")
    print(f"Retention: {best.crs:.4f}")
    print(f"Transformations: {' → '.join(best.transformation_history)}")
    print(f"Masked items: {len(best.masked_manifest)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        test_single_case()
    else:
        # Run comprehensive FAccT evaluation
        try:
            final_report = run_comprehensive_facct_evaluation()
            print("\n✅ All FAccT evaluations completed successfully!")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise