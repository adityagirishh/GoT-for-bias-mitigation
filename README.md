# Graph of Thoughts for Bias Mitigation (GoT-BiasM)

A novel approach to detecting and mitigating bias in text using Graph of Thoughts (GoT) methodology combined with deterministic masking and comprehensive bias evaluation metrics.

## Overview

This project implements a sophisticated bias mitigation system that leverages the Graph of Thoughts framework to systematically reduce various forms of bias in textual content. The system uses a graph-based search approach to explore different bias mitigation strategies and find optimal solutions that balance bias reduction with content preservation.

### Key Features

- **Multi-dimensional Bias Detection**: Detects toxicity, stereotypes, sentiment polarization, demographic imbalance, and privacy violations
- **Graph-based Optimization**: Uses depth-limited best-first search to explore bias mitigation strategies
- **Deterministic Masking**: Applies typed masks for different bias categories while preserving content structure
- **Comprehensive Evaluation**: Includes benchmarking framework with multiple baseline comparisons
- **Privacy Protection**: Automatically detects and masks PII including emails, phone numbers, and SSNs
- **Semantic Preservation**: Maintains semantic similarity while reducing bias through context-aware transformations

## Architecture

The system consists of several key components:

### 1. Bias Detection Engine (`got/got.py`)
- **BiasDetector**: Multi-category bias scoring system
- **BiasScores**: Structured bias metrics including:
  - Toxicity (0-1): Fraction of toxic language
  - Sentiment Polarization (0-1): Degree of sentiment imbalance
  - Stereotypes (0-1): Presence of stereotypical patterns
  - Imbalance (0-1): Demographic representation imbalance
  - Context Shift (0-1): Semantic similarity preservation

### 2. Graph of Thoughts Framework
- **GraphNode**: Represents text states with bias scores and transformation history
- **Graph Traversal**: Depth-limited best-first search for optimal bias mitigation paths
- **Reward System**: Balances bias reduction with content retention

### 3. Deterministic Masking System
- **Privacy Masking**: Detects and masks PII with typed tokens
- **Toxicity Filtering**: Identifies and masks harmful language
- **Stereotype Removal**: Targets biased patterns and generalizations
- **Span Merging**: Prevents overlapping masks for cleaner output

### 4. Benchmarking Framework (`benchmarks/`)
- **Dataset Support**: Synthetic data generation and CrowS-Pairs integration
- **Baseline Comparisons**: Multiple mitigation strategies for comparison
- **Comprehensive Metrics**: Evaluation across bias, retention, and semantic similarity

## Installation

```bash
git clone https://github.com/adityagirishh/GoT-for-bias-mitigation.git
cd GoT-for-bias-mitigation
pip install -r requirements.txt
```

### Requirements
- Python 3.7+
- sentence-transformers (for semantic similarity)
- numpy, networkx, matplotlib, scipy
- datasets (for CrowS-Pairs benchmarking)

## Quick Start

### Basic Usage

```python
from got.got import GraphOfThought

# Initialize the system
got = GraphOfThought()

# Process biased text
biased_text = "Your biased text here..."
root = got.create_root_node(biased_text)

# Find optimal bias mitigation
best_node = got.traverse_graph(root, max_depth=5, bias_threshold=0.15)

# Generate comprehensive report
report = got.generate_report(best_node)
print(f"Bias reduced from {root.cbs:.4f} to {best_node.cbs:.4f}")
print(f"Content retention: {best_node.crs:.4f}")
```

### Command Line Usage

```bash
# Run the main demonstration
python -m got.got

# Run benchmarks
python -m benchmarks.runner --dataset synthetic --samples 100
python -m benchmarks.runner --dataset crows --samples 200
```

## Dataset Support

The project includes several bias evaluation datasets:

### WinoBias Dataset (`Biased_Datasets/winoBias/`)
- **Pro-stereotyped variants**: Type1 and Type2 test/dev splits
- **Anti-stereotyped variants**: Corresponding counter-examples
- **StereoSet**: Comprehensive stereotype evaluation
- **WinoGender**: Gender bias evaluation dataset

### CrowS-Pairs Dataset
- Stereotypical and anti-stereotypical sentence pairs
- Covers multiple bias dimensions (race, gender, religion, etc.)
- Integrated benchmarking support

## Results and Evaluation

The system generates comprehensive outputs for each run:

### Output Structure
```
bias_mitigation_output_YYYYMMDD_HHMMSS/
├── results.txt                    # Final debiased text
├── results_report.json           # Comprehensive metrics report
├── results_traversal.json        # Graph traversal history
├── bias_mitigation_graph.png     # Network visualization
└── results_privacy_manifest.json # Privacy masking details
```

### Key Metrics
- **Composite Bias Score (CBS)**: Weighted combination of all bias dimensions (lower is better)
- **Content Retention Score (CRS)**: Text similarity preservation (higher is better)
- **Transformation Path**: Sequential bias mitigation steps applied
- **Processing Time**: Execution performance metrics

### Example Results
The system achieves significant bias reduction while maintaining high content retention:
- **Bias Reduction**: Up to 62% improvement in composite bias scores
- **Content Preservation**: Maintains >90% semantic similarity
- **Processing Efficiency**: Real-time processing for most text lengths

## Visualization

The system generates network visualizations showing the bias mitigation search process:

[22]

The graph visualization displays:
- **Nodes**: Text states with varying bias levels (color-coded)
- **Edges**: Transformation relationships between states
- **Color Scale**: Bias intensity (red = high bias, blue = low bias)
- **Layout**: Spring layout optimized for clarity

## Advanced Configuration

### Custom Bias Weights
```python
custom_weights = {
    "toxicity": 0.3,
    "sentiment_polarization": 0.1,
    "stereotypes": 0.3,
    "imbalance": 0.1,
    "context_shift": 0.2
}
got = GraphOfThought(weights=custom_weights)
```

### Embedding Models
```python
# Use different sentence transformer models
import os
os.environ["EMBED_MODEL"] = "all-mpnet-base-v2"
os.environ["EMBED_SIM_THRESHOLD"] = "0.9"
```

### Search Parameters
```python
# Customize graph traversal
best = got.traverse_graph(
    root,
    max_depth=7,           # Maximum search depth
    bias_threshold=0.1     # Stop when bias is sufficiently low
)
```

## Benchmarking

The project includes a comprehensive benchmarking framework:

### Running Benchmarks
```bash
# Quick synthetic evaluation
python -m benchmarks.runner --dataset synthetic --samples 50

# Full CrowS-Pairs evaluation
python -m benchmarks.runner --dataset crows --samples 1000 --max_depth 6
```

### Benchmark Outputs
- **JSONL Results**: Per-sample detailed results
- **CSV Summary**: Aggregated metrics for analysis
- **Markdown Report**: Human-readable evaluation summary

### Baseline Comparisons
The system compares against multiple baseline approaches:
- Simple word filtering
- Fixed-order masking sequences
- Random mitigation strategies

## Technical Details

### Bias Detection Patterns
The system uses comprehensive pattern libraries for detecting:
- **Gender Bias**: Occupational stereotypes, capability assumptions
- **Racial Bias**: Ethnic generalizations and discriminatory language
- **Age Bias**: Generational stereotypes and assumptions
- **Religious Bias**: Faith-based prejudices and generalizations
- **Socioeconomic Bias**: Class-based discrimination patterns

### Privacy Protection
Automatic detection and masking of:
- **Email Addresses**: Full pattern matching with case-insensitive support
- **Phone Numbers**: Various formatting patterns (XXX-XXX-XXXX)
- **Social Security Numbers**: XXX-XX-XXXX pattern recognition
- **Personal Names**: Context-aware name detection

### Semantic Preservation
The system uses multiple strategies to maintain meaning:
- **Sentence Transformers**: BERT-based semantic similarity
- **Context-aware Masking**: Preserves syntactic structure
- **Span Merging**: Prevents fragmentation of related content
- **Fallback Mechanisms**: SequenceMatcher for offline operation

## Research Applications

This project is suitable for:
- **Academic Research**: Bias detection and mitigation studies
- **Content Moderation**: Automated text filtering systems  
- **Fairness Evaluation**: ML model bias assessment
- **Privacy Protection**: PII detection and anonymization
- **Educational Tools**: Bias awareness and training

## Limitations and Future Work

### Current Limitations
- English-only language support
- Computational complexity scales with text length
- Requires internet connection for optimal semantic similarity
- Pattern-based detection may miss context-dependent bias

### Future Enhancements
- Multilingual support
- Real-time streaming processing
- Integration with large language models
- Advanced semantic understanding
- Custom bias category definitions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Development Guidelines
- Follow existing code style and documentation patterns
- Add comprehensive tests for new features
- Update benchmarks when adding new mitigation strategies
- Ensure backward compatibility with existing APIs

## License

This project is released under the MIT License. See the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{got-bias-mitigation,
  title={Graph of Thoughts for Bias Mitigation},
  author={Aditya Girish},
  year={2025},
  url={https://github.com/adityagirishh/GoT-for-bias-mitigation}
}
```

## Acknowledgments

This work builds upon the Graph of Thoughts framework introduced by Besta et al. (2023) and incorporates bias detection methodologies from fairness-aware machine learning research. Special thanks to the open-source community for providing the foundational libraries and datasets that make this research possible.

---

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the maintainers directly.
