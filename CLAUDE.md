# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **ml-from-scratch**, a comprehensive 4-year PhD-level machine learning curriculum designed to take learners from mathematical foundations to cutting-edge research. The curriculum follows a "first principles" approach, implementing everything from scratch before using libraries.

## Architecture and Structure

### Learning Philosophy

- **First principles**: All concepts derived from mathematical foundations
- **Implementation first**: Code algorithms from scratch before using libraries
- **Theory ↔ Practice**: Connect mathematical insights to empirical results
- **Research mindset**: Question assumptions, propose extensions

### Curriculum Organization

The repository is structured as a 4-year progressive curriculum:

**Year 1**: Mathematical foundations (00-02) + Statistical learning theory (01)
**Year 2**: Deep learning fundamentals (04) + Architectures (05) + Optimization (03)
**Year 3**: Generative models (06) + Reinforcement learning (07) + Specialization tracks
**Year 4**: Frontier research (08-12)

### Standard File Structure Pattern

Each topic follows a consistent pattern:

- `lesson.md`: Theoretical foundations and mathematical treatment
- `exercise.py`: Implementation template with TODO functions to complete
- `test_implementation.py`: Test suite to verify implementations
- `solutions/solution.py`: Reference implementation (learners should attempt solo first)
- `CHAPTER_INTRO.md`: Chapter overview and approach guidance

### Learning Progression

Sub-topics within each chapter are numbered sequentially (01-, 02-, etc.) to indicate the recommended learning order. Dependencies flow from lower to higher numbers within each chapter.

## Dependencies and Environment

### Core Dependencies (requirements.txt)

- **Scientific Computing**: numpy, scipy, matplotlib
- **Traditional ML**: scikit-learn  
- **Deep Learning**: PyTorch ecosystem (torch, torchvision, torchaudio)
- **Geometric Deep Learning**: torch-geometric, torch-sparse, torch-cluster, torch-scatter
- **Optimization**: cvxpy
- **Testing**: pytest, pytest-cov
- **Utilities**: networkx, seaborn, pandas, tqdm

### Installation

```bash
pip install -r requirements.txt
```

## Development Commands

### Testing

Run tests for specific implementations:

```bash
# Test specific implementation
cd 00-mathematical-foundations/01-analysis/01-real-analysis/
python -m pytest test_implementation.py -v

# Run all tests in a chapter
cd 04-deep-learning-fundamentals/
python -m pytest */test_implementation.py -v
```

### Verification

Each exercise includes verification scripts:

```bash
# Check implementation correctness
python exercise.py  # Most exercise files have __main__ verification
```

### Common Utilities

The `utils/common_functions.py` module provides shared functionality:

- Numerical gradient checking (`check_gradient`, `numerical_gradient`)
- Synthetic data generation (`generate_synthetic_data`)
- Visualization helpers (`plot_loss_landscape`, `plot_convergence`)
- Mathematical utilities (`eigendecomposition`, `is_positive_definite`)
- ML utilities (`normalize_data`, `create_mini_batches`, `one_hot_encode`)

## Implementation Guidelines

### Code Style for Exercises

- Implement TODO functions in `exercise.py` files
- Follow existing code patterns and mathematical notation
- Include docstrings explaining the mathematical concepts
- Add type hints for function signatures
- Use numpy for numerical computations
- Verify implementations against theoretical properties

### Mathematical Rigor

- Implementations should reflect the underlying mathematical theory
- Include numerical stability considerations
- Implement gradient checks for optimization algorithms
- Test edge cases and boundary conditions
- Connect implementations to theoretical guarantees

### Research Integration

- Each implementation connects to research literature
- Exercise templates include citations to seminal papers
- Extensions and research directions provided in lessons
- Encourage questioning assumptions and proposing improvements

## Key Architecture Components

### Mathematical Foundations (00-)

Base classes and utilities for analysis, linear algebra, probability theory, and optimization theory that underpin all ML algorithms.

### Learning Theory (01-)

PAC learning framework, VC dimension, Rademacher complexity - the theoretical foundations for understanding when and why ML works.

### Classical ML (02-)

Traditional algorithms implemented from scratch - linear models, trees, kernels, Bayesian methods, ensemble methods.

### Optimization (03-)

First-order methods (SGD variants, momentum, adaptive), second-order methods (Newton, quasi-Newton), variance reduction techniques.

### Deep Learning (04-05)

Neural network theory, backpropagation, initialization, normalization, and modern architectures (CNNs, RNNs, Transformers, GNNs).

### Advanced Topics (06-12)

Generative models, reinforcement learning, meta-learning, continual learning, theoretical deep learning, and frontier research areas.

## Curriculum Navigation

### Starting Point

Begin with `00-mathematical-foundations/01-analysis/01-real-analysis/` and follow the numbered sequence within each chapter.

### Prerequisites

Each chapter's `CHAPTER_INTRO.md` explains prerequisites and connections to other chapters.

### Progress Tracking

Complete all TODO functions in `exercise.py` before moving to the next topic. Use test suites to verify correctness.

### Research Integration

After completing implementations, read the research papers referenced in `lesson.md` to understand current state-of-the-art and open problems.
