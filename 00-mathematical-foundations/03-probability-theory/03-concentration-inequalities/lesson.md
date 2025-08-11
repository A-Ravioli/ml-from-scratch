# Concentration Inequalities for Machine Learning

## Prerequisites

- Measure-theoretic probability and expectations

## Learning Objectives

- Apply Markov, Chebyshev, Hoeffding, Bernstein, and Chernoff bounds
- Understand sub-Gaussian/sub-exponential variables
- Use concentration to justify generalization bounds

## Core Concepts

### 1. Basic Inequalities

- Markov: P(X ≥ a) ≤ E[X]/a for X ≥ 0
- Chebyshev: P(|X−E[X]| ≥ t) ≤ Var(X)/t²

### 2. Hoeffding’s Inequality

For bounded independent Xᵢ ∈ [a, b]:
P(\bar{X}−E[\bar{X}] ≥ t) ≤ exp(−2n t²/(b−a)²)

### 3. Bernstein’s Inequality

Sharpens Hoeffding with variance information.

### 4. Chernoff Bounds

MGF-based tail bounds for sums of independent variables.

## ML Connections

- Generalization error bounds; uniform convergence; PAC learning.

## Implementation Details

See `exercise.py` for:

1. `is_subgaussian`, `subgaussian_parameter`
2. `hoeffding_bound`, `bernstein_bound`, `chernoff_bound`
3. Empirical verification via simulation


