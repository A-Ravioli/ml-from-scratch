# Nonconvex Optimization for Machine Learning

## Prerequisites

- Convex optimization basics; gradients/Hessians

## Learning Objectives

- Understand challenges: saddle points, local minima, landscape geometry
- Analyze gradient descent with noise and escape from saddle points
- Use trust-region and line-search methods in nonconvex settings

## Core Concepts

### 1. Critical Points

∇f(x*) = 0; classify via eigenvalues of ∇²f(x*): minima, maxima, saddles.

### 2. Saddle-Point Escapes

Noise and momentum help escape strict saddles (negative curvature directions).

### 3. Trust-Region and Line-Search

Model subproblem in a region; ensure sufficient decrease.

## ML Connections

- Deep learning loss landscapes; overparameterization and benign optimization.

## Implementation Details

See `exercise.py` for:

1. `find_critical_points_2d` (grid-based) and classify via Hessian
2. `noisy_gradient_descent` with saddle-escape demo
3. `trust_region_step` (Cauchy point) and globalization


