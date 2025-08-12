# Information Geometry for Machine Learning

## Prerequisites

- Probability and statistics (exponential families)
- Differential geometry intuition (manifolds, metric, geodesic)

## Learning Objectives

- Understand statistical manifolds and the Fisher information metric
- Derive natural gradients and relate to preconditioning
- Explore exponential families, dual connections, and Bregman geometry

## Core Concepts

### 1. Statistical Manifolds and Fisher Metric

Parameter space Θ of distributions p(x; θ) forms a manifold with metric:
g_{ij}(θ) = E_θ[∂_i log p ∂_j log p] (Fisher information).

#### Intuition

- Measures distinguishability of nearby distributions; induces geometry for optimization.

### 2. Exponential Families and Duality

p(x; θ) = exp(⟨θ, T(x)⟩ − A(θ)) h(x). Natural (θ) and expectation (η) parameters are dual.

### 3. Natural Gradient

Update: θ_{k+1} = θ_k − α G(θ_k)^{-1} ∇_θ L(θ_k). Invariant to reparameterization.

## ML Connections

- Natural gradient descent, Amari’s theory; connections to second-order methods.
- KL divergence and Bregman divergences link to mirror descent.

## Implementation Details

See `exercise.py` for:

1. Fisher information for simple families (Bernoulli, Gaussian)
2. Natural gradient step using Fisher inverse
3. Geodesic approximation on Gaussian mean–variance manifold

## Exercises

- Compute Fisher information for Bernoulli and Gaussian families
- Show natural gradient equals preconditioned gradient for quadratic losses
- Visualize geodesics under Fisher metric in Gaussian family


