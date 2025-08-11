# Stochastic Optimization for Machine Learning

## Prerequisites

- Probability and convex optimization basics

## Learning Objectives

- Understand SGD, variance reduction, and adaptive methods
- Analyze convergence rates under smooth/strongly convex assumptions
- Implement mini-batch, momentum, and variance reduction techniques

## Core Concepts

### 1. Stochastic Gradient Descent (SGD)

Unbiased gradient estimates; step-size schedules; convergence in expectation.

### 2. Variance Reduction

SVRG, SAGA reduce variance using control variates; faster convergence.

### 3. Adaptive Methods

AdaGrad, RMSProp, Adam; per-coordinate step sizes; practical considerations.

## Implementation Details

See `exercise.py` for:

1. `sgd` with step-size schedules and convergence plots
2. `svrg` for finite-sum problems
3. `adam` with bias correction and comparisons


