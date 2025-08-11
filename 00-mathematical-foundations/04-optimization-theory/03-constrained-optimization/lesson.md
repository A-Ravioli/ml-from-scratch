# Constrained Optimization for Machine Learning

## Prerequisites

- Convex optimization basics, KKT conditions

## Learning Objectives

- Formulate problems with equality/inequality constraints
- Apply KKT conditions and barrier/penalty methods
- Implement projected and proximal methods for constraints

## Core Concepts

### 1. KKT for Constrained Problems

Stationarity, primal/dual feasibility, complementary slackness.

### 2. Barrier and Penalty Methods

- Interior-point (log barriers)
- Quadratic penalties and augmented Lagrangian

### 3. Projections

- Onto simplex, ℓ₂/ℓ₁ balls, boxes

## Implementation Details

See `exercise.py` for:

1. `solve_quadratic_program` (simple QP via cvxpy or custom KKT)
2. `augmented_lagrangian` for equality-constrained problems
3. Projections onto common sets


