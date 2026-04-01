# Module 7: Positive Definite Matrices

## Prerequisites

- Modules 1-6

## Learning Objectives

- Understand PD and PSD matrices through quadratic forms, eigenvalues, and Cholesky
- Connect positive definiteness to bowl-shaped objectives and covariance structure
- Use Cholesky for fast solves and Gaussian sampling

## Lesson 7.1: PD and PSD Matrices

### Concept

A symmetric matrix `A` is positive definite if

`x^T A x > 0`

for every non-zero vector `x`. It is positive semidefinite if the inequality is non-strict.

### Intuition

The quadratic form `x^T A x` describes an energy surface. Positive definite means the surface curves upward in every direction, like a bowl.

### ML Connection

Covariance matrices, kernel matrices, Gauss-Newton approximations, and many Hessians are symmetric and often PSD or PD.

### Code Focus

- Generate PD matrices as `A^T A + epsilon I`
- Verify PD using eigenvalues and Cholesky
- Visualize the quadratic form surface for a 2D matrix

## Lesson 7.2: Cholesky Decomposition

### Concept

For symmetric positive definite `A`, Cholesky gives

`A = L L^T`

with `L` lower triangular.

### Intuition

Cholesky is a matrix square root specialized to the PD setting. Because it exploits symmetry, it is faster and more stable than generic LU for this class.

### Code Focus

- Solve `Ax = b` with Cholesky
- Sample from a multivariate Gaussian with `L z + mu`
- Benchmark Cholesky against LU

## Study Questions

1. Why does symmetry matter in the definition of positive definiteness?
2. How does the quadratic form reveal curvature?
3. Why can Cholesky fail exactly when positive definiteness fails?
