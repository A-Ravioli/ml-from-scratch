# Module 5: Eigenvalues and Eigenvectors

## Prerequisites

- Modules 1-4

## Learning Objectives

- Interpret eigenvectors as invariant directions of a linear transformation
- Use eigendecomposition to diagonalize matrices and compute matrix powers efficiently
- Understand why symmetric matrices have especially clean geometry

## Lesson 5.1: What Are Eigenvectors?

### Concept

An eigenvector of `A` is a non-zero vector `v` such that `Av = lambda v`. The transformation acts on that direction by pure scaling, without changing the line it lies on.

### Intuition

Most vectors get rotated, sheared, or mixed. Eigenvectors are the rare directions the transformation leaves aligned with themselves.

### ML Connection

Principal components, covariance analysis, Hessian curvature, graph spectra, and diffusion dynamics all rely on eigen-geometry.

### Code Focus

- Compute eigendecompositions with NumPy
- Compare how selected vectors move under a matrix
- Identify which directions get only scaled

## Lesson 5.2: Eigendecomposition

### Concept

If a matrix is diagonalizable, we can write

`A = V Lambda V^{-1}`

This turns repeated applications of `A` into a change of basis, a diagonal scaling, and a change back.

### Intuition

Decompose, scale, and recompose. That is why powers like `A^k` become easy:

`A^k = V Lambda^k V^{-1}`

### Code Focus

- Implement matrix powers via eigendecomposition
- Compare to `numpy.linalg.matrix_power`
- Diagonalize symmetric matrices where the basis is orthogonal

## Lesson 5.3: Spectral Theorem and Symmetric Matrices

### Concept

Real symmetric matrices have real eigenvalues and orthogonal eigenvectors. They are diagonalized by an orthogonal matrix:

`A = Q Lambda Q^T`

### Intuition

Symmetry removes the messy complex behavior that general matrices can have. That is why covariance matrices and many Hessians are so pleasant to analyze.

### Code Focus

- Generate random symmetric matrices
- Verify real eigenvalues and orthogonality
- Connect symmetry to positive semidefiniteness

## Study Questions

1. Why is diagonalization so useful for repeated transformations?
2. Why are symmetric matrices easier to reason about than general square matrices?
3. How does eigendecomposition differ from SVD, which you will meet next?
