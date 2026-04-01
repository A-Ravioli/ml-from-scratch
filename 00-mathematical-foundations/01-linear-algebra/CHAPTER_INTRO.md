# Linear Algebra for Machine Learning

## Overview

This chapter turns linear algebra from a prerequisite into a working language for machine learning. The sequence starts with geometric intuition, moves through systems and subspaces, and then builds toward the decompositions, matrix calculus, and numerical stability ideas that appear everywhere in modern ML.

## Learning Goals

- See vectors and matrices as transformations, not just containers of numbers
- Solve systems reliably and understand when solutions are unstable
- Move comfortably between span, basis, rank, null spaces, and coordinate systems
- Use orthogonality, eigendecomposition, SVD, and Cholesky as practical tools
- Build intuition for gradients, Hessians, conditioning, and sparse computation
- Apply the whole toolkit in end-to-end ML projects

## Module Map

### Module 1. The Building Blocks

Vectors, basis vectors, matrix-vector multiplication, and special matrix families.

### Module 2. Solving Linear Systems

Geometric interpretations of `Ax = b`, Gaussian elimination, and LU reuse.

### Module 3. Vector Spaces

Span, independence, basis, the four fundamental subspaces, and change of basis.

### Module 4. Orthogonality

Projections, Gram-Schmidt, QR, and norm-preserving transforms.

### Module 5. Eigenvalues and Eigenvectors

Invariant directions, diagonalization, and the spectral theorem for symmetric matrices.

### Module 6. Singular Value Decomposition

SVD, low-rank approximation, PCA, and pseudoinverses.

### Module 7. Positive Definite Matrices

Quadratic forms, PD/PSD tests, and Cholesky decomposition.

### Module 8. Matrix Calculus

Gradients, Jacobians, Hessians, and backprop as structured chain rule.

### Module 9. Norms, Conditioning, and Sparse Linear Algebra

Norms, condition numbers, numerical sensitivity, and sparse storage formats.

### Module 10. Capstone Projects

PCA from scratch, linear regression solved four ways, attention as matrix operations, and a neural net without autograd.

## How To Study This Chapter

1. Read each `lesson.md` for geometric intuition, formal definitions, and ML connections.
2. Work through `exercise.py` in order. Many functions are small enough to re-derive from the math.
3. Use `test_implementation.py` to verify correctness and catch numerical mistakes early.
4. Read `solutions/solution.py` only after giving the exercises a serious attempt.
5. Re-run the visualization helpers with your own examples. Linear algebra becomes much clearer when you see the transformations.

## Suggested Pace

- Core path: Modules 1-6 first
- Stability and optimization path: Modules 7-9 next
- Integration path: Module 10 last

## ML Connections To Watch For

- PCA, whitening, and embeddings use basis changes and SVD
- Linear regression relies on least squares, QR, LU, and pseudoinverses
- Attention is mostly matrix multiplication, projections, and change of basis
- Optimization depends on gradients, Hessians, and conditioning
- Covariance matrices, kernels, and Gaussians rely on symmetry and positive definiteness
