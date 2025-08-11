# Matrix Decompositions for Machine Learning

## Prerequisites

- Vector spaces, norms, orthogonality (from 01-vector-spaces)
- Basic matrix algebra and numerical linear algebra intuition

## Learning Objectives

- Understand LU, QR, and Cholesky decompositions and when to use each
- Use decompositions to solve linear systems, compute determinants, and least squares
- Build geometric intuition for QR (orthogonal bases) and Cholesky (energy)

## Core Concepts

### 1. LU Decomposition (with Pivoting)

#### Idea

Factor A ≈ P L U with P permutation, L unit-lower-triangular, U upper-triangular. Pivoting improves stability.

#### Intuition

- Gaussian elimination written as a product of simple eliminations.
- L records how we combine rows, U is the echelon form, P reorders rows to avoid tiny pivots.

#### ML Connection

- Efficient solves for multiple right-hand sides (Ax=b) during training/inference.
- Determinant and log-determinant from U’s diagonal (useful in normalizing flows).

### 2. QR Decomposition (Orthogonal-Upper)

#### Idea

A = Q R with Q orthogonal (QᵀQ = I) and R upper-triangular.

#### Intuition

- Gram–Schmidt/Householder builds an orthonormal basis of the column space.
- Least squares: minimize ||Ax − b|| via Rx = Qᵀb (numerically stable).

#### ML Connection

- Least squares solvers, numerical stability in regression, orthogonalization layers.

### 3. Cholesky (SPD Matrices)

#### Idea

For symmetric positive definite A, A = L Lᵀ.

#### Intuition

- “Energy” matrix admits a square-root; halves the work vs LU.

#### ML Connection

- Gaussian models (covariance factorizations), kernel methods, Kalman filtering.

## Implementation Details

See `exercise.py` for:

1. `lu_decomposition`, `solve_lu`, `determinant_via_lu`
2. `qr_decomposition` (Gram–Schmidt or Householder) and `least_squares_qr`
3. `cholesky_decomposition` and SPD solves

## Exercises

- Implement LU with partial pivoting and test A ≈ P L U and solve Ax=b
- Implement QR via modified Gram–Schmidt and compare to Householder (np.linalg.qr)
- Implement Cholesky and verify correctness on SPD matrices


