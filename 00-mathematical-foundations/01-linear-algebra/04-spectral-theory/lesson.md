# Spectral Theory for Machine Learning

## Prerequisites

- Linear algebra (eigenvalues/eigenvectors), symmetric matrices, orthogonality

## Learning Objectives

- Apply spectral theorem to symmetric matrices
- Use iterative methods (power/inverse, Rayleigh quotient) to find eigenpairs
- Understand Gershgorin bounds and matrix functions via eigen-decomposition

## Core Concepts

### 1. Spectral Theorem (Symmetric Case)

Symmetric A ∈ ℝ^{n×n} has an orthonormal eigenbasis: A = Q Λ Qᵀ.

#### Intuition

- Orthogonal directions that A scales by eigenvalues; geometry is axis-aligned in the right basis.

### 2. Iterative Eigen Methods

- Power iteration: dominant eigenpair
- Inverse iteration / Rayleigh quotient iteration: interior eigenvalues

### 3. Gershgorin Disks and Matrix Functions

- Disks in ℂ containing eigenvalues; quick spectrum bounds.
- For diagonalizable A = V Λ V^{-1}, f(A) = V f(Λ) V^{-1}.

## ML Connections

- PCA and spectral clustering rely on eigen-structure
- Stability and conditioning relate to spectral radii and gaps

## Implementation Details

See `exercise.py` for:

1. `symmetric_eigendecomposition`, `diagonalize`
2. `power_iteration`, `inverse_iteration`, `rayleigh_quotient`, `rayleigh_quotient_iteration`
3. `gershgorin_disks`, `matrix_function_via_eigen`, `graph_laplacian_spectrum`

## Exercises

- Verify spectral decomposition and orthogonality of eigenvectors
- Implement Gershgorin disks and compare with true eigenvalues
- Implement matrix exponential via eigen-decomposition on diagonalizable matrices


