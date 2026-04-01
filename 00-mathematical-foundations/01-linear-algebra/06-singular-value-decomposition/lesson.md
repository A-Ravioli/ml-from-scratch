# Module 6: SVD, the Master Decomposition

## Prerequisites

- Modules 1-5

## Learning Objectives

- Understand every matrix as rotation, stretch, rotation through SVD
- Build low-rank approximations and image compression pipelines
- Connect SVD to PCA, pseudoinverses, and rank estimation

## Lesson 6.1: SVD From Scratch

### Concept

Every matrix `A` can be written as

`A = U Sigma V^T`

where `U` and `V` are orthogonal and `Sigma` stores non-negative singular values.

### Intuition

SVD says a linear map can always be understood as:

1. rotate into a convenient coordinate system
2. stretch or squash along orthogonal axes
3. rotate again

That is why SVD generalizes eigendecomposition to non-square matrices.

### Code Focus

- Compute SVD with NumPy
- Reconstruct the matrix from `U`, `Sigma`, and `V^T`
- Visualize or inspect the transformation in 2D

## Lesson 6.2: Low-Rank Approximation

### Concept

Keeping only the top `k` singular values gives the best rank-`k` approximation in Frobenius norm.

### Intuition

The largest singular values capture the dominant structure of the data. The smaller ones often represent fine detail or noise.

### Code Focus

- Truncate the SVD
- Plot reconstruction error versus rank
- Compress a grayscale image with truncated SVD
- Plot the singular value spectrum

## Lesson 6.3: SVD Applications

### Concept

SVD powers several fundamental tools:

- PCA via orthogonal basis change
- Pseudoinverse for least squares
- Numerical rank estimation from singular values

### Intuition

PCA rotates data toward directions of maximum variance. The pseudoinverse behaves like the “best possible inverse” even when exact inversion is impossible.

### Code Focus

- Implement PCA with SVD
- Solve overdetermined systems with the pseudoinverse
- Compare to `numpy.linalg.pinv`

## Study Questions

1. Why does SVD exist even when eigendecomposition does not?
2. Why do singular values reveal both rank and conditioning information?
3. Why is truncated SVD optimal among rank-limited approximations?
