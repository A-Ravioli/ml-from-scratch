# Module 4: Orthogonality

## Prerequisites

- Modules 1-3

## Learning Objectives

- Understand projection as “keeping only the aligned component”
- Build orthonormal bases with Gram-Schmidt and factor matrices with QR
- Interpret orthogonal matrices as rotations or reflections that preserve norms

## Lesson 4.1: Projections

### Concept

Projecting a vector onto another vector or subspace extracts the component aligned with that direction or space. The residual is orthogonal to the target.

### Intuition

Projection is the algebraic version of a shadow. Least squares works because the best approximation makes the error orthogonal to the model subspace.

### Code Focus

- Implement projection onto a vector
- Build the projection matrix `P = A(A^T A)^{-1}A^T`
- Visualize the original vector, its projection, and the residual in 2D

## Lesson 4.2: Gram-Schmidt and QR

### Concept

Gram-Schmidt turns a basis into an orthonormal basis by repeatedly removing components already explained by earlier vectors. QR packages that idea for whole matrices.

### Intuition

Each step says: keep only the new direction that has not already been accounted for.

### ML Connection

QR solves least-squares problems more stably than the normal equations and appears in numerical optimization, PCA pipelines, and orthogonalization layers.

### Code Focus

- Implement modified Gram-Schmidt
- Build a QR factorization from it
- Compare to `numpy.linalg.qr`

## Lesson 4.3: Orthogonal Matrices

### Concept

Orthogonal matrices satisfy `Q^T Q = I`. They preserve inner products, norms, and angles.

### Intuition

Rotations and reflections move vectors around without stretching them. That makes orthogonal transforms ideal for stable coordinate changes.

### ML Connection

Attention begins with dot products between query and key vectors. Thinking in orthogonal frames helps explain why dot-product geometry is meaningful and why certain projections preserve scale better than others.

### Code Focus

- Generate rotation matrices
- Compose rotations
- Verify norm preservation numerically

## Study Questions

1. Why does projection solve a best-approximation problem?
2. Why is modified Gram-Schmidt preferred over the naive version numerically?
3. What information can an orthogonal transformation change, and what must it preserve?
