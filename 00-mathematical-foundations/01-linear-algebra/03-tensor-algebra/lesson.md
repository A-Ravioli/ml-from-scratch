# Tensor Algebra for Machine Learning

## Prerequisites

- Linear algebra, Kronecker/outer products, index notation

## Learning Objectives

- Work with tensor products, contractions, and mode-n operations
- Understand Kronecker and Khatri–Rao products and their ML uses
- Build intuition for low-rank tensor approximations (rank-1 case)

## Core Concepts

### 1. Tensor Products and Contractions

#### Intuition

- Tensor as multi-dimensional arrays with multilinear structure.
- Contraction sums over paired indices (generalized matrix multiplication).

### 2. Kronecker and Khatri–Rao

- Kronecker: block-wise product (A ⊗ B). Expands features, used in structured layers.
- Khatri–Rao: column-wise Kronecker; core in CP decompositions.

### 3. Mode-n Matricization and Products

- Unfold tensor along a mode and multiply by a matrix on that mode.

## Implementation Details

See `exercise.py` for:

1. `kronecker_product`, `khatri_rao_product`
2. `tensor_contract` via np.tensordot
3. `matricize_mode_n` and `mode_n_product`
4. `rank1_approximation` for simple CP-1 case

## Exercises

- Implement contractions and verify shapes/values against einsum
- Implement Khatri–Rao and test with small examples
- Recover factors from a rank-1 synthetic tensor


