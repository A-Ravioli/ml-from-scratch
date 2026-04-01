# Module 9: Norms, Conditioning, and Numerical Stability

## Prerequisites

- Modules 1-8

## Learning Objectives

- Compare common vector and matrix norms geometrically and computationally
- Understand condition numbers as sensitivity measures
- Work with sparse matrices and see why storage format matters in practice

## Lesson 9.1: Vector and Matrix Norms

### Concept

Norms measure size. Different norms emphasize different aspects of a vector or matrix.

### Intuition

- `L1`: diamond unit ball, encourages sparsity
- `L2`: circular unit ball, Euclidean geometry
- `Linf`: square unit ball, dominated by the largest coordinate
- Frobenius norm: treats a matrix like one long vector
- Spectral norm: measures the largest stretch factor

### Code Focus

- Compute vector and matrix norms
- Sample unit balls in 2D
- Compare Frobenius and spectral norms of weight matrices

## Lesson 9.2: Condition Number

### Concept

For the 2-norm,

`kappa(A) = sigma_max / sigma_min`

It quantifies how much a problem can amplify perturbations.

### Intuition

Nearly singular matrices flatten some directions so strongly that tiny changes in `b` can cause large changes in the solution.

### Code Focus

- Construct matrices with chosen condition numbers
- Perturb `b` and measure solution sensitivity
- Show that optimization slows down on ill-conditioned quadratics

## Lesson 9.3: Sparse Matrices and Efficient Storage

### Concept

Sparse matrices store only non-zero entries. Different formats optimize different operations:

- COO: easy to build
- CSR: fast row slicing and matvec
- CSC: fast column operations

### Intuition

Graphs, bag-of-words matrices, and recommendation systems are mostly zeros. Dense storage wastes memory and time.

### Code Focus

- Build sparse matrices in SciPy
- Benchmark sparse versus dense matmul
- Visualize sparsity patterns
- Construct a graph Laplacian

## Study Questions

1. Why does the spectral norm matter for worst-case amplification?
2. How does conditioning affect both solving systems and optimization speed?
3. When would you choose CSR over COO or CSC?
