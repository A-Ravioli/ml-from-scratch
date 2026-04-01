# Module 1: The Building Blocks

## Prerequisites

- High-school algebra
- Basic NumPy arrays

## Learning Objectives

- Distinguish a vector as a geometric object from an array as a storage object
- Interpret matrix-vector multiplication as a transformation built from matrix columns
- Recognize important special matrix families and the transformations they encode

## Lesson 1.1: Vectors, More Than a List of Numbers

### Concept

A vector is not merely a row of numbers. In linear algebra it represents a point, displacement, direction, or coordinate description relative to a basis. The same coordinates can mean different geometric objects depending on the basis.

### Intuition

- In `R^2`, a vector is an arrow from the origin to a point
- The standard basis vectors `e_1 = [1, 0]^T` and `e_2 = [0, 1]^T` act like the x- and y-axis building blocks
- Unit vectors have norm 1 and isolate pure direction without scale

### ML Connection

Feature vectors, parameter vectors, embeddings, and gradients are all vectors. When we normalize an embedding or follow a gradient direction, we are working directly with vector geometry.

### Code Focus

- Implement vector addition, scalar multiplication, and dot product from scratch in NumPy
- Plot 2D vectors as arrows to build intuition

## Lesson 1.2: Matrices, Spreadsheets That Do Things

### Concept

A matrix can be read two ways:

- Column view: a stack of output directions telling us where each basis vector lands
- Row view: a list of linear measurements taken from the input vector

### Geometric Meaning of `Ax`

If `x = [x_1, ..., x_n]^T`, then

`Ax = x_1 a_1 + x_2 a_2 + ... + x_n a_n`

where `a_i` are the columns of `A`. Matrix-vector multiplication forms a weighted combination of the matrix columns. This is why the column space tells us all possible outputs of `Ax`.

### ML Connection

Every linear layer in a neural network is a matrix acting on a vector. Understanding columns, rows, and multiplication is the foundation for attention, PCA, regression, and optimization.

### Code Focus

- Implement matrix multiplication with explicit loops
- Benchmark your implementation against NumPy
- Visualize a 2D column space by sampling linear combinations of columns

## Lesson 1.3: Special Matrices

### Concept

Several matrix families matter because they restrict what a transformation can do:

- Identity: do nothing
- Diagonal: scale axes independently
- Symmetric: align with clean eigenspaces
- Orthogonal: rotate or reflect without stretching
- Triangular: encode ordered dependencies and efficient solves

### Intuition

Each special form removes degrees of freedom from a generic matrix and gives us structure we can exploit algorithmically.

### ML Connection

- Orthogonal matrices help preserve norms in optimization and representation learning
- Symmetric matrices appear as covariance matrices and Hessians
- Triangular matrices show up in Gaussian elimination, QR, LU, and autoregressive models

### Code Focus

- Generate each matrix family
- Verify the defining identities programmatically

## Study Questions

1. Why does the column view of a matrix make `Ax` feel more geometric than the row view?
2. What information is lost when you normalize a vector?
3. Why can an orthogonal matrix change coordinates without changing vector length?
