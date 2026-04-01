# Module 3: Vector Spaces

## Prerequisites

- Modules 1-2

## Learning Objectives

- Understand span, independence, and basis as the language of “reachable directions”
- Compute the four fundamental subspaces of a matrix and verify rank-nullity numerically
- Convert vectors between coordinate systems and connect basis change to PCA

## Lesson 3.1: Span, Independence, Basis

### Concept

The span of a set of vectors is the collection of all linear combinations they can produce. Linear independence says none of the vectors is redundant. A basis is the smallest non-redundant set that still reaches the whole space.

### Intuition

- Two non-collinear vectors in `R^2` span the plane
- Two collinear vectors only span a line
- Rank is the matrix-language summary of how many genuinely new directions are present

### ML Connection

Feature redundancy, rank deficiency, and dimensionality reduction all start here. If your features live in a lower-dimensional subspace than you thought, your model can become unstable or non-identifiable.

### Code Focus

- Check independence with matrix rank
- Visualize the span of two 2D vectors
- Demonstrate rank deficiency with redundant columns

## Lesson 3.2: The Four Fundamental Subspaces

### Concept

For a matrix `A`, the four fundamental subspaces organize everything the transformation can do:

- Column space: all reachable outputs `Ax`
- Row space: all linear measurements encoded by rows
- Null space: inputs that get sent to zero
- Left null space: output directions orthogonal to the column space

### Intuition

The null space tells you what the matrix “forgets.” The column space tells you where the matrix can land.

### Rank-Nullity

For `A in R^{m x n}`,

`n = rank(A) + nullity(A)`

This is the bookkeeping identity that explains why every lost dimension in the output creates freedom in the solution space.

### Code Focus

- Compute all four subspaces with SciPy
- Verify that null-space vectors satisfy `Ax = 0`
- Check rank-nullity numerically

## Lesson 3.3: Change of Basis

### Concept

The vector itself does not change when we change basis; only its coordinates do. A basis matrix translates between coordinate descriptions and the underlying geometric object.

### Intuition

- Pixel coordinates and world coordinates describe the same point in different frames
- PCA changes basis to align coordinates with directions of maximum variance

### Code Focus

- Convert vectors to and from arbitrary bases
- Transform coordinates between two bases
- Use SVD to build a PCA basis

## Study Questions

1. Why does rank capture the number of independent directions?
2. What does the null space tell you about ambiguity in solving `Ax = b`?
3. Why is PCA best understood as a change of basis rather than just a compression trick?
