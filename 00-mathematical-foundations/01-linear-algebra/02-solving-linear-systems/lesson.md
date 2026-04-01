# Module 2: Solving Linear Systems

## Prerequisites

- Module 1: vectors, matrices, and matrix multiplication

## Learning Objectives

- Interpret `Ax = b` geometrically in two and three dimensions
- Solve systems with Gaussian elimination and understand why pivoting matters
- Use LU factorization efficiently when the same matrix is paired with many right-hand sides

## Lesson 2.1: What Does `Ax = b` Mean?

### Concept

The equation `Ax = b` asks for an input `x` whose transformed version lands exactly at `b`. In coordinates it is a system of linear equations; geometrically it is an intersection problem.

### Intuition

- In 2D, each equation is a line
- In 3D, each equation is a plane
- A system can be:
  - Consistent: at least one solution exists
  - Inconsistent: no common intersection
  - Underdetermined: infinitely many solutions

### ML Connection

Least squares, linear regression, and normal equations all begin with systems of the form `Ax = b`. Sensitivity to perturbations in `b` foreshadows conditioning and numerical stability.

### Code Focus

- Visualize line intersections in 2D and planes in 3D
- Solve systems with `numpy.linalg.solve`
- Perturb `b` and observe how the solution changes

## Lesson 2.2: Gaussian Elimination

### Concept

Gaussian elimination applies row operations that preserve the solution set while driving the system into upper-triangular form.

### Intuition

Each elimination step removes one variable from the equations below the pivot row. Back substitution then recovers the variables from bottom to top.

### Why Pivoting Matters

Without pivoting, dividing by tiny numbers can amplify roundoff error. Partial pivoting swaps in the largest available pivot in the current column and dramatically improves stability.

### Code Focus

- Implement forward elimination with partial pivoting
- Finish with back substitution
- Compare your solver against NumPy

## Lesson 2.3: LU Decomposition

### Concept

LU stores Gaussian elimination as a factorization. Instead of redoing elimination for every new `b`, we factor once and solve repeatedly.

### Intuition

`A = LU` means:

- `L` records the elimination multipliers
- `U` stores the resulting upper-triangular system

When many targets share the same `A`, LU amortizes the expensive part.

### ML Connection

Repeated solves appear in regression, Kalman filtering, Gaussian models, and many iterative methods that reuse a fixed system matrix.

### Code Focus

- Use `scipy.linalg.lu_factor` and `lu_solve`
- Time LU reuse versus repeated direct solves

## Study Questions

1. What geometric change happens when `b` moves while `A` stays fixed?
2. Why does row swapping not change the solution set?
3. When is LU more valuable than calling `solve` repeatedly?
