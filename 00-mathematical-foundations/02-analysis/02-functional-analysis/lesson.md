# Functional Analysis for Machine Learning

## Prerequisites

- Linear algebra (vector spaces, inner products, eigenvalues)
- Real analysis basics (limits, continuity, sequences)
- Some familiarity with norms and metrics

## Learning Objectives

- Understand normed and inner product spaces and their role in ML
- Work with linear operators, operator norms, and continuity/boundedness
- Use projections and orthogonality in Hilbert spaces
- Connect functional analytic ideas to optimization and generalization

## Mathematical Foundations

### 1. Normed and Inner Product Spaces

#### Definition 1.1 (Normed Space)

A normed vector space is a pair (V, \|\cdot\|) where V is a vector space over ℝ (or ℂ) and \|\cdot\|: V → ℝ satisfies:

1. \|x\| ≥ 0 and \|x\| = 0 ⟺ x = 0
2. \|αx\| = |α|\|x\| for all scalars α
3. \|x + y\| ≤ \|x\| + \|y\| (triangle inequality)

The metric induced by a norm is d(x, y) = \|x − y\|.

#### Definition 1.2 (Inner Product Space)

An inner product space (V, ⟨·,·⟩) has an inner product satisfying:

1. ⟨x, x⟩ ≥ 0 and ⟨x, x⟩ = 0 ⟺ x = 0
2. ⟨x, y⟩ = ⟨y, x⟩ (conjugate symmetry in ℂ)
3. ⟨αx + βy, z⟩ = α⟨x, z⟩ + β⟨y, z⟩ (linearity)

The induced norm is \|x\| = sqrt(⟨x, x⟩).

#### Intuitive Understanding: Norms and Inner Products

- Norms measure “size” or “length.” Different norms emphasize different geometry (e.g., L¹ promotes sparsity; L² is rotationally invariant).
- Inner products measure “angles” and “projections.” Orthogonality captures “independent directions” useful for decomposition.
- Parallelogram law links norms to inner products; in Hilbert spaces, geometry is Euclidean-like.

#### ML Connection: Norms and Inner Products

- Choice of norm changes regularization and generalization (L¹ vs L²).
- Inner products underpin kernels, similarity, and projections (e.g., least squares).

### 2. Linear Operators and Operator Norms

#### Definition 2.1 (Linear Operator)

T: V → W is linear if T(x + y) = T(x) + T(y) and T(αx) = αT(x).

#### Definition 2.2 (Bounded Operator and Operator Norm)

T is bounded if ∃C such that \|T x\| ≤ C\|x\| for all x. The smallest such C is the operator norm:

\[ \|T\| = sup_{x≠0} \frac{\|T x\|}{\|x\|}. \]

In normed spaces, linearity + boundedness ⇔ continuity.

#### Intuitive Understanding: Operator Norm

- “Worst-case amplification”: how much T can stretch any vector.
- For matrices in ℝⁿ, this is the largest singular value.
- Continuity means small input changes can’t be blown up arbitrarily.

#### ML Connection: Operator Norms

- Lipschitz constants bound sensitivity; control robustness and generalization.
- Spectral norm regularization stabilizes training and adversarial robustness.

### 3. Hilbert Spaces and Projections

#### Theorem 3.1 (Projection Onto Closed Subspace)

In a Hilbert space, each x has a unique closest point P_U x in a closed subspace U. The error x − P_U x is orthogonal to U.

#### Intuitive Understanding: Orthogonal Projection

- “Shadow on a subspace”: best approximation in least-squares sense.
- Decomposes signal into “explained” and “residual” parts.

#### ML Connection: Projections

- Least squares, PCA, and many estimators are orthogonal projections.

### 4. Banach Spaces and Completeness

#### Definition 4.1 (Banach Space)

A normed space is Banach if every Cauchy sequence converges in the space.

#### Intuitive Understanding: Why Completeness?

- Guarantees limits live inside the space → existence of solutions to iterative methods.
- Prevents “chasing” limits that fall outside (no missing points).

#### ML Connection

- Convergence of optimization methods and fixed-point iterations rely on completeness.

## Implementation Details

See `exercise.py` for:

1. Normed/inner product space utilities and verifiers
2. Linear operator checks and operator norm estimation
3. Orthogonal projection onto a subspace in ℝⁿ
4. Continuity vs boundedness checks for linear maps

## Experiments

1. Compare effects of L¹ vs L² normalization on simple regressors
2. Estimate operator norms of random matrices and compare with SVD
3. Visualize orthogonal projections in 2D/3D

## Resources

- Kreyszig, “Introductory Functional Analysis with Applications”
- Lax, “Functional Analysis”
- Boyd & Vandenberghe, “Convex Optimization” (operator norms and duality)

## Exercises

- Prove Cauchy–Schwarz and derive triangle inequality from it
- Show linear + bounded ⇔ continuous for linear operators
- Implement projection onto span of given vectors and verify P² = P, Pᵀ = P for orthogonal projection
