# Vector Spaces and Linear Transformations for Machine Learning

## Prerequisites
- Basic matrix operations
- Real analysis fundamentals (from previous lesson)
- Elementary linear algebra

## Learning Objectives
- Master abstract vector spaces beyond ℝⁿ
- Understand linear transformations deeply
- Connect linear algebra to ML algorithms
- Build geometric intuition for high-dimensional spaces

## Mathematical Foundations

### 1. Vector Spaces

#### Definition 1.1 (Vector Space)
A vector space V over a field F (usually ℝ) is a set equipped with two operations:
- Addition: V × V → V
- Scalar multiplication: F × V → V

These operations satisfy eight axioms:

**Addition Axioms:**
1. Closure: ∀u, v ∈ V, u + v ∈ V
2. Associativity: (u + v) + w = u + (v + w)
3. Identity: ∃0 ∈ V such that v + 0 = v
4. Inverse: ∀v ∈ V, ∃(-v) ∈ V such that v + (-v) = 0
5. Commutativity: u + v = v + u

**Scalar Multiplication Axioms:**
6. Closure: ∀a ∈ F, v ∈ V, av ∈ V
7. Distributivity: a(u + v) = au + av and (a + b)v = av + bv
8. Associativity: a(bv) = (ab)v
9. Identity: 1v = v

#### ML Examples of Vector Spaces

1. **ℝⁿ**: Parameter spaces, feature vectors
2. **Function spaces**: L²[0,1] for kernel methods
3. **Matrix spaces**: ℝᵐˣⁿ for weight matrices
4. **Polynomial spaces**: For polynomial features
5. **Probability distributions**: Simplex for categorical distributions

### 2. Subspaces and Span

#### Definition 2.1 (Subspace)
A subset W ⊆ V is a subspace if:
1. 0 ∈ W
2. W is closed under addition
3. W is closed under scalar multiplication

#### Definition 2.2 (Linear Combination and Span)
Given vectors v₁, ..., vₖ ∈ V:
- Linear combination: ∑ᵢ aᵢvᵢ where aᵢ ∈ F
- Span: span{v₁, ..., vₖ} = {∑ᵢ aᵢvᵢ : aᵢ ∈ F}

#### Theorem 2.1 (Span is a Subspace)
For any set S ⊆ V, span(S) is the smallest subspace containing S.

**Proof**: 
1. 0 ∈ span(S) (take all coefficients = 0)
2. If u = ∑aᵢvᵢ and w = ∑bᵢvᵢ, then u + w = ∑(aᵢ + bᵢ)vᵢ ∈ span(S)
3. If u = ∑aᵢvᵢ and c ∈ F, then cu = ∑(caᵢ)vᵢ ∈ span(S) □

### 3. Linear Independence and Basis

#### Definition 3.1 (Linear Independence)
Vectors v₁, ..., vₖ are linearly independent if:
∑ᵢ aᵢvᵢ = 0 ⟹ aᵢ = 0 for all i

#### Definition 3.2 (Basis)
A set B ⊆ V is a basis if:
1. B is linearly independent
2. span(B) = V

#### Theorem 3.1 (Dimension)
All bases of a finite-dimensional vector space have the same cardinality, called the dimension.

#### ML Connection: Feature Selection
Linear independence relates to feature redundancy. If features are linearly dependent, we can reduce dimensionality without information loss.

### 4. Inner Product Spaces

#### Definition 4.1 (Inner Product)
An inner product on V is a function ⟨·,·⟩: V × V → ℝ satisfying:
1. Symmetry: ⟨u, v⟩ = ⟨v, u⟩
2. Linearity: ⟨au + bw, v⟩ = a⟨u, v⟩ + b⟨w, v⟩
3. Positive definiteness: ⟨v, v⟩ ≥ 0 with equality iff v = 0

#### Induced Norm
||v|| = √⟨v, v⟩

#### Theorem 4.1 (Cauchy-Schwarz Inequality)
|⟨u, v⟩| ≤ ||u|| · ||v||

**Proof**: Consider f(t) = ||u + tv||² ≥ 0 for all t ∈ ℝ.
f(t) = ⟨u + tv, u + tv⟩ = ||u||² + 2t⟨u, v⟩ + t²||v||²

This quadratic in t is non-negative, so its discriminant ≤ 0:
4⟨u, v⟩² - 4||u||²||v||² ≤ 0 □

### 5. Linear Transformations

#### Definition 5.1 (Linear Transformation)
T: V → W is linear if:
1. T(u + v) = T(u) + T(v)
2. T(cv) = cT(v)

Equivalently: T(au + bv) = aT(u) + bT(v)

#### Theorem 5.1 (Matrix Representation)
Every linear transformation T: ℝⁿ → ℝᵐ can be represented as matrix multiplication:
T(x) = Ax where A ∈ ℝᵐˣⁿ

The columns of A are T(e₁), ..., T(eₙ) where {eᵢ} is the standard basis.

#### Definition 5.2 (Kernel and Image)
- ker(T) = {v ∈ V : T(v) = 0} (null space)
- im(T) = {T(v) : v ∈ V} (range)

#### Theorem 5.2 (Rank-Nullity Theorem)
For T: V → W with V finite-dimensional:
dim(V) = dim(ker(T)) + dim(im(T))

### 6. Eigenvalues and Eigenvectors

#### Definition 6.1
For T: V → V, λ ∈ F is an eigenvalue with eigenvector v ≠ 0 if:
T(v) = λv

#### Characteristic Polynomial
For matrix A, the characteristic polynomial is:
p(λ = det(λI - A)

Eigenvalues are roots of p(λ).

#### Theorem 6.1 (Spectral Theorem for Symmetric Matrices)
If A ∈ ℝⁿˣⁿ is symmetric, then:
1. All eigenvalues are real
2. Eigenvectors for distinct eigenvalues are orthogonal
3. A is diagonalizable: A = QΛQ^T where Q is orthogonal

### 7. Singular Value Decomposition (SVD)

#### Theorem 7.1 (SVD)
Every matrix A ∈ ℝᵐˣⁿ has a decomposition:
A = UΣV^T

where:
- U ∈ ℝᵐˣᵐ is orthogonal (left singular vectors)
- Σ ∈ ℝᵐˣⁿ is diagonal (singular values σᵢ ≥ 0)
- V ∈ ℝⁿˣⁿ is orthogonal (right singular vectors)

#### Connection to Eigendecomposition
- A^T A = VΣ²V^T (eigendecomposition)
- AA^T = UΣ²U^T
- σᵢ = √λᵢ(A^T A)

### 8. Matrix Norms and Condition Numbers

#### Definition 8.1 (Operator Norm)
||A||_op = max_{||x||=1} ||Ax||

For the 2-norm: ||A||₂ = σ_max(A)

#### Definition 8.2 (Condition Number)
κ(A) = ||A|| · ||A⁻¹|| = σ_max(A) / σ_min(A)

High condition number → ill-conditioned → sensitive to perturbations

## Conceptual Understanding

### Geometric Intuition

1. **Vector spaces**: Generalize our intuition from ℝ² and ℝ³
2. **Linear transformations**: Preserve lines and origin
3. **Eigenspaces**: Invariant directions under transformation
4. **SVD**: Reveals the "action" of a matrix - rotate, scale, rotate

### Why This Matters for ML

1. **Data representation**: Features live in vector spaces
2. **Model parameters**: Weight spaces have geometric structure
3. **Optimization**: Gradient descent navigates these spaces
4. **Dimensionality reduction**: PCA uses eigendecomposition
5. **Regularization**: Constraints define subspaces

## Implementation Details

See `exercise.py` for implementations of:
1. Vector space operations and verification
2. Gram-Schmidt orthogonalization
3. Eigendecomposition from scratch
4. SVD algorithm
5. Condition number estimation
6. Applications to PCA and linear regression

## Experiments

1. **Numerical Stability**: Compare different orthogonalization methods
2. **Eigenvalue Algorithms**: Power method vs QR algorithm convergence
3. **SVD Applications**: Image compression, denoising
4. **Condition Numbers**: Effect on gradient descent convergence

## Research Connections

### Seminal Papers
1. Golub & Van Loan (1996) - "Matrix Computations"
   - Comprehensive numerical linear algebra

2. Trefethen & Bau (1997) - "Numerical Linear Algebra"
   - Modern perspective on algorithms

3. Halko, Martinsson & Tropp (2011) - "Finding Structure with Randomness"
   - Randomized algorithms for matrix decompositions

### ML Applications
1. **PCA**: Pearson (1901) - First principal component analysis
2. **Kernel Methods**: Schölkopf & Smola (2002) - Feature spaces
3. **Deep Learning**: Saxe et al. (2014) - "Exact solutions to nonlinear dynamics"

## Resources

### Primary Sources
1. **Axler - Linear Algebra Done Right**
   - Conceptual approach without determinants
2. **Strang - Linear Algebra and Its Applications**
   - Engineering perspective with applications
3. **MIT 18.06 Linear Algebra**
   - Gilbert Strang's legendary lectures

### Video Resources
1. **3Blue1Brown - Essence of Linear Algebra**
   - Beautiful geometric visualizations
2. **Khan Academy - Linear Algebra**
   - Step-by-step explanations
3. **Stanford CS229 - Linear Algebra Review**
   - ML-focused treatment

### Advanced Reading
1. **Horn & Johnson - Matrix Analysis**
   - Deep theoretical treatment
2. **Golub & Van Loan - Matrix Computations**
   - Numerical algorithms
3. **Boyd & Vandenberghe - Introduction to Applied Linear Algebra**
   - Modern applications

## Socratic Questions

### Understanding
1. Why do we need abstract vector spaces beyond ℝⁿ?
2. What's the geometric meaning of linear independence?
3. How does the choice of basis affect computations?

### Extension
1. Can you define vector spaces where "vectors" are functions?
2. What happens to SVD when the matrix is rank-deficient?
3. How do eigenvalues relate to matrix invertibility?

### Research
1. How can randomized algorithms speed up matrix computations?
2. What's the connection between condition number and learning rates?
3. How do neural networks learn representations in vector spaces?

## Exercises

### Theoretical
1. Prove that the set of symmetric matrices forms a vector space
2. Show that eigenvalues of a real symmetric matrix are real
3. Derive the optimal rank-k approximation using SVD

### Implementation
1. Implement Gram-Schmidt with numerical stability (modified version)
2. Code the power method for finding dominant eigenvalues
3. Build SVD using eigendecomposition of A^T A and AA^T

### Research
1. Investigate randomized SVD algorithms
2. Explore the geometry of loss surfaces using eigenvalues
3. Study how batch normalization affects condition numbers