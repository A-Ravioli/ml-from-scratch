# Kernel Methods and Reproducing Kernel Hilbert Spaces

## Prerequisites
- Linear algebra (inner products, eigendecompositions)
- Functional analysis (Hilbert spaces, operators)
- Optimization theory (convex optimization, duality)

## Learning Objectives
- Master kernel theory and RKHS fundamentals
- Understand kernel algorithms (SVM, kernel ridge regression)
- Connect kernels to feature maps and function spaces
- Apply kernel methods to modern ML problems

## Mathematical Foundations

### 1. Kernels and Feature Maps

#### Definition 1.1 (Kernel Function)
A function k: 𝒳 × 𝒳 → ℝ is a kernel if there exists a feature map φ: 𝒳 → ℋ into a Hilbert space ℋ such that:
k(x, x') = ⟨φ(x), φ(x')⟩_ℋ

#### Definition 1.2 (Positive Definite Kernel)
k is positive definite if for all n ∈ ℕ, x₁, ..., xₙ ∈ 𝒳, the Gram matrix K with Kᵢⱼ = k(xᵢ, xⱼ) is positive semidefinite.

#### Theorem 1.1 (Mercer's Theorem)
A continuous kernel k on compact 𝒳 is positive definite ⟺ it has an expansion:
k(x, x') = ∑ᵢ₌₁^∞ λᵢφᵢ(x)φᵢ(x')

where λᵢ ≥ 0 and φᵢ are eigenfunctions of the integral operator.

### 2. Examples of Kernels

#### Linear Kernel
k(x, x') = x^T x'

**Feature map**: φ(x) = x (identity map)
**Properties**: Simplest kernel, equivalent to linear methods

#### Polynomial Kernel
k(x, x') = (x^T x' + c)^d

**Feature map**: All monomials of degree ≤ d
**Dimension**: (n+d choose d) for n-dimensional input

#### RBF/Gaussian Kernel
k(x, x') = exp(-||x - x'||²/(2σ²))

**Feature map**: Infinite-dimensional
**Properties**: Universal kernel, smooth functions

#### String Kernels
For sequences/text data:
- Subsequence kernels
- Gap-weighted kernels
- Mismatch kernels

### 3. Reproducing Kernel Hilbert Spaces

#### Definition 3.1 (RKHS)
A Hilbert space ℋ of functions f: 𝒳 → ℝ is an RKHS if the evaluation functional δₓ(f) = f(x) is continuous for all x ∈ 𝒳.

#### Theorem 3.1 (Riesz Representation)
For RKHS ℋ, ∃!kₓ ∈ ℋ such that f(x) = ⟨f, kₓ⟩_ℋ for all f ∈ ℋ.

#### Definition 3.2 (Reproducing Kernel)
k(x, x') = ⟨kₓ, kₓ'⟩_ℋ = kₓ'(x)

#### Reproducing Property
f(x) = ⟨f, kₓ⟩_ℋ for all f ∈ ℋ, x ∈ 𝒳

### 4. Moore-Aronszajn Theorem

#### Theorem 4.1 (Moore-Aronszajn)
There is a bijection between positive definite kernels and RKHSs.

**Construction**: Given kernel k, define:
ℋ = span{kₓ : x ∈ 𝒳}

with inner product ⟨∑ᵢ αᵢkₓᵢ, ∑ⱼ βⱼkᵧⱼ⟩ = ∑ᵢⱼ αᵢβⱼk(xᵢ, yⱼ)

**Completion**: Take closure to get Hilbert space.

### 5. Support Vector Machines

#### Primal Problem
minimize (1/2)||w||² + C∑ᵢ ξᵢ
subject to yᵢ(w^T φ(xᵢ) + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

#### Dual Problem
maximize ∑ᵢ αᵢ - (1/2)∑ᵢⱼ αᵢαⱼyᵢyⱼk(xᵢ, xⱼ)
subject to 0 ≤ αᵢ ≤ C, ∑ᵢ αᵢyᵢ = 0

#### Representer Theorem
The solution has the form:
w = ∑ᵢ αᵢyᵢφ(xᵢ)

So f(x) = ∑ᵢ αᵢyᵢk(xᵢ, x) + b

#### KKT Conditions
- αᵢ = 0 ⟹ yᵢf(xᵢ) ≥ 1 (non-support vectors)
- 0 < αᵢ < C ⟹ yᵢf(xᵢ) = 1 (support vectors on margin)
- αᵢ = C ⟹ yᵢf(xᵢ) ≤ 1 (support vectors inside margin)

### 6. Kernel Ridge Regression

#### Problem
minimize ∑ᵢ (yᵢ - f(xᵢ))² + λ||f||²_ℋ

#### Solution
By representer theorem:
f(x) = ∑ᵢ αᵢk(xᵢ, x)

where α = (K + λI)⁻¹y and K is the Gram matrix.

#### Connection to Gaussian Processes
Kernel ridge regression is equivalent to GP regression with:
- Prior: f ~ GP(0, k)
- Likelihood: y|f ~ N(f, σ²I)

### 7. Kernel PCA

#### Problem
Find principal components in feature space φ(𝒳).

#### Solution
1. Center kernel matrix: K̃ = K - 1ₙK - K1ₙ + 1ₙK1ₙ
2. Solve eigenvalue problem: K̃α = λα
3. Principal components: ∑ᵢ αᵢᵏφ(xᵢ) where αᵏ is k-th eigenvector

#### Projections
Project new point x onto k-th component:
⟨φ(x), vₖ⟩ = ∑ᵢ αᵢᵏk(x, xᵢ)

### 8. Kernel Construction

#### Closure Properties
If k₁, k₂ are kernels:
- k₁ + k₂ is a kernel
- ck₁ is a kernel for c ≥ 0
- k₁k₂ is a kernel
- exp(k₁) is a kernel

#### Tensor Products
For kernels k₁ on 𝒳₁, k₂ on 𝒳₂:
k((x₁, x₂), (x₁', x₂')) = k₁(x₁, x₁')k₂(x₂, x₂')

#### String Kernels
For sequences s, t:
k(s, t) = ∑_{u∈Σ*} φᵤ(s)φᵤ(t)

where φᵤ(s) weights occurrences of subsequence u in s.

### 9. Learning Theory for Kernels

#### Rademacher Complexity
For RKHS with radius R:
ℛₘ(𝔹_R) ≤ R√(tr(K)/m)

where 𝔹_R = {f ∈ ℋ : ||f||_ℋ ≤ R}.

#### Generalization Bound
With probability ≥ 1 - δ:
R(f) ≤ R̂(f) + 2R√(tr(K)/m) + √(log(1/δ)/(2m))

#### Capacity Control
- Small ||f||_ℋ ⟹ simple function ⟹ good generalization
- Regularization λ||f||²_ℋ controls complexity

### 10. Computational Aspects

#### Kernel Matrix Properties
- Size: n × n (expensive for large n)
- Positive semidefinite
- Often dense (no sparsity)

#### Approximation Methods

##### Nyström Approximation
Sample m ≪ n points, approximate:
K ≈ K_{nm}K_{mm}⁻¹K_{mn}

##### Random Features
For translation-invariant kernels:
k(x, x') = E_ω[φ_ω(x)φ_ω(x')]

Sample ω₁, ..., ωₘ and use φ(x) = [φ_ω₁(x), ..., φ_ωₘ(x)].

##### Structured Kernels
- Circulant matrices: Fast via FFT
- Hierarchical methods: Fast multipole
- Toeplitz structure: Leveraged for efficiency

### 11. Multiple Kernel Learning

#### Problem
Learn optimal combination of kernels:
k(x, x') = ∑ⱼ βⱼkⱼ(x, x')

subject to constraints on β.

#### Approaches
- Convex combinations: ∑ⱼ βⱼ = 1, βⱼ ≥ 0
- ℓₚ regularization: ||β||ₚ ≤ 1
- Group regularization: Mixed norms

### 12. Modern Connections

#### Deep Networks as Kernels
In infinite width limit, neural networks correspond to specific kernels:
- Neural Tangent Kernel (NTK)
- Gaussian Process limit

#### Kernel Mean Embeddings
Embed distributions into RKHS:
μₚ = E_{x~P}[φ(x)]

Applications:
- Two-sample testing
- Independence testing
- Causal inference

#### Optimal Transport
Wasserstein distance can be kernelized for efficient computation.

## Implementation Details

See `exercise.py` for implementations of:
1. Basic kernel functions and properties
2. SVM solver using SMO algorithm
3. Kernel ridge regression
4. Kernel PCA
5. Multiple kernel learning
6. Kernel approximation methods

## Experiments

1. **Kernel Comparison**: Different kernels on same dataset
2. **Parameter Sensitivity**: Effect of kernel parameters
3. **Computational Scaling**: Exact vs approximate methods
4. **Feature Visualization**: Kernel PCA projections

## Research Connections

### Foundational Papers
1. Vapnik (1995) - "The Nature of Statistical Learning Theory"
2. Schölkopf et al. (1998) - "Nonlinear Component Analysis"
3. Shawe-Taylor & Cristianini (2004) - "Kernel Methods for Pattern Analysis"

### Modern Developments
1. Rahimi & Recht (2007) - "Random Features for Large-Scale Kernel Machines"
2. Bach (2008) - "Exploring Large Feature Spaces with Hierarchical Multiple Kernel Learning"
3. Jacot et al. (2018) - "Neural Tangent Kernel"

## Resources

### Primary Sources
1. **Schölkopf & Smola - Learning with Kernels**
   - Comprehensive kernel methods textbook
2. **Shawe-Taylor & Cristianini - Kernel Methods**
   - Theoretical foundations
3. **Steinwart & Christmann - Support Vector Machines**
   - Rigorous mathematical treatment

### Advanced Reading
1. **Berlinet & Thomas-Agnan - RKHS in Probability**
   - Deep theoretical treatment
2. **Cucker & Smale - Learning Theory**
   - Mathematical foundations
3. **Sriperumbudur et al. - Kernel Embeddings**
   - Modern applications

## Socratic Questions

### Understanding
1. Why do kernels enable nonlinear learning with linear algorithms?
2. What's the connection between kernels and feature maps?
3. How does the reproducing property work?

### Extension
1. Can you design kernels for structured data?
2. How do you choose appropriate kernel parameters?
3. What are the computational tradeoffs of different kernels?

### Research
1. How do modern deep networks relate to kernel methods?
2. Can we design adaptive kernels that learn from data?
3. What's the role of kernels in modern ML theory?

## Exercises

### Theoretical
1. Prove that the Gaussian kernel is positive definite
2. Derive the dual SVM formulation from the primal
3. Show that kernel ridge regression minimizes regularized risk

### Implementation
1. Implement SVM from scratch using SMO
2. Build kernel PCA for dimensionality reduction
3. Create multiple kernel learning algorithm

### Research
1. Study neural tangent kernels empirically
2. Investigate kernel approximation methods
3. Explore applications to structured prediction