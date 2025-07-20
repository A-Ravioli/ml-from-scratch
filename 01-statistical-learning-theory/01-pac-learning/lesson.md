# PAC Learning Theory for Machine Learning

## Prerequisites
- Probability theory (concentration inequalities)
- Real analysis (uniform convergence)
- Linear algebra basics

## Learning Objectives
- Understand PAC learning framework and sample complexity
- Master VC dimension and its implications
- Connect theory to practical ML algorithms
- Build intuition for generalization bounds

## Mathematical Foundations

### 1. The PAC Learning Framework

#### Setup
- **Input space**: 𝒳 (e.g., ℝᵈ for real vectors)
- **Output space**: 𝒴 (e.g., {0,1} for binary classification)
- **Unknown distribution**: D over 𝒳 × 𝒴
- **Hypothesis class**: ℋ ⊆ {h: 𝒳 → 𝒴}
- **Training set**: S = {(x₁,y₁), ..., (xₘ,yₘ)} ~ Dᵐ

#### Definition 1.1 (True Risk)
For hypothesis h and distribution D:
R(h) = P_{(x,y)~D}[h(x) ≠ y] = E[ℓ(h(x), y)]

where ℓ is the 0-1 loss.

#### Definition 1.2 (Empirical Risk)
R̂_S(h) = (1/m) ∑ᵢ₌₁ᵐ ℓ(h(xᵢ), yᵢ)

#### Definition 1.3 (PAC Learnable)
A hypothesis class ℋ is PAC learnable if there exists an algorithm A and polynomial p such that:
∀ε, δ ∈ (0,1), ∀D: if m ≥ p(1/ε, 1/δ, size(x)), then
P[R(A(S)) ≤ inf_{h∈ℋ} R(h) + ε] ≥ 1 - δ

**Interpretation**: With high probability (1-δ), algorithm finds hypothesis within ε of optimal.

### 2. Finite Hypothesis Classes

#### Theorem 2.1 (Fundamental Theorem of PAC Learning - Finite Case)
Any finite hypothesis class ℋ is PAC learnable with sample complexity:
m ≥ (1/ε)[log|ℋ| + log(1/δ)]

**Proof Sketch**:
1. Fix h* = argmin_{h∈ℋ} R(h)
2. For any h with R(h) ≥ R(h*) + ε:
   P[R̂_S(h) ≤ R̂_S(h*)] ≤ P[R̂_S(h) ≤ R(h) - ε] ≤ e^{-2mε²} (Hoeffding)
3. Union bound over all "bad" hypotheses
4. ERM algorithm achieves the bound □

#### Corollary 2.1 (Realizable Case)
If ∃h* ∈ ℋ with R(h*) = 0 (realizable), then:
m ≥ (1/ε)[log|ℋ| + log(1/δ)]

ensures P[R(ERM(S)) ≤ ε] ≥ 1 - δ.

### 3. VC Dimension

#### Definition 3.1 (Shattering)
A set C ⊆ 𝒳 is shattered by ℋ if:
∀b ∈ {0,1}^{|C|}, ∃h ∈ ℋ such that h(x) = b_x for all x ∈ C

#### Definition 3.2 (VC Dimension)
VCdim(ℋ) = max{|C| : C is shattered by ℋ}

#### Examples
1. **Linear classifiers in ℝᵈ**: VCdim = d + 1
2. **Axis-aligned rectangles in ℝ²**: VCdim = 4
3. **k-NN classifiers**: VCdim = ∞
4. **Neural networks**: Generally exponential in parameters

#### Theorem 3.1 (Sauer-Shelah Lemma)
For finite VC dimension d:
|{h|_C : h ∈ ℋ}| ≤ ∑ᵢ₌₀ᵈ (|C| choose i) ≤ (e|C|/d)ᵈ

where h|_C denotes the restriction of h to set C.

### 4. PAC Learning with Infinite Hypothesis Classes

#### Theorem 4.1 (Fundamental Theorem of PAC Learning - General Case)
A hypothesis class ℋ is PAC learnable if and only if VCdim(ℋ) < ∞.

Moreover, the sample complexity is:
m = O((d + log(1/δ))/ε)

where d = VCdim(ℋ).

#### Proof Strategy
**Upper bound**: Use uniform convergence + union bound over finite ε-covers
**Lower bound**: Construct hard distribution using shattering

#### Theorem 4.2 (VC Generalization Bound)
With probability ≥ 1 - δ:
R(h) ≤ R̂_S(h) + √((8d log(2m/d) + 8log(4/δ))/m)

for all h ∈ ℋ simultaneously, where d = VCdim(ℋ).

### 5. Rademacher Complexity

#### Definition 5.1 (Empirical Rademacher Complexity)
R̂_S(ℱ) = (1/m) E_σ[sup_{f∈ℱ} ∑ᵢ₌₁ᵐ σᵢf(xᵢ)]

where σᵢ are independent Rademacher random variables (±1 with equal probability).

#### Theorem 5.1 (Rademacher Generalization Bound)
With probability ≥ 1 - δ:
R(h) ≤ R̂_S(h) + 2R̂_S(ℋ ∘ S) + √(log(2/δ)/2m)

#### Connection to VC Dimension
For ℋ with VCdim(ℋ) = d:
R̂_S(ℋ) ≤ c√(d/m)

### 6. Structural Risk Minimization

When ℋ is too complex (large VC dimension), use nested sequence:
ℋ₁ ⊆ ℋ₂ ⊆ ℋ₃ ⊆ ...

#### Theorem 6.1 (SRM Bound)
Choose ĥ = argmin_{h∈ℋₖ} [R̂_S(h) + √((VCdim(ℋₖ) + log k + log(1/δ))/m)]

Then with probability ≥ 1 - δ:
R(ĥ) ≤ inf_{k,h∈ℋₖ} [R(h) + 2√((VCdim(ℋₖ) + log k + log(1/δ))/m)]

### 7. Agnostic Learning

#### Definition 7.1 (Agnostic PAC Learning)
ℋ is agnostically PAC learnable if ∃ algorithm A such that:
R(A(S)) ≤ inf_{h∈ℋ} R(h) + ε

with probability ≥ 1 - δ, for polynomial sample complexity.

#### Theorem 7.1 (Agnostic Learning = Uniform Convergence)
ℋ is agnostically PAC learnable ⟺ ℋ has uniform convergence property ⟺ VCdim(ℋ) < ∞.

### 8. Online Learning Connections

#### Definition 8.1 (Littlestone Dimension)
Maximum depth of binary tree that can be shattered by ℋ.

#### Theorem 8.1 (Online to Batch Conversion)
If ℋ has Littlestone dimension d, then mistake bound in online learning ≤ 2^d.

This connects to sample complexity: m = O(d log(1/δ)/ε).

## Applications to Machine Learning

### Linear Classifiers
- **Hypothesis class**: {x ↦ sign(w·x + b) : w ∈ ℝᵈ, b ∈ ℝ}
- **VC dimension**: d + 1
- **Sample complexity**: O((d + log(1/δ))/ε)

### Decision Trees
- **Hypothesis class**: Trees of depth ≤ k
- **VC dimension**: O(2^k)
- **Implication**: Need regularization to prevent overfitting

### Neural Networks
- **VC dimension**: Generally Θ(W log W) where W = # weights
- **Modern theory**: PAC-Bayes, compression bounds, implicit regularization

### Support Vector Machines
- **Key insight**: VC dimension depends on margin, not dimension
- **Fat-shattering dimension**: Refined complexity measure
- **Margin bounds**: Better than naive VC bounds

## Conceptual Understanding

### The Bias-Complexity Tradeoff

1. **Small ℋ (low complexity)**:
   - Low estimation error (few hypotheses to choose from)
   - High approximation error (might not contain good hypothesis)

2. **Large ℋ (high complexity)**:
   - High estimation error (many hypotheses, overfitting risk)
   - Low approximation error (likely contains good hypothesis)

### Why VC Dimension Matters

1. **Combinatorial complexity**: Captures "effective size" of infinite classes
2. **Distribution-free**: Bounds hold for all distributions
3. **Tight**: Both upper and lower bounds match (up to constants)
4. **Algorithmic**: ERM is optimal for VC classes

### Modern Perspectives

1. **Deep Learning**: Classical bounds often vacuous
2. **Implicit regularization**: Algorithms prefer "simple" solutions
3. **Interpolation**: Modern ML often fits training data perfectly
4. **Double descent**: Complexity vs performance not monotonic

## Implementation Details

See `exercise.py` for implementations of:
1. VC dimension computation for simple classes
2. Empirical risk minimization
3. Sample complexity calculators
4. Rademacher complexity estimation
5. Structural risk minimization

## Experiments

1. **Sample Complexity**: Verify theoretical predictions empirically
2. **VC Dimension**: Estimate VC dimension via shattering experiments
3. **Generalization Gap**: Compare training and test error vs complexity
4. **SRM**: Demonstrate model selection via complexity penalization

## Research Connections

### Seminal Papers
1. Valiant (1984) - "A Theory of the Learnable"
   - Introduced PAC learning framework

2. Vapnik & Chervonenkis (1971) - "On the Uniform Convergence"
   - VC theory foundations

3. Kearns & Vazirani (1994) - "An Introduction to Computational Learning Theory"
   - Comprehensive treatment

### Modern Extensions
1. **PAC-Bayes**: McAllester (1999) - Bayesian perspective
2. **Stability**: Bousquet & Elisseeff (2002) - Algorithmic stability
3. **Compression**: Littlestone & Warmuth (1986) - Sample compression

## Resources

### Primary Sources
1. **Shalev-Shwartz & Ben-David - Understanding Machine Learning**
   - Modern, comprehensive treatment
2. **Vapnik - The Nature of Statistical Learning Theory**
   - Original VC theory book
3. **Kearns & Vazirani - Computational Learning Theory**
   - Classic introduction

### Video Resources
1. **MIT 9.520 - Statistical Learning Theory**
   - Tomaso Poggio and team
2. **Caltech CS156 - Learning from Data**
   - Yaser Abu-Mostafa
3. **CMU 15-859 - Machine Learning Theory**
   - Nina Balcan

### Advanced Reading
1. **Mendelson - Learning without Concentration**
   - Modern high-dimensional perspective
2. **Bartlett & Mendelson - Rademacher Complexities**
   - Advanced complexity measures
3. **Mohri, Rostamizadeh & Talwalkar - Foundations of ML**
   - Algorithmic learning theory

## Socratic Questions

### Understanding
1. Why does finite VC dimension guarantee learnability?
2. How does sample complexity depend on confidence vs accuracy?
3. What's the difference between realizable and agnostic learning?

### Extension
1. Can we characterize learnability for other loss functions?
2. How do these bounds apply to modern deep learning?
3. What happens with infinite-dimensional feature spaces?

### Research
1. Are VC bounds tight for practical algorithms?
2. How does the choice of algorithm affect generalization?
3. Can we improve bounds using problem-specific structure?

## Exercises

### Theoretical
1. Prove that VCdim of linear classifiers in ℝᵈ is d+1
2. Show that k-NN has infinite VC dimension
3. Derive sample complexity for agnostic learning

### Implementation
1. Implement empirical VC dimension estimation
2. Code ERM for different hypothesis classes
3. Build sample complexity calculator

### Research
1. Study generalization in neural networks empirically
2. Investigate stability-based bounds
3. Explore PAC-Bayes theory applications