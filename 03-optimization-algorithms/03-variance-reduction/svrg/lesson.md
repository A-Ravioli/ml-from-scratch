# SVRG: Stochastic Variance Reduced Gradient

## Prerequisites
- Stochastic gradient descent fundamentals
- Convex optimization theory
- Concentration inequalities
- Basic understanding of variance reduction techniques

## Learning Objectives
- Master SVRG algorithm and its theoretical properties
- Understand variance reduction in finite-sum optimization
- Implement SVRG from scratch with convergence analysis
- Compare SVRG with other variance reduction methods

## Mathematical Foundations

### 1. Problem Setup

Consider the finite-sum optimization problem:
```
minimize f(x) = (1/n) ∑ᵢ₌₁ⁿ fᵢ(x)
```

where each fᵢ(x) is a smooth function.

**Examples**:
- **Empirical Risk Minimization**: fᵢ(x) = ℓ(h(x), yᵢ, zᵢ)
- **Regularized Learning**: fᵢ(x) = ℓᵢ(x) + (λ/n)||x||²

### 2. The Variance Problem in SGD

#### Standard SGD Update
xₖ₊₁ = xₖ - η∇fᵢₖ(xₖ) where iₖ ~ Uniform{1,...,n}

#### Variance Analysis
- **Bias**: E[∇fᵢₖ(xₖ)] = ∇f(xₖ) ✓ (unbiased)
- **Variance**: Var[∇fᵢₖ(xₖ)] = E[||∇fᵢₖ(xₖ) - ∇f(xₖ)||²]

**Key Issue**: Variance doesn't decrease as xₖ → x*, leading to O(1/√k) rates.

### 3. SVRG Algorithm

#### Core Idea
Reduce variance by using a control variate that becomes more accurate near the optimum.

#### Algorithm 3.1 (SVRG)
**Parameters**: m (inner loop length), η (step size)

For s = 0, 1, 2, ...
1. **Snapshot**: x̃ = x₀ˢ, μ = ∇f(x̃) = (1/n)∑ᵢ₌₁ⁿ ∇fᵢ(x̃)
2. **Initialize**: x₀ˢ⁺¹ = x̃
3. **Inner loop**: For t = 0, 1, ..., m-1:
   - Sample iₜ uniformly from {1, ..., n}
   - Compute variance-reduced gradient:
     ```
     gₜ = ∇fᵢₜ(xₜˢ⁺¹) - ∇fᵢₜ(x̃) + μ
     ```
   - Update: xₜ₊₁ˢ⁺¹ = xₜˢ⁺¹ - ηgₜ
4. **Set**: x₀ˢ⁺² = xₘˢ⁺¹

#### Variance Reduction Mechanism
E[gₜ|xₜˢ⁺¹] = ∇f(xₜˢ⁺¹) (unbiased)

Variance:
```
Var[gₜ|xₜˢ⁺¹] = E[||∇fᵢₜ(xₜˢ⁺¹) - ∇fᵢₜ(x̃)||²]
```

**Key insight**: As xₜˢ⁺¹ → x̃, variance → 0!

### 4. Convergence Analysis

#### Assumptions
1. **Smoothness**: Each fᵢ is L-smooth
2. **Strong convexity**: f is μ-strongly convex
3. **Bounded variance**: σ² = (1/n)∑ᵢ₌₁ⁿ ||∇fᵢ(x*)||² < ∞

#### Theorem 4.1 (SVRG Convergence)
Choose η ≤ 1/(8L) and m ≥ 8L/μ. Then:

E[f(x₀ˢ⁺¹) - f*] ≤ ρˢ[f(x₀⁰) - f*]

where ρ = 1/2 < 1.

**Convergence rate**: O(exp(-s)) = **linear convergence**!

#### Proof Sketch
1. **Inner loop analysis**: Show that inner loop makes progress on average
2. **Variance bound**: Control variance using smoothness
3. **Progress lemma**: Each epoch reduces optimality gap by constant factor
4. **Telescoping**: Combine over epochs

#### Corollary 4.1 (Iteration Complexity)
To achieve ε-accuracy: O((n + L/μ)log(1/ε)) gradient evaluations.

**Comparison**:
- **SGD**: O(1/(μη)) = O(L/μ²ε) 
- **SVRG**: O((n + L/μ)log(1/ε))

SVRG wins when n ≪ L/μ²ε.

### 5. Practical Considerations

#### Memory Requirements
- Store full gradient μ: O(d) memory
- No per-sample gradient storage (unlike SAGA)

#### Computational Cost per Iteration
- **Gradient evaluation**: Same as SGD
- **Full gradient**: Every m iterations (amortized O(1))
- **Extra operations**: O(d) vector operations

#### Hyperparameter Selection

**Step size η**:
- Theory: η ≤ 1/(8L)
- Practice: Use line search or adaptive methods

**Inner loop length m**:
- Theory: m ≥ 8L/μ  
- Practice: m = 2n often works well
- Tradeoff: Larger m → better variance reduction, more memory

**Update frequency**:
- **Too frequent**: Expensive full gradient computation
- **Too rare**: Stale snapshot, poor variance reduction

### 6. Variants and Extensions

#### SVRG++
Improved step size and better practical performance.

#### Proximal SVRG
For composite optimization: f(x) + g(x) where g is non-smooth.

Update: xₜ₊₁ = prox_{ηg}(xₜ - ηgₜ)

#### SVRG for Non-convex Functions
- No linear convergence guarantee
- Still provides variance reduction
- Convergence to stationary points

#### Accelerated SVRG
Combine with Nesterov acceleration for better rates.

### 7. Comparison with Other Methods

#### vs SGD
- **Pros**: Linear convergence, lower variance
- **Cons**: Memory overhead, complexity

#### vs SAGA
- **SVRG**: Simpler, less memory (O(d) vs O(nd))
- **SAGA**: Better constants, more stable

#### vs SAG
- Similar variance reduction
- SVRG easier to analyze theoretically

#### vs Full Gradient Methods
- **SVRG**: Better for large n
- **Full GD**: Better for small n, simpler

## Implementation Details

See `exercise.py` for implementations of:
1. Basic SVRG algorithm
2. Adaptive step size selection
3. Different snapshot update strategies
4. Proximal SVRG for composite problems
5. Comparison with SGD and other methods

## Experiments

1. **Convergence Verification**: Test linear convergence on strongly convex problems
2. **Hyperparameter Sensitivity**: Effect of m and η choices
3. **Problem Size Study**: When does SVRG outperform SGD?
4. **Memory vs Speed Tradeoff**: Different snapshot strategies

## Research Connections

### Seminal Papers
1. Johnson & Zhang (2013) - "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"
   - Original SVRG paper
   
2. Xiao & Zhang (2014) - "A Proximal Stochastic Gradient Method with Progressive Variance Reduction"
   - Theoretical improvements and proximal extension

3. Hofmann et al. (2015) - "Variance Reduced Stochastic Gradient Descent with Neighbors"
   - SVRG++ improvements

### Modern Developments
1. Allen-Zhu (2017) - "Katyusha: The First Direct Acceleration of Stochastic Gradient Methods"
2. Zhou et al. (2018) - "Direct Acceleration of SAGA using Sampled Negative Momentum"

## Resources

### Primary Sources
1. **Johnson & Zhang (2013)** - Original SVRG paper
2. **Schmidt et al. (2017)** - "Minimizing finite sums with the stochastic average gradient"
3. **Defazio et al. (2014)** - "SAGA: A fast incremental gradient method"

### Video Resources
1. **ICML 2013 Tutorial** - "Stochastic Optimization for Machine Learning"
2. **Francis Bach Lectures** - "Optimization for Machine Learning"
3. **Sébastien Bubeck** - "Convex Optimization: Algorithms and Complexity"

### Advanced Reading
1. **Bottou et al. (2018)** - "Optimization Methods for Large-Scale Machine Learning"
2. **Bubeck (2015)** - "Convex Optimization: Algorithms and Complexity"

## Socratic Questions

### Understanding
1. Why does SVRG achieve linear convergence while SGD only achieves O(1/√k)?
2. How does the choice of inner loop length m affect convergence?
3. What happens to SVRG when the functions fᵢ are very different?

### Extension  
1. Can you combine SVRG with momentum or acceleration?
2. How would you extend SVRG to mini-batches?
3. What modifications are needed for non-convex optimization?

### Research
1. How can we choose the snapshot update frequency adaptively?
2. What's the optimal way to select which samples to use for variance reduction?
3. Can we get better than linear convergence for finite-sum problems?

## Exercises

### Theoretical
1. Prove that the SVRG gradient estimator is unbiased
2. Derive the variance bound for the SVRG gradient estimator
3. Show how the choice of m affects the convergence constant

### Implementation
1. Implement SVRG with different snapshot strategies
2. Create adaptive hyperparameter selection methods
3. Compare SVRG variants on real datasets

### Research
1. Study the effect of problem conditioning on SVRG performance
2. Investigate SVRG behavior on non-convex neural network training
3. Design new variance reduction techniques inspired by SVRG