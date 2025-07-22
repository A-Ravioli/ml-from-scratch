# Stochastic Gradient Descent and Variants

## Prerequisites
- Convex optimization fundamentals
- Probability theory (concentration inequalities)
- Linear algebra (matrix norms, eigenvalues)

## Learning Objectives
- Master SGD theory and convergence analysis
- Understand variance reduction techniques
- Implement efficient stochastic optimization algorithms
- Connect theory to practical deep learning

## Mathematical Foundations

### 1. Stochastic Gradient Descent

#### Problem Setup
Minimize: f(x) = E_{ξ~D}[F(x, ξ)] = ∫ F(x, ξ) dP(ξ)

where F(x, ξ) is the loss for sample ξ.

**Examples**:
- **Empirical Risk**: f(x) = (1/n)∑ᵢ F(x, ξᵢ)
- **Population Risk**: f(x) = E[ℓ(h(x), y)]

#### Algorithm
At iteration k:
1. Sample ξₖ ~ D (or choose uniformly from dataset)
2. Compute stochastic gradient: gₖ = ∇F(xₖ, ξₖ)
3. Update: xₖ₊₁ = xₖ - ηₖ gₖ

#### Key Properties
- **Unbiased**: E[gₖ|xₖ] = ∇f(xₖ)
- **Bounded variance**: E[||gₖ - ∇f(xₖ)||²] ≤ σ²
- **Computational efficiency**: O(1) per iteration vs O(n) for full gradients

### 2. Convergence Analysis

#### Assumptions
1. **Lipschitz smoothness**: ||∇f(x) - ∇f(y)|| ≤ L||x - y||
2. **Bounded variance**: E[||gₖ - ∇f(xₖ)||²] ≤ σ²
3. **Strong convexity** (optional): f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²

#### Theorem 2.1 (SGD Convergence - Convex Case)
For convex f with constant step size η ≤ 1/L:
E[f(x̄ₙ) - f*] ≤ ||x₀ - x*||²/(2ηn) + ησ²/2

where x̄ₙ = (1/n)∑ₖ₌₀ⁿ⁻¹ xₖ.

**Proof Sketch**:
1. Use smoothness: f(xₖ₊₁) ≤ f(xₖ) + ∇f(xₖ)ᵀ(xₖ₊₁ - xₖ) + (L/2)||xₖ₊₁ - xₖ||²
2. Substitute SGD update: xₖ₊₁ - xₖ = -ηgₖ
3. Take expectation and use unbiasedness
4. Apply telescoping sum and convexity □

#### Corollary 2.1 (Strongly Convex Case)
For μ-strongly convex f with η = 1/(μn):
E[f(xₙ) - f*] ≤ (L||x₀ - x*||²)/(2μn) + σ²/(2μn)

### 3. Mini-batch SGD

#### Algorithm
Use batch of size b:
gₖ = (1/b) ∑ᵢ₌₁ᵇ ∇F(xₖ, ξₖᵢ)

#### Variance Reduction
Var(mini-batch gradient) = σ²/b

**Tradeoff**: Larger batches reduce variance but increase computation per iteration.

#### Parallel Efficiency
- **Linear speedup**: Possible with b processors
- **Communication overhead**: Becomes bottleneck for large b
- **Generalization**: Large batches may hurt generalization

### 4. Learning Rate Schedules

#### Constant Step Size
η = constant

**Pros**: Simple, good for strongly convex
**Cons**: Doesn't converge to optimum (oscillates)

#### Diminishing Step Size
ηₖ = η₀/√k or ηₖ = η₀/k

**Robbins-Monro conditions**:
∑ₖ ηₖ = ∞ and ∑ₖ ηₖ² < ∞

#### Exponential Decay
ηₖ = η₀ γᵏ where γ < 1

#### Cosine Annealing
ηₖ = η_min + (η_max - η_min)(1 + cos(πk/K))/2

### 5. Momentum Methods

#### Heavy Ball (Polyak 1964)
vₖ₊₁ = βvₖ + ηgₖ
xₖ₊₁ = xₖ - vₖ₊₁

**Intuition**: Accumulate gradients to accelerate progress and smooth oscillations.

#### Nesterov Accelerated Gradient
vₖ₊₁ = βvₖ + η∇f(xₖ - βvₖ)
xₖ₊₁ = xₖ - vₖ₊₁

**Key insight**: Look ahead before computing gradient.

#### Theorem 5.1 (Nesterov Acceleration)
For convex functions, Nesterov achieves O(1/k²) convergence vs O(1/k) for SGD.

### 6. Adaptive Methods

#### AdaGrad (Duchi et al. 2011)
Gₖ = ∑ᵢ₌₁ᵏ gᵢgᵢᵀ (outer product sum)
xₖ₊₁ = xₖ - η/√(diag(Gₖ) + ε) ⊙ gₖ

**Intuition**: Adapt learning rate per parameter based on historical gradients.

#### RMSprop (Hinton 2012)
vₖ = γvₖ₋₁ + (1-γ)gₖ²
xₖ₊₁ = xₖ - η/√(vₖ + ε) ⊙ gₖ

**Fix**: Exponential moving average instead of cumulative sum.

#### Adam (Kingma & Ba 2014)
mₖ = β₁mₖ₋₁ + (1-β₁)gₖ (momentum)
vₖ = β₂vₖ₋₁ + (1-β₂)gₖ² (second moment)

Bias correction:
m̂ₖ = mₖ/(1-β₁ᵏ)
v̂ₖ = vₖ/(1-β₂ᵏ)

Update:
xₖ₊₁ = xₖ - η m̂ₖ/(√v̂ₖ + ε)

### 7. Variance Reduction Methods

#### SVRG (Johnson & Zhang 2013)
Maintain full gradient μ = ∇f(x̃) where x̃ is snapshot point.

Update: xₖ₊₁ = xₖ - η[∇F(xₖ, ξₖ) - ∇F(x̃, ξₖ) + μ]

**Variance**: E[||gradient||²] decreases as xₖ → x*

#### SAGA (Defazio et al. 2014)
Store gradient for each sample: table[i] = ∇F(x, ξᵢ)

Update: 
- Compute ∇F(xₖ, ξₖ) 
- gradient = ∇F(xₖ, ξₖ) - table[k] + (1/n)∑ᵢ table[i]
- Update table[k] and parameters

### 8. Convergence Rates Summary

| Method | Convex | Strongly Convex | Notes |
|--------|--------|-----------------|-------|
| GD | O(1/k) | O(exp(-k)) | Full gradients |
| SGD | O(1/√k) | O(1/k) | Stochastic |
| SGD (averaging) | O(1/k) | O(1/k) | Polyak averaging |
| Nesterov | O(1/k²) | O(exp(-√k)) | Acceleration |
| SVRG | - | O(exp(-k)) | Variance reduction |
| SAGA | - | O(exp(-k)) | Variance reduction |

### 9. Practical Considerations

#### Hyperparameter Tuning
- **Learning rate**: Most critical hyperparameter
- **Batch size**: Affects convergence and generalization
- **Momentum**: β ∈ [0.9, 0.99] typically good
- **Adaptive methods**: Usually work with default parameters

#### Gradient Clipping
For training stability:
g = g * min(1, threshold/||g||)

#### Learning Rate Schedules
- **Warmup**: Start with small learning rate, increase gradually
- **Decay**: Reduce learning rate during training
- **Cyclical**: Vary learning rate cyclically

## Implementation Details

See `exercise.py` for implementations of:
1. SGD with various learning rate schedules
2. Momentum and Nesterov acceleration
3. Adaptive methods (AdaGrad, RMSprop, Adam)
4. Variance reduction methods (SVRG, SAGA)
5. Gradient clipping and normalization
6. Comparison tools and visualization

## Experiments

1. **Convergence Comparison**: Different optimizers on quadratic functions
2. **Learning Rate Sensitivity**: Effect of learning rate choice
3. **Batch Size Study**: Variance vs computation tradeoff
4. **Adaptive Methods**: Performance on different problem types

## Research Connections

### Seminal Papers
1. Robbins & Monro (1951) - "A Stochastic Approximation Method"
2. Polyak (1964) - "Some Methods of Speeding up Convergence"
3. Nesterov (1983) - "A Method for Unconstrained Convex Minimization"
4. Duchi et al. (2011) - "Adaptive Subgradient Methods"

### Modern Developments
1. Johnson & Zhang (2013) - "Accelerating Stochastic Gradient Descent"
2. Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization"
3. Reddi et al. (2018) - "On the Convergence of Adam and Beyond"

## Resources

### Primary Sources
1. **Shalev-Shwartz & Ben-David - Understanding Machine Learning**
2. **Bubeck - Convex Optimization: Algorithms and Complexity**
3. **Bottou et al. - Optimization Methods for Large-Scale ML**

### Video Resources
1. **CMU 10-725 - Convex Optimization**
2. **Stanford CS229 - Machine Learning**
3. **NYU DS-GA 1003 - Machine Learning**

## Socratic Questions

### Understanding
1. Why does SGD converge slower than gradient descent?
2. How does momentum help optimization?
3. When do adaptive methods fail?

### Extension
1. Can we get better than O(1/√k) rates for non-convex functions?
2. How do different batch sizes affect generalization?
3. What's the optimal learning rate schedule?

### Research
1. Why do large batch methods generalize poorly?
2. How can we design better adaptive methods?
3. What's the role of noise in SGD for generalization?

## Exercises

### Theoretical
1. Prove SGD convergence for strongly convex functions
2. Derive the variance of mini-batch gradients
3. Analyze the effect of momentum on convergence

### Implementation
1. Implement all major SGD variants from scratch
2. Build hyperparameter tuning framework
3. Create convergence visualization tools

### Research
1. Study the effect of learning rate schedules empirically
2. Investigate adaptive methods on non-convex problems
3. Explore the connection between batch size and generalization