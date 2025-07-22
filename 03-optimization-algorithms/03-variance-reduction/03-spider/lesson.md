# SPIDER: Stochastic Path-Integrated Differential Estimator

## Prerequisites
- SVRG and SAGA algorithms
- Non-convex optimization theory
- Stochastic gradient methods
- Variance reduction techniques

## Learning Objectives
- Understand SPIDER's path-integrated variance reduction
- Master non-convex optimization convergence analysis
- Implement SPIDER with optimal hyperparameter selection
- Compare SPIDER with other variance reduction methods

## Mathematical Foundations

### 1. Problem Setup and Motivation

#### Non-convex Finite-sum Problem
```
minimize f(x) = (1/n) ∑ᵢ₌₁ⁿ fᵢ(x)
```

**Key challenge**: SVRG and SAGA assume convexity for their best results. What about non-convex problems like neural networks?

#### Limitations of Existing Methods
- **SVRG**: Requires full gradient computation periodically
- **SAGA**: O(nd) memory requirement  
- **Both**: Analysis relies heavily on convexity

### 2. The SPIDER Algorithm

#### Core Innovation: Path-Integrated Estimator
Instead of using single snapshots (SVRG) or stored gradients (SAGA), SPIDER integrates gradient differences along the optimization path.

#### Algorithm 2.1 (SPIDER)
**Parameters**: 
- Batch sizes: b₁ (for estimator), b₂ (for updates)
- Update frequency: q
- Step size: η

**Initialization**:
```
v₀ = ∇f_S₀(x₀) where |S₀| = b₁
```

**For k = 0, 1, 2, ...**:
1. **If k mod q = 0** (estimator update):
   ```
   vₖ = ∇f_Sₖ(xₖ) where |Sₖ| = b₁
   ```
2. **Else** (path-integrated update):
   ```
   Sample Bₖ with |Bₖ| = b₂
   vₖ = ∇f_Bₖ(xₖ) - ∇f_Bₖ(xₖ₋₁) + vₖ₋₁
   ```
3. **Parameter update**:
   ```
   xₖ₊₁ = xₖ - η vₖ
   ```

#### Key Insight: Telescoping Property
The estimator vₖ telescopes:
```
vₖ = ∇f_S₀(x₀) + ∑ᵢ₌₁ᵏ [∇f_Bᵢ(xᵢ) - ∇f_Bᵢ(xᵢ₋₁)]
```

This provides a **path-integrated** estimate of the current gradient.

### 3. Theoretical Analysis

#### Assumptions
1. **Smoothness**: Each fᵢ is L-smooth
2. **Bounded variance**: σ² = E[‖∇fᵢ(x) - ∇f(x)‖²] ≤ σ² 
3. **Lower bound**: f(x) ≥ f* > -∞

#### Theorem 3.1 (SPIDER Convergence - Non-convex)
Choose parameters appropriately. Then:

E[‖∇f(x_output)‖²] ≤ ε

with gradient complexity:
```
O(n + √(nσ²/ε) + σ⁴/ε²)
```

#### Comparison with Other Methods

| Method | Gradient Complexity (Non-convex) |
|--------|----------------------------------|
| **SGD** | O(σ²/ε²) |
| **SVRG** | O(n + σ²/ε²) |
| **SPIDER** | O(n + √(nσ²/ε) + σ⁴/ε²) |

**Key insight**: SPIDER interpolates between SGD and full gradient methods!

#### Theorem 3.2 (SPIDER Convergence - Finite-sum Convex)
For convex finite-sum problems, SPIDER achieves:
```
O((n + √n/ε) log(1/ε))
```

This matches the best known rates for variance-reduced methods.

### 4. Algorithm Variants

#### SPIDER-SFO (Stochastic First-Order)
Simplified version with b₂ = 1:
```
vₖ = ∇fᵢₖ(xₖ) - ∇fᵢₖ(xₖ₋₁) + vₖ₋₁
```

**Advantages**:
- Minimal memory overhead
- Simple implementation
- Good practical performance

#### Proximal SPIDER
For composite problems f(x) + g(x):
```
xₖ₊₁ = prox_{ηg}(xₖ - η vₖ)
```

#### SPIDER with Momentum
Incorporate momentum for acceleration:
```
yₖ = xₖ + β(xₖ - xₖ₋₁)
vₖ = ∇f_Bₖ(yₖ) - ∇f_Bₖ(yₖ₋₁) + vₖ₋₁
xₖ₊₁ = yₖ - η vₖ
```

### 5. Implementation Considerations

#### Hyperparameter Selection

**Batch sizes**:
- b₁ (estimator): Larger → better variance reduction, more expensive
- b₂ (updates): Usually b₂ = 1 for efficiency

**Update frequency q**:
- Smaller q → more frequent resets, better variance control
- Larger q → fewer expensive estimator updates

**Step size η**:
- Theory: η ∝ 1/L
- Practice: Use line search or adaptive methods

#### Memory Requirements
- **Estimator storage**: O(d)
- **Previous point**: O(d) 
- **Total**: O(d) - much better than SAGA!

#### Computational Cost per Iteration
- **Estimator update**: O(b₁) gradient evaluations (every q iterations)
- **Regular update**: O(b₂) gradient evaluations
- **Amortized cost**: O(b₂ + b₁/q) per iteration

### 6. Practical Advantages

#### Flexibility
- Works for both convex and non-convex problems
- Handles composite optimization naturally
- Adapts to problem structure automatically

#### Scalability  
- Low memory overhead O(d)
- Parallelizable mini-batch computation
- Suitable for large-scale problems

#### Robustness
- Less sensitive to hyperparameter choice than SVRG
- Natural adaptation to problem difficulty
- Works well with adaptive step sizes

### 7. Theoretical Insights

#### Why Path Integration Works
1. **Telescoping cancellation**: Adjacent gradients nearly cancel
2. **Variance accumulation**: Path differences have lower variance
3. **Natural adaptation**: Method adapts to optimization trajectory

#### Connection to Other Methods
- **q = 1**: Reduces to full gradient method
- **q = ∞**: Similar to SGD with growing variance
- **Optimal q**: Balances computation and variance

#### Non-convex Analysis Techniques
- **Descent lemma**: Control function value decrease
- **Variance bound**: Track estimator variance evolution
- **Martingale analysis**: Handle stochastic terms carefully

### 8. Extensions and Research Directions

#### Accelerated SPIDER
Combine with Nesterov momentum for better rates.

#### Adaptive SPIDER
Automatically adjust q and batch sizes based on progress.

#### Distributed SPIDER
Parallelize across multiple machines with communication efficiency.

#### Non-smooth SPIDER
Extend to non-smooth optimization using subgradients.

## Implementation Details

See `exercise.py` for implementations of:
1. Basic SPIDER algorithm with configurable parameters
2. SPIDER-SFO simplified variant
3. Proximal SPIDER for composite problems
4. Adaptive hyperparameter selection
5. Comparison with SVRG, SAGA, and SGD
6. Non-convex optimization examples

## Experiments

1. **Convergence Verification**: Test on convex and non-convex problems
2. **Hyperparameter Study**: Optimal choice of q, b₁, b₂
3. **Memory Efficiency**: Compare memory usage with other methods
4. **Large-scale Performance**: Scalability analysis

## Research Connections

### Seminal Papers
1. Fang et al. (2018) - "SPIDER: Near-Optimal Non-Convex Optimization via Stochastic Path-Integrated Differential Estimator"
   - Original SPIDER paper

2. Wang et al. (2018) - "SpiderBoost and Momentum: Faster Variance Reduction Algorithms"
   - Accelerated variants

3. Li & Li (2018) - "On the Convergence Rate of Stochastic Mirror Descent for Nonsmooth Nonconvex Optimization"
   - Extensions to non-smooth problems

### Recent Developments
1. Nguyen et al. (2019) - "SARAH: A Novel Method for Machine Learning Problems Using Stochastic Recursive Gradient"
2. Horváth & Richtárik (2019) - "Nonconvex Variance Reduced Optimization with Arbitrary Sampling"
3. Cutkosky & Orabona (2019) - "Momentum-Based Variance Reduction in Non-Convex SGD"

## Resources

### Primary Sources
1. **Fang et al. (2018)** - Original SPIDER paper
2. **Reddi et al. (2016)** - "Stochastic Variance Reduction for Nonconvex Optimization"
3. **Allen-Zhu (2018)** - "How To Make the Gradients Small Stochastically"

### Video Resources
1. **Cong Fang** - "SPIDER: Near-Optimal Non-Convex Optimization" (NeurIPS 2018)
2. **Zeyuan Allen-Zhu** - "Variance Reduction for Non-Convex Optimization"

### Advanced Reading
1. **Arjevani et al. (2019)** - "Lower Bounds for Non-Convex Stochastic Optimization"
2. **Carmon et al. (2020)** - "Gradient Descent Finds the Cubic-Regularized Non-Convex Newton Step"

## Socratic Questions

### Understanding
1. How does path integration differ from the snapshot approach of SVRG?
2. Why is SPIDER particularly well-suited for non-convex optimization?
3. What role does the update frequency q play in balancing computation and variance?

### Extension
1. How would you design an adaptive method to choose q automatically?
2. Can SPIDER be extended to constrained optimization problems?
3. What happens to SPIDER in the online/streaming setting?

### Research
1. Can we achieve better complexity bounds for specific problem classes?
2. How does SPIDER perform on problems with sparse gradients?
3. What's the optimal way to parallelize SPIDER across multiple machines?

## Exercises

### Theoretical
1. Prove that the SPIDER estimator telescopes correctly
2. Derive the variance bound for the path-integrated estimator
3. Analyze the effect of different choices of q on convergence

### Implementation
1. Implement SPIDER with adaptive hyperparameter selection
2. Create efficient implementations for sparse problems
3. Build proximal SPIDER for regularized learning

### Research
1. Compare SPIDER variants on deep learning problems
2. Study the effect of mini-batch size on convergence
3. Investigate hybrid methods combining SPIDER with other techniques