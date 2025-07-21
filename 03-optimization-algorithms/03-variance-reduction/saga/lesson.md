# SAGA: Stochastic Average Gradient Algorithm

## Prerequisites
- Finite-sum optimization fundamentals
- SVRG algorithm and variance reduction concepts
- Convex optimization theory
- Linear convergence analysis

## Learning Objectives
- Master SAGA algorithm and its theoretical guarantees
- Understand memory-based variance reduction
- Compare SAGA with SVRG and other variance reduction methods
- Implement SAGA with optimal hyperparameter selection

## Mathematical Foundations

### 1. Problem Setup

Consider the finite-sum optimization problem:
```
minimize f(x) = (1/n) ∑ᵢ₌₁ⁿ fᵢ(x)
```

**Key assumption**: Each fᵢ is convex and L-smooth.

### 2. The SAGA Algorithm

#### Core Idea
Maintain a table of gradients for each sample and use these stored gradients as control variates to reduce variance.

#### Algorithm 2.1 (SAGA)
**Initialize**: 
- x₀ ∈ ℝᵈ
- Table φᵢ⁰ = ∇fᵢ(x₀) for i = 1, ..., n
- ȳ⁰ = (1/n) ∑ᵢ₌₁ⁿ φᵢ⁰

**For k = 0, 1, 2, ...**:
1. Sample jₖ uniformly from {1, ..., n}
2. Compute current gradient: gₖ = ∇fⱼₖ(xₖ)
3. Update table: φⱼₖᵏ⁺¹ = gₖ, φᵢᵏ⁺¹ = φᵢᵏ for i ≠ jₖ
4. Update average: ȳᵏ⁺¹ = ȳᵏ + (1/n)(gₖ - φⱼₖᵏ)
5. **SAGA update**: xₖ₊₁ = xₖ - η(gₖ - φⱼₖᵏ + ȳᵏ)

#### Variance Reduction Mechanism
The SAGA gradient estimator is:
```
vₖ = gₖ - φⱼₖᵏ + ȳᵏ
```

**Properties**:
- **Unbiased**: E[vₖ|xₖ] = ∇f(xₖ)
- **Variance decreases**: As xₖ → x*, stored gradients become more accurate
- **Memory efficient updates**: Only one gradient per iteration

### 3. Convergence Analysis

#### Assumptions
1. **Convexity**: Each fᵢ is convex
2. **Smoothness**: Each fᵢ is L-smooth  
3. **Strong convexity**: f is μ-strongly convex (for linear convergence)

#### Theorem 3.1 (SAGA Convergence - Strongly Convex)
For step size η = 1/(2(L + μn)), SAGA satisfies:

E[f(xₖ) - f*] ≤ (1 - μη)ᵏ [f(x₀) - f* + (η/2)‖∇f(x₀)‖²]

**Convergence rate**: O(exp(-μη k)) = **linear convergence**

#### Corollary 3.1 (Iteration Complexity)
To achieve ε-accuracy:
- **Expected iterations**: O((n + L/μ) log(1/ε))
- **Gradient evaluations**: Same (one per iteration)

#### Theorem 3.2 (SAGA Convergence - Convex Case)
For convex functions with η = 1/(3L):

E[f(x̄ₖ) - f*] ≤ O(1/k)

where x̄ₖ is the averaged iterate.

### 4. Comparison with Other Methods

#### vs SVRG
| Aspect | SAGA | SVRG |
|--------|------|------|
| **Memory** | O(nd) | O(d) |
| **Gradient evaluations/iteration** | 1 | 1 (+ periodic full gradient) |
| **Convergence rate** | O((n + L/μ) log(1/ε)) | O((n + L/μ) log(1/ε)) |
| **Constants** | Better | Worse |
| **Implementation** | More complex | Simpler |
| **Storage updates** | Every iteration | Periodic snapshots |

#### vs SGD
- **SAGA**: Linear convergence, higher memory cost
- **SGD**: O(1/√k) convergence, O(d) memory

#### vs SAG (Stochastic Average Gradient)
- **SAGA**: Unbiased estimator, works for general step sizes
- **SAG**: Biased estimator, requires specific step size choice

### 5. Practical Considerations

#### Memory Requirements
- **Gradient table**: O(nd) storage
- **Average maintenance**: O(d) per iteration
- **Total**: O(nd) memory overhead

#### Step Size Selection

**Theoretical optimum**:
η* = 1/(2(L + μn))

**Practical choices**:
1. **Conservative**: η = 1/(3L) (works for convex case)
2. **Adaptive**: Line search or backtracking
3. **Decreasing**: ηₖ = η₀/(1 + γk)

#### Initialization Strategy

**Cold start**: φᵢ⁰ = ∇fᵢ(x₀)
- Requires n gradient evaluations initially
- Ensures unbiasedness from start

**Warm start**: φᵢ⁰ = 0 or random
- Faster initialization
- Temporary bias that disappears

### 6. Variants and Extensions

#### Proximal SAGA
For composite problems: f(x) + g(x) where g is non-smooth.

**Update**: xₖ₊₁ = prox_{ηg}(xₖ - η(gₖ - φⱼₖᵏ + ȳᵏ))

#### SAGA with Mini-batches
Use mini-batch of size b:
- Sample batch Bₖ ⊆ {1, ..., n}
- Update multiple table entries per iteration
- Improved constants and parallelization

#### Accelerated SAGA
Combine with Nesterov momentum:
```
yₖ = xₖ + β(xₖ - xₖ₋₁)
xₖ₊₁ = yₖ - η · SAGA_gradient(yₖ)
```

#### SAGA++
Improved constants and better practical performance through:
- Better step size selection
- Adaptive variance estimation
- Hybrid with other methods

### 7. Implementation Tricks

#### Efficient Table Updates
```python
# Instead of storing full gradients
grad_table = np.zeros((n, d))

# Store only differences from average
diff_table = np.zeros((n, d))  
average_grad = np.zeros(d)
```

#### Memory Management
- **Sparse storage**: For high-dimensional problems
- **Compression**: Low-rank approximation of gradient table
- **Streaming**: For very large n

#### Parallel Implementation
- **Asynchronous updates**: Multiple workers update table
- **Mini-batch variants**: Parallel gradient computation
- **Lock-free algorithms**: Avoid synchronization overhead

### 8. Theoretical Insights

#### Why Linear Convergence?
1. **Variance reduction**: Better gradient estimates near optimum
2. **Memory effect**: Past information guides current updates
3. **Control variates**: Systematic bias-variance tradeoff

#### Role of Problem Structure
- **Condition number**: κ = L/μ affects convergence rate
- **Sample similarity**: Similar fᵢ → better variance reduction
- **Dimension**: Higher d increases memory cost

#### Optimality Results
SAGA achieves optimal dependence on:
- Problem dimension d
- Condition number κ = L/μ  
- Dataset size n

## Implementation Details

See `exercise.py` for implementations of:
1. Basic SAGA algorithm with efficient table management
2. Proximal SAGA for composite optimization
3. Mini-batch SAGA variants
4. Adaptive step size selection
5. Memory-efficient implementations
6. Comparison with SVRG and SGD

## Experiments

1. **Convergence Verification**: Test linear convergence theory
2. **Memory vs Performance**: Trade-off analysis
3. **Hyperparameter Sensitivity**: Step size and initialization effects
4. **Large-scale Problems**: Scalability analysis

## Research Connections

### Seminal Papers
1. Defazio et al. (2014) - "SAGA: A Fast Incremental Gradient Method with Support for Non-Strongly Convex Composite Objectives"
   - Original SAGA paper

2. Reddi et al. (2016) - "Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization"
   - Extensions to non-convex settings

3. Hofmann et al. (2015) - "Variance Reduced Stochastic Gradient Descent with Neighbors"
   - Connections between variance reduction methods

### Modern Developments
1. Allen-Zhu (2017) - "Katyusha: Direct Acceleration of Variance-Reduced Stochastic Methods"
2. Qian et al. (2019) - "SAGA with Arbitrary Sampling"
3. Gorbunov et al. (2020) - "Unified Analysis of Stochastic Gradient Methods"

## Resources

### Primary Sources
1. **Defazio et al. (2014)** - Original SAGA paper
2. **Schmidt et al. (2017)** - "Minimizing finite sums with the stochastic average gradient"
3. **Bottou et al. (2018)** - "Optimization Methods for Large-Scale Machine Learning"

### Video Resources
1. **Aaron Defazio** - "SAGA: A Fast Incremental Gradient Method" (NIPS 2014)
2. **Francis Bach** - "Stochastic Variance Reduction Methods" 
3. **Simon Lacoste-Julien** - "Convergence Rates for Variance-Reduced Stochastic Gradient"

### Advanced Reading
1. **Bubeck (2015)** - "Convex Optimization: Algorithms and Complexity"
2. **Bach & Moulines (2013)** - "Non-asymptotic analysis of stochastic approximation algorithms"

## Socratic Questions

### Understanding
1. Why does SAGA require O(nd) memory while SVRG only needs O(d)?
2. How does the gradient table update mechanism reduce variance?
3. What happens to SAGA when the functions fᵢ have very different scales?

### Extension
1. Can we reduce SAGA's memory requirements without losing convergence guarantees?
2. How would you extend SAGA to streaming/online settings?
3. What's the optimal way to initialize the gradient table?

### Research
1. Can we achieve better than O((n + L/μ) log(1/ε)) complexity for finite-sum problems?
2. How does mini-batching affect SAGA's theoretical guarantees?
3. What's the connection between SAGA and other optimization methods?

## Exercises

### Theoretical
1. Prove that the SAGA gradient estimator is unbiased
2. Derive the variance reduction property of SAGA
3. Compare SAGA and SVRG convergence constants

### Implementation
1. Implement memory-efficient SAGA variants
2. Create adaptive step size selection algorithms
3. Build proximal SAGA for sparse regularization

### Research
1. Study SAGA behavior on ill-conditioned problems
2. Investigate optimal initialization strategies
3. Design hybrid methods combining SAGA with other techniques