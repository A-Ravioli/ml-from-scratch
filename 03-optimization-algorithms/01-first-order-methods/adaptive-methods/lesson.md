# Adaptive Gradient Methods

## Prerequisites
- Stochastic gradient descent and convergence analysis
- Second-order optimization concepts (Hessian, curvature)
- Probability theory (martingales, concentration inequalities)
- Matrix analysis (eigenvalues, condition numbers)

## Learning Objectives
- Master the theory and practice of adaptive learning rate methods
- Understand the trade-offs between adaptation and convergence guarantees
- Implement adaptive optimizers from scratch with proper numerical stability
- Analyze when adaptive methods excel vs when they fail
- Connect adaptive methods to natural gradient and second-order information

## Mathematical Foundations

### 1. The Motivation for Adaptation

#### Problems with Fixed Learning Rates
Standard SGD: x_{k+1} = x_k - η∇f(x_k)

**Issues**:
- Same learning rate for all parameters
- No adaptation to local curvature
- Poor performance on ill-conditioned problems
- Manual learning rate scheduling required

#### Ideal Adaptive Behavior
Want learning rate inversely proportional to curvature:
- Large steps in flat directions
- Small steps in steep directions
- Automatic adjustment during training

### 2. AdaGrad: The Foundation

#### Algorithm (Duchi, Hazan & Singer 2011)
```
G_k = ∑_{t=1}^k g_t g_t^T    # Accumulate squared gradients
x_{k+1} = x_k - η/√(diag(G_k) + ε) ⊙ g_k
```

where ⊙ denotes element-wise multiplication.

#### Diagonal Approximation
Instead of full matrix G_k, use diagonal:
```
g_{i,k}^2 = ∑_{t=1}^k (∇f(x_t))_i^2
x_{i,k+1} = x_{i,k} - η/√(g_{i,k}^2 + ε) · (∇f(x_k))_i
```

#### Theoretical Guarantees

**Theorem 2.1 (AdaGrad Regret Bound)**
For convex functions with bounded gradients ||g_t|| ≤ G:

R_T = ∑_{t=1}^T [f(x_t) - f(x*)] ≤ (G√d/η)√(∑_{t=1}^T ||g_t||^2) + η∑_{t=1}^T ||g_t||^2/(2√(∑_{s=1}^t ||g_s||^2))

This gives O(√T) regret, which is optimal for online convex optimization.

**Key Properties**:
- Regret bounds independent of problem conditioning
- Automatic adaptation to gradient sparsity
- No need for prior knowledge of gradients

#### Problems with AdaGrad
1. **Diminishing Learning Rates**: ∑_{t=1}^∞ g_t^2 → ∞ makes learning rates → 0
2. **Aggressive Decay**: May stop learning too early
3. **Memory Requirements**: Stores all historical gradients

### 3. RMSprop: Exponential Moving Average

#### Algorithm (Hinton 2012, unpublished)
Replace cumulative sum with exponential moving average:
```
v_k = γv_{k-1} + (1-γ)g_k^2    # Element-wise
x_{k+1} = x_k - η/√(v_k + ε) ⊙ g_k
```

**Advantages**:
- Prevents aggressive decay of learning rates
- Adapts to recent gradient behavior
- Fixed memory requirements

#### Theoretical Analysis
Less well-understood than AdaGrad, but empirically very effective.

**Intuition**: γ controls the "memory" of the adaptive scaling:
- γ → 1: Long memory (like AdaGrad)
- γ → 0: Short memory (like SGD)

### 4. Adam: Combining Momentum and Adaptation

#### Algorithm (Kingma & Ba 2014)
Combines first and second moment estimation:

```python
# First moment (momentum)
m_k = β_1 m_{k-1} + (1-β_1)g_k

# Second moment (adaptive learning rate)  
v_k = β_2 v_{k-1} + (1-β_2)g_k^2

# Bias correction
m̂_k = m_k / (1 - β_1^k)
v̂_k = v_k / (1 - β_2^k)

# Update
x_{k+1} = x_k - η m̂_k / (√v̂_k + ε)
```

#### Bias Correction Necessity
Initial estimates are biased toward zero:
- E[m_1] = (1-β_1)E[g_1] ≠ E[g_1] if β_1 ≠ 0
- E[v_1] = (1-β_2)E[g_1^2] ≠ E[g_1^2] if β_2 ≠ 0

Bias correction factors: 1/(1-β_1^k) and 1/(1-β_2^k) fix this.

#### Default Hyperparameters
- β_1 = 0.9 (momentum decay)
- β_2 = 0.999 (second moment decay)  
- η = 0.001 (learning rate)
- ε = 10^{-8} (numerical stability)

These work well across many problems.

### 5. Theoretical Issues with Adam

#### Non-Convergence Example (Reddi et al. 2018)
Simple convex function where Adam fails to converge:

f_t(x) = {
  1010x,  if t mod 3 = 1
  -x,     otherwise
}

Adam oscillates and doesn't converge to optimum.

#### Root Cause
The issue is the exponential moving average in second moment:
- Heavy-tailed gradient distributions
- Bias toward recent large gradients
- Loss of long-term gradient information

### 6. AMSGrad: Fixing Adam

#### Algorithm (Reddi, Kale & Kumar 2018)
Maintain maximum of second moment estimates:
```python
v_k = β_2 v_{k-1} + (1-β_2)g_k^2
v̂_k = max(v̂_{k-1}, v_k)  # Key difference
x_{k+1} = x_k - η m̂_k / (√v̂_k + ε)
```

**Why This Helps**:
- Prevents learning rates from increasing
- Maintains memory of large gradients
- Guarantees convergence for convex functions

#### Convergence Theorem
**Theorem 6.1**: For convex functions, AMSGrad achieves O(1/√T) convergence rate.

### 7. AdaBelief: Adapting to Gradient Prediction Error

#### Motivation (Adapting Stepsizes by the Belief in Gradient Direction)
Instead of adapting to gradient magnitude, adapt to prediction error:

```python
m_k = β_1 m_{k-1} + (1-β_1)g_k              # Momentum
s_k = β_2 s_{k-1} + (1-β_2)(g_k - m_k)^2    # Prediction error
x_{k+1} = x_k - η m̂_k / (√ŝ_k + ε)
```

**Intuition**: If gradients are predictable (small prediction error), take larger steps.

### 8. Natural Gradient Connection

#### Natural Gradient Descent
x_{k+1} = x_k - η F_k^{-1} ∇f(x_k)

where F_k is the Fisher Information Matrix.

#### Diagonal Approximation
Adaptive methods approximate F^{-1} with diagonal matrices:
- AdaGrad: F^{-1} ≈ diag(1/√(∑g_i^2))
- RMSprop/Adam: F^{-1} ≈ diag(1/√(EMA(g_i^2)))

This provides second-order-like adaptation with first-order computational cost.

### 9. Practical Adaptive Methods

#### AdamW: Decoupled Weight Decay
Separates L2 regularization from gradient-based adaptation:
```python
x_{k+1} = x_k - η(m̂_k / (√v̂_k + ε) + λx_k)
```

Better than adding weight decay to gradients.

#### RAdam: Rectified Adam
Adaptive learning rate warm-up based on variance of adaptive term:
```python
ρ_k = ρ_∞ - 2k β_2^k / (1 - β_2^k)  # Variance estimate
r_k = √((ρ_k - 4)(ρ_k - 2)ρ_∞ / ((ρ_∞ - 4)(ρ_∞ - 2)ρ_k))
```

Use momentum-only updates when variance too high.

#### Lookahead: Slow and Fast Weights
```python
# Fast weights: standard optimizer (Adam, SGD, etc.)
φ_{t+1} = opt_update(φ_t, ∇L(φ_t))

# Slow weights: interpolation every k steps  
if t % k == 0:
    θ_{t+1} = θ_t + α(φ_{t+1} - θ_t)
    φ_{t+1} = θ_{t+1}
```

Reduces variance of adaptive methods.

### 10. When Adaptive Methods Fail

#### Generalization Issues
- Often worse generalization than SGD with momentum
- May converge to different (worse) local minima
- Hypothesis: Adaptive methods find "sharp" minima

#### Learning Rate Schedules
- Fixed adaptive learning rates can be suboptimal
- May benefit from decay schedules
- Warmup often necessary

#### Problem-Dependent Performance
**Good for**:
- Sparse gradients (NLP, recommender systems)
- Initial training phases
- Hyperparameter insensitive applications

**Problematic for**:
- Computer vision (CNNs often prefer SGD+momentum)
- Final training phases
- When generalization is critical

## Implementation Details

See `exercise.py` for implementations of:
1. AdaGrad with numerical stability considerations
2. RMSprop with proper exponential moving averages
3. Adam with bias correction and numerical checks
4. AMSGrad and AdaBelief variants
5. AdamW with decoupled weight decay
6. Lookahead wrapper for any base optimizer
7. Learning rate scheduling for adaptive methods
8. Convergence analysis and visualization tools

## Experiments

1. **Convergence Comparison**: Adaptive vs SGD on various function types
2. **Hyperparameter Sensitivity**: Effect of β₁, β₂, ε on performance
3. **Generalization Study**: Training vs validation performance across optimizers
4. **Sparse Gradients**: Performance on problems with sparse gradient structure
5. **Learning Rate Schedules**: Impact of decay on adaptive methods
6. **Numerical Stability**: Behavior with extreme gradient values

## Research Connections

### Foundational Papers
1. **Duchi, Hazan & Singer (2011)** - "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
   - Introduced AdaGrad with theoretical guarantees

2. **Kingma & Ba (2014)** - "Adam: A Method for Stochastic Optimization"  
   - Most popular adaptive method

3. **Reddi, Kale & Kumar (2018)** - "On the Convergence of Adam and Beyond"
   - Identified Adam's convergence issues, proposed AMSGrad

### Modern Developments
4. **Loshchilov & Hutter (2017)** - "Decoupled Weight Decay Regularization"
   - AdamW and proper regularization

5. **Liu et al. (2019)** - "On the Variance of the Adaptive Learning Rate and Beyond"
   - RAdam with variance-based warm-up

6. **Zhang et al. (2019)** - "Lookahead Optimizer: k steps forward, 1 step back"
   - Meta-optimizer framework

### Theoretical Analysis
7. **Wilson et al. (2017)** - "The Marginal Value of Adaptive Gradient Methods in Machine Learning"
   - Critical analysis of adaptive methods

8. **Zhou et al. (2018)** - "On the Convergence of Adaptive Gradient Methods for Nonconvex Optimization"
   - Non-convex convergence theory

## Resources

### Primary Sources
1. **Ruder (2016)** - "An Overview of Gradient Descent Optimization Algorithms"
   - Comprehensive survey of optimization methods
2. **Bottou et al. (2018)** - "Optimization Methods for Large-Scale Machine Learning"
   - Practical perspective on large-scale optimization

### Video Resources
1. **Geoffrey Hinton - Coursera Neural Networks Course**
   - Original RMSprop presentation
2. **Sebastian Ruder - Optimization for Deep Learning**
   - Modern survey of optimization methods
3. **Diederik Kingma - Adam and Beyond**
   - Original Adam author's presentations

### Advanced Reading
1. **McMahan & Streeter (2010)** - "Adaptive Bound Optimization for Online Convex Optimization"
2. **Orabona (2019)** - "A Modern Introduction to Online Learning"
3. **Hazan (2016)** - "Introduction to Online Convex Optimization"

## Socratic Questions

### Understanding
1. Why do adaptive methods often work well initially but may hurt final performance?
2. How does the choice of ε affect numerical stability vs adaptation quality?
3. When might you prefer AdaGrad over Adam?

### Extension
1. How would you design an adaptive method that maintains good generalization?
2. Can adaptive methods be extended to constrained optimization naturally?
3. What's the relationship between adaptive methods and preconditioning?

### Research
1. Why do adaptive methods seem to find different local minima than SGD?
2. How can we combine the best aspects of adaptive methods and momentum?
3. Is there a principled way to choose β₁ and β₂ for a given problem?

## Exercises

### Theoretical
1. Derive the regret bound for AdaGrad on strongly convex functions
2. Analyze the bias in Adam's moment estimates without correction
3. Prove convergence of RMSprop under appropriate assumptions

### Implementation  
1. Implement all major adaptive methods with careful numerical considerations
2. Build adaptive learning rate schedulers
3. Create visualization tools for learning rate evolution during training

### Research
1. Compare adaptive methods on different types of neural networks
2. Study the effect of batch size on adaptive method performance  
3. Investigate hybrid approaches combining adaptive and non-adaptive methods