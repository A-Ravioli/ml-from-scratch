# Bilevel Optimization for Machine Learning

## Prerequisites
- Convex and non-convex optimization fundamentals
- Gradient-based optimization methods
- Meta-learning concepts
- Implicit function theorem

## Learning Objectives
- Master bilevel optimization theory and algorithms
- Understand hypergradient computation methods
- Implement bilevel optimization algorithms from scratch
- Apply bilevel optimization to meta-learning and hyperparameter optimization

## Mathematical Foundations

### 1. Problem Setup and Motivation

#### Bilevel Optimization Problem
```
minimize_{x} F(x, y*(x))
subject to: y*(x) ∈ argmin_{y} G(x, y)
```

**Upper level**: Optimize outer objective F w.r.t. upper-level variable x
**Lower level**: For each x, solve inner optimization problem for y*(x)

#### Machine Learning Applications

**Hyperparameter Optimization**:
- x: hyperparameters (learning rate, regularization)
- y: model parameters
- F: validation loss
- G: training loss

**Meta-Learning**:
- x: meta-parameters (initialization, learning algorithm)
- y: task-specific parameters  
- F: meta-objective across tasks
- G: task-specific loss

**Neural Architecture Search**:
- x: architecture parameters
- y: network weights
- F: validation performance
- G: training loss

**Data Cleaning/Poisoning**:
- x: data weights or perturbations
- y: model parameters
- F: attack objective or cleaning metric
- G: training loss

### 2. Theoretical Foundations

#### Assumptions
1. **Lower-level optimality**: y*(x) is well-defined
2. **Smoothness**: F and G are smooth in both arguments
3. **Strong convexity**: G is strongly convex in y (for unique y*)
4. **Implicit function theorem**: ∇²ᵧG(x, y*(x)) is invertible

#### Implicit Function Theorem
If ∇ᵧG(x, y*(x)) = 0 and ∇²ᵧG is invertible, then:
```
dy*/dx = -[∇²ᵧG(x, y*)]⁻¹ ∇²ₓᵧG(x, y*)
```

#### Hypergradient Computation
The gradient of the upper-level objective:
```
dF/dx = ∇ₓF(x, y*) + ∇ᵧF(x, y*) · dy*/dx
```

Substituting the implicit function theorem:
```
dF/dx = ∇ₓF(x, y*) - ∇ᵧF(x, y*) · [∇²ᵧG(x, y*)]⁻¹ ∇²ₓᵧG(x, y*)
```

### 3. Hypergradient Computation Methods

#### Method 1: Implicit Differentiation (Exact)

**Algorithm 3.1 (Implicit Differentiation)**
1. Solve lower-level problem: y* = argmin G(x, y)
2. Compute Hessian: H = ∇²ᵧG(x, y*)
3. Solve linear system: H⁻¹ ∇²ₓᵧG(x, y*)
4. Compute hypergradient using formula above

**Computational cost**: O(d³) for Hessian inversion

#### Method 2: Iterative Differentiation (IFT)

**Algorithm 3.2 (Iterative Differentiation)**
Instead of solving H⁻¹v, solve Hv = b iteratively:
1. Use conjugate gradient or other iterative solver
2. Avoid explicit Hessian inversion
3. Memory efficient for large problems

**Computational cost**: O(d²) per CG iteration

#### Method 3: Approximate Implicit Differentiation (AID)

**Algorithm 3.3 (AID)**
Approximate y*(x) using finite optimization steps:
1. Run K steps of lower-level optimization
2. Differentiate through the optimization trajectory
3. Trade accuracy for computational efficiency

#### Method 4: Reverse-mode Automatic Differentiation

**Algorithm 3.4 (Reverse-mode AD)**
1. Forward pass: Compute y* by running optimization
2. Backward pass: Backpropagate through optimization steps
3. Automatic but can be memory intensive

### 4. Practical Algorithms

#### MAML (Model-Agnostic Meta-Learning)

**Algorithm 4.1 (MAML)**
For meta-learning with gradient-based adaptation:

```
For each meta-iteration:
  1. Sample batch of tasks Tᵢ
  2. For each task i:
     - Compute adapted parameters: φᵢ = θ - α∇θL_Tᵢ(θ)
     - Evaluate on query set: L_query(φᵢ)
  3. Meta-update: θ ← θ - β∇θ Σᵢ L_query(φᵢ)
```

**Key insight**: Differentiate through the gradient update step.

#### DrNAS (Differentiable Neural Architecture Search)

**Algorithm 4.2 (DARTS)**
1. **Continuous relaxation**: Use weighted sum of operations
2. **Joint optimization**: Alternate between architecture and weights
3. **Bilevel formulation**: Architecture on validation, weights on training

#### Hyperparameter Optimization with Hypergradients

**Algorithm 4.3 (HyperGrad)**
1. **Unroll optimization**: Track gradient steps for T iterations
2. **Reverse-mode AD**: Backpropagate through unrolled computation
3. **Memory vs accuracy**: Trade-off between T and memory usage

### 5. Convergence Analysis

#### Assumptions for Convergence
1. **Lipschitz smoothness**: Upper and lower level objectives
2. **Strong convexity**: Lower-level problem
3. **Bounded iterates**: Optimization stays in bounded region

#### Theorem 5.1 (Bilevel SGD Convergence)
Under appropriate conditions, bilevel SGD achieves:
```
E[‖∇F(x_T)‖²] ≤ O(1/√T)
```

where T is the number of upper-level iterations.

#### Complexity Results
- **Exact hypergradients**: O(d³) per iteration
- **Approximate methods**: O(d²) per iteration  
- **Sample complexity**: Similar to single-level optimization
- **Memory complexity**: Can grow with optimization trajectory length

### 6. Implementation Considerations

#### Computational Challenges
1. **Hessian computation**: Expensive for large models
2. **Memory usage**: Storing optimization trajectory
3. **Numerical stability**: Condition number of Hessian
4. **Approximation errors**: Trade-offs in practical methods

#### Hyperparameter Selection
- **Learning rates**: Different scales for upper/lower levels
- **Optimization steps**: How many lower-level steps?
- **Approximation quality**: Accuracy vs efficiency trade-offs

#### Software Implementation
- **Automatic differentiation**: Use modern AD frameworks
- **Memory management**: Checkpointing and gradient accumulation
- **Numerical precision**: Handle ill-conditioned problems

### 7. Advanced Topics

#### Non-convex Lower Levels
- **Multiple local minima**: Which solution to choose?
- **Approximation theory**: When is approximate solution good enough?
- **Practical algorithms**: Heuristics for non-convex case

#### Stochastic Bilevel Optimization
- **Noisy gradients**: Both levels have stochastic gradients
- **Variance reduction**: Apply SVRG/SAGA to bilevel problems
- **Sample complexity**: Analysis with stochastic oracles

#### Constrained Bilevel Problems
- **Feasibility constraints**: Both upper and lower levels
- **KKT conditions**: Optimality conditions for constrained case
- **Algorithms**: Penalty methods and barrier approaches

#### Multi-level Optimization
- **Hierarchical problems**: More than two levels
- **Compositional optimization**: Chain of optimization problems
- **Applications**: Deep meta-learning, hierarchical models

### 8. Applications in Detail

#### Hyperparameter Optimization
```python
# Upper level: hyperparameters λ
# Lower level: model parameters θ
def bilevel_hyperopt(λ):
    θ_star = minimize(train_loss(θ, λ))  # Lower level
    return val_loss(θ_star, λ)           # Upper level
```

#### Few-Shot Learning (MAML)
```python
# Upper level: initialization θ
# Lower level: task-specific adaptation
def maml_objective(θ):
    total_loss = 0
    for task in tasks:
        φ = θ - α * grad(task_loss(θ, task))  # Lower level
        total_loss += query_loss(φ, task)    # Upper level
    return total_loss
```

#### Neural Architecture Search
```python
# Upper level: architecture weights α  
# Lower level: network weights w
def nas_objective(α):
    w_star = minimize(train_loss(w, α))  # Lower level
    return val_loss(w_star, α)           # Upper level
```

## Implementation Details

See `exercise.py` for implementations of:
1. Implicit differentiation for hypergradient computation
2. MAML for few-shot learning
3. Hyperparameter optimization with bilevel methods
4. Simple neural architecture search
5. Comparison of different hypergradient methods
6. Memory-efficient approximation techniques

## Experiments

1. **Hypergradient Accuracy**: Compare exact vs approximate methods
2. **Computational Efficiency**: Time/memory trade-offs
3. **Meta-Learning**: MAML on few-shot classification
4. **Hyperparameter Optimization**: Automated learning rate tuning

## Research Connections

### Seminal Papers
1. Finn et al. (2017) - "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
   - MAML algorithm and meta-learning

2. Liu et al. (2018) - "DARTS: Differentiable Architecture Search"
   - Neural architecture search via bilevel optimization

3. Franceschi et al. (2018) - "Bilevel Programming for Hyperparameter Optimization and Meta-Learning"
   - Theoretical foundations and algorithms

### Modern Developments
1. Rajeswaran et al. (2019) - "Meta-Learning with Implicit Gradients"
   - Improved implicit differentiation methods

2. Lorraine et al. (2020) - "Optimizing Millions of Hyperparameters by Implicit Differentiation"
   - Large-scale hyperparameter optimization

3. Ji et al. (2021) - "Bilevel Optimization: Convergence Analysis and Enhanced Design"
   - Recent theoretical advances

## Resources

### Primary Sources
1. **Colson et al. (2007)** - "An overview of bilevel optimization"
2. **Franceschi et al. (2018)** - "Bilevel Programming for Hyperparameter Optimization"
3. **Liu et al. (2021)** - "Investigating Bi-Level Optimization for Learning and Vision"

### Video Resources
1. **Chelsea Finn** - "Model-Agnostic Meta-Learning" (ICML 2017)
2. **Hanxiao Liu** - "DARTS: Differentiable Architecture Search" (ICLR 2019)
3. **Luca Franceschi** - "Bilevel Optimization in Machine Learning"

### Advanced Reading
1. **Dempe (2002)** - "Foundations of Bilevel Programming"
2. **Sinha et al. (2017)** - "A Review on Bilevel Optimization: From Classical to Evolutionary Approaches"

## Socratic Questions

### Understanding
1. When is the implicit function theorem applicable to bilevel problems?
2. How does the condition number of the lower-level Hessian affect convergence?
3. What's the trade-off between exact and approximate hypergradient methods?

### Extension
1. How would you extend bilevel optimization to handle multiple objectives?
2. Can bilevel optimization be applied to adversarial training problems?
3. What happens when the lower-level problem has multiple solutions?

### Research
1. How can we design bilevel algorithms that scale to modern deep learning?
2. What are the fundamental limits of bilevel optimization complexity?
3. How does noise in the lower-level problem affect upper-level convergence?

## Exercises

### Theoretical
1. Derive the hypergradient formula using the implicit function theorem
2. Analyze the computational complexity of different hypergradient methods
3. Prove convergence for bilevel SGD under strong convexity

### Implementation
1. Implement MAML for few-shot learning from scratch
2. Create a differentiable hyperparameter optimization framework
3. Build a simple neural architecture search algorithm

### Research
1. Compare bilevel methods for automated machine learning
2. Study the effect of approximation quality on final performance
3. Investigate bilevel optimization for adversarial robustness