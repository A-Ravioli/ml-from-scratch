# Quasi-Newton Methods

## Prerequisites
- Newton's method and second-order optimization theory
- Linear algebra (matrix updates, eigenvalue properties)
- Optimization fundamentals (line search, convergence analysis)
- Understanding of the secant condition and gradient information

## Learning Objectives
- Master quasi-Newton methods and their theoretical foundations
- Understand how to approximate Hessian information efficiently
- Implement BFGS, L-BFGS, and other quasi-Newton variants from scratch
- Analyze the trade-offs between Newton and quasi-Newton approaches
- Apply quasi-Newton methods to machine learning problems

## Mathematical Foundations

### 1. The Quasi-Newton Idea

#### Motivation: Avoiding Hessian Computation
Newton's method: x_{k+1} = x_k - H_k^{-1}g_k where H_k = ∇²f(x_k)

**Problems with Newton**:
- Computing Hessian: O(n²) storage, expensive evaluation
- Inverting Hessian: O(n³) operations per iteration
- Hessian may not be positive definite

**Quasi-Newton Approach**:
- Approximate H_k or H_k^{-1} using gradient information
- Maintain positive definiteness
- Update approximation efficiently using rank-1 or rank-2 updates

#### The Secant Condition
Key insight: Use gradient changes to infer curvature information.

For quadratic function f(x) = (1/2)x^T H x - b^T x:
∇f(x_k) - ∇f(x_{k-1}) = H(x_k - x_{k-1})

**Secant Condition**: B_{k+1}s_k = y_k

where:
- s_k = x_{k+1} - x_k (step)
- y_k = ∇f(x_{k+1}) - ∇f(x_k) (gradient change)
- B_{k+1} ≈ ∇²f(x_{k+1}) (Hessian approximation)

### 2. BFGS Method

#### The BFGS Update Formula
**Broyden-Fletcher-Goldfarb-Shanno (BFGS)** update for Hessian approximation:

B_{k+1} = B_k - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k) + (y_k y_k^T)/(y_k^T s_k)

#### Inverse BFGS Update
More commonly, we maintain H_k ≈ [∇²f(x_k)]^{-1}:

H_{k+1} = (I - ρ_k s_k y_k^T) H_k (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T

where ρ_k = 1/(y_k^T s_k).

#### BFGS Algorithm
```
1. Choose x_0, H_0 = I (usually)
2. For k = 0, 1, 2, ...
   a. Compute d_k = -H_k ∇f(x_k)
   b. Line search: x_{k+1} = x_k + α_k d_k  
   c. Compute s_k = x_{k+1} - x_k, y_k = ∇f(x_{k+1}) - ∇f(x_k)
   d. Update H_{k+1} using BFGS formula
```

#### Theoretical Properties

**Theorem 2.1 (BFGS Convergence)**: For strongly convex f with Lipschitz gradients, BFGS with exact line search converges superlinearly:

||x_k - x*|| = o(||x_{k-1} - x*||)

**Theorem 2.2 (Positive Definiteness)**: If H_k ≻ 0 and y_k^T s_k > 0, then H_{k+1} ≻ 0.

**Self-correcting property**: BFGS "forgets" bad initial approximations and converges to true inverse Hessian on quadratic functions in finite steps.

### 3. Limited Memory BFGS (L-BFGS)

#### Motivation for L-BFGS
BFGS requires O(n²) storage for H_k. For large-scale problems, this is prohibitive.

**L-BFGS idea**: 
- Don't store H_k explicitly
- Store last m correction pairs {s_i, y_i}
- Compute H_k ∇f(x_k) using two-loop recursion

#### Two-Loop Recursion Algorithm
```python
def lbfgs_two_loop(grad, s_history, y_history, gamma):
    """
    Compute H_k * grad using L-BFGS two-loop recursion
    
    Args:
        grad: Current gradient
        s_history: List of last m steps s_i
        y_history: List of last m gradient changes y_i  
        gamma: Initial Hessian scaling
    """
    q = grad.copy()
    rho = []
    alpha = []
    
    # First loop (backward)
    for i in reversed(range(len(s_history))):
        rho_i = 1.0 / (y_history[i].T @ s_history[i])
        alpha_i = rho_i * s_history[i].T @ q
        q = q - alpha_i * y_history[i]
        rho.append(rho_i)
        alpha.append(alpha_i)
    
    # Apply initial Hessian approximation
    r = gamma * q
    
    # Second loop (forward)  
    for i in range(len(s_history)):
        beta = rho[-(i+1)] * y_history[i].T @ r
        r = r + s_history[i] * (alpha[-(i+1)] - beta)
    
    return -r  # Return search direction
```

#### L-BFGS Storage and Complexity
- **Storage**: O(mn) where m is memory parameter (typically m = 5-20)
- **Computation**: O(mn) per iteration
- **Convergence**: Superlinear in practice, linear in theory

#### Choosing Initial Scaling γ_k
Common choices:
1. γ_k = 1 (identity)
2. γ_k = (s_{k-1}^T y_{k-1})/(y_{k-1}^T y_{k-1}) (Barzilai-Borwein)
3. γ_k = (y_{k-1}^T y_{k-1})/(s_{k-1}^T y_{k-1}) (inverse BB)

### 4. Other Quasi-Newton Methods

#### DFP Method
**Davidon-Fletcher-Powell** - the first quasi-Newton method:

H_{k+1} = H_k + (s_k s_k^T)/(s_k^T y_k) - (H_k y_k y_k^T H_k)/(y_k^T H_k y_k)

DFP is dual to BFGS but generally less robust in practice.

#### Broyden Class
General family of quasi-Newton updates:

H_{k+1} = H_k - (H_k y_k y_k^T H_k)/(y_k^T H_k y_k) + (s_k s_k^T)/(s_k^T y_k) + φ_k (y_k^T H_k y_k) v_k v_k^T

where v_k = s_k/(s_k^T y_k) - H_k y_k/(y_k^T H_k y_k)

- φ_k = 0: DFP method
- φ_k = 1: BFGS method

#### SR1 Method
**Symmetric Rank-1** update:

B_{k+1} = B_k + ((y_k - B_k s_k)(y_k - B_k s_k)^T)/((y_k - B_k s_k)^T s_k)

**Properties**:
- Maintains symmetry
- May lose positive definiteness
- Good for trust region methods
- Can better approximate indefinite Hessians

### 5. Quasi-Newton for Machine Learning

#### L-BFGS for Neural Networks
L-BFGS has been successfully applied to neural network training:

**Advantages**:
- Fast convergence near optimum
- No hyperparameter tuning (learning rate automatically adapted)
- Memory efficient compared to full Newton

**Challenges**:
- Requires multiple gradient evaluations per iteration (line search)
- Not naturally suited for mini-batches
- Can struggle with very non-convex landscapes

#### Stochastic Quasi-Newton Methods

**Online L-BFGS**:
- Maintain correction pairs from recent iterations
- Use different step sizes for different iterations
- Challenge: y_k^T s_k > 0 not guaranteed with noise

**Sample Average Approximation**:
```
s_k = x_{k+1} - x_k
y_k = ∇f_{S_k}(x_{k+1}) - ∇f_{S_k}(x_k)
```
where ∇f_{S_k} is gradient over mini-batch S_k.

**Variance Reduction**:
Combine quasi-Newton with variance reduction techniques (SVRG, SAGA) to get better gradient estimates.

### 6. Practical Considerations

#### Curvature Condition
For BFGS to work well, need y_k^T s_k > 0 (curvature condition).

**When violated**:
- Skip the update (keep H_k = H_{k-1})
- Use Powell's modification
- Restart with H_k = I

#### Powell's Modification
When y_k^T s_k ≤ 0.2 s_k^T B_k s_k, modify y_k:

ỹ_k = θ y_k + (1-θ) B_k s_k

where θ = 0.8 s_k^T B_k s_k / (s_k^T B_k s_k - y_k^T s_k)

#### Line Search Requirements
Quasi-Newton methods typically require:
1. **Wolfe conditions**: Sufficient decrease + curvature condition
2. **Strong Wolfe conditions**: For superlinear convergence

**Armijo condition**: f(x_k + α_k d_k) ≤ f(x_k) + c_1 α_k ∇f(x_k)^T d_k

**Curvature condition**: ∇f(x_k + α_k d_k)^T d_k ≥ c_2 ∇f(x_k)^T d_k

Typical values: c_1 = 10^{-4}, c_2 = 0.9

### 7. Advanced Quasi-Newton Methods

#### Limited Memory SR1 (L-SR1)
Symmetric rank-1 updates with limited memory:
- Better approximation for indefinite Hessians
- Used in trust region frameworks
- More robust to noise than L-BFGS

#### Quasi-Newton with Regularization
For ill-conditioned problems:

H_{k+1} + λ_k I where λ_k is chosen adaptively

#### Block BFGS
For structured problems, update Hessian blocks separately:
- Partition variables into blocks
- Apply BFGS to each block
- Maintain sparsity structure

#### Factorized Quasi-Newton
For very large problems, maintain factorization:
H_k = L_k L_k^T + D_k

Update factors rather than full matrix.

### 8. Convergence Analysis

#### Superlinear Convergence Theorem
**Theorem 8.1**: For strongly convex f with Lipschitz continuous Hessian, if BFGS is used with exact line search, then:

lim_{k→∞} ||x_{k+1} - x*|| / ||x_k - x*|| = 0

#### Rate of Convergence
- **Newton**: ||x_{k+1} - x*|| ≤ M ||x_k - x*||² (quadratic)
- **BFGS**: ||x_{k+1} - x*|| ≤ M_k ||x_k - x*|| where M_k → 0 (superlinear)
- **L-BFGS**: Linear convergence in theory, superlinear in practice

#### Finite Termination Property
**Theorem 8.2**: On quadratic functions, BFGS with exact line search terminates in at most n+1 iterations, where n is the dimension.

### 9. Implementation Details

#### Numerical Stability
- Check for y_k^T s_k > machine epsilon
- Scale initial Hessian appropriately  
- Use robust line search implementation
- Handle edge cases (zero gradients, etc.)

#### Memory Management for L-BFGS
```python
class LBFGSBuffer:
    def __init__(self, memory_size, dimension):
        self.m = memory_size
        self.s_buffer = np.zeros((memory_size, dimension))
        self.y_buffer = np.zeros((memory_size, dimension))
        self.rho_buffer = np.zeros(memory_size)
        self.current_size = 0
        self.newest_idx = 0
    
    def add_pair(self, s, y):
        # Add new s, y pair and maintain circular buffer
        pass
```

#### Stopping Criteria
Common criteria:
1. ||∇f(x_k)|| ≤ tolerance
2. |f(x_k) - f(x_{k-1})| ≤ tolerance  
3. ||x_k - x_{k-1}|| ≤ tolerance
4. Combination of above

## Implementation Details

See `exercise.py` for implementations of:
1. BFGS method with line search
2. L-BFGS with two-loop recursion
3. SR1 method with trust region
4. Stochastic quasi-Newton variants
5. Line search algorithms (Wolfe conditions)
6. Robust curvature condition checking
7. Performance comparison tools
8. Convergence analysis utilities

## Experiments

1. **BFGS vs Newton**: Comparison on quadratic and non-quadratic functions
2. **Memory Size Study**: Effect of m in L-BFGS on convergence
3. **Stochastic vs Deterministic**: Quasi-Newton with mini-batches
4. **Line Search Comparison**: Different line search strategies
5. **Initialization Effects**: Impact of initial Hessian approximation
6. **Robustness Study**: Behavior on ill-conditioned problems

## Research Connections

### Foundational Papers
1. **Broyden (1970)** - "The Convergence of a Class of Double-rank Minimization Algorithms"
   - Original quasi-Newton convergence theory

2. **Fletcher & Powell (1963)** - "A Rapidly Convergent Descent Method"
   - DFP method introduction

3. **Broyden, Fletcher, Goldfarb & Shanno (1970)** - Multiple papers leading to BFGS

### Modern Developments
4. **Liu & Nocedal (1989)** - "On the Limited Memory BFGS Method"
   - L-BFGS algorithm and analysis

5. **Byrd, Nocedal & Schnabel (1994)** - "Representations of Quasi-Newton Matrices"
   - Compact representations for limited memory methods

6. **Schraudolph, Yu & Günter (2007)** - "A Stochastic Quasi-Newton Method"
   - Early work on stochastic quasi-Newton

### Machine Learning Applications
7. **Andrew & Gao (2007)** - "Scalable Training of L1-Regularized Log-Linear Models"
   - L-BFGS for sparse learning

8. **Moritz, Nishihara & Jordan (2016)** - "A Linearly-Convergent Stochastic L-BFGS Algorithm"
   - Theoretical analysis of stochastic L-BFGS

9. **Gower, Goldfarb & Richtárik (2016)** - "Stochastic Block BFGS"
   - Block-wise quasi-Newton updates

## Resources

### Primary Sources
1. **Nocedal & Wright - Numerical Optimization** (Chapters 6-7)
   - Comprehensive treatment of quasi-Newton methods
2. **Fletcher - Practical Methods of Optimization** (Chapter 3)
   - Practical perspective on implementation
3. **Dennis & Schnabel - Numerical Methods for Unconstrained Optimization** 
   - Classical reference with detailed proofs

### Video Resources
1. **Jorge Nocedal - Quasi-Newton Methods Lectures**
   - From the co-developer of L-BFGS
2. **Stephen Wright - Large-Scale Optimization** 
   - Modern perspective on scalable methods
3. **Katya Scheinberg - Stochastic Quasi-Newton Methods**
   - Recent developments in stochastic settings

### Advanced Reading
1. **Byrd, Hansen, Nocedal & Singer (2016)** - "A Stochastic Quasi-Newton Method for Large-Scale Optimization"
2. **Wang, Ma & Goldfarb (2017)** - "Stochastic Quasi-Newton Methods for Nonconvex Stochastic Optimization"
3. **Lucchi, McWilliams & Hofmann (2015)** - "A Variance Reduced Stochastic Newton Method"

## Socratic Questions

### Understanding
1. Why do quasi-Newton methods achieve superlinear convergence while first-order methods are only linear?
2. What information is lost when using L-BFGS instead of full BFGS?
3. When might quasi-Newton methods perform worse than gradient descent?

### Extension
1. How would you design a quasi-Newton method for constrained optimization?
2. Can quasi-Newton ideas be applied to non-smooth optimization?
3. What's the relationship between quasi-Newton methods and Krylov subspace methods?

### Research
1. How can we make quasi-Newton methods more robust to stochastic noise?
2. What are the fundamental limits of second-order approximations in deep learning?
3. Can we develop quasi-Newton methods that work well with very large mini-batches?

## Exercises

### Theoretical
1. Prove that BFGS maintains positive definiteness under the curvature condition
2. Derive the L-BFGS two-loop recursion from first principles
3. Analyze the convergence rate of L-BFGS with limited memory

### Implementation
1. Implement BFGS with backtracking line search from scratch
2. Build L-BFGS with circular buffer for memory management
3. Create stochastic quasi-Newton method with variance reduction
4. Implement SR1 method with trust region framework

### Research
1. Compare quasi-Newton methods on various ML optimization problems
2. Study the effect of memory size in L-BFGS on different problem types
3. Investigate hybrid approaches combining quasi-Newton with first-order methods