# Newton Methods and Second-Order Optimization

## Prerequisites
- Multivariate calculus (Hessian matrices, Taylor expansions)
- Linear algebra (matrix inversions, eigenvalue decomposition, condition numbers)
- Convex optimization fundamentals
- First-order optimization methods (gradient descent)

## Learning Objectives
- Master Newton's method and its theoretical foundations
- Understand the role of curvature information in optimization
- Implement various Newton method variants from scratch
- Analyze convergence rates and computational trade-offs
- Connect Newton methods to natural gradients and quasi-Newton approaches

## Mathematical Foundations

### 1. Newton's Method Fundamentals

#### Motivation: Second-Order Taylor Approximation
For a twice-differentiable function f: ℝⁿ → ℝ, the second-order Taylor expansion around x_k is:

f(x_k + p) ≈ f(x_k) + ∇f(x_k)ᵀp + (1/2)pᵀ∇²f(x_k)p

To minimize this quadratic approximation, we set the gradient to zero:
∇_p[f(x_k) + ∇f(x_k)ᵀp + (1/2)pᵀ∇²f(x_k)p] = ∇f(x_k) + ∇²f(x_k)p = 0

#### Newton Step
Solving for the optimal step p:
p_k = -[∇²f(x_k)]⁻¹∇f(x_k)

#### Newton's Method Algorithm
```
x_{k+1} = x_k + p_k = x_k - [∇²f(x_k)]⁻¹∇f(x_k)
```

This is the pure Newton step with unit step size.

### 2. Theoretical Properties

#### Quadratic Convergence
**Theorem 2.1**: For twice continuously differentiable f with Lipschitz continuous Hessian, if x_k is sufficiently close to x* where ∇f(x*) = 0 and ∇²f(x*) ≻ 0, then Newton's method converges quadratically:

||x_{k+1} - x*|| ≤ M||x_k - x*||²

for some constant M > 0.

**Proof Sketch**:
1. Use Taylor expansion of ∇f around x*
2. Apply Newton update rule  
3. Use Lipschitz continuity of Hessian
4. Show error contracts quadratically □

#### Affine Invariance
**Theorem 2.2**: Newton's method is affine invariant.

If we transform the problem by x = Ay + b, Newton's method in the y-space gives the same iterates (after transformation) as in the x-space.

**Significance**: Newton's method is independent of problem scaling, unlike gradient descent.

#### Convergence Basin
Newton's method only has local convergence guarantees. The basin of attraction can be small for non-convex functions.

### 3. Practical Newton Method

#### Damped Newton Method
To ensure global convergence, combine Newton direction with line search:
```
p_k = -[∇²f(x_k)]⁻¹∇f(x_k)    # Newton direction
α_k = argmin_α f(x_k + αp_k)    # Line search
x_{k+1} = x_k + α_k p_k
```

#### Backtracking Line Search
**Armijo condition**: Choose α such that
f(x_k + αp_k) ≤ f(x_k) + c₁α∇f(x_k)ᵀp_k

where c₁ ∈ (0, 1), typically c₁ = 10⁻⁴.

#### Trust Region Newton Method
Instead of line search, constrain step size:
```
min    ∇f(x_k)ᵀp + (1/2)pᵀ∇²f(x_k)p
s.t.   ||p|| ≤ Δ_k
```

Adjust trust region radius Δ_k based on agreement between model and actual reduction.

### 4. Computational Aspects

#### Solving the Newton System
The core computational challenge: solve ∇²f(x_k)p = -∇f(x_k)

**Direct Methods**:
- Cholesky decomposition (when Hessian is positive definite)
- LU decomposition (general case)
- Cost: O(n³) operations

**Iterative Methods**:
- Conjugate Gradient (CG) for positive definite systems
- GMRES for indefinite systems
- Cost: O(n²) per iteration, fewer iterations

#### Hessian Computation
**Exact Hessian**:
- Analytical computation: O(n²) space, expensive for large n
- Automatic differentiation: Practical for moderate n

**Finite Differences**:
- Forward differences: ∇²f(x)ᵢⱼ ≈ [∇f(x + hεᵢ) - ∇f(x)]ⱼ/h
- Central differences: More accurate but 2x more expensive

### 5. Modified Newton Methods

#### Regularized Newton Method
When ∇²f(x_k) is not positive definite or ill-conditioned:
```
(∇²f(x_k) + λI)p_k = -∇f(x_k)
```

Choose λ > 0 to ensure positive definiteness and good conditioning.

#### Levenberg-Marquardt Algorithm
Adaptive regularization parameter:
```
(∇²f(x_k) + λ_k I)p_k = -∇f(x_k)
```

- If step reduces objective: decrease λ_k (more Newton-like)
- If step increases objective: increase λ_k (more gradient descent-like)

#### Gauss-Newton Method
For least squares problems f(x) = (1/2)||r(x)||²:
```
∇f(x) = J(x)ᵀr(x)
∇²f(x) = J(x)ᵀJ(x) + ∑ᵢ rᵢ(x)∇²rᵢ(x)
```

Gauss-Newton approximation: ∇²f(x) ≈ J(x)ᵀJ(x)
- Always positive semidefinite
- Good when residuals are small
- Used in neural network training (Gauss-Newton optimization)

### 6. Newton Methods for Machine Learning

#### Hessian-Free Optimization
Avoid storing full Hessian by using Hessian-vector products:
```
H(x)v ≈ [∇f(x + εv) - ∇f(x)]/ε
```

Use CG to solve Newton system without forming H explicitly.

#### Natural Gradient Descent
For probability distributions parameterized by θ:
```
θ_{k+1} = θ_k - α F(θ_k)⁻¹∇L(θ_k)
```

where F(θ) is the Fisher Information Matrix.

Connection: F(θ) = E[∇log p(x|θ)∇log p(x|θ)ᵀ] ≈ ∇²L(θ) in some settings.

#### Newton Methods for Neural Networks
Challenges:
- Hessian size: O(P²) where P is number of parameters
- Non-convexity: Many saddle points
- Computational cost: O(P³) per iteration

Solutions:
- Block-diagonal approximations
- Kronecker-factored approximations (K-FAC)
- Diagonal approximations (similar to adaptive methods)

### 7. Advanced Newton Variants

#### Cubic Newton Method
Include third-order term in Taylor expansion:
```
min ∇f(x_k)ᵀp + (1/2)pᵀ∇²f(x_k)p + (σ/6)||p||³
```

- Better global convergence
- More robust to negative curvature
- Adaptive σ based on problem properties

#### Inexact Newton Methods
Solve Newton system approximately:
```
||∇²f(x_k)p_k + ∇f(x_k)|| ≤ η_k||∇f(x_k)||
```

Choose η_k (forcing sequence) to balance accuracy vs computational cost.

#### Stochastic Newton Methods
For finite sum problems f(x) = (1/n)∑ᵢ fᵢ(x):

**Sample Average Approximation**:
```
H_k ≈ (1/|S_k|) ∑_{i∈S_k} ∇²fᵢ(x_k)
```

**Subsampled Newton**:
- Use subset for Hessian estimation
- Combine with variance reduction techniques
- Balance sample size with convergence rate

### 8. Convergence Analysis

#### Local Convergence Rate
Under standard assumptions:
- **Gradient Descent**: ||x_k - x*|| = O(ρᵏ) where ρ = (κ-1)/(κ+1)
- **Newton's Method**: ||x_k - x*|| = O(μᵏ) where μ → 0 quadratically

Here κ is the condition number of ∇²f(x*).

#### Global Convergence
With proper line search or trust region:
- Newton method achieves global convergence
- Asymptotically reduces to pure Newton steps
- Combines robustness of first-order with speed of second-order

#### Complexity Results
**Theorem 8.1**: Damped Newton with backtracking achieves ε-accuracy in:
- O(log(1/ε)) iterations (compared to O(1/ε) for gradient descent)
- O(n³log(1/ε)) total computational cost

### 9. Practical Considerations

#### When to Use Newton Methods
✅ **Good for**:
- Small to medium dimensional problems (n ≤ 10³)
- Problems where Hessian is cheap to compute
- High accuracy requirements
- Well-conditioned problems near optimum

❌ **Problematic for**:
- Large-scale problems (n > 10⁶)
- Highly non-convex landscapes  
- Ill-conditioned or indefinite Hessians
- When gradient evaluation is expensive

#### Numerical Stability
- Check positive definiteness of Hessian
- Use regularization when Hessian is indefinite
- Monitor condition number of linear systems
- Use iterative solvers for large systems

#### Hybrid Approaches
- Start with first-order methods (SGD, Adam)
- Switch to Newton methods near convergence
- Use quasi-Newton methods as middle ground

## Implementation Details

See `exercise.py` for implementations of:
1. Basic Newton method with line search
2. Trust region Newton method
3. Regularized Newton variants
4. Gauss-Newton for least squares
5. Hessian-free optimization using CG
6. Newton method for logistic regression
7. Numerical Hessian computation methods
8. Convergence analysis and visualization tools

## Experiments

1. **Convergence Rate Comparison**: Newton vs gradient descent on quadratics
2. **Line Search vs Trust Region**: Compare globalization strategies
3. **Hessian Approximation**: Exact vs finite difference vs Hessian-free
4. **Regularization Effects**: Impact of different regularization strategies
5. **Scaling Study**: Performance vs problem dimension
6. **Condition Number Analysis**: Behavior on ill-conditioned problems

## Research Connections

### Foundational Papers
1. **Dennis & Moré (1977)** - "Quasi-Newton Methods, Motivation and Theory"
   - Comprehensive treatment of Newton and quasi-Newton methods

2. **Nocedal & Wright (2006)** - "Numerical Optimization" 
   - Modern reference for optimization theory

3. **Conn, Gould & Toint (2000)** - "Trust Region Methods"
   - Definitive treatment of trust region approaches

### Modern Developments
4. **Martens (2010)** - "Deep Learning via Hessian-free Optimization"
   - Hessian-free methods for neural networks

5. **Grosse & Martens (2016)** - "A Kronecker-factored Approximate Curvature Method"
   - K-FAC for practical second-order deep learning

6. **Roosta-Khorasani & Mahoney (2016)** - "Sub-sampled Newton Methods"
   - Stochastic Newton methods for machine learning

### Theoretical Advances
7. **Nesterov & Polyak (2006)** - "Cubic Regularization of Newton Method"
   - Improved global convergence properties

8. **Cartis, Gould & Toint (2011)** - "Adaptive Cubic Regularisation Methods"
   - Optimal complexity bounds for second-order methods

## Resources

### Primary Sources
1. **Nocedal & Wright - Numerical Optimization** (Chapters 3-4)
   - Standard reference for Newton methods
2. **Boyd & Vandenberghe - Convex Optimization** (Chapter 9)
   - Convex perspective on Newton methods  
3. **Bertsekas - Nonlinear Programming** (Chapter 1)
   - Theoretical foundations

### Video Resources
1. **Stephen Boyd - Convex Optimization Lectures** (Stanford)
   - Clear exposition of Newton methods
2. **Jorge Nocedal - Optimization Methods** (Northwestern)
   - Advanced topics in numerical optimization
3. **Benjamin Recht - Optimization for Machine Learning** (Berkeley)
   - ML perspective on second-order methods

### Advanced Reading
1. **Byrd, Lu, Nocedal & Zhu (2012)** - "A Limited Memory Algorithm for Bound Constrained Optimization"
2. **Gould, Orban, Sartenaer & Toint (2005)** - "Superlinear Convergence of Primal-Dual Interior Point Algorithms"
3. **Curtis, Robinson & Samadi (2017)** - "A Trust Region Algorithm with a Worst-Case Iteration Complexity"

## Socratic Questions

### Understanding
1. Why does Newton's method achieve quadratic convergence while gradient descent only achieves linear convergence?
2. What role does the condition number of the Hessian play in Newton method convergence?
3. When might Newton's method perform worse than gradient descent?

### Extension
1. How would you modify Newton's method for constrained optimization problems?
2. Can you design a Newton method that works well for non-convex neural network training?
3. What's the relationship between Newton methods and interior point methods?

### Research
1. How can we make Newton methods more robust to saddle points in deep learning?
2. What are the fundamental limits of second-order optimization in high dimensions?
3. Can quantum computing provide advantages for solving Newton linear systems?

## Exercises

### Theoretical
1. Prove the quadratic convergence rate of Newton's method under standard assumptions
2. Derive the Newton method for logistic regression
3. Analyze the effect of Hessian regularization on convergence properties

### Implementation
1. Implement Newton method with backtracking line search from scratch
2. Build trust region Newton method with dogleg step computation
3. Create Hessian-free Newton method using conjugate gradient
4. Implement Gauss-Newton method for nonlinear least squares

### Research
1. Compare Newton vs quasi-Newton methods on various ML problems
2. Study the effect of inexact Hessian computation on convergence
3. Investigate hybrid first/second-order optimization strategies