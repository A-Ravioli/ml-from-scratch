# Convex Optimization for Machine Learning

## Prerequisites
- Linear algebra (eigenvalues, quadratic forms)
- Real analysis (gradients, convexity)
- Basic calculus of variations

## Learning Objectives
- Master convex sets, functions, and optimization problems
- Understand duality theory and KKT conditions
- Connect optimization theory to ML algorithms
- Build geometric intuition for optimization landscapes

## Mathematical Foundations

### 1. Convex Sets

#### Definition 1.1 (Convex Set)
A set C ⊆ ℝⁿ is convex if for all x, y ∈ C and λ ∈ [0,1]:
λx + (1-λ)y ∈ C

Geometrically: line segments between any two points lie entirely in C.

#### Examples
1. **Affine sets**: {x : Ax = b}
2. **Halfspaces**: {x : aᵀx ≤ b}
3. **Norm balls**: {x : ||x|| ≤ r}
4. **Positive semidefinite cone**: S₊ⁿ = {X : X ⪰ 0}
5. **Probability simplex**: {x : ∑xᵢ = 1, xᵢ ≥ 0}

#### Theorem 1.1 (Convex Combinations)
C is convex ⟺ C contains all convex combinations of its points.

#### Theorem 1.2 (Operations Preserving Convexity)
1. Intersection: ∩ᵢ Cᵢ is convex if each Cᵢ is convex
2. Affine transformation: {Ax + b : x ∈ C} is convex if C is convex
3. Cartesian product: C₁ × C₂ is convex if C₁, C₂ are convex

### 2. Convex Functions

#### Definition 2.1 (Convex Function)
f: ℝⁿ → ℝ is convex if dom(f) is convex and for all x, y ∈ dom(f), λ ∈ [0,1]:
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)

Strict inequality (except at endpoints) defines strict convexity.

#### Theorem 2.1 (First-Order Characterization)
If f is differentiable, then f is convex ⟺
f(y) ≥ f(x) + ∇f(x)ᵀ(y - x) for all x, y ∈ dom(f)

**Geometric interpretation**: Function lies above its tangent planes.

#### Theorem 2.2 (Second-Order Characterization)
If f is twice differentiable, then f is convex ⟺ ∇²f(x) ⪰ 0 for all x ∈ dom(f)

#### Examples in ML
1. **Linear**: f(x) = aᵀx + b
2. **Quadratic**: f(x) = ½xᵀPx + qᵀx + r with P ⪰ 0
3. **Norm**: f(x) = ||x||ₚ for p ≥ 1
4. **Log-sum-exp**: f(x) = log(∑ᵢ eˣⁱ)
5. **Negative entropy**: f(x) = ∑ᵢ xᵢ log xᵢ on probability simplex

### 3. Convex Optimization Problems

#### Standard Form
minimize f₀(x)
subject to fᵢ(x) ≤ 0, i = 1,...,m
         Ax = b

where f₀, f₁,...,fₘ are convex and dom(f₀) ∩ ∩ᵢ dom(fᵢ) ≠ ∅.

#### Theorem 3.1 (Global Optimality)
For convex problems, any local minimum is a global minimum.

**Proof**: Suppose x* is local minimum but not global. Then ∃y with f(y) < f(x*).
By convexity of feasible set, the line segment [x*, y] is feasible.
By convexity of f, f has no local minimum between f(x*) and f(y), contradiction. □

#### Theorem 3.2 (Optimality Conditions)
For problem with differentiable f₀, x* is optimal iff:
∇f₀(x*) = 0 (unconstrained case)

### 4. Lagrangian Duality

#### Definition 4.1 (Lagrangian)
L(x, λ, ν) = f₀(x) + ∑ᵢ λᵢfᵢ(x) + νᵀ(Ax - b)

where λᵢ ≥ 0 are inequality multipliers, ν are equality multipliers.

#### Definition 4.2 (Dual Function)
g(λ, ν) = inf_{x} L(x, λ, ν)

Properties:
1. g is concave (even if original problem is not convex)
2. g(λ, ν) ≤ p* for all λ ⪰ 0, ν (weak duality)

#### Definition 4.3 (Dual Problem)
maximize g(λ, ν)
subject to λ ⪰ 0

#### Theorem 4.1 (Strong Duality)
If the primal problem is convex and satisfies Slater's condition (∃x strictly feasible), then:
p* = d* (strong duality holds)

### 5. KKT Conditions

#### Theorem 5.1 (KKT Necessary Conditions)
If x* is optimal and constraint qualifications hold, then ∃λ*, ν* such that:
1. ∇f₀(x*) + ∑ᵢ λᵢ*∇fᵢ(x*) + Aᵀν* = 0 (stationarity)
2. fᵢ(x*) ≤ 0, i = 1,...,m (primal feasibility)
3. Ax* = b (primal feasibility)
4. λᵢ* ≥ 0, i = 1,...,m (dual feasibility)
5. λᵢ*fᵢ(x*) = 0, i = 1,...,m (complementary slackness)

#### Theorem 5.2 (KKT Sufficient Conditions)
For convex problems, KKT conditions are sufficient for optimality.

### 6. Applications to Machine Learning

#### Linear Programming
minimize cᵀx
subject to Ax ≤ b

**ML applications**: 
- ℓ₁ regularization (Lasso)
- Support vector classification (hard margin)
- Optimal transport

#### Quadratic Programming
minimize ½xᵀPx + qᵀx
subject to Ax ≤ b

**ML applications**:
- Ridge regression
- Support vector machines
- Portfolio optimization

#### Second-Order Cone Programming (SOCP)
minimize cᵀx
subject to ||Aᵢx + bᵢ||₂ ≤ cᵢᵀx + dᵢ

**ML applications**:
- Robust optimization
- ℓ₂ regularization with constraints

#### Semidefinite Programming (SDP)
minimize ⟨C, X⟩
subject to ⟨Aᵢ, X⟩ = bᵢ
         X ⪰ 0

**ML applications**:
- Matrix completion
- Relaxations of combinatorial problems
- Kernel learning

### 7. Algorithms for Convex Optimization

#### Gradient Descent
For unconstrained minimization of f:
xₖ₊₁ = xₖ - αₖ∇f(xₖ)

**Convergence**: For L-smooth, μ-strongly convex f:
f(xₖ) - f* ≤ (1 - μ/L)ᵏ(f(x₀) - f*)

#### Projected Gradient Descent
For constrained problems:
xₖ₊₁ = P_C(xₖ - αₖ∇f(xₖ))

where P_C is projection onto constraint set C.

#### Newton's Method
xₖ₊₁ = xₖ - α∇²f(xₖ)⁻¹∇f(xₖ)

**Convergence**: Quadratic near optimum if ∇²f is Lipschitz.

#### Interior Point Methods
Handle inequality constraints by adding barrier functions:
minimize f₀(x) - μ∑ᵢ log(-fᵢ(x))

### 8. Subdifferentials and Non-smooth Optimization

#### Definition 8.1 (Subdifferential)
For convex f, the subdifferential at x is:
∂f(x) = {g : f(y) ≥ f(x) + gᵀ(y-x) ∀y}

#### Subgradient Method
xₖ₊₁ = xₖ - αₖgₖ where gₖ ∈ ∂f(xₖ)

**ML applications**:
- ℓ₁ regularization
- Hinge loss optimization
- Non-smooth regularizers

## Conceptual Understanding

### Geometric Intuition

1. **Convex sets**: "No dents" - straight lines stay inside
2. **Convex functions**: "Bowl-shaped" - unique global minimum
3. **Duality**: Provides lower bounds and alternative formulations
4. **KKT conditions**: Necessary and sufficient optimality conditions

### Why Convexity Matters in ML

1. **Guaranteed Global Optimum**: No local minima to get stuck in
2. **Efficient Algorithms**: Polynomial-time solvable
3. **Theory**: Rich theory for convergence analysis
4. **Robustness**: Small data changes → small solution changes

### Non-Convex ML Problems

While many ML problems are non-convex (deep learning), convex relaxations and convex subproblems are crucial:
- Initialization strategies
- Regularization techniques
- Optimization algorithm design

## Implementation Details

See `exercise.py` for implementations of:
1. Convexity verification functions
2. Gradient descent variants
3. Proximal operators
4. Interior point methods
5. Duality gap computation
6. Applications to ML problems

## Experiments

1. **Convergence Rates**: Compare algorithms on quadratic functions
2. **Conditioning**: Effect of condition number on convergence
3. **Regularization Paths**: Trace solutions as λ varies
4. **Duality Gaps**: Monitor convergence using dual problems

## Research Connections

### Foundational Papers
1. Boyd & Vandenberghe (2004) - "Convex Optimization"
   - Comprehensive treatment of theory and algorithms

2. Rockafellar (1970) - "Convex Analysis"
   - Mathematical foundations

3. Nesterov (2003) - "Introductory Lectures on Convex Optimization"
   - Modern algorithmic perspective

### ML Applications
1. **SVM**: Vapnik (1995) - Convex formulation of classification
2. **Lasso**: Tibshirani (1996) - ℓ₁ regularized regression
3. **Matrix Completion**: Candès & Recht (2009) - Nuclear norm relaxation

## Resources

### Primary Sources
1. **Boyd & Vandenberghe - Convex Optimization**
   - THE reference for the field
2. **Bertsekas - Nonlinear Programming**
   - Comprehensive treatment including non-convex
3. **Beck - First-Order Methods in Optimization**
   - Modern algorithmic focus

### Video Resources
1. **Stanford EE364A - Convex Optimization**
   - Stephen Boyd's legendary course
2. **CMU 10-725 - Optimization for ML**
   - Ryan Tibshirani's course
3. **MIT 6.253 - Convex Analysis and Optimization**
   - Dimitri Bertsekas

### Advanced Reading
1. **Rockafellar & Wets - Variational Analysis**
   - Advanced theory
2. **Hiriart-Urruty & Lemaréchal - Convex Analysis and Minimization**
   - Two-volume comprehensive treatment
3. **Bubeck - Convex Optimization: Algorithms and Complexity**
   - Modern complexity-theoretic view

## Socratic Questions

### Understanding
1. Why does convexity guarantee global optimality?
2. What's the geometric interpretation of KKT conditions?
3. How does strong duality differ from weak duality?

### Extension
1. Can you construct convex relaxations of non-convex problems?
2. What happens to optimality when we add non-convex constraints?
3. How do we handle stochastic/online convex optimization?

### Research
1. What's the role of convexity in deep learning optimization?
2. How can we exploit partial convexity in ML problems?
3. What are efficient algorithms for large-scale convex optimization?

## Exercises

### Theoretical
1. Prove that the intersection of convex sets is convex
2. Show that strong convexity implies unique global minimum
3. Derive the dual of the SVM optimization problem

### Implementation
1. Code gradient descent with line search for quadratic functions
2. Implement ADMM for ℓ₁ regularized regression
3. Build interior point method for linear programming

### Research
1. Investigate convex relaxations of matrix factorization
2. Study optimization landscapes of neural networks
3. Explore connections between convex optimization and game theory