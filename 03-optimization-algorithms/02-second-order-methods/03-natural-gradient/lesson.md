# Natural Gradient Methods

## Prerequisites
- Information geometry and differential geometry fundamentals
- Fisher Information Matrix and statistical manifolds
- Probability theory and statistical inference
- Optimization on Riemannian manifolds
- Neural networks and probabilistic models

## Learning Objectives
- Master the theory of natural gradients and information geometry
- Understand the geometric interpretation of optimization on statistical manifolds
- Implement natural gradient methods for various probabilistic models
- Connect natural gradients to second-order optimization and adaptive methods
- Apply natural gradients to modern deep learning architectures

## Mathematical Foundations

### 1. Information Geometry Fundamentals

#### Statistical Manifolds
A **statistical manifold** is a family of probability distributions {p(x|θ) : θ ∈ Θ} where:
- θ ∈ ℝⁿ are parameters
- Each θ corresponds to a probability distribution
- The space forms a Riemannian manifold

#### Riemannian Metric: Fisher Information Matrix
The **Fisher Information Matrix** defines the Riemannian metric:

F(θ)ᵢⱼ = E_{x~p(x|θ)}[∂log p(x|θ)/∂θᵢ · ∂log p(x|θ)/∂θⱼ]

Alternative forms:
- F(θ) = E[∇log p(x|θ)∇log p(x|θ)ᵀ]
- F(θ) = -E[∇²log p(x|θ)] (when regularity conditions hold)

#### Geometric Properties
1. **Positive definiteness**: F(θ) ≻ 0 under regularity conditions
2. **Coordinate invariance**: Natural gradients are independent of parameterization
3. **Curvature**: F(θ) encodes the intrinsic curvature of the statistical manifold

### 2. Natural Gradient Descent

#### Standard vs Natural Gradient
**Standard gradient descent**: θ_{k+1} = θ_k - α∇L(θ_k)

**Natural gradient descent**: θ_{k+1} = θ_k - αF(θ_k)⁻¹∇L(θ_k)

#### Geometric Interpretation
- Standard gradient: Steepest descent in Euclidean space
- Natural gradient: Steepest descent on the statistical manifold with Fisher metric
- Natural gradient follows the direction of steepest descent in the "information geometry sense"

#### Invariance Property
**Theorem 2.1 (Parameterization Invariance)**: Natural gradient updates are invariant under smooth reparameterizations of the model.

If φ = h(θ) is a smooth reparameterization, then natural gradient updates in θ-space correspond to natural gradient updates in φ-space.

**Proof Sketch**:
1. Fisher matrix transforms as: F_φ = J⁻ᵀF_θJ⁻¹ where J = ∂h/∂θ
2. Gradient transforms as: ∇L_φ = J⁻ᵀ∇L_θ  
3. Natural gradient: F_φ⁻¹∇L_φ = JF_θ⁻¹∇L_θ
4. Transform back to θ-space gives same update □

### 3. Natural Gradients for Specific Models

#### Linear Regression
Model: y = θᵀx + ε where ε ~ N(0, σ²)

Fisher Information: F(θ) = (1/σ²)E[xxᵀ] = (1/σ²)X ᵀX/n

Natural gradient: Δθ = -(XᵀX)⁻¹Xᵀ(Xθ - y)

This is exactly the Newton method update!

#### Logistic Regression  
Model: p(y=1|x,θ) = σ(θᵀx) where σ is sigmoid

Fisher Information: F(θ) = E[σ(θᵀx)(1-σ(θᵀx))xxᵀ]

Empirical Fisher: F̂(θ) = (1/n)∑ᵢ σ(θᵀxᵢ)(1-σ(θᵀxᵢ))xᵢxᵢᵀ

Natural gradient update:
θ_{k+1} = θ_k - αF̂(θ_k)⁻¹∇L(θ_k)

#### Gaussian Distributions
For multivariate Gaussian p(x|μ,Σ) = N(μ,Σ):

**Mean parameter**: Natural gradient for μ is standard gradient
**Covariance parameter**: More complex due to constraint Σ ≻ 0

### 4. Natural Gradients in Neural Networks

#### Multilayer Perceptrons
For a neural network with parameters θ, the Fisher matrix is:

F(θ) = E[∇log p(y|x,θ)∇log p(y|x,θ)ᵀ]

**Challenges**:
- Fisher matrix is huge: O(P²) where P is number of parameters
- Computing and inverting F is intractable for large networks
- Fisher matrix changes at every iteration

#### Block-Diagonal Approximations
Approximate F as block-diagonal over layers:

F ≈ diag(F₁, F₂, ..., F_L)

where F_l is Fisher matrix for layer l parameters.

#### Kronecker-Factored Approximation (K-FAC)
**Key insight**: For layer l with weight matrix W_l ∈ ℝᵐˣⁿ, approximate:

F_l ≈ A_l ⊗ G_l

where:
- A_l ∈ ℝⁿˣⁿ: Covariance of layer inputs
- G_l ∈ ℝᵐˣᵐ: Covariance of layer output gradients  
- ⊗: Kronecker product

**Advantages**:
- Storage: O(m²+n²) instead of O((mn)²)
- Inversion: Use Kronecker product properties
- Computation: Significantly more efficient

#### K-FAC Algorithm
```
1. Compute A_l = E[a_l a_l^T] (input covariance)
2. Compute G_l = E[g_l g_l^T] (output gradient covariance)
3. Approximate: F_l^{-1} ≈ A_l^{-1} ⊗ G_l^{-1}
4. Natural gradient: Δw_l = (A_l^{-1} ⊗ G_l^{-1}) vec(∇L/∂W_l)
5. Equivalently: ΔW_l = G_l^{-1} (∇L/∂W_l) A_l^{-1}
```

### 5. Practical Natural Gradient Algorithms

#### Empirical Fisher Information
Instead of true Fisher matrix, use empirical approximation:

F̂(θ) = (1/n)∑ᵢ ∇log p(yᵢ|xᵢ,θ)∇log p(yᵢ|xᵢ,θ)ᵀ

**Note**: This uses actual labels yᵢ, not samples from model.

#### Gauss-Newton Approximation
For models with squared loss, use Gauss-Newton approximation:

F̂(θ) ≈ JᵀJ

where J is Jacobian of model outputs w.r.t. parameters.

#### Natural Gradient with Regularization
Add damping for numerical stability:

θ_{k+1} = θ_k - α(F(θ_k) + λI)⁻¹∇L(θ_k)

#### Trust Region Natural Gradients
Constrain update size in natural metric:

min_Δθ ∇L(θ)ᵀΔθ + (1/2)ΔθᵀF(θ)Δθ
s.t.  ΔθᵀF(θ)Δθ ≤ δ²

### 6. Connections to Other Methods

#### Natural Gradient vs Newton's Method
For maximum likelihood estimation:
- **Newton**: θ_{k+1} = θ_k - H⁻¹∇L where H = ∇²L
- **Natural**: θ_{k+1} = θ_k - F⁻¹∇L where F is Fisher matrix

Under regularity conditions: E[H] = F (Fisher information equality)

#### Natural Gradient vs Adaptive Methods
**Adam-style adaptation**: Diagonal approximation of natural gradient
- Adam: Uses empirical second moments of gradients
- Natural gradient: Uses Fisher information matrix
- Connection: Fisher diagonal ≈ E[g²] in some settings

#### Mirror Descent Connection
Natural gradient can be viewed as mirror descent with entropy regularization:

min_θ L(θ) + D_F(θ, θ_k)

where D_F is Bregman divergence induced by log-partition function.

### 7. Advanced Natural Gradient Methods

#### Stochastic Natural Gradients
**Challenge**: Fisher matrix estimation with mini-batches

**Solutions**:
1. **Running averages**: F_k = βF_{k-1} + (1-β)F̂_k
2. **Block-wise updates**: Update Fisher approximation less frequently
3. **Variance reduction**: Combine with SVRG-style techniques

#### Distributed Natural Gradients
For distributed learning:
1. **Local Fisher computation**: Each worker computes local Fisher approximation
2. **Aggregation**: Combine Fisher matrices across workers
3. **Synchronization**: Coordinate natural gradient updates

#### Natural Gradients for Deep Reinforcement Learning
**Policy gradient methods**: Natural policy gradients for RL

θ_{k+1} = θ_k + αF⁻¹∇J(θ_k)

where F is Fisher matrix of policy distribution and J is expected return.

**TRPO (Trust Region Policy Optimization)**:
- Use natural gradients with trust region constraints
- Approximately solve: max_Δθ ∇J^T Δθ s.t. (1/2)Δθ^T F Δθ ≤ δ

### 8. Computational Considerations

#### Fisher Matrix Estimation
**Exact computation**: F(θ) = E[∇log p(x|θ)∇log p(x|θ)ᵀ]
- Requires sampling from model distribution
- Computationally expensive for large models

**Empirical approximation**: Use training data
- F̂(θ) = (1/n)∑ᵢ ∇log p(xᵢ|θ)∇log p(xᵢ|θ)ᵀ
- Biased estimator but computationally tractable

#### Matrix Inversion Strategies
1. **Direct inversion**: O(n³) - only for small problems
2. **Conjugate gradient**: Solve F(θ)d = ∇L iteratively
3. **Low-rank approximation**: F ≈ UU^T + λI
4. **Block-diagonal**: Ignore off-diagonal interactions
5. **Kronecker factorization**: For structured problems

#### Numerical Stability
- **Regularization**: Add λI to Fisher matrix
- **Eigen-clipping**: Remove small eigenvalues
- **Gradual updates**: Slowly update Fisher approximation

### 9. Modern Developments

#### Shampoo Algorithm
**Idea**: Use different preconditioners for different tensor modes

For gradient tensor G ∈ ℝᵐˣⁿ:
- Left preconditioner: P_L = (∑ G G^T)^{-1/4}
- Right preconditioner: P_R = (∑ G^T G)^{-1/4}
- Update: W ← W - α P_L G P_R

#### AdaHessian
Diagonal approximation using Hessian information:

v_t = β v_{t-1} + (1-β) ⊙ diag(H_t)
θ_{t+1} = θ_t - α/√(v_t + ε) ⊙ ∇L_t

where H_t is Hessian diagonal.

#### Natural Evolution Strategies
Apply natural gradients to evolution strategies:
- Parameterize search distribution (e.g., Gaussian)
- Use Fisher matrix of search distribution
- Natural gradient update of distribution parameters

### 10. Applications and Case Studies

#### Variational Inference
**Variational objective**: L(θ) = E_{q(z|θ)}[log p(x,z) - log q(z|θ)]

Natural gradients for variational parameters:
- Fisher matrix of variational distribution q(z|θ)
- Often leads to coordinate ascent updates
- Connection to natural gradient VI

#### GANs and Variational Autoencoders
**Generator networks**: Natural gradients for likelihood-based training
**VAE encoder/decoder**: Natural gradients for variational parameters

#### Large Language Models
**Recent applications**:
- K-FAC for transformer training
- Shampoo for large-scale optimization
- Natural gradients for fine-tuning

## Implementation Details

See `exercise.py` for implementations of:
1. Fisher Information Matrix computation for various models
2. K-FAC algorithm for neural networks
3. Natural gradient descent with regularization
4. Empirical vs true Fisher approximations
5. Block-diagonal Fisher approximations
6. Conjugate gradient solver for Fisher systems
7. Natural gradients for policy optimization
8. Performance comparison with standard methods

## Experiments

1. **Fisher vs Hessian**: Compare natural gradients with Newton's method
2. **K-FAC Evaluation**: Natural gradients for deep networks
3. **Parameterization Invariance**: Verify invariance properties
4. **Approximation Quality**: Empirical vs true Fisher performance
5. **Computational Efficiency**: Speed and memory comparisons
6. **Convergence Analysis**: Natural vs standard gradients on various problems

## Research Connections

### Foundational Papers
1. **Amari (1998)** - "Natural Gradient Works Efficiently in Learning"
   - Seminal paper introducing natural gradients

2. **Amari & Nagaoka (2000)** - "Methods of Information Geometry"
   - Comprehensive treatment of information geometry

3. **Kakade (2001)** - "A Natural Policy Gradient"
   - Natural gradients for reinforcement learning

### Modern Developments
4. **Martens & Grosse (2015)** - "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"
   - K-FAC algorithm for practical natural gradients

5. **Grosse & Martens (2016)** - "A Kronecker-factored Approximate Curvature Method for Convolution Layers"
   - Extension of K-FAC to convolutional layers

6. **George et al. (2018)** - "Fast Approximate Natural Gradient Descent in a Kronecker Factored Eigenbasis"
   - Efficient eigendecomposition for K-FAC

### Recent Advances
7. **Gupta et al. (2018)** - "Shampoo: Preconditioned Stochastic Tensor Optimization"
   - Matrix-free preconditioning methods

8. **Anil et al. (2020)** - "Scalable Second Order Optimization for Deep Learning"
   - Large-scale natural gradient methods

9. **Yao et al. (2021)** - "AdaHessian: An Adaptive Second Order Optimizer for Machine Learning"
   - Diagonal Hessian approximations

## Resources

### Primary Sources
1. **Amari & Nagaoka - Methods of Information Geometry**
   - Definitive reference for information geometry
2. **Martens - New Insights and Perspectives on Natural Gradients**
   - Modern survey of natural gradient methods
3. **Nielsen - An Elementary Introduction to Information Geometry**
   - Accessible introduction to the mathematical foundations

### Video Resources
1. **Shun-ichi Amari - Information Geometry Lectures**
   - From the founder of natural gradient methods
2. **James Martens - K-FAC and Natural Gradients**
   - Practical implementation perspectives
3. **Roger Grosse - Second Order Optimization**
   - Modern view on natural gradients in deep learning

### Advanced Reading
1. **Ollivier (2015)** - "Riemannian Metrics for Neural Networks I: Feedforward Networks"
2. **Pascanu & Bengio (2013)** - "Revisiting Natural Gradient for Deep Networks"
3. **Zhang et al. (2017)** - "Noisy Natural Gradients as Variational Inference"

## Socratic Questions

### Understanding
1. Why are natural gradients invariant to parameter reparameterization while standard gradients are not?
2. How does the Fisher Information Matrix encode the geometry of the statistical manifold?
3. When do natural gradients reduce to Newton's method?

### Extension
1. How would you extend natural gradients to constrained optimization problems?
2. Can natural gradient ideas be applied to non-probabilistic models?
3. What's the relationship between natural gradients and mirror descent?

### Research
1. How can we make natural gradient methods more scalable to modern deep learning?
2. What are the fundamental limitations of Kronecker factorization in K-FAC?
3. Can natural gradients help with the optimization challenges in training very large language models?

## Exercises

### Theoretical
1. Prove the parameterization invariance property of natural gradients
2. Derive the Fisher Information Matrix for multivariate Gaussian distributions
3. Show the connection between natural gradients and Newton's method for exponential families

### Implementation
1. Implement natural gradient descent for logistic regression from scratch
2. Build K-FAC algorithm for multilayer perceptrons
3. Create Fisher matrix computation for various probability distributions
4. Implement natural policy gradients for simple RL environments

### Research
1. Compare natural gradients with adaptive methods on neural network training
2. Study the effect of Fisher matrix approximation quality on convergence
3. Investigate natural gradient methods for transformer architectures