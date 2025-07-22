# Wasserstein GAN: Optimal Transport for Stable Adversarial Training

## Prerequisites
- Measure theory and probability distributions
- Optimal transport theory (Earth Mover's distance)
- Functional analysis (Lipschitz functions, dual spaces)
- Standard GAN theory and training difficulties
- Convex optimization and duality theory

## Learning Objectives
- Master the mathematical foundations of optimal transport for generative modeling
- Understand the Wasserstein distance and its advantages over f-divergences
- Implement WGAN with Lipschitz constraints and analyze training dynamics
- Connect optimal transport theory to modern generative modeling
- Analyze the theoretical guarantees and practical benefits of WGAN

## Mathematical Foundations

### 1. Limitations of Standard GANs

#### Problems with Jensen-Shannon Divergence
**Standard GAN objective**:
min_G max_D V(G,D) = 𝔼_p_data[log D(x)] + 𝔼_p_g[log(1-D(x))]

**Optimal discriminator**: D*(x) = p_data(x)/(p_data(x) + p_g(x))

**Connection to JS divergence**:
C(G) = -log(4) + 2·JS(p_data || p_g)

#### Issues with JS Divergence
**Theorem 1.1**: If p_data and p_g have disjoint supports, then:
JS(p_data || p_g) = log(2)

**Consequence**: Gradient of JS divergence is zero almost everywhere when supports don't overlap.

#### Vanishing Gradients
**Problem**: In high dimensions, supports are typically disjoint
**Result**: Discriminator becomes perfect, generator gradients vanish
**Manifestation**: Training instability, mode collapse

### 2. Optimal Transport Theory

#### Earth Mover's Distance
**Intuitive definition**: Minimum cost to transport mass from one distribution to another

**Formal definition**: For distributions μ and ν on metric space (M,d):
W(μ, ν) = inf_{γ∈Π(μ,ν)} ∫ d(x,y) dγ(x,y)

where Π(μ,ν) is the set of couplings between μ and ν.

#### Couplings and Transport Plans
**Coupling**: Joint distribution γ on X×Y with marginals μ and ν
**Transport plan**: How to move mass from μ to ν
**Optimal transport**: Coupling that minimizes transport cost

#### Kantorovich Duality
**Primal problem**: Find optimal coupling
**Dual problem**: Maximize over Lipschitz functions

**Theorem 2.1 (Kantorovich-Rubinstein)**:
W(μ, ν) = sup_{||f||_L ≤ 1} [∫ f dμ - ∫ f dν]

where ||f||_L is the Lipschitz constant of f.

### 3. Wasserstein Distance for Probability Distributions

#### 1-Wasserstein Distance
**Definition**: For probability distributions P_r and P_g:
W(P_r, P_g) = inf_{γ∈Π(P_r,P_g)} 𝔼_{(x,y)~γ}[||x - y||]

#### Kantorovich-Rubinstein Duality
**Dual formulation**:
W(P_r, P_g) = sup_{||f||_L ≤ 1} [𝔼_{x~P_r}[f(x)] - 𝔼_{x~P_g}[f(x)]]

**Key insight**: Wasserstein distance can be computed via optimization over Lipschitz functions.

#### Advantages over JS Divergence

**Theorem 3.1**: Wasserstein distance provides meaningful gradients even when supports are disjoint.

**Continuous dependence**: W(P_r, P_g) changes continuously with the distributions

**Weak convergence**: W(P_n, P) → 0 implies weak convergence P_n → P

### 4. WGAN Formulation

#### The WGAN Objective
**Discriminator becomes critic**: No sigmoid activation, output real values
**Objective**:
W(P_r, P_g) ≈ max_{w: ||f_w||_L ≤ 1} [𝔼_{x~P_r}[f_w(x)] - 𝔼_{x~P_g}[f_w(x)]]

**Generator**: min_θ W(P_r, P_{g_θ})

#### Algorithm 4.1 (WGAN Training)
```
for iteration in training_loop:
    # Train critic more than generator
    for i in range(n_critic):
        # Sample batch
        x_real ~ P_data
        z ~ p(z)
        x_fake = G(z)
        
        # Critic loss (maximize)
        L_critic = -[mean(C(x_real)) - mean(C(x_fake))]
        
        # Update critic
        C_optimizer.step(L_critic)
        
        # Enforce Lipschitz constraint
        clip_weights(C, c)
    
    # Train generator (every n_critic steps)
    z ~ p(z)
    x_fake = G(z)
    L_generator = -mean(C(x_fake))
    G_optimizer.step(L_generator)
```

#### Weight Clipping for Lipschitz Constraint
**Simple approach**: Clip weights to [-c, c] after each update
**Rationale**: Restricts function class to approximately satisfy Lipschitz constraint

**Limitations**:
- Crude approximation of Lipschitz constraint
- Can lead to capacity underutilization
- May cause gradient flow issues

### 5. Theoretical Analysis

#### Convergence Guarantees
**Theorem 5.1**: Under mild conditions, WGAN training converges to optimal transport solution.

**Proof sketch**: 
- Wasserstein distance is continuous and differentiable
- Gradient provides meaningful direction even with disjoint supports
- No saturation unlike JS divergence

#### Meaningful Loss Metric
**Benefit**: WGAN loss correlates with sample quality
**Standard GAN**: Loss doesn't indicate generation quality
**WGAN**: Lower loss → better samples (empirically verified)

#### Mode Coverage
**Theoretical advantage**: Wasserstein distance penalizes mode dropping
**Explanation**: Must cover entire support to minimize transport cost
**Empirical**: Better mode coverage than standard GANs

### 6. WGAN-GP (Gradient Penalty)

#### Problems with Weight Clipping
- Capacity underutilization
- Exploding/vanishing gradients
- Biases critic toward simple functions

#### Gradient Penalty Solution
**Idea**: Enforce Lipschitz constraint via gradient penalty

**Modified objective**:
L = 𝔼[C(x̃)] - 𝔼[C(x)] + λ 𝔼[(||∇_x̂ C(x̂)||₂ - 1)²]

where x̂ = εx + (1-ε)x̃ with ε ~ U[0,1]

#### Theoretical Justification
**Theorem 6.1**: If C is differentiable and 1-Lipschitz, then ||∇C(x)||₂ ≤ 1 almost everywhere.

**Enforcement**: Penalize deviations from unit gradient norm

#### Algorithm 6.1 (WGAN-GP)
```python
def gradient_penalty(critic, real_data, fake_data, lambda_gp=10):
    batch_size = real_data.size(0)
    
    # Sample random interpolation points
    epsilon = torch.rand(batch_size, 1, 1, 1)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated.requires_grad_(True)
    
    # Compute critic output
    critic_interpolated = critic(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Gradient penalty
    gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return penalty
```

### 7. Spectral Normalization Alternative

#### Spectral Normalization for Lipschitz Constraint
**Idea**: Control Lipschitz constant by normalizing spectral norm

**For linear layer**: W_SN = W / σ(W)
where σ(W) is the largest singular value

#### Power Iteration Method
**Efficient computation**:
```python
def spectral_norm(W, n_power_iterations=1):
    u = torch.randn(1, W.size(0))
    v = torch.randn(1, W.size(1))
    
    for _ in range(n_power_iterations):
        v = F.normalize(torch.matmul(u, W), dim=1)
        u = F.normalize(torch.matmul(v, W.t()), dim=1)
    
    sigma = torch.matmul(torch.matmul(v, W.t()), u.t())
    return W / sigma
```

#### Advantages over Gradient Penalty
- Computationally efficient
- Stable during training
- Easier to implement
- Better scaling to large networks

### 8. Extensions and Variants

#### Sliced Wasserstein Distance
**Motivation**: Computing Wasserstein distance is expensive in high dimensions
**Idea**: Project to 1D, compute Wasserstein distance, average over projections

**Formula**:
SW(μ, ν) = ∫_{S^{d-1}} W₁(P_θ#μ, P_θ#ν) dθ

where P_θ is projection onto direction θ.

#### Sinkhorn Divergences
**Regularized optimal transport**: Add entropy regularization
**Benefits**: 
- Faster computation via Sinkhorn iterations
- Differentiable approximation
- GPU-friendly parallel computation

#### Unbalanced Optimal Transport
**Generalization**: Allow mass creation/destruction
**Applications**: 
- Different total masses
- Partial transport
- Robust optimal transport

### 9. Practical Implementation

#### Hyperparameter Guidelines
**n_critic**: 5 for WGAN, 1-2 for WGAN-GP
**Learning rates**: Often lr_G = lr_C (unlike standard GAN)
**λ_GP**: 10 is standard choice
**Clip value**: 0.01 for weight clipping

#### Architecture Considerations
**No batch normalization**: In critic (breaks Lipschitz constraint)
**Layer normalization**: Alternative that preserves constraints
**Activation functions**: Avoid saturation (LeakyReLU preferred)

#### Training Stability
**Benefits over standard GAN**:
- Less sensitive to hyperparameters
- More stable training
- Meaningful loss metric
- Reduced mode collapse

### 10. Theoretical Connections

#### Optimal Transport and Generative Modeling
**Transportation metaphor**: Learn to transport noise to data distribution
**Continuous normalizing flows**: Implement optimal transport via ODEs
**Gradient flows**: WGAN as gradient flow in Wasserstein space

#### Connection to VAEs
**Optimal transport VAE**: Use Wasserstein distance in VAE objective
**Wasserstein autoencoders**: Replace KL with Wasserstein penalty

#### Information Geometry
**Riemannian manifolds**: Probability distributions as manifold points
**Geodesics**: Optimal transport paths as shortest paths
**Curvature**: Geometry affects optimization dynamics

### 11. Computational Aspects

#### Complexity Analysis
**Standard WGAN**: O(n²) for exact Wasserstein computation
**Approximations**: O(n) with neural network critics
**Gradient penalty**: Additional O(n) for gradient computation

#### GPU Implementation
**Parallel computation**: Batch processing of transport plans
**Memory efficiency**: Tricks for large-scale transport
**Mixed precision**: FP16 for faster training

#### Scalability
**Large datasets**: Mini-batch optimal transport
**High resolution**: Progressive training techniques
**Distributed training**: Synchronization challenges

### 12. Empirical Studies

#### Sample Quality
**Inception Score**: WGAN often outperforms standard GAN
**FID scores**: Better distribution matching
**Mode coverage**: Reduced mode collapse
**Visual quality**: Subjectively better samples

#### Training Dynamics
**Loss curves**: More stable and meaningful
**Convergence**: Faster and more reliable
**Hyperparameter sensitivity**: Less sensitive than standard GAN

#### Ablation Studies
**Weight clipping vs GP**: GP generally superior
**Spectral norm**: Competitive with GP, more efficient
**n_critic values**: 5 works well for most cases

### 13. Applications and Impact

#### High-Resolution Image Generation
**Progressive GAN**: Built on WGAN-GP foundation
**StyleGAN**: Incorporates WGAN-GP training
**BigGAN**: Uses spectral normalization from WGAN research

#### Domain Transfer
**Optimal transport for domain adaptation**
**CycleGAN variants**: With Wasserstein distance
**Style transfer**: Using optimal transport geometry

#### Scientific Applications
**Single-cell analysis**: Transport between cell populations
**Climate modeling**: Transport between weather patterns
**Economics**: Market matching and assignment problems

### 14. Limitations and Challenges

#### Computational Cost
**Gradient penalty**: Expensive gradient computation
**Multiple critic updates**: Slower than standard GAN
**Memory usage**: Higher for gradient tracking

#### Theoretical Gaps
**Finite sample**: Theory assumes infinite data
**Neural network**: Approximation quality unclear
**Optimization**: Non-convex optimization challenges

#### Practical Issues
**Hyperparameter tuning**: λ_GP requires careful selection
**Architecture sensitivity**: Some architectures work better
**Mode collapse**: Not completely eliminated

### 15. Recent Advances

#### Improved Lipschitz Constraints
**Convex potential flows**: Provably optimal transport
**Input convex networks**: Guarantee convexity
**Quadratic activation**: New activation functions

#### Computational Improvements
**Sliced variants**: Multiple slicing strategies
**Neural optimal transport**: End-to-end learning
**Continuous normalizing flows**: Connection to optimal transport

#### Theoretical Understanding
**Generalization bounds**: Better understanding of WGAN generalization
**Optimization theory**: Convergence analysis improvements
**Dual formulations**: New perspectives on optimal transport

## Implementation Details

See `exercise.py` for implementations of:
1. WGAN with weight clipping
2. WGAN-GP with gradient penalty
3. Spectral normalization for Lipschitz constraint
4. Sliced Wasserstein distance
5. Training monitoring and loss visualization
6. Comparison with standard GAN training

## Experiments

1. **Training Stability**: Compare WGAN vs standard GAN stability
2. **Hyperparameter Sensitivity**: Effect of λ_GP, n_critic
3. **Constraint Methods**: Weight clipping vs GP vs spectral norm
4. **Sample Quality**: Quantitative evaluation on standard datasets
5. **Computational Cost**: Training time and memory analysis

## Research Connections

### Foundational Papers
1. Arjovsky, Chintala & Bottou (2017) - "Wasserstein GAN"
2. Gulrajani et al. (2017) - "Improved Training of Wasserstein GANs"
3. Miyato et al. (2018) - "Spectral Normalization for Generative Adversarial Networks"
4. Kolouri et al. (2018) - "Sliced Wasserstein Autoencoders"

### Optimal Transport Theory
1. Villani (2008) - "Optimal Transport: Old and New"
2. Peyré & Cuturi (2019) - "Computational Optimal Transport"
3. Santambrogio (2015) - "Optimal Transport for Applied Mathematicians"

### Modern Applications
1. Karras et al. (2018) - "Progressive Growing of GANs for Improved Quality" 
2. Brock et al. (2019) - "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
3. Cuturi & Doucet (2014) - "Fast Computation of Wasserstein Barycenters"

## Resources

### Primary Sources
1. **Arjovsky et al. (2017)** - Original WGAN paper
2. **Villani (2008)** - Comprehensive optimal transport theory
3. **Peyré & Cuturi (2019)** - Computational optimal transport survey

### Software Resources
1. **PyTorch WGAN-GP**: Reference implementations
2. **POT Library**: Python optimal transport library
3. **GeomLoss**: Efficient optimal transport on GPU

### Video Resources
1. **Marco Cuturi - Optimal Transport Tutorial**
2. **Martin Arjovsky - WGAN Explanation** 
3. **ICML 2019 - Optimal Transport Workshop**

## Socratic Questions

### Understanding
1. Why does Wasserstein distance provide meaningful gradients when JS divergence fails?
2. How does the Lipschitz constraint relate to the discriminator's power?
3. What are the trade-offs between different Lipschitz enforcement methods?

### Extension
1. How would you design a WGAN for sequential data or graphs?
2. Can you derive connections between WGAN and other generative models?
3. What modifications would improve WGAN for discrete data?

### Research
1. What are the fundamental limitations of optimal transport for generative modeling?
2. How can we make optimal transport more computationally efficient?
3. What new applications of optimal transport in ML are most promising?

## Exercises

### Theoretical
1. Derive the Kantorovich-Rubinstein duality theorem
2. Prove that WGAN provides non-vanishing gradients
3. Analyze the effect of gradient penalty on the optimization landscape

### Implementation
1. Implement WGAN with multiple Lipschitz enforcement methods
2. Create visualization tools for optimal transport plans
3. Compare computational costs of different WGAN variants
4. Build evaluation pipeline for transport distance approximation

### Research
1. Study the relationship between Lipschitz constant and generation quality
2. Investigate novel architectures for optimal transport
3. Explore connections between WGAN and causal inference

## Advanced Topics

### Computational Optimal Transport
- **Sinkhorn algorithms**: Entropic regularization for fast computation
- **Unbalanced transport**: Handling different mass distributions
- **Multi-marginal transport**: Extensions to multiple distributions

### Geometric Deep Learning
- **Transport on manifolds**: Optimal transport for non-Euclidean data
- **Graph optimal transport**: Transport between graph distributions
- **Riemannian optimization**: Optimization on manifolds

### Continuous Normalizing Flows
- **Neural ODEs**: Implement optimal transport via differential equations
- **Flow matching**: Modern approach to optimal transport training
- **Rectified flows**: Straight-line transport paths