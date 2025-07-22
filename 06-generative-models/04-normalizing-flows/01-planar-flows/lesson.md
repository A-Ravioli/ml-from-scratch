# Planar Flows: Fundamentals of Normalizing Flows

## Prerequisites
- Multivariate calculus (Jacobians, change of variables)
- Probability theory (density transformations)
- Variational inference and ELBO
- Linear algebra (matrix determinants, eigenvalues)
- Basic understanding of deep learning

## Learning Objectives
- Master the mathematical foundations of normalizing flows
- Understand invertible transformations and Jacobian determinants
- Implement planar flows for variational posterior approximation
- Analyze expressiveness and limitations of simple flow transformations
- Connect flows to modern generative modeling approaches

## Mathematical Foundations

### 1. The Change of Variables Formula

#### Transformation of Densities
**Setup**: Let z₀ have density p₀(z₀) and z₁ = f(z₀) where f is invertible

**Change of variables**:
p₁(z₁) = p₀(z₀) |det J_f⁻¹(z₁)|

where J_f⁻¹ is the Jacobian of f⁻¹.

**Equivalent form**:
p₁(z₁) = p₀(f⁻¹(z₁)) |det J_f⁻¹(z₁)|

#### Jacobian and Volume Change
**Geometric interpretation**: |det J_f| measures volume scaling
- |det J_f| > 1: Volume expansion
- |det J_f| < 1: Volume contraction  
- |det J_f| = 1: Volume preservation

**Log-density transformation**:
log p₁(z₁) = log p₀(z₀) - log |det J_f(z₀)|

### 2. Normalizing Flows Framework

#### Definition 2.1 (Normalizing Flow)
A **normalizing flow** is a sequence of invertible transformations:
z₀ → z₁ → z₂ → ... → zₖ

where zᵢ = fᵢ(zᵢ₋₁) and each fᵢ is invertible with tractable Jacobian.

#### Flow Composition
**Density after K transformations**:
log pₖ(zₖ) = log p₀(z₀) - ∑ᵢ₌₁ᴷ log |det J_fᵢ(zᵢ₋₁)|

**Key insight**: Start with simple distribution, apply sequence of transformations to create complex distribution.

#### Requirements for Flow Transformations
1. **Invertibility**: f must have inverse f⁻¹
2. **Tractable Jacobian**: |det J_f| computable efficiently
3. **Differentiability**: For gradient-based optimization

### 3. Planar Flow Transformation

#### Definition 3.1 (Planar Flow)
A planar flow transformation is:
f(z) = z + u h(wᵀz + b)

where:
- u, w ∈ ℝᵈ are learnable vectors
- b ∈ ℝ is learnable scalar
- h: ℝ → ℝ is smooth activation function

#### Geometric Interpretation
**Planar transformation**: Translates points along direction u
**Magnitude**: Depends on h(wᵀz + b)
**Hyperplane**: wᵀz + b = const defines hyperplanes
**Effect**: Pushes/pulls points relative to hyperplanes

#### Jacobian Computation
**Jacobian matrix**:
J_f(z) = I + u ∇_z h(wᵀz + b) = I + u h'(wᵀz + b) wᵀ

**Determinant** (matrix determinant lemma):
det J_f(z) = det(I + u h'(wᵀz + b) wᵀ) = 1 + h'(wᵀz + b) uᵀw

#### Invertibility Constraint
**Condition**: For invertibility, we need det J_f(z) > 0 for all z

**Sufficient condition**: h'(·) ≥ 0 and uᵀw ≥ -1

**Practical enforcement**: 
- Use h with non-negative derivative (e.g., tanh, sigmoid)
- Constrain uᵀw ≥ -1 during training

### 4. Theoretical Analysis

#### Expressiveness of Single Planar Flow
**Theorem 4.1**: A single planar flow can represent any continuous distribution with support on a hyperplane.

**Limitation**: Cannot model distributions with support in higher dimensions without multiple flows.

#### Universal Approximation
**Theorem 4.2**: Compositions of planar flows can approximate any smooth diffeomorphism arbitrarily well.

**Proof sketch**: Each planar flow performs local deformation; composition allows global shaping.

#### Volume Preservation Analysis
**Volume change**: ∫ det J_f(z) p₀(z) dz measures total volume change
**Conservation**: For probability distributions, this integral equals 1

### 5. Implementation Details

#### Forward Pass
```python
def planar_flow_forward(z, u, w, b, activation='tanh'):
    """
    Forward pass through planar flow
    z: input tensor [batch_size, dim]
    u, w: parameter vectors [dim]
    b: bias scalar
    """
    if activation == 'tanh':
        h = torch.tanh
        h_prime = lambda x: 1 - torch.tanh(x)**2
    
    # Linear combination
    linear = torch.sum(w * z, dim=1, keepdim=True) + b  # [batch_size, 1]
    
    # Apply nonlinearity
    h_linear = h(linear)  # [batch_size, 1]
    
    # Planar transformation
    z_new = z + u.unsqueeze(0) * h_linear  # [batch_size, dim]
    
    # Jacobian determinant
    h_prime_linear = h_prime(linear)  # [batch_size, 1]
    u_dot_w = torch.sum(u * w)  # scalar
    log_det_jacobian = torch.log(torch.abs(1 + h_prime_linear * u_dot_w))
    
    return z_new, log_det_jacobian.squeeze()
```

#### Invertibility Enforcement
```python
def enforce_invertibility(u, w):
    """Ensure invertibility constraint uᵀw ≥ -1"""
    u_dot_w = torch.sum(u * w)
    
    if u_dot_w < -1:
        # Project u to satisfy constraint
        u_norm_sq = torch.sum(u**2)
        u = u + ((-1 - u_dot_w) / u_norm_sq) * u
        
    return u
```

#### Multi-Flow Composition
```python
class PlanarFlowSequence(nn.Module):
    def __init__(self, dim, num_flows):
        super().__init__()
        self.flows = nn.ModuleList([
            PlanarFlow(dim) for _ in range(num_flows)
        ])
    
    def forward(self, z0):
        z = z0
        sum_log_det_jacobian = 0
        
        for flow in self.flows:
            z, log_det_jac = flow(z)
            sum_log_det_jacobian += log_det_jac
            
        return z, sum_log_det_jacobian
```

### 6. Training Objective

#### Variational Inference with Flows
**Goal**: Approximate intractable posterior p(z|x) with flow q_φ(z|x)

**Flow posterior**: q_φ(z|x) starts from simple q₀(z₀|x), applies flows
zₖ = fₖ ∘ ... ∘ f₁(z₀)

**Flow density**:
log q_φ(zₖ|x) = log q₀(z₀|x) - ∑ᵢ₌₁ᴷ log |det J_fᵢ(zᵢ₋₁)|

#### ELBO with Normalizing Flows
**Variational lower bound**:
ℒ = 𝔼_{q_φ(z|x)}[log p(x|z)] - KL(q_φ(z|x) || p(z))

**Flow-based ELBO**:
ℒ = 𝔼_{q₀(z₀|x)}[log p(x|fₖ(...f₁(z₀)))] - KL(q_φ(zₖ|x) || p(z))

#### Training Algorithm
```python
def train_flow_vae(model, data_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for batch in data_loader:
            x = batch.to(device)
            
            # Encode to base distribution
            mu, log_var = model.encoder(x)
            z0 = model.reparameterize(mu, log_var)
            
            # Apply normalizing flows
            zK, sum_log_det_jac = model.flow(z0)
            
            # Decode
            x_recon = model.decoder(zK)
            
            # Compute losses
            recon_loss = reconstruction_loss(x, x_recon)
            
            # KL with flow correction
            kl_loss = kl_divergence(mu, log_var) - sum_log_det_jac.mean()
            
            loss = recon_loss + kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 7. Extensions and Variants

#### Radial Flows
**Alternative transformation**:
f(z) = z + β h(α, r)(z - z₀)

where r = ||z - z₀|| and h(α, r) controls radial expansion/contraction.

**Advantage**: Natural for radially symmetric transformations
**Limitation**: Still limited expressiveness

#### Sylvester Flows
**Higher-rank transformation**:
f(z) = z + A h(Bz + c)

where A ∈ ℝᵈˣᴹ, B ∈ ℝᴹˣᵈ

**Benefit**: More expressive than planar flows
**Cost**: More parameters and computation

#### Inverse Autoregressive Flows (IAF)
**Autoregressive structure**: Each dimension depends on previous dimensions
**Advantage**: Triangular Jacobian for efficient determinant
**Use case**: Expressive posterior approximation

### 8. Applications

#### Variational Autoencoders
**Problem**: Simple Gaussian posterior may be too restrictive
**Solution**: Use flows to increase posterior flexibility
**Result**: Better approximation of true posterior

#### Bayesian Neural Networks
**Posterior over weights**: Use flows for weight posterior
**Benefit**: More flexible uncertainty quantification
**Challenge**: Computational scaling

#### Density Estimation
**Direct approach**: Model data distribution with flows
**Training**: Maximum likelihood on observed data
**Inference**: Generate samples by sampling base distribution and transforming

### 9. Computational Considerations

#### Jacobian Computation
**Planar flows**: O(d) time for determinant
**General flows**: O(d³) for arbitrary Jacobians
**Structured flows**: Design for efficient computation

#### Memory Requirements
**Forward pass**: Store intermediate values for backward pass
**Gradient computation**: Requires inverse transformations
**Scaling**: Memory grows with flow depth

#### Numerical Stability
**Determinant computation**: Risk of overflow/underflow
**Solution**: Work in log space when possible
**Regularization**: Avoid extreme Jacobian values

### 10. Limitations and Challenges

#### Expressiveness vs Efficiency Trade-off
**Simple flows**: Fast but limited expressiveness
**Complex flows**: Expressive but computationally expensive
**Design choice**: Balance based on application needs

#### Invertibility Requirements
**Constraint**: All transformations must be invertible
**Limitation**: Rules out many useful transformations
**Research direction**: Approximately invertible flows

#### Local vs Global Transformations
**Planar flows**: Perform local deformations
**Global structure**: May require many flows
**Alternative**: Coupling layers for global transformations

### 11. Modern Developments

#### Continuous Normalizing Flows
**Idea**: Flows as solutions to ODEs
**Benefit**: Adaptive computation and memory
**Challenge**: ODE solver overhead

#### Neural Ordinary Differential Equations
**Connection**: Flows as discrete-time approximation of continuous dynamics
**Advantage**: Flexible depth, memory efficiency
**Modern impact**: Foundation for many recent flow methods

#### Autoregressive Flows
**Masked architectures**: MADE, MAF for autoregressive flows
**Benefit**: Flexible transformations with tractable Jacobians
**Application**: State-of-the-art density models

### 12. Experimental Analysis

#### Visualization of Flow Effects
**2D examples**: Visualize how planar flows transform distributions
**Parameter effects**: How u, w, b parameters change transformation
**Flow composition**: Multiple flows creating complex distributions

#### Expressiveness Studies
**Approximation quality**: How well can planar flows approximate target distributions?
**Number of flows**: Relationship between flow depth and approximation quality
**Comparison**: Planar vs radial vs other simple flows

#### Training Dynamics
**Convergence**: How flow parameters evolve during training
**Stability**: Numerical issues and mitigation strategies
**Hyperparameter sensitivity**: Learning rates, initialization effects

### 13. Evaluation Metrics

#### Density Estimation Quality
**Log-likelihood**: Direct evaluation for density models
**Coverage**: How well flow covers true distribution support
**Mode coverage**: Ability to capture all modes

#### Posterior Approximation Quality
**KL divergence**: Distance from true posterior (when available)
**Moment matching**: First and second moment accuracy
**Effective sample size**: Quality of approximate samples

#### Computational Metrics
**Training time**: Speed of convergence
**Inference time**: Cost of sampling and density evaluation
**Memory usage**: Requirements for large-scale applications

### 14. Connections to Other Methods

#### Relationship to VAEs
**Standard VAE**: Fixed Gaussian posterior
**Flow VAE**: Flexible posterior via flows
**Trade-off**: Complexity vs computational cost

#### Connection to GANs
**Implicit vs explicit**: GANs implicit, flows explicit density
**Training**: MLE vs adversarial
**Evaluation**: Flows allow likelihood evaluation

#### Link to Autoregressive Models
**Factorization**: Both use chain rule, different approaches
**Flows**: Transform simple to complex
**Autoregressive**: Model conditional distributions directly

### 15. Future Directions

#### Learnable Flow Architectures
**Neural architecture search**: Automatically design flow architectures
**Adaptive flows**: Flows that adjust their structure during training
**Meta-learning**: Learn to design flows for new tasks

#### Discrete and Mixed Data
**Discrete flows**: Flows for categorical data
**Mixed data**: Handling continuous and discrete variables
**Graph data**: Flows for graph-structured data

#### Theoretical Understanding
**Expressiveness theory**: Formal characterization of flow expressiveness
**Optimization theory**: Understanding flow training dynamics
**Generalization**: How flows generalize to unseen data

## Implementation Details

See `exercise.py` for implementations of:
1. Basic planar flow transformation
2. Multi-flow composition for complex distributions
3. Flow-based VAE with planar flows
4. Visualization tools for 2D flow transformations
5. Training loop with proper gradient flow
6. Comparison with standard VAE

## Experiments

1. **2D Distribution Modeling**: Visualize planar flow effects on simple 2D distributions
2. **Flow Depth Study**: How many flows needed for different target distributions?
3. **Parameter Analysis**: Effect of u, w, b parameters on transformation
4. **VAE Comparison**: Flow VAE vs standard VAE on same dataset
5. **Computational Cost**: Training time vs flow complexity

## Research Connections

### Foundational Papers
1. Rezende & Mohamed (2015) - "Variational Inference with Normalizing Flows"
2. Jimenez Rezende et al. (2014) - "Stochastic Backpropagation and Approximate Inference in Deep Generative Models"
3. Kingma et al. (2016) - "Improved Variational Inference with Inverse Autoregressive Flow"

### Theoretical Developments
1. Papamakarios et al. (2019) - "Normalizing Flows for Probabilistic Modeling and Inference"
2. Kobyzev et al. (2020) - "Normalizing Flows: An Introduction and Review of Current Methods"
3. Grathwohl et al. (2019) - "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models"

### Modern Applications
1. Durkan et al. (2019) - "Neural Spline Flows"
2. Ho et al. (2019) - "Flow++: Improving Flow-Based Generative Models"
3. Nielsen et al. (2020) - "SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows"

## Resources

### Primary Sources
1. **Rezende & Mohamed (2015)** - Original normalizing flows paper
2. **Papamakarios et al. (2019)** - Comprehensive survey
3. **Cranmer et al. (2020)** - "The frontier of simulation-based inference"

### Software Resources
1. **pyro**: Probabilistic programming with flows
2. **nflows**: PyTorch normalizing flows library
3. **FrEIA**: Framework for easily invertible architectures

### Video Resources
1. **ICML 2020 - Normalizing Flows Tutorial**
2. **NeurIPS 2019 - Flow-based Deep Generative Models**
3. **Danilo Rezende - Introduction to Normalizing Flows**

## Socratic Questions

### Understanding
1. Why do planar flows require invertible transformations for density modeling?
2. How does the Jacobian determinant relate to probability density changes?
3. What limits the expressiveness of a single planar flow transformation?

### Extension
1. How would you design flows for high-dimensional data efficiently?
2. Can you modify planar flows to handle discrete variables?
3. What happens to flow training dynamics as the number of flows increases?

### Research
1. What are the fundamental trade-offs in normalizing flow design?
2. How can we automatically discover good flow architectures?
3. What theoretical guarantees can we provide for flow-based inference?

## Exercises

### Theoretical
1. Derive the Jacobian determinant formula for planar flows
2. Prove the invertibility condition for planar transformations
3. Analyze the expressiveness limitations of single planar flows

### Implementation
1. Implement planar flows from scratch with proper gradients
2. Create visualization tools for 2D flow transformations
3. Build flow-based VAE and compare with standard VAE
4. Implement numerical stability improvements

### Research
1. Study the relationship between flow depth and approximation quality
2. Investigate novel activation functions for planar flows
3. Explore connections between flows and other generative models

## Advanced Topics

### Theoretical Foundations
- **Diffeomorphisms**: Mathematical foundations of smooth invertible mappings
- **Measure theory**: Rigorous treatment of density transformations
- **Information geometry**: Geometric perspective on flow transformations

### Computational Improvements
- **Efficient Jacobians**: Tricks for fast determinant computation
- **Memory optimization**: Reducing memory requirements for deep flows
- **Parallel computation**: Strategies for flow parallelization

### Connections to Physics
- **Hamiltonian flows**: Energy-preserving transformations
- **Optimal transport**: Connecting flows to optimal transport theory
- **Statistical mechanics**: Flows as equilibrium processes