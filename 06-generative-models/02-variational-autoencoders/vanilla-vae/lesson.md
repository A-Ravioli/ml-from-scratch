# Variational Autoencoders: Deep Generative Models with Latent Variables

## Prerequisites
- Variational inference and Bayesian statistics
- Information theory (KL divergence, mutual information)
- Neural network fundamentals
- Probability theory (Gaussian distributions, reparameterization)
- Basic understanding of autoencoders

## Learning Objectives
- Master the mathematical foundations of variational inference
- Understand the Evidence Lower Bound (ELBO) derivation
- Implement the reparameterization trick for backpropagation
- Analyze the latent space structure and disentanglement
- Connect VAEs to modern generative modeling and representation learning

## Mathematical Foundations

### 1. The Generative Modeling Problem

#### Latent Variable Models
**Motivation**: Real data often has underlying structure

**Generative model**:
- Latent variables: z ∈ ℝᵈ (unobserved)
- Observations: x ∈ ℝᴰ (observed data)
- Joint model: p(x, z) = p(x|z)p(z)

#### The Inference Problem
**Goal**: Infer posterior p(z|x) for given observations

**Challenge**: Posterior is intractable for complex models
p(z|x) = p(x|z)p(z) / p(x) = p(x|z)p(z) / ∫ p(x|z)p(z) dz

**Solution**: Variational approximation q(z|x) ≈ p(z|x)

### 2. Variational Inference Framework

#### Variational Lower Bound Derivation
**Evidence (marginal likelihood)**:
log p(x) = log ∫ p(x|z)p(z) dz

**Key insight**: Introduce variational distribution q(z|x)

log p(x) = log ∫ q(z|x) · [p(x|z)p(z)/q(z|x)] dz
         ≥ ∫ q(z|x) log [p(x|z)p(z)/q(z|x)] dz    (Jensen's inequality)
         = 𝔼_{q(z|x)}[log p(x|z)] - KL(q(z|x) || p(z))

#### Evidence Lower Bound (ELBO)
**Definition**:
ℒ(x; θ, φ) = 𝔼_{q_φ(z|x)}[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))

where:
- θ: generative model parameters
- φ: variational parameters

#### Theorem 2.1 (ELBO Properties)
1. **Lower bound**: ℒ(x) ≤ log p(x)
2. **Gap**: log p(x) - ℒ(x) = KL(q(z|x) || p(z|x)) ≥ 0
3. **Optimality**: ℒ(x) = log p(x) iff q(z|x) = p(z|x)

**Proof**: Direct from Jensen's inequality and properties of KL divergence. □

### 3. VAE Architecture and Parameterization

#### Neural Network Parameterization
**Encoder (Recognition Model)**: q_φ(z|x)
- Maps observations to latent distribution parameters
- φ represents neural network weights

**Decoder (Generative Model)**: p_θ(x|z)
- Maps latent codes to data distribution parameters  
- θ represents neural network weights

#### Gaussian Assumption
**Prior**: p(z) = 𝒩(0, I)
**Posterior approximation**: q_φ(z|x) = 𝒩(μ_φ(x), σ²_φ(x)I)
**Likelihood**: p_θ(x|z) = 𝒩(μ_θ(z), σ²I) or Bernoulli for binary data

#### ELBO with Gaussian Distributions
ℒ(x) = 𝔼_{z~q_φ(z|x)}[log p_θ(x|z)] - KL(q_φ(z|x) || 𝒩(0, I))

**KL term (analytical)**:
KL(q_φ(z|x) || 𝒩(0, I)) = ½ ∑ⱼ (1 + log σⱼ² - μⱼ² - σⱼ²)

### 4. The Reparameterization Trick

#### Problem: Non-Differentiable Sampling
**Issue**: Cannot backpropagate through stochastic sampling z ~ q_φ(z|x)

**Original form**:
𝔼_{q_φ(z|x)}[f(z)] requires Monte Carlo with z ~ q_φ(z|x)

#### Reparameterization Solution
**Key insight**: Express z as deterministic function of parameters and noise

**Gaussian case**:
z = μ_φ(x) + σ_φ(x) ⊙ ε,  where ε ~ 𝒩(0, I)

**General form**:
z = g_φ(ε, x),  where ε ~ p(ε)

#### Theorem 4.1 (Reparameterization Gradient)
For z = g_φ(ε, x) with ε ~ p(ε):
∇_φ 𝔼_{q_φ(z|x)}[f(z)] = 𝔼_{p(ε)}[∇_φ f(g_φ(ε, x))]

**Significance**: Gradients can flow through the sampling process.

### 5. Training Algorithm

#### Algorithm 5.1 (VAE Training)
```
for each batch {x⁽ⁱ⁾} in dataset:
    # Encoder forward pass
    μ, log_σ² = encoder(x⁽ⁱ⁾)
    
    # Reparameterization
    ε ~ N(0, I)
    z = μ + exp(log_σ²/2) ⊙ ε
    
    # Decoder forward pass  
    x_recon = decoder(z)
    
    # Compute ELBO
    recon_loss = -log p(x⁽ⁱ⁾|z)
    kl_loss = KL(q(z|x⁽ⁱ⁾) || p(z))
    loss = recon_loss + kl_loss
    
    # Backpropagation
    loss.backward()
    optimizer.step()
```

#### Loss Terms Interpretation
**Reconstruction loss**: How well decoder reconstructs input
**KL regularization**: Keeps posterior close to prior

**Trade-off**: Perfect reconstruction vs regularized latent space

### 6. Theoretical Analysis

#### Information-Theoretic Interpretation
**ELBO decomposition**:
ℒ = 𝔼[log p(x|z)] - KL(q(z|x) || p(z))
  = -ℒ_recon - ℒ_reg

**Alternative decomposition**:
ℒ = log p(x) - KL(q(z|x) || p(z|x))
  = log p(x) - I_q(x; z) + H_q(z)

where I_q(x; z) is mutual information under q.

#### Rate-Distortion Perspective
**Connection**: VAE optimizes rate-distortion trade-off
- **Rate**: KL(q(z|x) || p(z)) (information cost)
- **Distortion**: -𝔼[log p(x|z)] (reconstruction error)

#### Posterior Collapse
**Problem**: q(z|x) ≈ p(z) (posterior ignores x)
**Causes**: Strong decoder, weak encoder, optimization dynamics
**Symptoms**: KL term → 0, poor latent representations

**Solutions**:
- β-VAE: Weight KL term differently
- Cyclical annealing: Gradually increase KL weight
- Skip connections: Stronger encoder

### 7. Latent Space Properties

#### Continuity and Interpolation
**Smoothness**: Small changes in z produce smooth changes in x
**Interpolation**: Linear interpolation in latent space
z_interp = (1-α)z₁ + αz₂

#### Arithmetic in Latent Space
**Vector arithmetic**: z_king - z_man + z_woman ≈ z_queen
**Concept vectors**: Direction vectors for semantic concepts

#### Latent Space Topology
**Manifold structure**: Data lies on lower-dimensional manifold
**Holes**: Regions with no corresponding data
**Mode coverage**: How well latent space covers data modes

### 8. Architecture Considerations

#### Encoder Design
**CNN encoder** (for images):
```
x → Conv → ReLU → Conv → ReLU → Flatten → Dense → [μ, log_σ²]
```

**Fully connected** (for tabular):
```
x → Dense → ReLU → Dense → ReLU → [μ, log_σ²]
```

#### Decoder Design
**Symmetry**: Often mirror of encoder
**Upsampling**: ConvTranspose2d, PixelShuffle
**Output activation**: Sigmoid for Bernoulli, none for Gaussian

#### Latent Dimension Choice
**Trade-offs**: 
- Too small: Underfitting, information bottleneck
- Too large: Overfitting, posterior collapse
- Typical: 2-512 dimensions depending on data complexity

### 9. Extensions and Variants

#### Conditional VAE (CVAE)
**Modification**: Condition on labels or other information
- Encoder: q_φ(z|x, c)
- Decoder: p_θ(x|z, c)

**Applications**: Class-conditional generation, semi-supervised learning

#### β-VAE
**Motivation**: Control disentanglement vs reconstruction trade-off
**Objective**: ℒ_β = 𝔼[log p(x|z)] - β · KL(q(z|x) || p(z))

**Effects**:
- β > 1: More disentangled representations
- β < 1: Better reconstruction quality

#### Importance Weighted VAE (IWAE)
**Idea**: Tighter lower bound using importance sampling
**Bound**: ℒ_IW = 𝔼[log (1/K) ∑ₖ wₖ]

where wₖ = p(x|zₖ)p(zₖ)/q(zₖ|x)

**Property**: ℒ_IW ≥ ℒ_VAE with equality when K=1

### 10. Practical Considerations

#### Optimization Challenges
**KL vanishing**: KL term becomes zero too quickly
**Balancing**: Reconstruction vs regularization
**Mode collapse**: Decoder becomes too powerful

#### Hyperparameter Tuning
**Learning rate**: Often need different rates for encoder/decoder
**β parameter**: Critical for β-VAE performance
**Architecture depth**: Affects expressiveness and training stability

#### Evaluation Metrics
**Likelihood**: ELBO as proxy (but not exact likelihood)
**Reconstruction quality**: MSE, SSIM for images
**Sample quality**: FID, IS for generated samples
**Disentanglement**: β-VAE metric, MIG, SAP

### 11. Applications

#### Generative Modeling
**Unconditional generation**: Sample z ~ p(z), decode x = decoder(z)
**Conditional generation**: Control generation with labels/attributes
**Interpolation**: Smooth transitions between data points

#### Representation Learning
**Dimensionality reduction**: Encoder as feature extractor
**Clustering**: Latent space clustering
**Anomaly detection**: Reconstruction error for outliers

#### Data Augmentation
**Synthetic data**: Generate new training examples
**Adversarial examples**: Perturb in latent space
**Missing data imputation**: Encode partial data, decode complete

### 12. Connections to Other Models

#### Relationship to PCA
**Linear VAE**: With linear encoder/decoder, recovers PCA
**Nonlinear extension**: VAE as nonlinear dimensionality reduction

#### Connection to Autoencoders
**Deterministic**: Standard autoencoder has deterministic bottleneck
**Stochastic**: VAE introduces stochasticity for better generalization

#### Link to GANs
**Complementary**: VAE has tractable training, GAN has better samples
**Hybrid models**: VAE-GAN combines both approaches

### 13. Modern Developments

#### Vector Quantized VAE (VQ-VAE)
**Discrete latents**: Replace continuous z with discrete codes
**Benefits**: More interpretable, better for sequential data
**Challenges**: Non-differentiable quantization

#### Hierarchical VAE
**Multi-scale**: Multiple levels of latent variables
**Architecture**: z₁ → z₂ → ... → x
**Benefits**: Better modeling of complex dependencies

#### Normalizing Flow VAE
**Flexible posterior**: Use normalizing flows for q(z|x)
**Benefits**: More expressive posterior approximation
**Cost**: Increased computational complexity

### 14. Evaluation and Metrics

#### Likelihood-Based Metrics
**ELBO**: Lower bound on log-likelihood
**Importance Weighted Bound**: Tighter approximation
**Annealed Importance Sampling**: Better likelihood estimates

#### Sample Quality Metrics
**Inception Score (IS)**: Quality and diversity of samples
**Fréchet Inception Distance (FID)**: Distance between real/fake distributions
**Precision/Recall**: Coverage vs quality trade-off

#### Disentanglement Metrics
**β-VAE metric**: Mutual information between factors and latents
**MIG (Mutual Information Gap)**: Modularity of representations
**SAP (Separated Attribute Predictability)**: Predictability of factors

### 15. Theoretical Limitations

#### Approximation Gap
**Issue**: ELBO is only a lower bound
**Consequence**: May underestimate true likelihood
**Solutions**: Tighter bounds (IWAE), better posteriors

#### Posterior Collapse
**Fundamental issue**: Optimization pathologies
**Theory**: Connection to information bottleneck
**Open problem**: Guaranteed avoidance methods

#### Expressiveness Limits
**Decoder limitations**: Limited by neural network capacity
**Prior mismatch**: Simple prior may not match true posterior
**Research direction**: More flexible priors and posteriors

## Implementation Details

See `exercise.py` for implementations of:
1. Standard VAE with Gaussian encoder/decoder
2. β-VAE with controllable regularization
3. Conditional VAE for class-conditional generation
4. Evaluation metrics (ELBO, reconstruction quality)
5. Latent space visualization and interpolation
6. Training loop with proper loss balancing

## Experiments

1. **β Parameter Study**: Effect of β on reconstruction vs disentanglement
2. **Latent Dimension**: How latent size affects performance
3. **Architecture Comparison**: Different encoder/decoder designs
4. **Dataset Scaling**: Performance on datasets of varying complexity
5. **Interpolation Quality**: Smoothness of latent space transitions

## Research Connections

### Foundational Papers
1. Kingma & Welling (2014) - "Auto-Encoding Variational Bayes"
2. Rezende et al. (2014) - "Stochastic Backpropagation and Approximate Inference"
3. Higgins et al. (2017) - "β-VAE: Learning Basic Visual Concepts with a Constrained VAE"
4. Burda et al. (2016) - "Importance Weighted Autoencoders"

### Theoretical Developments
1. Alemi et al. (2018) - "Fixing a Broken ELBO"
2. Lucas et al. (2019) - "Don't Blame the ELBO! A Linear VAE Perspective"
3. Dai & Wipf (2019) - "Diagnosing and Enhancing VAE Models"

### Modern Applications
1. Van Den Oord et al. (2017) - "Neural Discrete Representation Learning" (VQ-VAE)
2. Razavi et al. (2019) - "Generating Diverse High-Fidelity Images with VQ-VAE-2"
3. Child (2021) - "Very Deep VAEs Generalize Autoregressive Models"

## Resources

### Primary Sources
1. **Kingma & Welling (2014)** - Original VAE paper
2. **Blei, Kucukelbir & McAuliffe (2017) - "Variational Inference: A Review"**
   - Broader context of variational methods
3. **Doersch (2016) - "Tutorial on Variational Autoencoders"**
   - Accessible introduction

### Video Resources
1. **Stanford CS236 - Variational Autoencoders**
   - Stefano Ermon's generative models course
2. **MIT 6.S191 - Deep Generative Modeling**
   - Practical implementation focus
3. **Two Minute Papers - VAE Explained**
   - Visual intuitions

### Software Resources
1. **PyTorch VAE Examples**: Official tutorials and implementations
2. **Disentanglement Library**: Google Research evaluation framework
3. **β-VAE Implementation**: Reference implementations with hyperparameters

## Socratic Questions

### Understanding
1. Why is the reparameterization trick necessary for training VAEs?
2. How does the choice of prior p(z) affect the learned representations?
3. What causes posterior collapse and how can it be prevented?

### Extension
1. How would you design a VAE for sequential data like text or audio?
2. Can you derive the ELBO for non-Gaussian distributions?
3. What happens to VAE training in the limit of infinite data?

### Research
1. What are the fundamental limitations of amortized variational inference?
2. How can we design better priors that adapt to the data?
3. What new architectures might replace the standard encoder-decoder design?

## Exercises

### Theoretical
1. Derive the ELBO for Bernoulli observations with Gaussian latents
2. Prove that IWAE provides a tighter bound than standard VAE
3. Analyze the effect of β on the information-theoretic properties

### Implementation
1. Build a VAE from scratch for MNIST digit generation
2. Implement β-VAE and study the β parameter effect
3. Create latent space visualization and interpolation tools
4. Compare VAE with standard autoencoder on same dataset

### Research
1. Study the relationship between latent dimension and disentanglement
2. Investigate novel architectures for better posterior approximation
3. Explore VAE applications to your domain of interest

## Advanced Topics

### Amortized Variational Inference
- **Amortization gap**: Difference between amortized and optimal inference
- **Meta-learning**: Learning to do inference across tasks
- **Iterative refinement**: Improving posterior approximation

### Disentangled Representation Learning
- **Causal factors**: Learning representations of causal variables
- **Identifiability**: When can we recover true factors?
- **Evaluation**: How to measure disentanglement objectively

### VAEs for Structured Data
- **Graph VAE**: Variational autoencoders for graph data
- **Molecular VAE**: Chemical compound generation and optimization
- **Music VAE**: Hierarchical generation of musical sequences