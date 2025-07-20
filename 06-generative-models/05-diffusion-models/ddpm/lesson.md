# Denoising Diffusion Probabilistic Models: State-of-the-Art Generation

## Prerequisites
- Stochastic processes and Markov chains
- Variational inference and ELBO
- Score-based models and Langevin dynamics
- Neural network architectures (U-Net, attention)
- Advanced probability theory (SDEs, Fokker-Planck equations)

## Learning Objectives
- Master the mathematical foundations of diffusion processes for generation
- Understand forward and reverse diffusion processes
- Implement DDPM from scratch with proper training and sampling
- Connect diffusion models to score-based generative models
- Analyze the theoretical properties and state-of-the-art performance

## Mathematical Foundations

### 1. The Diffusion Process Framework

#### Forward Diffusion Process
**Definition**: A Markov chain that gradually adds noise to data

q(x₁:T | x₀) = ∏ᵗ₌₁ᵀ q(xₜ | xₜ₋₁)

where q(xₜ | xₜ₋₁) = 𝒩(xₜ; √(1-βₜ)xₜ₋₁, βₜI)

**Properties**:
- βₜ ∈ (0,1): Noise schedule
- x₀: Original data
- xₜ: Noisy version at step T
- As T → ∞: xₜ → 𝒩(0,I)

#### Closed-Form Forward Process
**Key insight**: Can sample xₜ directly from x₀

Define αₜ = 1 - βₜ, ᾱₜ = ∏ₛ₌₁ᵗ αₛ

**Theorem 1.1**: q(xₜ | x₀) = 𝒩(xₜ; √ᾱₜ x₀, (1-ᾱₜ)I)

**Proof**: By induction on the Markov property and Gaussian closure. □

**Reparameterization**: xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε, where ε ~ 𝒩(0,I)

### 2. Reverse Diffusion Process

#### The Reverse Process
**Goal**: Learn to reverse the forward process

p_θ(x₀:T) = p(xT) ∏ᵗ₌₁ᵀ p_θ(xₜ₋₁ | xₜ)

where p_θ(xₜ₋₁ | xₜ) = 𝒩(xₜ₋₁; μ_θ(xₜ,t), Σ_θ(xₜ,t))

#### Tractable Reverse Process
**Theorem 2.1**: For small βₜ, the reverse process is approximately Gaussian:

q(xₜ₋₁ | xₜ, x₀) = 𝒩(xₜ₋₁; μ̃ₜ(xₜ, x₀), β̃ₜI)

where:
μ̃ₜ(xₜ, x₀) = (√ᾱₜ₋₁ βₜ)/(1-ᾱₜ) x₀ + (√αₜ (1-ᾱₜ₋₁))/(1-ᾱₜ) xₜ

β̃ₜ = (1-ᾱₜ₋₁)/(1-ᾱₜ) βₜ

### 3. Training Objective Derivation

#### Variational Lower Bound
**Goal**: Maximize log p_θ(x₀)

log p_θ(x₀) ≥ 𝔼_q[log p_θ(x₀:T)/q(x₁:T|x₀)]

**ELBO decomposition**:
ℒ = 𝔼_q[-log p_θ(xT) + ∑ᵗ₌₂ᵀ KL(q(xₜ₋₁|xₜ,x₀) || p_θ(xₜ₋₁|xₜ)) + log p_θ(x₀|x₁)]

#### Simplified Objective
**Key insight**: KL divergence between Gaussians has closed form

**Theorem 3.1**: The training objective can be written as:
ℒ_simple = 𝔼_{t,x₀,ε}[||ε - ε_θ(√ᾱₜ x₀ + √(1-ᾱₜ) ε, t)||²]

where ε_θ predicts the noise added to x₀.

**Significance**: Train a neural network to predict noise!

### 4. DDPM Algorithm

#### Algorithm 4.1 (DDPM Training)
```
repeat:
    x₀ ~ q(x₀)           # Sample data
    t ~ Uniform(1, T)     # Sample timestep
    ε ~ N(0, I)          # Sample noise
    
    # Forward process (reparameterization)
    xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε
    
    # Predict noise
    ε_pred = ε_θ(xₜ, t)
    
    # Compute loss
    loss = ||ε - ε_pred||²
    
    # Update parameters
    θ ← θ - ∇_θ loss
```

#### Algorithm 4.2 (DDPM Sampling)
```
xₜ ~ N(0, I)             # Start from noise

for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else 0
    
    # Predict noise
    ε_pred = ε_θ(xₜ, t)
    
    # Compute mean
    μ = (1/√αₜ)(xₜ - βₜ/√(1-ᾱₜ) ε_pred)
    
    # Sample previous step
    xₜ₋₁ = μ + √β̃ₜ z

return x₀
```

### 5. Connection to Score-Based Models

#### Score Function
**Definition**: ∇_x log p(x)

**Connection**: Noise prediction is related to score:
ε_θ(xₜ, t) = -√(1-ᾱₜ) s_θ(xₜ, t)

where s_θ is the score function.

#### Langevin Dynamics
**Classical**: Sample from p(x) using ∇_x log p(x)

x ← x + η∇_x log p(x) + √(2η) z

where z ~ 𝒩(0,I).

#### Theorem 5.1 (Song et al.)
DDPM sampling is a discretization of reverse-time SDE:
dx = [f(x,t) - g²(t)∇_x log p_t(x)]dt + g(t)dw̄

### 6. Noise Schedules

#### Linear Schedule
βₜ = β₁ + (t-1)/(T-1)(β_T - β₁)

**Parameters**: β₁ = 10⁻⁴, β_T = 0.02, T = 1000

#### Cosine Schedule  
ᾱₜ = cos²(π(t/T + s)/(1 + s))

where s = 0.008 (small offset)

**Benefits**: More noise at beginning, less at end

#### Learned Schedules
**Idea**: Learn βₜ as part of training
**Challenge**: Optimization can be unstable
**Solutions**: Constrain to valid ranges

### 7. Architecture Design

#### U-Net Backbone
**Why U-Net**: 
- Skip connections preserve high-frequency details
- Multi-scale processing
- Proven effectiveness for image-to-image tasks

**Modifications for DDPM**:
- Time embedding: Sinusoidal + MLP
- Group normalization instead of batch norm
- Self-attention at multiple resolutions

#### Time Embedding
**Sinusoidal encoding**:
PE(t, 2i) = sin(t/10000^(2i/d))
PE(t, 2i+1) = cos(t/10000^(2i/d))

**Integration**: Add to each layer via shift and scale

#### Attention Mechanisms
**Self-attention**: At 16×16 and 8×8 resolutions
**Benefit**: Capture long-range dependencies
**Cost**: Quadratic in spatial dimensions

### 8. Advanced Training Techniques

#### Parameterization Variants
**ε-parameterization**: Predict noise (standard)
**x₀-parameterization**: Predict original image
**v-parameterization**: Predict velocity field

**v-parameterization**:
v = αₜ ε - σₜ x₀

where αₜ = √ᾱₜ, σₜ = √(1-ᾱₜ)

#### Improved Training
**Importance sampling**: Weight loss by t
ℒ_weighted = 𝔼[w(t) ||ε - ε_θ(xₜ, t)||²]

**EMA**: Exponential moving average of parameters
θ_ema ← μ θ_ema + (1-μ) θ

### 9. Sampling Improvements

#### DDIM (Denoising Diffusion Implicit Models)
**Key insight**: Use deterministic sampling process

xₜ₋₁ = √ᾱₜ₋₁ (xₜ - √(1-ᾱₜ) ε_θ(xₜ,t))/√ᾱₜ + √(1-ᾱₜ₋₁ - σₜ²) ε_θ(xₜ,t) + σₜ ε

**Benefits**: 
- Faster sampling (fewer steps)
- Interpolation in latent space
- Deterministic process

#### Classifier Guidance
**Idea**: Use pretrained classifier to guide generation

ε̃ = ε_θ(xₜ,t) - √(1-ᾱₜ) ∇_xₜ log p_φ(y|xₜ)

**Benefits**: Better sample quality and controllability
**Cost**: Requires classifier for each class

#### Classifier-Free Guidance
**Innovation**: Avoid need for separate classifier

ε̃ = ε_θ(xₜ,t,∅) + w(ε_θ(xₜ,t,c) - ε_θ(xₜ,t,∅))

where w > 1 is guidance scale, c is condition.

### 10. Theoretical Analysis

#### Sample Quality vs Likelihood
**Observation**: DDPMs generate high-quality samples but have poor likelihood

**Explanation**: 
- Likelihood includes all frequencies
- Human perception focuses on low frequencies
- DDPMs excel at perceptually important features

#### Convergence Analysis
**Theorem 10.1**: Under Lipschitz conditions, DDPM sampling converges to true distribution as T → ∞ and network capacity → ∞.

#### Mode Coverage
**Empirical**: DDPMs show excellent mode coverage
**Theory**: Langevin dynamics is ergodic under regularity conditions

### 11. Extensions and Variants

#### Latent Diffusion Models
**Motivation**: Reduce computational cost
**Approach**: Apply diffusion in latent space of autoencoder

**Architecture**:
1. Train VAE: x ↔ z
2. Train diffusion on z
3. Sample: z ~ DDPM, x = decoder(z)

#### Cascaded Diffusion
**Strategy**: Low-res → High-res pipeline
**Implementation**: 
1. Generate 64×64 image
2. Super-resolve to 256×256
3. Super-resolve to 1024×1024

#### Text-to-Image Diffusion
**DALL-E 2**: CLIP embeddings + diffusion
**Imagen**: T5 text encoder + cascaded diffusion
**Stable Diffusion**: Latent diffusion + CLIP

### 12. Computational Considerations

#### Training Efficiency
**Memory**: Store intermediate activations for gradient checkpointing
**Computation**: O(T) forward passes per training step
**Parallelization**: Batch over t dimension

#### Sampling Efficiency
**Standard**: T=1000 sampling steps
**DDIM**: 50-200 steps with similar quality
**Progressive distillation**: Train few-step models
**Consistency models**: One-step generation

#### Hardware Optimization
**Mixed precision**: FP16 for memory and speed
**Gradient checkpointing**: Trade compute for memory
**Compile optimization**: XLA, TorchScript

### 13. Evaluation and Metrics

#### Sample Quality
**FID**: Fréchet Inception Distance
**IS**: Inception Score  
**Precision/Recall**: Mode coverage vs quality
**CLIP Score**: For text-to-image models

#### Diversity Metrics
**LPIPS**: Learned perceptual distance
**MS-SSIM**: Multi-scale structural similarity
**Intra-class diversity**: Within-class variation

#### Human Evaluation
**Gold standard**: Human preference studies
**Aspects**: Realism, diversity, text alignment
**Challenges**: Expensive, subjective

### 14. Applications and Impact

#### Image Generation
**High-resolution**: Up to 1024×1024 and beyond
**Conditional**: Class-conditional, text-to-image
**Style transfer**: Zero-shot style manipulation

#### Text-to-Image
**DALL-E 2**: OpenAI's system
**Imagen**: Google's text-to-image
**Stable Diffusion**: Open-source alternative
**Midjourney**: Commercial artistic tool

#### Other Domains
**Audio**: WaveGrad for audio synthesis
**Video**: Video diffusion models
**3D**: Point cloud and mesh generation
**Molecular**: Drug discovery applications

#### Scientific Applications
**Medical imaging**: MRI, CT scan synthesis
**Climate modeling**: Weather pattern generation
**Astronomy**: Galaxy and star formation simulation

### 15. Societal Implications

#### Positive Applications
**Creative tools**: Artists and designers
**Data augmentation**: Training data synthesis
**Accessibility**: Image generation for visually impaired
**Education**: Visual learning aids

#### Challenges and Concerns
**Deepfakes**: Realistic fake content
**Copyright**: Training on copyrighted images
**Bias**: Reflecting training data biases
**Computational cost**: Environmental impact

#### Ethical Considerations
**Content filtering**: Preventing harmful generation
**Attribution**: Credit for artistic styles
**Regulation**: Governance frameworks needed
**Digital watermarking**: Identifying synthetic content

## Implementation Details

See `exercise.py` for implementations of:
1. Forward and reverse diffusion processes
2. DDPM training loop with proper loss computation
3. U-Net architecture with time embeddings
4. Various noise schedules (linear, cosine)
5. DDIM sampling for faster generation
6. Evaluation metrics and visualization tools

## Experiments

1. **Noise Schedule Comparison**: Linear vs cosine vs learned schedules
2. **Architecture Ablation**: Effect of attention, skip connections, time embedding
3. **Sampling Methods**: DDPM vs DDIM vs accelerated methods
4. **Guidance Analysis**: Effect of classifier-free guidance strength
5. **Scale Study**: Performance vs model size and training time

## Research Connections

### Foundational Papers
1. Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"
2. Song et al. (2021) - "Denoising Diffusion Implicit Models"
3. Dhariwal & Nichol (2021) - "Diffusion Models Beat GANs on Image Synthesis"
4. Ho & Salimans (2022) - "Classifier-Free Diffusion Guidance"

### Theoretical Foundations
1. Song et al. (2021) - "Score-Based Generative Modeling through SDEs"
2. Anderson (1982) - "Reverse-time diffusion equation models"
3. Sohl-Dickstein et al. (2015) - "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"

### Modern Applications
1. Ramesh et al. (2022) - "Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALL-E 2)
2. Saharia et al. (2022) - "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Imagen)
3. Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)

## Resources

### Primary Sources
1. **Ho et al. (2020)** - Original DDPM paper
2. **Song et al. (2021)** - Score-based perspective
3. **Yang Song's Blog** - "Generative Modeling by Estimating Gradients of the Data Distribution"

### Video Resources
1. **Yannic Kilcher - DDPM Explained**
   - Clear mathematical explanation
2. **Outlier - Diffusion Models Explained**
   - Intuitive understanding
3. **Ari Seff - Diffusion Models from Scratch**
   - Implementation details

### Software Resources
1. **Hugging Face Diffusers**: Production-ready implementations
2. **OpenAI DALL-E 2**: API access
3. **Stability AI**: Open-source Stable Diffusion

## Socratic Questions

### Understanding
1. Why do diffusion models generate higher quality images than GANs for many tasks?
2. How does the noise schedule affect sample quality and training dynamics?
3. What's the connection between diffusion models and score-based generative models?

### Extension
1. How would you design a diffusion model for non-Euclidean data (graphs, manifolds)?
2. Can you derive the continuous-time limit of the diffusion process?
3. What happens to diffusion models in very high dimensions?

### Research
1. How can we make diffusion models sample faster while maintaining quality?
2. What new conditioning mechanisms might improve controllable generation?
3. How do diffusion models compare to other generative models in terms of mode coverage and sample quality?

## Exercises

### Theoretical
1. Derive the forward process closed-form formula
2. Prove the equivalence between DDPM and score-based models
3. Analyze the effect of different noise schedules on training dynamics

### Implementation
1. Build DDPM from scratch for MNIST or CIFAR-10
2. Implement DDIM sampling for faster generation
3. Add classifier-free guidance for conditional generation
4. Create evaluation pipeline with FID and IS metrics

### Research
1. Study the relationship between sample quality and number of diffusion steps
2. Investigate novel architectures beyond U-Net for diffusion models
3. Explore applications of diffusion models to your domain of interest

## Advanced Topics

### Stochastic Differential Equations
- **Continuous formulation**: Diffusion as SDE solution
- **Ito vs Stratonovich**: Different SDE formulations
- **Numerical methods**: Discretization schemes for SDEs

### Optimal Transport Perspective
- **Schrödinger bridge**: Connecting diffusion to optimal transport
- **Flow matching**: Alternative to diffusion for generation
- **Wasserstein geodesics**: Optimal paths between distributions

### Accelerated Sampling
- **Progressive distillation**: Multi-step to few-step models
- **Consistency models**: Direct mapping from noise to data
- **Flow-based acceleration**: Using normalizing flows for speed