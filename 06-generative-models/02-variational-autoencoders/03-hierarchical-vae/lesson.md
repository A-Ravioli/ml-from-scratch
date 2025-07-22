# Hierarchical Variational Autoencoders

## Prerequisites
- Understanding of vanilla VAEs and β-VAE
- Knowledge of hierarchical Bayesian models
- Familiarity with ladder networks and skip connections
- Information theory and mutual information

## Learning Objectives
- Master hierarchical latent variable models
- Understand multi-scale representation learning
- Implement Ladder VAEs and other hierarchical architectures
- Analyze the theoretical benefits of hierarchical modeling
- Connect to modern hierarchical generative models

## Mathematical Foundations

### 1. Hierarchical Latent Variable Models

#### Definition 1.1 (Hierarchical VAE)
A hierarchical VAE extends the standard VAE with multiple levels of latent variables:

p(x, z₁, z₂, ..., zₗ) = p(x | z₁)∏ᵢ₌₁ᴸ p(zᵢ | zᵢ₊₁)p(zₗ)

where z₁ is closest to observations and zₗ is the highest-level representation.

#### Top-Down Generation
**Generative process**: Sample from top level and condition downward
1. Sample zₗ ~ p(zₗ)
2. For i = L-1 down to 1: Sample zᵢ ~ p(zᵢ | zᵢ₊₁)
3. Generate x ~ p(x | z₁)

#### Bottom-Up Inference  
**Inference process**: Extract features from bottom up
q(z₁, ..., zₗ | x) = q(zₗ | x)∏ᵢ₌₁ᴸ⁻¹ q(zᵢ | zᵢ₊₁, x)

### 2. Ladder Variational Autoencoders

#### Architecture Design
**Key innovation**: Bidirectional information flow
- **Bottom-up path**: Extract increasingly abstract features
- **Top-down path**: Generate from abstract to concrete
- **Skip connections**: Direct information flow between levels

#### Mathematical Formulation
For Ladder VAE with L levels:

**Encoder (Bottom-up)**:
hᵢᵇᵘ = fᵢᵇᵘ(hᵢ₋₁ᵇᵘ, x)  for i = 1, ..., L

**Prior network**: 
μᵢᵖʳⁱᵒʳ, σᵢᵖʳⁱᵒʳ = fᵢᵖʳⁱᵒʳ(hᵢ₊₁ᵗᵈ)

**Inference network**:
μᵢⁱⁿᶠ, σᵢⁱⁿᶠ = fᵢⁱⁿᶠ(hᵢᵇᵘ, hᵢ₊₁ᵗᵈ)

**Decoder (Top-down)**:
hᵢᵗᵈ = fᵢᵗᵈ(hᵢ₊₁ᵗᵈ, zᵢ)

#### ELBO Decomposition
The ELBO decomposes across hierarchy levels:

ℒ = 𝔼q[log p(x|z₁)] - ∑ᵢ₌₁ᴸ KL[q(zᵢ|z₁:ᵢ₊₁, x) || p(zᵢ|zᵢ₊₁)]

**Benefits**:
- More stable training than deep VAEs
- Better posterior approximation
- Hierarchical feature learning

### 3. Multi-Scale Feature Learning

#### Scale-Specific Representations
Different hierarchy levels capture different scales:
- **Low levels**: Local patterns, textures
- **Mid levels**: Object parts, spatial arrangements  
- **High levels**: Global structure, semantic content

#### Information Processing Theory
**Theorem 3.1 (Hierarchical Information Processing)**: 
In optimal hierarchical models, each level processes information at its natural scale, minimizing redundancy between levels.

**Proof sketch**: Information-theoretic analysis shows that minimizing total description length leads to scale-specific representations.

#### Mutual Information Analysis
For hierarchical representations z₁, ..., zₗ:
- I(zᵢ; x) decreases with hierarchy level i
- I(zᵢ; zⱼ) decreases with |i-j| (locality)
- Total mutual information I(z₁:ₗ; x) increases with depth L

### 4. Advanced Architectures

#### Very Deep VAEs (VDVAE)
**Innovation**: Residual connections in hierarchical VAE
- **Residual blocks**: Enable very deep hierarchies (30+ levels)
- **Squeeze-and-excite**: Adaptive channel attention
- **Progressive training**: Start shallow, gradually increase depth

**Architecture**:
```
x → ResBlock → ... → ResBlock → z₁
    ↓           ↓         ↓
    ResBlock → ... → ResBlock → z₂
         ↓         ↓         ↓
         ResBlock → ... → ResBlock → zₗ
```

#### Normalizing Flows + Hierarchical VAE
**Idea**: Replace simple posteriors with normalizing flows
q(zᵢ | z₁:ᵢ₊₁, x) = q₀(z₀)∏ⱼ₌₁ᵏ |det ∂fⱼ/∂zⱼ₋₁|⁻¹

**Benefits**:
- More flexible posterior approximation
- Better fit to true posterior
- Improved generation quality

#### Bidirectional Inference Networks (BiGAN + Hierarchical)
**Extension**: Learn inference and generation jointly
- **Generator**: G: Z → X with hierarchical Z
- **Encoder**: E: X → Z with hierarchical inference
- **Discriminator**: D: (X,Z) → {0,1}

### 5. Training Dynamics

#### Posterior Collapse Prevention
**Challenge**: Higher levels may be ignored during training
**Solutions**:
1. **β-scheduling**: Gradually increase KL weight
2. **Free bits**: Minimum KL divergence per dimension
3. **Skip connections**: Direct information flow
4. **Spectral normalization**: Stabilize training dynamics

#### Balancing Hierarchy Levels
**Objective**: Ensure all levels are utilized
**Metrics**:
- KL divergence per level: KL[q(zᵢ|x) || p(zᵢ)]
- Mutual information: I(zᵢ; x)
- Reconstruction contribution per level

**Algorithm 5.1 (Hierarchical β-VAE Training)**:
```
for epoch in range(num_epochs):
    for level i in range(L):
        βᵢ = schedule_beta(epoch, level=i)
        KL_i = KL[q(zᵢ|x) || p(zᵢ|zᵢ₊₁)]
        loss += βᵢ * KL_i
```

### 6. Theoretical Analysis

#### Expressivity of Hierarchical Models
**Theorem 6.1 (Universal Approximation)**: 
Hierarchical VAEs with sufficient depth and width can approximate any hierarchical distribution p(x, z₁:ₗ).

**Proof**: Extension of universal approximation theorem to hierarchical case using composition of universal approximators.

#### Sample Complexity
**Theorem 6.2**: 
Hierarchical models require O(log d) fewer samples than flat models for d-dimensional hierarchical data.

**Intuition**: Hierarchical inductive bias matches natural data structure.

#### Posterior Contraction
**Analysis**: How quickly does posterior concentrate around true parameters?
- **Flat VAE**: O(n⁻¹/²) convergence rate
- **Hierarchical VAE**: O(n⁻¹/²⁺ᵋ) for hierarchical data (faster)

### 7. Advanced Techniques

#### Hierarchical Normalizing Flows
**Innovation**: Flow-based posteriors at each hierarchy level
q(zᵢ | z₁:ᵢ₊₁, x) = q₀(zᵢ⁽⁰⁾)∏ⱼ₌₁ᴷ |det J_fⱼ|⁻¹

**Benefits**:
- Exact likelihood computation per level
- More flexible posterior approximation
- Better mode coverage

#### Hierarchical Attention Mechanisms
**Idea**: Attention between hierarchy levels
- **Cross-level attention**: zᵢ attends to zⱼ for j ≠ i
- **Temporal attention**: For sequential hierarchical data
- **Spatial attention**: For image hierarchies

#### Neural ODE Hierarchies
**Extension**: Continuous hierarchy levels
dz(t)/dt = f(z(t), t, θ)

**Interpretation**: Hierarchy level as continuous time
**Benefits**: 
- Adaptive depth
- Memory efficient
- Smoother transitions

### 8. Applications

#### Multi-Resolution Image Generation
**Task**: Generate images at multiple scales simultaneously
**Architecture**: 
- Level 1: 4×4 global structure
- Level 2: 16×16 object layout
- Level 3: 64×64 details and textures

#### Hierarchical Text Modeling
**Task**: Model documents with paragraph/sentence/word hierarchy
**Levels**:
- Document level: Topic and genre
- Paragraph level: Subtopics and flow
- Sentence level: Syntax and semantics
- Word level: Local dependencies

#### Video Generation with Temporal Hierarchies
**Task**: Generate videos with multi-timescale dynamics
**Hierarchy**:
- Slow variables: Scene, lighting, camera
- Medium variables: Object motion, interactions
- Fast variables: Appearance details, noise

### 9. Evaluation Metrics

#### Hierarchical Evaluation
**Disentanglement metrics per level**:
- SAP (Separated Attribute Predictability)
- MIG (Mutual Information Gap)
- DCI (Disentanglement, Completeness, Informativeness)

**Cross-level evaluation**:
- Information flow between levels
- Redundancy across hierarchy
- Reconstruction quality per level

#### Hierarchical Interpolation
**Test**: Interpolate at different hierarchy levels
- High-level interpolation: Semantic changes
- Low-level interpolation: Surface appearance
- Cross-level consistency: Semantic-appearance alignment

### 10. Modern Developments

#### VQ-VAE-2 and Hierarchical VQ
**Innovation**: Hierarchical vector quantization
- **Bottom level**: Local patterns (high resolution)
- **Top level**: Global structure (low resolution)
- **Attention**: Cross-level attention for coherence

#### Hierarchical Diffusion Models
**Connection**: Diffusion process at multiple scales
- **Cascaded diffusion**: Generate coarse-to-fine
- **Latent hierarchical diffusion**: Diffusion in hierarchical latent space

#### Foundation Model Hierarchies
**Modern trend**: Hierarchical representations in large models
- **Language models**: Token/phrase/sentence/document hierarchy
- **Vision transformers**: Patch/region/image hierarchy
- **Multimodal models**: Modality/concept/instance hierarchy

## Implementation Details

See `exercise.py` for implementations of:
1. Basic Ladder VAE architecture
2. Very Deep VAE (VDVAE) with residual connections  
3. Hierarchical β-VAE with level-specific β scheduling
4. Multi-scale evaluation metrics
5. Hierarchical interpolation and generation

## Experiments

1. **Depth vs Performance**: How does hierarchy depth affect generation quality?
2. **Information Flow**: Visualize information usage across hierarchy levels
3. **Multi-Scale Generation**: Generate and evaluate at different resolutions
4. **Ablation Studies**: Skip connections, residual blocks, attention mechanisms
5. **Disentanglement**: Compare disentanglement quality across hierarchy levels

## Research Connections

### Foundational Papers
1. Sønderby et al. (2016) - "Ladder Variational Autoencoders"
2. Child (2020) - "Very Deep VAEs Generalize Autoregressive Models"
3. Vahdat & Kautz (2020) - "NVAE: A Deep Hierarchical Variational Autoencoder"

### Modern Developments  
1. Razavi et al. (2019) - "Generating Diverse High-Fidelity Images with VQ-VAE-2"
2. Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
3. Ramesh et al. (2022) - "Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALL-E 2)

### Theoretical Analysis
1. Brekelmans et al. (2019) - "All in the Exponential Family: Bregman Duality in Thermodynamic VAE"
2. Wu et al. (2020) - "On the Mutual Information in Variational Autoencoders"
3. Rezende & Viola (2018) - "Taming VAEs"

## Resources

### Primary Sources
1. **Kingma & Welling (2019) - "An Introduction to Variational Autoencoders"**
   - Comprehensive VAE foundation
2. **Tomczak (2022) - "Deep Generative Modeling"**
   - Modern perspective including hierarchical models
3. **NIPS 2016 Tutorial - "Generative Adversarial Networks"**
   - Covers hierarchical extensions

### Video Resources
1. **DeepMind - "Hierarchical Latent Variable Models"**
   - Technical deep dive
2. **Pieter Abbeel - CS 294-158 Deep Unsupervised Learning**
   - Berkeley course covering hierarchical VAEs
3. **Montreal Deep Learning Summer School - VAE Lectures**
   - Theoretical foundations

### Software Resources
1. **Pyro**: Probabilistic programming with hierarchical models
2. **TensorFlow Probability**: Hierarchical Bayesian modeling
3. **PyTorch Lightning**: Scalable hierarchical VAE implementations

## Socratic Questions

### Understanding
1. Why do hierarchical VAEs suffer less from posterior collapse than deep flat VAEs?
2. How do skip connections in Ladder VAEs affect information flow?
3. What determines the optimal number of hierarchy levels for a given dataset?

### Extension  
1. How would you design a hierarchical VAE for 3D scene understanding?
2. Can you prove convergence guarantees for hierarchical VAE training?
3. How do hierarchical VAEs relate to multiscale wavelet representations?

### Research
1. What are the fundamental limits of hierarchical representation learning?
2. How can hierarchical VAEs be combined with foundation model architectures?
3. What novel applications emerge from hierarchical generative modeling?

## Exercises

### Theoretical
1. Derive the ELBO decomposition for a 3-level hierarchical VAE
2. Prove that Ladder VAE posterior approximation is strictly better than factorized
3. Analyze the information-theoretic properties of hierarchical representations

### Implementation
1. Implement Ladder VAE with bidirectional inference
2. Build Very Deep VAE with 20+ hierarchy levels
3. Create multi-scale image generation with hierarchical control
4. Compare flat vs hierarchical VAE on structured datasets

### Research
1. Design novel hierarchical architectures for your domain of interest
2. Study the emergence of hierarchical representations in large models
3. Investigate connections between hierarchical VAEs and human perception

## Advanced Topics

### Theoretical Connections
- **Hierarchical Bayesian Models**: Connection to classical statistics
- **Renormalization Group Theory**: Multi-scale physics inspiration  
- **Category Theory**: Mathematical foundations of hierarchical structure
- **Information Geometry**: Geometric view of hierarchical inference

### Cutting-Edge Research
- **Neural Architecture Search**: Automatic hierarchy design
- **Meta-Learning**: Learning to learn hierarchical representations
- **Causal Hierarchies**: Hierarchical causal representation learning
- **Quantum Hierarchies**: Quantum extensions of hierarchical models