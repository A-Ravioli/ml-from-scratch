# Generative Adversarial Networks: Game-Theoretic Deep Learning

## Prerequisites
- Game theory (minimax games, Nash equilibria)
- Deep neural networks and backpropagation
- Probability theory and optimal transport basics
- Optimization theory (gradient descent, convergence analysis)
- Understanding of generative modeling fundamentals

## Learning Objectives
- Master the mathematical foundations of adversarial training
- Understand the minimax game formulation and Nash equilibria
- Implement GANs from scratch with stable training techniques
- Analyze training dynamics, mode collapse, and convergence issues
- Connect GANs to optimal transport and density estimation theory

## Mathematical Foundations

### 1. The Adversarial Principle

#### Two-Player Zero-Sum Game
**Players**:
- **Generator G**: Creates fake data to fool discriminator
- **Discriminator D**: Distinguishes real from fake data

**Objective**: Generator minimizes what discriminator maximizes

#### Game-Theoretic Formulation
**Discriminator objective**: Maximize ability to classify real vs fake
**Generator objective**: Minimize discriminator's ability to detect fakes

**Value function**:
V(G, D) = 𝔼_{x~p_{data}(x)}[log D(x)] + 𝔼_{z~p_z(z)}[log(1 - D(G(z)))]

#### Minimax Problem
min_G max_D V(G, D) = min_G max_D {𝔼_{x~p_{data}}[log D(x)] + 𝔼_{z~p_z}[log(1 - D(G(z)))]}

### 2. Theoretical Analysis

#### Optimal Discriminator
**Theorem 2.1**: For fixed G, the optimal discriminator is:
D*_G(x) = p_{data}(x) / (p_{data}(x) + p_g(x))

where p_g is the generator distribution.

**Proof**: 
Taking functional derivative of V(G, D) w.r.t. D(x):
δV/δD = p_{data}(x)/D(x) - p_g(x)/(1-D(x)) = 0

Solving: D*(x) = p_{data}(x)/(p_{data}(x) + p_g(x)) □

#### Global Optimum
**Theorem 2.2**: The global minimum of the minimax game is achieved when p_g = p_{data}, and the minimum value is -log(4).

**Proof**:
At p_g = p_{data}: D*(x) = 1/2 for all x
V(G, D*) = 𝔼[log(1/2)] + 𝔼[log(1/2)] = -log(4) □

#### Connection to Jensen-Shannon Divergence
**Theorem 2.3**: The generator objective is equivalent to minimizing Jensen-Shannon divergence:
C(G) = -log(4) + 2·JS(p_{data} || p_g)

where JS is Jensen-Shannon divergence.

### 3. Training Algorithm

#### Algorithm 3.1 (Standard GAN Training)
```
for epoch in range(num_epochs):
    for batch in dataloader:
        # Train Discriminator
        real_data = batch
        fake_data = G(sample_noise())
        
        D_real = D(real_data)
        D_fake = D(fake_data.detach())  # Stop gradients to G
        
        D_loss = -[log(D_real) + log(1 - D_fake)]
        D_loss.backward()
        D_optimizer.step()
        
        # Train Generator
        fake_data = G(sample_noise())
        D_fake = D(fake_data)
        
        G_loss = -log(D_fake)  # or log(1 - D_fake)
        G_loss.backward()
        G_optimizer.step()
```

#### Training Challenges
**Mode collapse**: Generator produces limited variety
**Training instability**: Oscillations, divergence
**Gradient vanishing**: When discriminator becomes too good
**Hyperparameter sensitivity**: Learning rates, architectures

### 4. Architecture Design

#### Generator Architecture
**Input**: Random noise z ∼ p(z) (typically Gaussian)
**Output**: Generated data x̂ = G(z)

**Common design** (for images):
```
z → Dense → Reshape → ConvTranspose → BatchNorm → ReLU → 
    ConvTranspose → BatchNorm → ReLU → ConvTranspose → Tanh
```

#### Discriminator Architecture
**Input**: Real or generated data x
**Output**: Probability that x is real

**Common design**:
```
x → Conv → LeakyReLU → Conv → BatchNorm → LeakyReLU → 
    Conv → BatchNorm → LeakyReLU → Flatten → Dense → Sigmoid
```

#### Architectural Guidelines
**Generator**:
- Use transposed convolutions for upsampling
- Batch normalization helps (except output layer)
- ReLU activation in hidden layers, Tanh for output

**Discriminator**:
- Use strided convolutions for downsampling
- Batch normalization helps (except first layer)  
- LeakyReLU activation throughout
- No batch norm in output layer

### 5. Loss Functions and Variants

#### Original GAN Loss
**Generator**: min_G 𝔼_{z~p_z}[log(1 - D(G(z)))]
**Discriminator**: max_D 𝔼_{x~p_{data}}[log D(x)] + 𝔼_{z~p_z}[log(1 - D(G(z)))]

**Issue**: Generator gradients vanish when D is optimal

#### Alternative Generator Loss
**Non-saturating**: min_G -𝔼_{z~p_z}[log D(G(z))]

**Advantage**: Provides stronger gradients when D(G(z)) ≈ 0
**Standard practice**: Use this instead of original formulation

#### Least Squares GAN (LSGAN)
**Discriminator**: min_D ½𝔼_{x~p_{data}}[(D(x) - 1)²] + ½𝔼_{z~p_z}[D(G(z))²]
**Generator**: min_G ½𝔼_{z~p_z}[(D(G(z)) - 1)²]

**Benefits**: More stable training, better gradient behavior

### 6. Training Dynamics Analysis

#### Nash Equilibrium
**Definition**: (G*, D*) is Nash equilibrium if:
- G* = argmin_G V(G, D*)
- D* = argmax_D V(G*, D)

**Challenge**: Finding Nash equilibria is computationally hard

#### Convergence Analysis
**Theorem 6.1**: Under ideal conditions (convex-concave game), simultaneous gradient descent converges to Nash equilibrium.

**Reality**: GANs are non-convex, convergence not guaranteed

#### Training Pathologies
**Mode collapse**: Generator maps multiple z to same x
**Training instability**: Loss oscillations, divergence
**Gradient explosion/vanishing**: Poor gradient flow

### 7. Stabilization Techniques

#### Spectral Normalization
**Idea**: Control Lipschitz constant of discriminator
**Method**: Normalize weights by spectral norm

W_SN = W / σ(W)

where σ(W) is largest singular value.

#### Progressive Growing
**Strategy**: Start with low resolution, gradually increase
**Benefits**: More stable training, higher quality results
**Implementation**: Add layers progressively during training

#### Self-Attention
**Motivation**: Capture long-range dependencies
**Mechanism**: Attention mechanism in generator/discriminator
**SAGAN**: Self-Attention GAN for better image generation

### 8. Evaluation Metrics

#### Inception Score (IS)
**Definition**: IS = exp(𝔼_x[KL(p(y|x) || p(y))])

where p(y|x) is classifier output, p(y) is marginal.

**Measures**: Quality (sharp p(y|x)) and diversity (uniform p(y))
**Higher is better**: Typical range 1-10 for natural images

#### Fréchet Inception Distance (FID)
**Definition**: FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r Σ_g))

where (μ_r, Σ_r) and (μ_g, Σ_g) are moments of real and generated features.

**Lower is better**: Measures distance between real and generated distributions

#### Precision and Recall
**Precision**: Fraction of generated samples that are realistic
**Recall**: Fraction of real distribution covered by generator
**Trade-off**: High precision vs high recall

### 9. Mode Collapse Analysis

#### Types of Mode Collapse
**Complete collapse**: All z map to single x
**Partial collapse**: Limited diversity in outputs
**Sequential collapse**: Cycling through limited modes

#### Mathematical Understanding
**Reverse KL**: Generator minimizes KL(p_g || p_{data})
**Issue**: Heavily penalizes mass where p_{data} = 0
**Consequence**: Prefers to cover fewer modes well

#### Mitigation Strategies
**Unrolled GANs**: Consider future discriminator updates
**Minibatch discrimination**: Encourage diversity in batches
**Mode regularization**: Explicit diversity penalties

### 10. Theoretical Connections

#### Optimal Transport
**Wasserstein distance**: Alternative to JS divergence
**Earth Mover's distance**: Cost of transforming one distribution to another
**Connection**: WGAN uses Wasserstein distance as objective

#### f-divergences
**General framework**: JS divergence is special case
**f-GAN**: Generalize GAN to any f-divergence
**Examples**: KL, Pearson χ², Total Variation

#### Density Estimation
**Implicit models**: GANs define distribution implicitly
**vs Explicit**: VAEs, flows define likelihood explicitly
**Trade-off**: Sample quality vs likelihood computation

### 11. Advanced GAN Variants

#### Conditional GAN (cGAN)
**Modification**: Condition on additional information
- G(z, c): Generator takes noise and condition
- D(x, c): Discriminator sees data and condition

**Applications**: Class-conditional generation, image-to-image translation

#### CycleGAN
**Problem**: Unpaired image-to-image translation
**Innovation**: Cycle consistency loss
**Objective**: G_{X→Y}, G_{Y→X} such that G_{Y→X}(G_{X→Y}(x)) ≈ x

#### StyleGAN
**Innovation**: Style-based generator architecture
**Key ideas**: 
- Mapping network from z to w
- Adaptive instance normalization (AdaIN)
- Progressive growing

### 12. Practical Implementation

#### Hyperparameter Guidelines
**Learning rates**: Often different for G and D (e.g., lr_G = 0.0002, lr_D = 0.0002)
**Batch size**: Larger often better (32-128 typical)
**Noise dimension**: 100-512 typical
**Training ratio**: Sometimes train D multiple times per G update

#### Initialization
**Weights**: Normal(0, 0.02) or Xavier/He initialization
**Biases**: Usually zero
**Batch norm**: γ=1, β=0

#### Monitoring Training
**Loss curves**: Both losses should decrease over time
**Visual inspection**: Generated samples quality
**Metric tracking**: IS, FID during training

### 13. Common Failure Modes

#### Discriminator Wins
**Symptom**: D loss → 0, G loss explodes
**Cause**: D becomes too powerful too quickly
**Solution**: Reduce D learning rate, add noise to real data

#### Generator Wins
**Symptom**: G loss → 0, D loss explodes  
**Cause**: G fools D too easily
**Solution**: Increase D capacity, improve D architecture

#### Mode Collapse
**Symptom**: Limited diversity in generated samples
**Detection**: Low recall, cycling through few modes
**Solutions**: Unrolled GANs, minibatch features, WGAN

#### Training Instability
**Symptom**: Oscillating losses, poor convergence
**Causes**: Learning rate mismatch, architecture issues
**Solutions**: Spectral normalization, progressive training

### 14. Recent Advances

#### BigGAN
**Scaling**: Very large batch sizes and model capacity
**Techniques**: Self-attention, spectral normalization, truncation trick
**Results**: State-of-the-art image generation quality

#### Progressive GAN
**Strategy**: Grow network progressively during training
**Benefits**: Stable training, high-resolution results
**Architecture**: Careful layer addition and blending

#### StyleGAN Series
**StyleGAN**: Style-based generation with high-quality results
**StyleGAN2**: Improved architecture and training
**StyleGAN3**: Translation and rotation equivariance

### 15. Applications and Impact

#### Image Generation
**High-resolution**: PhotoRealistic face generation
**Art and creativity**: Style transfer, artistic generation
**Data augmentation**: Synthetic training data

#### Image-to-Image Translation
**Paired**: pix2pix for supervised translation
**Unpaired**: CycleGAN for unsupervised translation
**Applications**: Colorization, super-resolution, domain adaptation

#### Beyond Images
**Text generation**: SeqGAN, variants for sequences
**Audio synthesis**: WaveGAN for audio generation
**Video generation**: Temporal consistency challenges

#### Scientific Applications
**Drug discovery**: Molecular generation
**Materials science**: Crystal structure generation
**Climate modeling**: Weather pattern simulation

## Implementation Details

See `exercise.py` for implementations of:
1. Standard GAN with DCGAN architecture
2. Training loop with proper loss computation
3. Various loss functions (standard, non-saturating, LSGAN)
4. Evaluation metrics (IS, FID) computation
5. Visualization tools for training monitoring
6. Techniques for avoiding common failure modes

## Experiments

1. **Architecture Study**: Compare different G/D architectures
2. **Loss Function Comparison**: Standard vs non-saturating vs LSGAN
3. **Training Dynamics**: Effect of learning rate ratios
4. **Mode Collapse Investigation**: Conditions that cause/prevent collapse
5. **Evaluation Metrics**: Correlation between different metrics

## Research Connections

### Foundational Papers
1. Goodfellow et al. (2014) - "Generative Adversarial Nets"
2. Radford et al. (2016) - "Unsupervised Representation Learning with DCGANs"
3. Arjovsky et al. (2017) - "Wasserstein GAN"
4. Miyato et al. (2018) - "Spectral Normalization for GANs"

### Theoretical Analysis
1. Arjovsky & Bottou (2017) - "Towards Principled Methods for Training GANs"
2. Mescheder et al. (2018) - "The Numerics of GANs"
3. Nagarajan & Kolter (2017) - "Gradient Descent GAN Optimization is Locally Stable"

### Modern Developments
1. Karras et al. (2018) - "Progressive Growing of GANs"
2. Karras et al. (2019) - "StyleGAN"
3. Brock et al. (2019) - "Large Scale GAN Training for High Fidelity Natural Image Synthesis" (BigGAN)

## Resources

### Primary Sources
1. **Goodfellow et al. (2014)** - Original GAN paper
2. **Goodfellow (2016) - "NIPS 2016 Tutorial: Generative Adversarial Networks"**
   - Comprehensive tutorial by the inventor
3. **Salimans et al. (2016) - "Improved Techniques for Training GANs"**
   - Practical training improvements

### Video Resources
1. **Stanford CS236 - Generative Adversarial Networks**
   - Stefano Ermon's course on generative models
2. **Ian Goodfellow - GANs Tutorial (NIPS 2016)**
   - Creator's explanation of the method
3. **Two Minute Papers - GAN Progress**
   - Visual overview of GAN developments

### Software Resources
1. **PyTorch GAN Examples**: Official implementations
2. **TensorFlow GAN Library**: Google's GAN toolkit
3. **Papers with Code - GANs**: Implementations of latest papers

## Socratic Questions

### Understanding
1. Why do GANs often produce sharper images than VAEs?
2. How does the choice of generator architecture affect the types of images produced?
3. What role does the discriminator play beyond just classification?

### Extension
1. How would you design a GAN for sequential data like text or music?
2. Can you prove conditions under which GAN training is guaranteed to converge?
3. What happens to GAN dynamics in the limit of infinite discriminator capacity?

### Research
1. How can we design objective functions that avoid mode collapse entirely?
2. What new architectures might replace the current generator-discriminator paradigm?
3. How do GANs relate to other areas of AI like reinforcement learning or causal inference?

## Exercises

### Theoretical
1. Derive the optimal discriminator for the GAN objective
2. Prove the connection between GAN training and Jensen-Shannon divergence
3. Analyze the effect of different f-divergences on training dynamics

### Implementation
1. Build a DCGAN from scratch for MNIST generation
2. Implement training monitoring and early stopping
3. Compare different loss functions on the same dataset
4. Create tools for detecting mode collapse during training

### Research
1. Study the effect of architecture choices on training stability
2. Investigate novel techniques for preventing mode collapse
3. Explore GAN applications in your domain of interest

## Advanced Topics

### Game-Theoretic Analysis
- **Stackelberg games**: Leader-follower dynamics in GAN training
- **Evolutionary dynamics**: Population-based training approaches
- **Multi-agent learning**: Connections to multi-agent reinforcement learning

### Differential Privacy
- **Private GANs**: Generating data while preserving privacy
- **Differential privacy**: Formal privacy guarantees
- **Trade-offs**: Privacy vs utility in synthetic data

### Adversarial Robustness
- **Connection to adversarial examples**: Shared mathematical foundations
- **Robust training**: GANs that are robust to input perturbations
- **Certified defenses**: Provable robustness guarantees

### Causal GANs
- **Causal generation**: Generating data that respects causal structure
- **Interventional data**: Simulating interventions and counterfactuals
- **Structural equation models**: Combining causality with deep generation