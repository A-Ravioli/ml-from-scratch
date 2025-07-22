# Vector Quantized Variational Autoencoders (VQ-VAE)

## Prerequisites
- Understanding of vanilla VAEs and variational inference
- Knowledge of discrete latent variable models
- Familiarity with k-means clustering and vector quantization
- Information theory and mutual information

## Learning Objectives
- Master discrete latent variable modeling with VQ-VAE
- Understand vector quantization in deep learning
- Implement VQ-VAE, VQ-VAE-2, and modern variants
- Analyze the benefits of discrete vs continuous representations
- Connect to modern foundation models and tokenization

## Mathematical Foundations

### 1. Vector Quantization Principle

#### Definition 1.1 (Vector Quantization)
Vector quantization maps continuous vectors to discrete codebook entries:

VQ(z) = argmin_{e_k ∈ ℰ} ||z - e_k||²

where ℰ = {e₁, e₂, ..., e_K} is a learnable codebook of K embedding vectors.

#### Key Innovation
**Discrete Latent Space**: Unlike VAEs with continuous latents, VQ-VAE uses discrete latents:
- **Continuous VAE**: z ~ q(z|x) where z ∈ ℝᵈ
- **VQ-VAE**: z ∈ {e₁, e₂, ..., e_K} where each e_k ∈ ℝᵈ

#### Benefits of Discretization
1. **No Posterior Collapse**: Discrete latents cannot collapse to priors
2. **Interpretable Representations**: Each code represents a distinct pattern
3. **Efficient Storage**: Log₂(K) bits per latent instead of floating point
4. **Autoregressive Generation**: Can model p(z) with autoregressive models

### 2. VQ-VAE Architecture

#### Overall Structure
**Encoder**: E: X → ℝᵈ (continuous representation)
**Quantizer**: Q: ℝᵈ → ℰ (discrete codebook lookup)
**Decoder**: D: ℰ → X (reconstruction from discrete codes)

#### Mathematical Formulation
Given input x:
1. Encode: z_e = E(x)
2. Quantize: z_q = VQ(z_e) = e_k where k = argmin_j ||z_e - e_j||²
3. Decode: x̂ = D(z_q)

#### Straight-Through Estimator
**Challenge**: Quantization is non-differentiable
**Solution**: Straight-through gradient estimation

During forward pass: z_q = VQ(z_e)
During backward pass: ∇z_e = ∇z_q (copy gradients)

**Implementation**:
```python
z_q = z_e + (quantized - z_e).detach()
```

### 3. Learning Objective

#### Complete Loss Function
The VQ-VAE loss combines three components:

ℒ = ℒ_recon + ℒ_vq + β·ℒ_commit

#### Reconstruction Loss
ℒ_recon = ||x - D(VQ(E(x)))||²

Standard reconstruction objective.

#### Vector Quantization Loss
ℒ_vq = ||sg(z_e) - z_q||²

where sg(·) is the stop-gradient operator.
**Purpose**: Updates codebook embeddings toward encoder outputs.

#### Commitment Loss
ℒ_commit = ||z_e - sg(z_q)||²

**Purpose**: Encourages encoder outputs to stay close to chosen codebook entries.
**Hyperparameter**: β typically set to 0.25.

#### Intuitive Understanding
- **VQ Loss**: Pulls codebook vectors toward encoder outputs
- **Commitment Loss**: Pulls encoder outputs toward codebook vectors
- **Balance**: Ensures encoder and codebook co-adapt effectively

### 4. Codebook Learning Dynamics

#### Exponential Moving Average (EMA) Updates
**Alternative to gradient-based updates**: Update codebook with EMA:

N_i^{(t)} = γ·N_i^{(t-1)} + (1-γ)·n_i^{(t)}
m_i^{(t)} = γ·m_i^{(t-1)} + (1-γ)·∑_{z_e: VQ(z_e)=e_i} z_e
e_i^{(t)} = m_i^{(t)} / N_i^{(t)}

where:
- N_i: Usage count for code i  
- m_i: Accumulated encoder outputs assigned to code i
- γ: EMA decay factor (typically 0.99)

**Benefits**:
- More stable than gradient updates
- Handles dead codes better
- Less sensitive to learning rate

#### Dead Code Problem
**Issue**: Some codebook entries never get used
**Solutions**:
1. **Random Restart**: Replace unused codes with random encoder outputs
2. **Code Splitting**: Split frequently used codes
3. **Entropy Regularization**: Encourage uniform code usage

#### Codebook Utilization
**Metrics**:
- **Active Codes**: Number of codes used in training
- **Usage Distribution**: Entropy of code usage frequencies
- **Perplexity**: exp(H(p)) where p is usage distribution

### 5. VQ-VAE-2: Hierarchical Vector Quantization

#### Motivation
**Limitation of VQ-VAE**: Single quantization level limits modeling capacity
**Solution**: Multi-level hierarchical quantization

#### Architecture Overview
**Two-level hierarchy**:
- **Bottom Level**: Local patterns, high resolution (32×32 codes)
- **Top Level**: Global structure, low resolution (8×8 codes)

#### Mathematical Formulation
**Bottom Encoder**: E_bottom: X → Z_bottom
**Top Encoder**: E_top: Z_bottom → Z_top  
**Bottom Quantizer**: Q_bottom: Z_bottom → ℰ_bottom
**Top Quantizer**: Q_top: Z_top → ℰ_top
**Decoder**: D: (ℰ_top, ℰ_bottom) → X

#### Hierarchical Generation Process
1. Generate top-level codes: z_top ~ p(z_top)
2. Generate bottom codes conditioned on top: z_bottom ~ p(z_bottom | z_top)
3. Decode: x = D(z_top, z_bottom)

#### Training Objective
ℒ = ℒ_recon + ℒ_vq_top + ℒ_vq_bottom + β(ℒ_commit_top + ℒ_commit_bottom)

**Key Insight**: Each level captures different scales of structure.

### 6. Advanced VQ Techniques

#### Improved Codebook Initialization
**Problem**: Poor initialization leads to slow convergence
**Solutions**:
1. **K-means initialization**: Run k-means on initial encoder outputs
2. **Xavier/Glorot**: Standard neural network initialization
3. **Data-driven**: Initialize with representative data samples

#### Multiple Codebooks
**Product Quantization**: Use multiple smaller codebooks:
VQ(z) = [VQ₁(z₁), VQ₂(z₂), ..., VQ_M(z_M)]

where z is split into M sub-vectors.

**Benefits**:
- Exponential increase in effective codebook size: K^M
- Better representational capacity
- Reduced memory usage per codebook

#### Gumbel-Softmax VQ
**Alternative**: Replace hard quantization with soft assignment:
VQ_soft(z_e) = ∑_k softmax((-||z_e - e_k||²)/τ) · e_k

**Benefits**:
- Fully differentiable
- Temperature annealing: τ → 0 for hard quantization

**Drawbacks**:
- No longer discrete during training
- Requires careful temperature scheduling

### 7. Theoretical Analysis

#### Representation Learning Theory
**Theorem 7.1**: VQ-VAE learns a rate-distortion optimal quantization in the limit of infinite data and perfect optimization.

**Proof Sketch**: 
- Reconstruction loss → distortion minimization
- Discrete codes → rate constraint  
- VQ loss + commitment loss → Lloyd's algorithm for optimal quantization

#### Information Theoretic View
**Mutual Information**: I(X; Z_q) ≤ log K bits
**Rate-Distortion**: VQ-VAE implicitly solves rate-distortion problem:
min_q 𝔼[d(X, X̂)] subject to I(X; Z_q) ≤ R

#### Generative Modeling Perspective
**Two-stage Generation**:
1. **Stage 1**: Learn discrete representation z ~ p(z|x)
2. **Stage 2**: Model prior p(z) with autoregressive model

**Benefits over continuous VAEs**:
- No assumption about prior distribution shape
- Can use powerful autoregressive models for p(z)
- No posterior collapse issues

### 8. Modern Extensions

#### VQ-GAN
**Innovation**: Replace VAE decoder with GAN discriminator
**Architecture**: VQ Encoder + GAN Generator/Discriminator
**Benefits**:
- Higher quality image generation
- Better perceptual losses
- Combines discrete representation with adversarial training

**Loss Function**:
ℒ = ℒ_VQ + λ_adv·ℒ_adv + λ_perceptual·ℒ_perceptual

#### Residual VQ (RVQ)
**Idea**: Multiple quantization stages with residuals:
z₁ = VQ₁(z_e)
r₁ = z_e - z₁
z₂ = VQ₂(r₁)
...

**Final representation**: z_total = z₁ + z₂ + ... + z_M

**Benefits**:
- Progressive refinement of quantization
- Better reconstruction quality
- Hierarchical code structure

#### Finite Scalar Quantization (FSQ)
**Innovation**: Quantize each dimension independently to finite set
**Example**: Each dimension ∈ {-1, 0, +1}
**Benefits**:
- No learnable codebook
- Guaranteed code utilization
- Simpler implementation

#### Neural Audio Codec (EnCodec)
**Application**: High-fidelity audio compression
**Architecture**: Convolutional VQ-VAE with multiple quantization stages
**Innovation**: Residual quantization + adversarial training
**Achievement**: High-quality audio at very low bitrates

### 9. Applications

#### High-Resolution Image Generation
**Pipeline**:
1. Train VQ-VAE on images → discrete image tokens
2. Train autoregressive model on image tokens
3. Generate image tokens → decode to images

**Examples**: DALL-E, Parti (text-to-image generation)

#### Speech and Audio
**Applications**:
- Speech synthesis (high-quality voice generation)
- Music generation (discrete audio tokens)
- Audio compression (neural codecs)

**Benefits**: Discrete tokens enable language model-style generation

#### Natural Language Processing
**Connection**: VQ-VAE principles inspire discrete text representation
**Applications**:
- Discrete sentence embeddings
- Hierarchical text generation
- Cross-modal alignment (text-image)

#### Video Modeling
**Challenge**: Long sequences, high-dimensional data
**Solution**: Hierarchical VQ with temporal and spatial quantization
**Applications**: Video prediction, video generation, video compression

### 10. Implementation Details

#### Codebook Size Selection
**Trade-offs**:
- **Larger codebook**: Better reconstruction, harder to train
- **Smaller codebook**: Simpler training, information bottleneck

**Guidelines**:
- Start with K = 512 or 1024
- Monitor codebook utilization
- Adjust based on reconstruction quality

#### Training Strategies
**Learning Rate Scheduling**:
- Separate learning rates for encoder/decoder vs codebook
- Warm-up period for codebook learning
- Exponential decay for fine-tuning

**Data Augmentation**:
- Standard image augmentations work well
- Careful with strong augmentations that change semantics

#### Architectural Choices
**Encoder/Decoder Architecture**:
- ResNet blocks for images
- Dilated convolutions for audio
- Attention for long sequences

**Quantization Layer Placement**:
- After encoder bottleneck (standard)
- Multiple locations for hierarchical quantization

### 11. Evaluation Metrics

#### Reconstruction Quality
**Standard metrics**:
- MSE/PSNR for pixel-level quality
- SSIM for structural similarity
- LPIPS for perceptual similarity

#### Codebook Analysis
**Utilization Metrics**:
- Percentage of active codes
- Entropy of code usage distribution
- Perplexity = exp(entropy)

**Example**: Perplexity of 256 with 512-size codebook → 50% utilization

#### Downstream Task Performance
**Representation Quality**:
- Classification accuracy on frozen VQ features
- Transfer learning performance
- Clustering quality in discrete space

#### Generation Quality (for VQ-VAE-2 and variants)
**Standard generative metrics**:
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- Precision and Recall

### 12. Comparison with Other Methods

#### VQ-VAE vs Standard VAE
**Advantages of VQ-VAE**:
- No posterior collapse
- Interpretable discrete codes
- Better for autoregressive generation

**Advantages of VAE**:
- Smooth latent space
- Continuous interpolation
- Simpler training (no quantization)

#### VQ-VAE vs GANs
**VQ-VAE Advantages**:
- Stable training
- Meaningful latent space
- Explicit likelihood modeling (via autoregressive prior)

**GAN Advantages**:
- Higher sample quality
- Faster inference
- More mature adversarial techniques

#### VQ-VAE vs Diffusion Models
**VQ-VAE + Autoregressive**:
- Faster generation (autoregressive on discrete codes)
- Better for discrete modalities
- Explicit likelihood computation

**Diffusion Models**:
- Higher quality samples
- More flexible generation process
- Better mode coverage

### 13. Recent Developments

#### Foundation Models with VQ
**DALL-E**: Text-to-image generation via VQ-VAE + transformer
**Parti**: Autoregressive text-to-image with VQ tokens
**Make-A-Video**: Text-to-video with VQ representations

#### Scaling Laws for VQ Models
**Empirical Findings**:
- Larger codebooks improve quality up to a point
- Hierarchical quantization scales better than single-level
- Training data quality matters more than quantity

#### Multi-Modal VQ
**Idea**: Shared discrete vocabulary across modalities
**Applications**: Image-text models, audio-visual models
**Challenge**: Aligning discrete spaces across modalities

## Implementation Details

See `exercise.py` for implementations of:
1. Basic VQ-VAE with EMA codebook updates
2. VQ-VAE-2 with hierarchical quantization
3. VQ-GAN with adversarial training
4. Residual Vector Quantization
5. Comprehensive evaluation suite

## Experiments

1. **Codebook Size Analysis**: How does codebook size affect reconstruction and generation?
2. **EMA vs Gradient Updates**: Compare different codebook learning strategies
3. **Hierarchical vs Single-level**: Benefits of hierarchical quantization
4. **Cross-Modal Transfer**: Train on one domain, test on another
5. **Compression Analysis**: Rate-distortion curves for different configurations

## Research Connections

### Foundational Papers
1. van den Oord et al. (2017) - "Neural Discrete Representation Learning" (VQ-VAE)
2. Razavi et al. (2019) - "Generating Diverse High-Fidelity Images with VQ-VAE-2"
3. Esser et al. (2021) - "Taming Transformers for High-Resolution Image Synthesis" (VQ-GAN)

### Modern Applications
1. Ramesh et al. (2021) - "Zero-Shot Text-to-Image Generation" (DALL-E)
2. Yu et al. (2022) - "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation" (Parti)
3. Défossez et al. (2022) - "High Fidelity Neural Audio Compression" (EnCodec)

### Theoretical Analysis
1. Bansal et al. (2021) - "Cold Posterior Effect in VAE"
2. Tomczak (2022) - "Deep Generative Modeling" (Chapter on VQ-VAE)
3. Huang et al. (2023) - "Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation"

## Resources

### Primary Sources
1. **Original VQ-VAE Paper (van den Oord et al., 2017)**
   - Foundational work introducing vector quantization to deep learning
2. **VQ-VAE-2 Paper (Razavi et al., 2019)**
   - Hierarchical extension and high-quality image generation
3. **VQ-GAN Paper (Esser et al., 2021)**
   - Combining vector quantization with adversarial training

### Video Resources
1. **DeepMind VQ-VAE Presentation**
   - Authors explaining the original work
2. **Yannic Kilcher - VQ-VAE-2 Explained**
   - Clear explanation of hierarchical quantization
3. **Two Minute Papers - DALL-E Explained**
   - VQ-VAE applications in text-to-image generation

### Code Resources
1. **Official VQ-VAE Implementation (DeepMind)**
   - Reference PyTorch implementation
2. **VQ-GAN-CLIP (Katherine Crowson)**
   - Popular artistic generation tool
3. **Transformers Library (Hugging Face)**
   - Pre-trained VQ models and tokenizers

## Socratic Questions

### Understanding
1. Why doesn't VQ-VAE suffer from posterior collapse like standard VAEs?
2. How does the straight-through estimator enable gradient flow through quantization?
3. What determines the optimal codebook size for a given dataset?

### Extension
1. How would you design a VQ-VAE for 3D point clouds or meshes?
2. Can you prove that VQ-VAE converges to optimal rate-distortion solution?
3. How do you extend VQ-VAE to continuous control or reinforcement learning?

### Research
1. What are the fundamental limits of discrete representation learning?
2. How can VQ-VAE be combined with modern foundation model architectures?
3. What new applications emerge from high-quality discrete representations?

## Exercises

### Theoretical
1. Derive the gradient flow through the straight-through estimator
2. Prove that EMA updates converge to k-means solution under certain conditions
3. Analyze the information-theoretic properties of hierarchical quantization

### Implementation
1. Implement VQ-VAE with both gradient and EMA codebook updates
2. Build VQ-VAE-2 with hierarchical quantization for images
3. Create audio VQ-VAE for speech synthesis
4. Compare different quantization strategies (standard, residual, product)

### Research
1. Design novel applications of VQ-VAE in your domain of interest
2. Study the emergence of discrete structure in continuous data
3. Investigate connections between VQ-VAE and human perceptual organization

## Advanced Topics

### Mathematical Connections
- **Information Theory**: Rate-distortion theory and optimal quantization
- **Signal Processing**: Vector quantization and source coding  
- **Machine Learning**: Clustering, prototype learning, and discrete optimization
- **Neuroscience**: Sparse coding and population vector decoding

### Cutting-Edge Research
- **Unified Multi-Modal Models**: Shared discrete vocabularies across modalities
- **Neural Compression**: Real-time audio/video codecs with VQ
- **Discrete Diffusion**: Combining VQ with discrete diffusion processes
- **Causal Representation Learning**: VQ for disentangled discrete factors