# Normalization Techniques for Deep Learning

## Prerequisites
- Neural network fundamentals (forward/backward pass)
- Statistics (mean, variance, standardization)
- Linear algebra (matrix operations)
- Understanding of gradient flow and optimization

## Learning Objectives
- Master the theory and implementation of batch normalization
- Understand layer normalization and its applications
- Compare different normalization techniques and their use cases
- Analyze the effects of normalization on optimization and generalization
- Implement normalization layers from scratch

## Mathematical Foundations

### 1. The Normalization Problem

#### Internal Covariate Shift Hypothesis
**Original motivation**: Input distributions to layers change during training, slowing convergence.

**Definition**: For layer ℓ with inputs x^(ℓ), the distribution P(x^(ℓ)) changes as parameters in previous layers update.

#### Modern Understanding
Recent research suggests normalization helps by:
1. **Smoothing optimization landscape**: Reducing sensitivity to initialization
2. **Enabling higher learning rates**: More stable gradients
3. **Regularization effect**: Implicit noise injection
4. **Gradient flow improvement**: Better signal propagation

### 2. Batch Normalization

#### Algorithm 2.1 (Batch Normalization)
For mini-batch B = {x₁, x₂, ..., xₘ}:

**Training (per feature)**:
1. μ_B = (1/m) ∑ᵢ₌₁ᵐ xᵢ (batch mean)
2. σ²_B = (1/m) ∑ᵢ₌₁ᵐ (xᵢ - μ_B)² (batch variance)
3. x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε) (normalize)
4. yᵢ = γx̂ᵢ + β (scale and shift)

**Parameters**:
- γ: learnable scale parameter
- β: learnable shift parameter
- ε: small constant for numerical stability (typically 1e-5)

#### Inference Mode
During inference, use population statistics:
- μ: running average of batch means during training
- σ²: running average of batch variances during training

#### Mathematical Analysis

**Jacobian of normalization**:
∂x̂ᵢ/∂xⱼ = {
  (1/σ_B)(1 - 1/m - (x̂ᵢx̂ⱼ)/m)  if i = j
  -(1/σ_B)(1/m + (x̂ᵢx̂ⱼ)/m)      if i ≠ j
}

**Gradient flow**:
∂L/∂xᵢ = (γ/σ_B)[∂L/∂yᵢ - (1/m)∑ⱼ ∂L/∂yⱼ - (x̂ᵢ/m)∑ⱼ x̂ⱼ ∂L/∂yⱼ]

#### Key Properties
1. **Zero mean, unit variance**: E[x̂] = 0, Var[x̂] = 1
2. **Learnable transformation**: Can recover any mean/variance via γ, β
3. **Batch dependence**: Normalization depends on current batch
4. **Centering effect**: Removes first-order statistics

### 3. Layer Normalization

#### Motivation
- **Batch independence**: Normalize across features, not batch
- **RNN applicability**: Sequence lengths can vary
- **Small batch robustness**: Works with batch size 1

#### Algorithm 3.1 (Layer Normalization)
For input x ∈ ℝᵈ:

1. μ = (1/d) ∑ⱼ₌₁ᵈ xⱼ (layer mean)
2. σ² = (1/d) ∑ⱼ₌₁ᵈ (xⱼ - μ)² (layer variance)
3. x̂ⱼ = (xⱼ - μ) / √(σ² + ε) (normalize)
4. yⱼ = γⱼx̂ⱼ + βⱼ (scale and shift)

#### Comparison with BatchNorm
| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalization axis | Batch dimension | Feature dimension |
| Batch dependence | Yes | No |
| Inference mode | Different from training | Same as training |
| RNN compatibility | Poor | Excellent |
| Small batch performance | Poor | Good |

#### Mathematical Properties
**Invariance**: Layer normalization is invariant to scaling and shifting of all features:
f(αx + β1) = f(x) for any α > 0, β ∈ ℝ

**Gradient properties**: More stable gradients for varying sequence lengths.

### 4. Normalization Variants

#### Group Normalization
**Motivation**: Combine benefits of LayerNorm and BatchNorm.

**Algorithm**: Divide channels into G groups, normalize within each group.
- Group size: C/G where C is number of channels
- Reduces to LayerNorm when G=1, InstanceNorm when G=C

#### Instance Normalization
**Application**: Style transfer, where each image should be normalized independently.

**Algorithm**: Normalize each channel of each sample independently.
yᵢⱼₖₗ = γⱼ(xᵢⱼₖₗ - μᵢⱼ)/σᵢⱼ + βⱼ

where i=sample, j=channel, k,l=spatial dimensions.

#### Root Mean Square Layer Normalization (RMSNorm)
**Simplification**: Remove mean centering, only normalize by RMS.

x̂ⱼ = xⱼ / √((1/d)∑ₖ₌₁ᵈ xₖ²)

**Benefits**: Faster computation, similar performance in many cases.

#### Weight Normalization
**Idea**: Normalize weight vectors instead of activations.

W = g(v/||v||) where g is learnable magnitude, v is direction.

**Benefits**: 
- Accelerates convergence
- Improves conditioning of optimization
- Reduces dependence on initialization

#### Spectral Normalization
**Application**: GANs, where we want to control Lipschitz constant.

**Method**: Normalize weights by their spectral norm (largest singular value).
W_SN = W/σ(W) where σ(W) is spectral norm.

### 5. Theoretical Analysis

#### Optimization Landscape Effects

**Theorem 5.1 (Santurkar et al. 2018)**: BatchNorm's primary benefit is making the optimization landscape significantly smoother, rather than reducing internal covariate shift.

**Evidence**:
- Loss surface has smaller Lipschitz constants
- Gradients are more predictive
- Allows larger learning rates

#### Implicit Regularization
BatchNorm acts as regularizer through:
1. **Noise injection**: Stochastic batch statistics
2. **Gradient noise**: Batch-dependent gradients
3. **Parameter coupling**: Dependencies between samples

#### Generalization Effects
**Double descent phenomenon**: Normalization can cause performance to improve again after initial overfitting.

**Implicit bias**: Normalization biases toward certain types of solutions.

### 6. Placement and Architecture Considerations

#### Pre-activation vs Post-activation
**Original**: Conv → BatchNorm → ReLU
**Pre-activation**: BatchNorm → ReLU → Conv

**Benefits of pre-activation**:
- Easier gradient flow
- Better regularization
- Improved performance in very deep networks

#### Normalization in Different Architectures

**ResNets**: 
- Original: Post-activation
- v2: Pre-activation
- Modern: Pre-activation with improved skip connections

**Transformers**:
- Original: Post-LayerNorm (after self-attention)
- Modern: Pre-LayerNorm (before self-attention)
- Benefits: More stable training, better convergence

**CNNs**:
- Standard: BatchNorm after conv, before activation
- Alternative: GroupNorm for small batches

### 7. Training vs Inference Behavior

#### Batch Normalization Training/Inference Gap
**Training**: Use batch statistics
**Inference**: Use population statistics

**Moving averages**:
μ_pop ← (1-α)μ_pop + αμ_batch
σ²_pop ← (1-α)σ²_pop + ασ²_batch

where α is momentum parameter (typically 0.1).

#### Common Issues
1. **Train/test mismatch**: Different behavior modes
2. **Batch size sensitivity**: Small batches have noisy statistics
3. **Domain shift**: Population stats may not match deployment data

#### Solutions
- **Layer Normalization**: Same behavior in train/test
- **Batch Renormalization**: Reduce train/test gap
- **Cross-validation**: Validate with same batch sizes as training

### 8. Implementation Considerations

#### Numerical Stability
**Issue**: Division by small variances causes instability.

**Solutions**:
- Add epsilon to variance: σ² + ε
- Typical values: ε ∈ [1e-5, 1e-3]
- Gradient clipping for extreme cases

#### Memory and Computation
**Memory overhead**:
- Store running means and variances
- Additional parameters γ, β

**Computation overhead**:
- Forward: 2 passes (mean, variance)
- Backward: Complex gradient computation

#### Efficient Implementation
```python
# Fused operations for efficiency
def batch_norm_forward_fast(x, gamma, beta, running_mean, running_var, eps):
    # Single pass computation when possible
    # Use numerically stable algorithms
    pass
```

### 9. Modern Developments

#### Adaptive Normalization
**AdaIN (Adaptive Instance Normalization)**:
Used in style transfer, where statistics come from style image.

**SPADE (Spatially-Adaptive Normalization)**:
Normalization parameters are spatially varying.

#### Learnable Normalization
**Switchable Normalization**: Learn combination of different normalizations.
**EvoNorm**: Evolutionary search for normalization functions.

#### Normalization-Free Networks
**NFNets**: Achieve SOTA without normalization using:
- Careful initialization
- Adaptive gradient clipping
- Modified architectures

#### Attention-Based Normalization
**SelfNorm**: Use attention to compute normalization statistics.
**ChannelNorm**: Channel-wise attention for normalization.

### 10. Application-Specific Guidelines

#### Computer Vision
- **Standard CNNs**: BatchNorm after conv layers
- **Small batches**: GroupNorm or LayerNorm
- **Style transfer**: InstanceNorm
- **Object detection**: Careful with multi-scale training

#### Natural Language Processing
- **Transformers**: LayerNorm (Pre-LN for stability)
- **RNNs**: LayerNorm within cells
- **Embeddings**: Usually no normalization

#### Generative Models
- **GANs**: Spectral normalization for discriminator
- **VAEs**: Careful with stochastic layers
- **Autoregressive**: LayerNorm for stability

#### Reinforcement Learning
- **Observation normalization**: Running statistics
- **Gradient normalization**: Clip by global norm
- **Value function**: LayerNorm for stability

### 11. Debugging and Monitoring

#### Diagnostic Tools
1. **Activation statistics**: Monitor means and variances
2. **Gradient norms**: Check for vanishing/exploding
3. **Parameter evolution**: Track γ, β values
4. **Batch effect**: Compare different batch sizes

#### Common Problems
- **Dead neurons**: All activations become zero
- **Gradient explosion**: Despite normalization
- **Slow convergence**: Wrong normalization choice
- **Overfitting**: Too much regularization from normalization

#### Solutions
- **Learning rate adjustment**: Normalization enables higher LR
- **Architecture changes**: Pre-activation vs post-activation
- **Hyperparameter tuning**: ε, momentum for running stats

### 12. Research Frontiers

#### Theoretical Understanding
- **Why does normalization work?**: Beyond internal covariate shift
- **Optimization theory**: Effect on loss landscape
- **Generalization bounds**: How normalization affects generalization

#### New Techniques
- **Normalization-free training**: Alternative approaches
- **Dynamic normalization**: Adaptive during training
- **Hardware-efficient**: Specialized for edge devices

#### Cross-domain Applications
- **Graph Neural Networks**: How to normalize on graphs
- **3D vision**: Normalization for point clouds, meshes
- **Time series**: Temporal normalization strategies

## Implementation Details

See `exercise.py` for implementations of:
1. Batch normalization (training and inference modes)
2. Layer normalization and variants
3. Group normalization and instance normalization
4. Weight normalization and spectral normalization
5. Comparison tools and visualization utilities
6. Architecture integration examples

## Experiments

1. **Normalization Comparison**: BatchNorm vs LayerNorm vs GroupNorm on same task
2. **Batch Size Sensitivity**: How performance varies with batch size
3. **Depth Study**: Normalization effects in very deep networks
4. **Learning Rate Interaction**: How normalization enables higher learning rates
5. **Regularization Analysis**: Compare dropout + normalization combinations

## Research Connections

### Foundational Papers
1. Ioffe & Szegedy (2015) - "Batch Normalization: Accelerating Deep Network Training"
2. Ba, Kiros & Hinton (2016) - "Layer Normalization"
3. Wu & He (2018) - "Group Normalization"
4. Salimans & Kingma (2016) - "Weight Normalization"

### Theoretical Analysis
1. Santurkar et al. (2018) - "How Does Batch Normalization Help Optimization?"
2. Kohler et al. (2019) - "Towards a Theoretical Understanding of Batch Normalization"
3. Daneshmand et al. (2021) - "Batch Normalization Provably Avoids Rank Collapse"

### Modern Developments
1. Brock et al. (2021) - "High-Performance Large-Scale Image Recognition Without Normalization"
2. Xu et al. (2019) - "Understanding and Improving Layer Normalization"

## Resources

### Primary Sources
1. **Goodfellow, Bengio & Courville - Deep Learning**
   - Normalization in context of optimization
2. **Deep Learning specialization - Andrew Ng**
   - Practical implementation guidelines

### Video Resources
1. **Stanford CS231n - Batch Normalization**
   - Clear explanations with visual aids
2. **Two Minute Papers - Normalization Techniques**
   - Recent developments overview

### Implementation References
1. **PyTorch documentation**: torch.nn normalization layers
2. **TensorFlow/Keras**: Normalization layer implementations
3. **Papers with Code**: Normalization benchmarks

## Socratic Questions

### Understanding
1. Why does batch normalization enable higher learning rates?
2. When would you choose LayerNorm over BatchNorm?
3. How does normalization interact with different activation functions?

### Extension
1. How would you design normalization for a new domain (e.g., graphs, point clouds)?
2. Can you prove that certain normalizations preserve important properties?
3. What happens to normalization in the infinite width limit?

### Research
1. Why do normalization-free networks work, and what does this tell us about normalization?
2. How can we design normalization techniques for hardware-constrained environments?
3. What are the fundamental trade-offs between different normalization approaches?

## Exercises

### Theoretical
1. Derive the backward pass equations for batch normalization
2. Prove that layer normalization is invariant to input scaling
3. Analyze the effect of batch size on BatchNorm gradient noise

### Implementation
1. Implement all major normalization techniques from scratch
2. Create tools for monitoring normalization statistics during training
3. Build comparison frameworks for different normalization schemes
4. Implement efficient fused operations for normalization

### Research
1. Study the interaction between normalization and different optimizers
2. Investigate normalization for novel architectures
3. Explore the connection between normalization and generalization

## Advanced Topics

### Normalization and Information Theory
- **Information bottleneck**: How normalization affects information flow
- **Mutual information**: Between layers in normalized networks
- **Entropy regularization**: Normalization as entropy constraint

### Normalization and Generative Models
- **Mode collapse**: How normalization affects GAN training
- **Style transfer**: Using normalization statistics for style
- **Variational inference**: Normalization in variational autoencoders

### Hardware-Aware Normalization
- **Quantization-friendly**: Normalization for low-precision training
- **Memory-efficient**: Techniques for limited memory
- **Parallel-friendly**: Normalization for distributed training