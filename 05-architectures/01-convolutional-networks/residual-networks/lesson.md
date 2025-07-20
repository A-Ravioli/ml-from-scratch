# Residual Networks (ResNet)

## Prerequisites
- Classical CNN architectures (convolution, pooling, basic deep networks)
- Gradient descent and backpropagation theory
- Understanding of vanishing gradient problem
- Optimization landscapes and critical points

## Learning Objectives
- Master the theory behind residual connections and skip connections
- Understand the degradation problem in deep networks
- Implement ResNet blocks and various ResNet architectures from scratch
- Analyze gradient flow and training dynamics in residual networks
- Connect ResNet to optimization theory and highway networks

## Mathematical Foundations

### 1. The Degradation Problem

#### Classical Deep Network Training
As network depth increases, accuracy first improves, then degrades - even on training data.

**Key Observation**: This is not overfitting (training error also increases).

#### Empirical Evidence (He et al., 2016)
- 20-layer CNN: 91.25% training accuracy  
- 56-layer CNN: 72.07% training accuracy
- Deeper network performs worse on same training set

#### Theoretical Analysis
Consider learning identity mapping H(x) = x with L layers:
- **Nonlinear activation**: ReLU(Wx + b) cannot represent identity exactly
- **Accumulating approximation errors**: Small errors compound across layers
- **Optimization difficulty**: Very deep networks are harder to optimize

### 2. Residual Learning Framework

#### Problem Formulation
Instead of learning mapping H(x), learn residual F(x) = H(x) - x.

**Original formulation**: H(x) = F(x) + x
**Residual block**: Output = F(x) + x

#### Hypothesis: Easier Optimization
**Claim**: It's easier to optimize residual function F(x) to zero than to fit identity with H(x).

**Intuition**: 
- If identity is optimal, F(x) → 0 is easier than H(x) → x
- Gradient flows through shortcut enable better optimization
- Multiple paths for information flow

### 3. Mathematical Framework

#### Basic Residual Block
Input: x ∈ ℝ^d
Residual function: F(x; W) = W₂σ(W₁x + b₁) + b₂  
Output: y = F(x; W) + x

Where σ is activation function (usually ReLU).

#### Bottleneck Block (ResNet-50/101/152)
```
x → 1×1 conv (reduce) → 3×3 conv → 1×1 conv (expand) → + x → output
```

Mathematical representation:
- F₁(x) = σ(W₁x + b₁)     [1×1, C/4 filters]
- F₂(x) = σ(W₂F₁(x) + b₂) [3×3, C/4 filters]  
- F₃(x) = W₃F₂(x) + b₃    [1×1, C filters]
- Output: y = F₃(x) + x

#### Dimension Matching
When input/output dimensions differ, use projection:
y = F(x; W) + W_s x

Where W_s is learned projection matrix or fixed downsampling.

### 4. Gradient Flow Analysis

#### Backpropagation Through Residual Blocks
For block with output y = F(x) + x:

∂L/∂x = ∂L/∂y · (∂F/∂x + I)

Where I is identity matrix.

#### Theorem 4.1 (Gradient Flow in ResNets)
In residual networks, gradients flow through two paths:
1. **Through residual function**: ∂L/∂y · ∂F/∂x
2. **Through skip connection**: ∂L/∂y

**Consequence**: Even if ∂F/∂x vanishes, gradients still flow via identity path.

#### Deep Network Gradient Analysis
For L-layer ResNet with blocks F_i:

∂L/∂x₀ = ∂L/∂x_L · ∏ᵢ₌₁ᴸ (∂F_i/∂x_{i-1} + I)

**Expansion**:
= ∂L/∂x_L · (∏ᵢ ∂F_i/∂x_{i-1} + ∑_terms_with_identity + I)

**Key insight**: Always has additive identity term, preventing vanishing.

### 5. Highway Networks Connection

#### Highway Network Formulation
Output: y = H(x,W_H) · T(x,W_T) + x · C(x,W_C)

Where:
- H(x,W_H): Transform gate
- T(x,W_T): Carry gate  
- C(x,W_C) = 1 - T(x,W_T): Typically

#### Relationship to ResNet
ResNet is special case of Highway Networks:
- T(x) = 1 (always transform)
- C(x) = 1 (always carry)
- No learned gating

#### Gating vs Fixed Shortcuts
**Highway**: Learned gating allows selective information flow
**ResNet**: Fixed shortcuts are simpler, work equally well empirically

### 6. ResNet Architecture Variants

#### ResNet Family
- **ResNet-18/34**: Basic blocks (3×3 → 3×3)
- **ResNet-50/101/152**: Bottleneck blocks for efficiency
- **ResNet-1000+**: Extremely deep variants

#### Pre-activation ResNet (ResNet v2)
**Original**: Conv → BN → ReLU → Conv → BN → ReLU
**Pre-activation**: BN → ReLU → Conv → BN → ReLU → Conv

**Advantages**:
- Better gradient flow
- Easier optimization
- Improved regularization

#### Wide ResNet
Increase width instead of depth:
- Wider residual blocks (more channels)
- Better accuracy vs computational cost trade-off
- WRN-28-10: 28 layers, 10× wider

### 7. Theoretical Analysis

#### Theorem 7.1 (Universal Approximation for ResNets)
ResNets with width Õ(n) can approximate any Lipschitz function to arbitrary accuracy with depth polynomial in the input dimension.

#### Expressivity Analysis
**Theorem 7.2**: The set of functions representable by L-layer ResNet is strictly larger than L-layer feedforward network.

**Proof Sketch**:
1. ResNet can represent any feedforward function (set F(x) = H(x) - x)
2. ResNet can represent functions unreachable by feedforward networks
3. Therefore: ResNet ⊃ Feedforward □

#### Optimization Landscape
**Empirical findings**:
- ResNets have smoother loss landscapes
- More linear paths between solutions
- Better local minima connectivity

### 8. Training Dynamics

#### Initialization in ResNets
Standard initialization works better due to skip connections:
- **Xavier/Glorot**: Works well for moderate depth
- **He initialization**: Preferred for ReLU activations
- **Zero initialization**: Initialize last layer of each block to zero

#### Batch Normalization Interaction
Batch norm + ResNet synergy:
- BN reduces internal covariate shift
- ResNet provides gradient highways
- Together enable very deep training

#### Learning Rate Schedules
ResNets enable higher learning rates:
- Better gradient flow allows aggressive optimization
- Step decay commonly used: LR = 0.1 → 0.01 → 0.001
- Warm-up + cosine annealing also effective

### 9. Advanced Residual Architectures

#### ResNeXt (Aggregated Residual Transformations)
Replace 3×3 conv with grouped convolutions:
- Split channels into groups
- Apply same transformation to each group
- Aggregate results
- **Cardinality**: Number of groups (new hyperparameter)

#### Squeeze-and-Excitation ResNet (SE-ResNet)
Add channel attention mechanism:
1. **Squeeze**: Global average pooling
2. **Excitation**: FC → ReLU → FC → Sigmoid
3. **Scale**: Multiply feature maps by attention weights

#### DenseNet Connection
DenseNet uses all previous features:
x_ℓ = H_ℓ([x₀, x₁, ..., x_{ℓ-1}])

**Comparison**:
- **ResNet**: Additive shortcuts (x + F(x))
- **DenseNet**: Concatenative shortcuts ([x, F(x)])

### 10. Computational Aspects

#### Parameter Count
**Basic Block**: 2 × (3×3×C×C) = 18C² parameters
**Bottleneck**: 1×1×C×C/4 + 3×3×C/4×C/4 + 1×1×C/4×C ≈ 3C²/2 parameters

**Efficiency**: Bottleneck reduces parameters by factor of 6.

#### Memory Requirements
Skip connections require storing activations:
- **Forward pass**: Store input for each residual block
- **Memory cost**: Proportional to depth
- **Trade-off**: Memory vs recomputation

#### Inference Optimization
**Block fusion**: Merge BN into convolution at inference
**Quantization**: 8-bit integer arithmetic
**Pruning**: Remove unnecessary connections

### 11. Residual Connections in Other Domains

#### NLP Applications
**Transformer residual connections**:
- Around attention layers: x + Attention(x)
- Around feed-forward: x + FFN(x)
- Enable training of very deep transformers

#### Time Series and RNNs
**Residual RNNs**:
h_t = h_{t-1} + F(h_{t-1}, x_t)

**Benefits**: Better long-term dependency modeling

#### Graph Neural Networks
**Residual GCN**:
H^{(ℓ+1)} = H^{(ℓ)} + σ(ÃH^{(ℓ)}W^{(ℓ)})

### 12. Modern Developments

#### EfficientNet Residual Blocks
Compound scaling with residual connections:
- Depth scaling: More residual blocks
- Width scaling: Wider residual blocks  
- Resolution scaling: Higher input resolution

#### Vision Transformers (ViTs) 
Residual connections around attention:
- x' = x + MultiHeadAttention(LN(x))
- x'' = x' + MLP(LN(x'))

#### Neural Architecture Search
Automated ResNet design:
- Search optimal block structures
- Find best skip connection patterns
- Optimize depth-width trade-offs

## Implementation Details

See `exercise.py` for implementations of:
1. Basic and bottleneck residual blocks
2. Complete ResNet architectures (ResNet-18, 34, 50, 101, 152)
3. Pre-activation ResNet variants
4. Wide ResNet implementation
5. Gradient flow analysis tools
6. ResNeXt with grouped convolutions
7. SE-ResNet with channel attention
8. Training utilities and optimization schedules

## Experiments

1. **Degradation Problem**: Train deep vs shallow networks, observe training accuracy
2. **Gradient Flow**: Visualize gradient magnitudes through ResNet vs plain networks
3. **Ablation Studies**: Remove skip connections, compare performance
4. **Architecture Comparison**: ResNet vs VGG vs plain deep networks
5. **Initialization Effects**: Test different initialization schemes
6. **Depth Analysis**: Performance vs depth for ResNets

## Research Connections

### Foundational Papers
1. **He, Zhang, Ren & Sun (2016)** - "Deep Residual Learning for Image Recognition"
   - Original ResNet paper introducing skip connections

2. **He, Zhang, Ren & Sun (2016)** - "Identity Mappings in Deep Residual Networks"  
   - Pre-activation ResNet with theoretical analysis

3. **Srivastava, Greff & Schmidhuber (2015)** - "Highway Networks"
   - Precursor to ResNet with learned gating

### Modern Developments
4. **Xie, Girshick, Dollár, Tu & He (2017)** - "Aggregated Residual Transformations for Deep Neural Networks"
   - ResNeXt architecture

5. **Hu, Shen & Sun (2018)** - "Squeeze-and-Excitation Networks"
   - Channel attention in residual networks

6. **Zagoruyko & Komodakis (2016)** - "Wide Residual Networks"
   - Width vs depth trade-offs

### Theoretical Understanding
7. **Veit, Wilber & Belongie (2016)** - "Residual Networks Behave Like Ensembles"
   - Ensemble perspective on ResNets

8. **Hardt & Ma (2016)** - "Identity Matters in Deep Learning"
   - Theoretical analysis of skip connections

9. **Li, Xu, Chen, Xu & Tao (2018)** - "Visualizing the Loss Landscape of Neural Nets"
   - Loss landscape analysis for ResNets

## Resources

### Primary Sources
1. **He et al. - Deep Residual Learning Papers**
   - Original theoretical and empirical foundations
2. **Goodfellow, Bengio & Courville - Deep Learning** (Section 8.5)
   - Mathematical treatment of skip connections
3. **Stanford CS231n Lecture Notes**
   - Practical ResNet implementation guide

### Video Resources
1. **Kaiming He - ICCV 2015 Presentation**
   - Original ResNet presentation by lead author
2. **Andrej Karpathy - ResNet Explanation**
   - Intuitive understanding of residual learning
3. **Two Minute Papers - ResNet Breakthrough**
   - High-level overview of significance

### Advanced Reading
1. **Orhan & Pitkow (2017)** - "Skip Connections Eliminate Singularities"
2. **Balduzzi et al. (2017)** - "The Shattered Gradients Problem"
3. **Yang et al. (2017)** - "Gated Residual Networks"

## Socratic Questions

### Understanding
1. Why do residual connections solve the degradation problem but not the overfitting problem?
2. How do skip connections change the optimization landscape of deep networks?
3. What is the relationship between residual learning and ensemble methods?

### Extension
1. How would you design residual connections for recurrent neural networks?
2. Can residual principles be applied to other optimization problems outside deep learning?
3. What are the trade-offs between different types of skip connections (additive vs concatenative)?

### Research
1. What are the fundamental limits of how deep we can make residual networks?
2. How do residual connections interact with different optimization algorithms?
3. Can we theoretically predict when residual connections will help vs hurt performance?

## Exercises

### Theoretical
1. Derive the gradient flow equations for a multi-layer ResNet
2. Prove that ResNets can represent any function representable by plain networks
3. Analyze the effect of skip connections on the Hessian conditioning

### Implementation
1. Implement ResNet-18 and ResNet-50 from scratch
2. Build gradient flow visualization tools
3. Create ablation study framework for testing skip connection variants
4. Implement pre-activation ResNet and compare with original

### Research
1. Study the effect of skip connection placement on network performance
2. Investigate optimal initialization schemes for very deep ResNets
3. Compare residual learning with other approaches to training deep networks