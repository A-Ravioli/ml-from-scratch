# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

## Prerequisites
- Mobile convolution architectures (MobileNet, MobileNetV2)
- Neural Architecture Search (NAS) fundamentals
- Squeeze-and-Excitation (SE) blocks
- Compound scaling principles
- AutoML and architecture optimization

## Learning Objectives
- Master compound scaling methodology for neural networks
- Understand mobile-optimized architecture design principles
- Implement EfficientNet building blocks from scratch
- Analyze scaling laws and efficiency trade-offs
- Connect architectural innovations to practical deployment constraints

## Mathematical Foundations

### 1. The Scaling Problem

#### Traditional Scaling Approaches
Conventional methods scale networks in one dimension:

**Depth Scaling**: 
```
d = α^φ · d₀
```
where d₀ is baseline depth, α > 1, φ is scaling coefficient.

**Width Scaling**:
```
w = β^φ · w₀  
```
where w₀ is baseline width, β > 1.

**Resolution Scaling**:
```
r = γ^φ · r₀
```
where r₀ is baseline resolution, γ > 1.

#### Limitations of Single-Dimension Scaling
- **Depth scaling**: Vanishing gradients, diminishing returns
- **Width scaling**: Expensive computation, limited capacity gains
- **Resolution scaling**: Memory intensive, hardware constraints

### 2. Compound Scaling Methodology

#### Definition 2.1 (Compound Scaling)
Scale all three dimensions simultaneously with fixed relationship:

```
depth: d = α^φ
width: w = β^φ  
resolution: r = γ^φ
```

**Constraint**: α · β² · γ² ≈ 2

**Rationale**: FLOPS ∝ d · w² · r², so constraint maintains roughly constant FLOPS increase per scaling step.

#### Scaling Coefficient φ
Controls overall resource usage:
- φ = 0: baseline model
- φ = 1: 2× more FLOPS
- φ = 2: 4× more FLOPS

#### Optimal Scaling Factors
For EfficientNet family:
- α = 1.2 (depth)
- β = 1.1 (width)  
- γ = 1.15 (resolution)

These satisfy: 1.2 × 1.1² × 1.15² ≈ 2

### 3. EfficientNet Base Architecture (B0)

#### Mobile Inverted Bottleneck (MBConv)
Core building block based on MobileNetV2 with enhancements:

```
MBConv Block:
Input → Expansion → Depthwise → SE → Projection → Output
  ↓       ↓          ↓        ↓        ↓         ↓
 BN    1×1 Conv   3×3 DWConv  SE    1×1 Conv   BN
```

#### Expansion and Projection
**Expansion**: 1×1 conv increases channels by expansion ratio (typically 6)
**Projection**: 1×1 conv projects back to desired output channels

#### Depthwise Separable Convolution
**Standard Conv**: O(k² · cin · cout · H · W)
**Depthwise Sep**: O(k² · cin · H · W + cin · cout · H · W)

**Efficiency gain**: k²/(k² + 1/cin · cout) when k = 3, often 8-9× reduction

#### Squeeze-and-Excitation (SE) Blocks
Adaptive channel-wise feature recalibration:

```
SE Block:
Input → Global Pool → FC → ReLU → FC → Sigmoid → Scale
```

**Squeeze**: Global average pooling to 1×1×C
**Excitation**: Two FC layers with reduction ratio (typically 4)
**Scale**: Element-wise multiplication with input

### 4. EfficientNet-B0 Architecture

#### Stage Configuration
| Stage | Operator | Resolution | Channels | Layers |
|-------|----------|------------|----------|---------|
| 1 | Conv3×3 | 224×224 | 32 | 1 |
| 2 | MBConv1, k3×3 | 112×112 | 16 | 1 |
| 3 | MBConv6, k3×3 | 112×112 | 24 | 2 |
| 4 | MBConv6, k5×5 | 56×56 | 40 | 2 |
| 5 | MBConv6, k3×3 | 28×28 | 80 | 3 |
| 6 | MBConv6, k5×5 | 14×14 | 112 | 3 |
| 7 | MBConv6, k5×5 | 14×14 | 192 | 4 |
| 8 | MBConv6, k3×3 | 7×7 | 320 | 1 |
| 9 | Conv1×1 & Pool | 7×7 | 1280 | 1 |

#### Design Principles
1. **Kernel diversity**: Mix of 3×3 and 5×5 kernels
2. **Progressive channel increase**: Gradual feature complexity growth
3. **SE integration**: Applied to all MBConv blocks
4. **Efficient downsampling**: Stride in first layer of each stage

### 5. Scaling Laws and Analysis

#### Theorem 5.1 (Scaling Efficiency)
For fixed computation budget, compound scaling achieves better accuracy than any single-dimension scaling.

**Proof Intuition**:
- Higher resolution requires more feature capacity (width)
- More capacity benefits from deeper processing (depth)
- Optimal balance maximizes representational power

#### Empirical Scaling Relationships
From extensive experiments:

**Accuracy ∝ (FLOPS)^α** where α ≈ 0.2-0.3

This sublinear relationship motivates efficient architecture design over pure scaling.

#### Resource Constraints
**Memory**: O(w · r²) for activations
**Computation**: O(d · w² · r²) for training
**Parameters**: O(d · w²) approximately

### 6. Advanced Components

#### Stochastic Depth
Randomly skip blocks during training:
```
P(skip block i) = 1 - i/L · (1 - p₀)
```
where L is total blocks, p₀ is survival probability at final layer.

**Benefits**:
- Reduces training time
- Implicit ensemble effect
- Improves gradient flow

#### Swish Activation
```
Swish(x) = x · σ(βx)
```
where σ is sigmoid, β is learnable parameter (often fixed to 1).

**Properties**:
- Smooth, non-monotonic
- Better than ReLU for deeper networks
- Self-gating mechanism

#### Progressive Resizing
Train with increasing input resolution:
1. Start with smaller images (e.g., 128×128)
2. Gradually increase to target resolution
3. Fine-tune with final resolution

**Benefits**:
- Faster training
- Better convergence properties
- Improved generalization

### 7. EfficientNet Variants

#### EfficientNet-B1 through B7
Systematic scaling using compound methodology:

| Model | φ | Resolution | Parameters | Top-1 Acc |
|-------|---|------------|------------|-----------|
| B0 | 0 | 224 | 5.3M | 77.3% |
| B1 | 0.5 | 240 | 7.8M | 79.2% |
| B2 | 1 | 260 | 9.2M | 80.3% |
| B3 | 2 | 300 | 12M | 81.7% |
| B4 | 3 | 380 | 19M | 83.0% |
| B5 | 4 | 456 | 30M | 83.7% |
| B6 | 5 | 528 | 43M | 84.1% |
| B7 | 6 | 600 | 66M | 84.5% |

#### EfficientNetV2
Improvements over original:
- **Fused-MBConv**: Replace some MBConv with fused convolutions
- **Smaller expansion ratios**: Reduce memory usage
- **Smaller 3×3 kernels**: More 3×3, fewer 5×5 kernels
- **Progressive learning**: Adaptive regularization with image size

### 8. Implementation Considerations

#### Mobile Optimization
**Quantization-Friendly Design**:
- Avoid operations that don't quantize well
- Use ReLU6 instead of Swish for quantization
- Channel numbers divisible by 8

**Hardware-Aware Design**:
- Consider memory access patterns
- Optimize for specific accelerators (TPU, GPU, mobile)
- Balance accuracy vs inference speed

#### Training Techniques
**AutoAugment**: Learned data augmentation policies
**Mixup**: Linear combination of training examples
**Cutmix**: Regional dropout and mixing
**Label Smoothing**: Soft targets for better calibration

### 9. Neural Architecture Search Background

#### Platform-Aware NAS
EfficientNet-B0 discovered using NAS with constraints:
- **Latency**: Target inference time on mobile devices
- **Accuracy**: ImageNet top-1 performance
- **FLOPS**: Computational efficiency
- **Parameters**: Model size constraints

#### Search Space Design
**Macro Search Space**:
- Number of stages
- Stage connections
- Overall architecture patterns

**Micro Search Space**:
- Block types (Conv, MBConv variants)
- Kernel sizes (3×3, 5×5, 7×7)
- Expansion ratios (1, 3, 6)
- SE ratios (0, 0.25)

### 10. Theoretical Analysis

#### Scaling Laws
**Power Law Relationship**:
```
Accuracy ∝ (Parameters)^α · (Data)^β · (Compute)^γ
```

**EfficientNet Insight**: Better base architecture improves all scaling curves.

#### Compound Scaling Optimality
**Theorem 10.1**: Under resource constraints, compound scaling approaches Pareto optimality in accuracy-efficiency trade-off.

#### Generalization Properties
- **Architecture regularization**: SE blocks provide implicit regularization
- **Scale consistency**: Performance improvements transfer across scales
- **Domain adaptation**: Scaling principles generalize to other domains

## Implementation Details

See `exercise.py` for implementations of:
1. MBConv blocks with SE integration
2. Compound scaling mechanisms
3. Complete EfficientNet architectures (B0-B7)
4. Progressive training strategies
5. Mobile optimization techniques
6. Architecture search utilities

## Experiments

1. **Scaling Analysis**: Compare single vs compound scaling effectiveness
2. **Component Ablation**: Study impact of SE blocks, Swish, stochastic depth
3. **Efficiency Benchmarks**: Measure accuracy vs FLOPS/parameters/latency
4. **Transfer Learning**: Evaluate performance on downstream tasks

## Research Connections

### Seminal Papers
1. Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
   - Original EfficientNet paper introducing compound scaling

2. Tan & Le (2021) - "EfficientNetV2: Smaller Models and Faster Training"
   - Improved architecture and training methods

3. Sandler et al. (2018) - "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
   - Foundation for MBConv blocks

### Influential Follow-ups
1. Howard et al. (2019) - "Searching for MobileNetV3"
   - Architecture search for mobile networks

2. Radosavovic et al. (2020) - "Designing Network Design Spaces"
   - RegNet family inspired by EfficientNet scaling

## Resources

### Primary Sources
1. **Tan & Le (2019)** - Original EfficientNet paper
2. **EfficientNet GitHub** - Official implementation and pretrained models
3. **Tan & Le (2021)** - EfficientNetV2 improvements

### Video Resources
1. **Mingxing Tan** - "EfficientNet: Rethinking Model Scaling" (Google AI)
2. **Two Minute Papers** - "EfficientNet Explained"
3. **CS231n Guest Lecture** - "Mobile and Efficient Architectures"

### Advanced Reading
1. **Real et al. (2019)** - "AutoAugment: Learning Augmentation Strategies"
2. **Cubuk et al. (2020)** - "RandAugment: Practical automated data augmentation"

## Socratic Questions

### Understanding
1. Why does compound scaling work better than scaling individual dimensions?
2. How do mobile constraints influence architecture design decisions?
3. What role does Neural Architecture Search play in discovering efficient architectures?

### Extension
1. How would you adapt compound scaling to other domains (NLP, speech)?
2. Can compound scaling principles be automated further?
3. How do efficiency constraints change with different hardware platforms?

### Research
1. What are the fundamental limits of model scaling for computer vision?
2. How can we design architectures that scale optimally across different modalities?
3. What new scaling dimensions (beyond depth, width, resolution) might be valuable?

## Exercises

### Theoretical
1. Derive the compound scaling constraint α·β²·γ² ≈ 2
2. Analyze the computational complexity of MBConv vs standard convolution
3. Prove that SE blocks are equivalent to learned attention over channels

### Implementation
1. Implement progressive training with increasing resolution
2. Build architecture search framework for finding efficient designs
3. Create mobile-optimized variants with quantization support

### Research
1. Study scaling behavior on tasks beyond ImageNet classification
2. Investigate compound scaling for video understanding architectures
3. Design novel efficient building blocks inspired by EfficientNet principles