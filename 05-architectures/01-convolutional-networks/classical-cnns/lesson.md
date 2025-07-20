# Classical Convolutional Neural Networks

## Prerequisites
- Neural network fundamentals (feedforward networks, backpropagation)
- Linear algebra (convolution, matrix operations, eigenvalues)
- Signal processing basics (filtering, frequency domain)
- Optimization theory (gradient descent, chain rule)

## Learning Objectives
- Master the mathematical foundations of convolution operations
- Understand the principles of translation invariance and local connectivity
- Implement convolutional layers and pooling operations from scratch
- Analyze the computational and memory complexity of CNNs
- Connect CNN design principles to computer vision applications

## Mathematical Foundations

### 1. Convolution Operation

#### Definition 1.1 (Discrete Convolution)
For discrete signals f: ℤ → ℝ and g: ℤ → ℝ, the convolution is:
(f * g)[n] = ∑_{m=-∞}^{∞} f[m]g[n-m]

#### 2D Convolution for Images
For image I ∈ ℝ^{H×W} and kernel K ∈ ℝ^{k×k}:
(I * K)[i,j] = ∑_{u=0}^{k-1} ∑_{v=0}^{k-1} I[i+u, j+v] · K[u,v]

#### Cross-Correlation vs Convolution
**Cross-correlation** (what's actually used in deep learning):
(I ⋆ K)[i,j] = ∑_{u=0}^{k-1} ∑_{v=0}^{k-1} I[i+u, j+v] · K[u,v]

**True convolution**:
(I * K)[i,j] = ∑_{u=0}^{k-1} ∑_{v=0}^{k-1} I[i+u, j+v] · K[k-1-u, k-1-v]

Note: Deep learning literature often uses "convolution" to mean cross-correlation.

### 2. Convolutional Layer Mathematics

#### Multi-Channel Convolution
Input: X ∈ ℝ^{H×W×C_{in}} (height × width × input channels)
Kernel: W ∈ ℝ^{k×k×C_{in}×C_{out}} (kernel_size × kernel_size × input_channels × output_channels)
Bias: b ∈ ℝ^{C_{out}}

**Forward Pass**:
Y[i,j,c] = ∑_{u=0}^{k-1} ∑_{v=0}^{k-1} ∑_{d=0}^{C_{in}-1} X[i+u, j+v, d] · W[u,v,d,c] + b[c]

#### Output Dimensions
For input size H×W, kernel size k×k, padding P, stride S:
- Output height: H_out = ⌊(H + 2P - k)/S⌋ + 1
- Output width: W_out = ⌊(W + 2P - k)/S⌋ + 1

#### Parameter Count
Parameters = k × k × C_in × C_out + C_out (weights + biases)

### 3. Translation Invariance

#### Definition 3.1 (Translation Equivariance)
A function f is translation equivariant if:
f(T_δ x) = T_δ f(x)
where T_δ is translation by δ.

#### Theorem 3.1 (CNN Translation Equivariance)
Convolutional layers are translation equivariant: if we shift the input by (δ_x, δ_y), the output shifts by (δ_x/S, δ_y/S) where S is the stride.

**Proof Sketch**:
1. Let Y = conv(X, W) and X' = shift(X, δ)
2. Y'[i,j] = ∑_u ∑_v X'[i·S + u, j·S + v] · W[u,v]
3. = ∑_u ∑_v X[i·S + u - δ_x, j·S + v - δ_y] · W[u,v]
4. = Y[i - δ_x/S, j - δ_y/S] □

#### Breaking Translation Invariance
- **Pooling operations**: Can break exact equivariance
- **Padding effects**: Boundary conditions affect equivariance
- **Different stride patterns**: Create sampling artifacts

### 4. Receptive Field Analysis

#### Definition 4.1 (Receptive Field)
The receptive field of a neuron is the region in the input space that affects its output.

#### Receptive Field Calculation
For layer ℓ with kernel size k_ℓ and stride s_ℓ:

**Receptive field size**:
RF_ℓ = RF_{ℓ-1} + (k_ℓ - 1) · ∏_{i=1}^{ℓ-1} s_i

**Jump (distance between adjacent receptive fields)**:
J_ℓ = ∏_{i=1}^ℓ s_i

#### Effective Receptive Field
**Theorem 4.1**: The effective receptive field (where gradients have significant magnitude) grows approximately as O(√depth) for networks with small kernels.

### 5. Pooling Operations

#### Max Pooling
For pool size p×p and stride s:
MaxPool(X)[i,j] = max_{0≤u<p, 0≤v<p} X[i·s+u, j·s+v]

#### Average Pooling
AvgPool(X)[i,j] = (1/p²) ∑_{u=0}^{p-1} ∑_{v=0}^{p-1} X[i·s+u, j·s+v]

#### Global Average Pooling
GAP(X)[c] = (1/HW) ∑_{i=0}^{H-1} ∑_{j=0}^{W-1} X[i,j,c]

**Properties**:
- Reduces spatial dimensions
- Provides translation invariance
- Reduces overfitting (fewer parameters)
- Computational efficiency

### 6. Backpropagation in CNNs

#### Convolution Backward Pass
Given gradient ∂L/∂Y, compute:

**Gradient w.r.t. input**:
∂L/∂X[i,j,d] = ∑_{c=0}^{C_{out}-1} ∑_{u,v} (∂L/∂Y)[⌊(i-u)/s⌋, ⌊(j-v)/s⌋, c] · W[u,v,d,c]

**Gradient w.r.t. weights**:
∂L/∂W[u,v,d,c] = ∑_{i,j} (∂L/∂Y)[i,j,c] · X[i·s+u, j·s+v, d]

**Gradient w.r.t. bias**:
∂L/∂b[c] = ∑_{i,j} (∂L/∂Y)[i,j,c]

#### Pooling Backward Pass

**Max Pooling**:
∂L/∂X[i,j] = {
  ∂L/∂Y[⌊i/s⌋, ⌊j/s⌋]  if X[i,j] = max in pool
  0                        otherwise
}

**Average Pooling**:
∂L/∂X[i,j] = (1/p²) · ∂L/∂Y[⌊i/s⌋, ⌊j/s⌋]

### 7. Classical CNN Architectures

#### LeNet-5 (LeCun et al., 1998)
Architecture: Input → Conv → Pool → Conv → Pool → FC → FC → Output
- First successful CNN for digit recognition
- Introduced key CNN principles
- Used sigmoid/tanh activations

#### AlexNet (Krizhevsky et al., 2012)
Key innovations:
- ReLU activations: f(x) = max(0, x)
- Dropout regularization
- Data augmentation
- GPU implementation
- Local Response Normalization (LRN)

#### VGGNet (Simonyan & Zisserman, 2014)
Design principles:
- Small (3×3) convolutions throughout
- Deep architecture (16-19 layers)
- Consistent channel doubling: 64 → 128 → 256 → 512
- Large number of parameters (138M for VGG-16)

### 8. Design Principles and Insights

#### Hierarchical Feature Learning
**Layer 1**: Edge detectors (Gabor-like filters)
**Layer 2**: Corners, junctions, texture patterns
**Layer 3**: Object parts, shapes
**Deep layers**: Complete objects, semantic concepts

#### Inductive Biases of CNNs
1. **Local connectivity**: Nearby pixels are more related
2. **Translation equivariance**: Features can appear anywhere
3. **Hierarchical composition**: Complex features from simple ones
4. **Shared parameters**: Same feature detector across spatial locations

#### Parameter Efficiency
CNN with shared weights vs fully connected:
- FC layer: H·W·C_in × H'·W'·C_out parameters
- Conv layer: k² · C_in · C_out parameters
- Reduction factor: (H·W·H'·W')/(k²) (often 1000x+)

### 9. Computational Complexity

#### Forward Pass Complexity
**Time Complexity**: O(H_out · W_out · C_out · k² · C_in)
**Space Complexity**: O(H_out · W_out · C_out) for output + O(k² · C_in · C_out) for weights

#### Memory Requirements
**Activations**: Store all intermediate feature maps for backprop
**Weights**: All convolutional kernels and biases
**Gradients**: Same size as weights during training

#### Optimizations
1. **Im2col**: Reshape convolution as matrix multiplication
2. **FFT convolution**: O(n log n) for large kernels
3. **Winograd**: Reduced multiplications for small kernels
4. **Quantization**: Lower precision arithmetic

### 10. Normalization in CNNs

#### Batch Normalization
For mini-batch B with μ_B = mean, σ²_B = variance:
BN(x) = γ · (x - μ_B)/√(σ²_B + ε) + β

**Benefits**:
- Reduces internal covariate shift
- Allows higher learning rates
- Provides regularization effect
- Reduces sensitivity to initialization

#### Layer Normalization
Normalize across channels instead of batch:
LN(x) = γ · (x - μ_L)/√(σ²_L + ε) + β

### 11. Advanced Convolution Variants

#### Dilated Convolution
Introduce gaps in the kernel:
(I *_d K)[i,j] = ∑_u ∑_v I[i + d·u, j + d·v] · K[u,v]

**Benefits**: Exponentially growing receptive field

#### Separable Convolution
**Depthwise**: Apply one filter per input channel
**Pointwise**: 1×1 convolution to combine channels
**Parameter reduction**: k² · C_in · C_out → k² · C_in + C_in · C_out

#### Grouped Convolution
Divide channels into groups, convolve within groups:
- Reduces parameters and computation
- Used in ResNeXt, ShuffleNet

## Implementation Details

See `exercise.py` for implementations of:
1. 2D convolution operation from scratch
2. Various pooling operations
3. Backpropagation through conv and pool layers
4. Classical CNN architectures (LeNet, AlexNet, VGG)
5. Batch normalization layer
6. Memory-efficient convolution implementations
7. Receptive field calculation utilities
8. Visualization tools for filters and feature maps

## Experiments

1. **Convolution Properties**: Verify translation equivariance empirically
2. **Receptive Field Analysis**: Compute and visualize receptive fields
3. **Filter Visualization**: Examine learned convolutional filters
4. **Feature Map Evolution**: Track features through network layers
5. **Architecture Comparison**: Compare LeNet, AlexNet, VGG on datasets
6. **Ablation Studies**: Effect of pooling, normalization, activation functions

## Research Connections

### Foundational Papers
1. **LeCun et al. (1989)** - "Backpropagation Applied to Handwritten Zip Code Recognition"
   - First practical CNN demonstration

2. **LeCun et al. (1998)** - "Gradient-Based Learning Applied to Document Recognition"
   - LeNet-5 and comprehensive CNN framework

3. **Krizhevsky, Sutskever & Hinton (2012)** - "ImageNet Classification with Deep Convolutional Neural Networks"
   - AlexNet breakthrough on ImageNet

### Modern Developments
4. **Simonyan & Zisserman (2014)** - "Very Deep Convolutional Networks for Large-Scale Image Recognition"
   - VGGNet and deep architecture principles

5. **Ioffe & Szegedy (2015)** - "Batch Normalization: Accelerating Deep Network Training"
   - Batch normalization revolution

6. **Yu & Koltun (2015)** - "Multi-Scale Context Aggregation by Dilated Convolutions"
   - Dilated convolutions for dense prediction

### Theoretical Understanding
7. **Zhang et al. (2016)** - "Understanding Deep Learning Requires Rethinking Generalization"
   - Generalization properties of CNNs

8. **Raghu et al. (2017)** - "On the Expressive Power of Deep Neural Networks"
   - Theoretical analysis of CNN expressivity

## Resources

### Primary Sources
1. **Goodfellow, Bengio & Courville - Deep Learning** (Chapter 9)
   - Comprehensive CNN treatment
2. **LeCun, Bengio & Hinton (2015)** - "Deep Learning" (Nature Review)
   - Historical perspective and key insights
3. **Stanford CS231n Course Notes**
   - Practical CNN implementation guide

### Video Resources
1. **Andrej Karpathy - CS231n Lectures**
   - Excellent visual explanations of CNN concepts
2. **3Blue1Brown - CNNs Explained**
   - Intuitive visualization of convolution operations
3. **Yann LeCun - Deep Learning Course (NYU)**
   - From the inventor of modern CNNs

### Advanced Reading
1. **Zeiler & Fergus (2014)** - "Visualizing and Understanding Convolutional Networks"
2. **Springenberg et al. (2014)** - "Striving for Simplicity: The All Convolutional Net"
3. **Lin, Chen & Yan (2013)** - "Network In Network"

## Socratic Questions

### Understanding
1. Why do CNNs work better than fully connected networks for image tasks?
2. How does the choice of kernel size affect the network's representational capacity?
3. What is the relationship between receptive field size and semantic understanding?

### Extension
1. How would you design a CNN for 3D medical imaging data?
2. Can CNN principles be applied to non-grid data like text or graphs?
3. What are the trade-offs between depth and width in CNN architectures?

### Research
1. How can we design CNNs that are more robust to adversarial attacks?
2. What are the fundamental limits of translation equivariance in real applications?
3. How might neuromorphic computing change CNN implementations?

## Exercises

### Theoretical
1. Prove that the composition of translation-equivariant operations is translation-equivariant
2. Derive the receptive field formula for arbitrary CNN architectures
3. Analyze the approximation capacity of CNNs vs fully connected networks

### Implementation
1. Implement 2D convolution and pooling operations from scratch without using any deep learning libraries
2. Build LeNet-5 and train it on MNIST, visualizing learned filters
3. Create a tool to compute and visualize receptive fields for any CNN architecture
4. Implement various normalization techniques and compare their effects

### Research
1. Study the effect of different initialization schemes on CNN training
2. Investigate the relationship between network depth and feature hierarchy
3. Compare CNN performance on natural vs synthetic images to understand inductive biases