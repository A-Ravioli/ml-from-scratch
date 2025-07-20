# Backpropagation Calculus for Deep Learning

## Prerequisites
- Multivariable calculus (chain rule, partial derivatives)
- Linear algebra (matrix operations, Jacobians)
- Basic neural network architecture understanding
- Computational graph concepts

## Learning Objectives
- Master the mathematical foundations of automatic differentiation
- Understand forward-mode vs reverse-mode differentiation
- Derive backpropagation algorithm from first principles
- Implement a basic automatic differentiation framework
- Connect theory to practical deep learning optimization
- Analyze computational complexity and memory requirements

## Mathematical Foundations

### 1. The Chain Rule and Composite Functions

#### Univariate Chain Rule
For composite function h(x) = f(g(x)):
dh/dx = (df/dg)(dg/dx)

#### Multivariate Chain Rule
For f(x₁, x₂, ..., xₙ) where each xᵢ = xᵢ(t):
df/dt = ∑ᵢ (∂f/∂xᵢ)(dxᵢ/dt)

#### Vector Chain Rule
For f: ℝⁿ → ℝᵐ and g: ℝᵏ → ℝⁿ, composition h = f ∘ g:
Jₕ(x) = Jf(g(x)) · Jg(x)

where J denotes the Jacobian matrix.

### 2. Computational Graphs

#### Definition 2.1 (Computational Graph)
A computational graph G = (V, E) is a directed acyclic graph where:
- **Vertices V**: Variables (inputs, intermediates, outputs)
- **Edges E**: Dependencies between variables
- **Operations**: Functions computing each vertex from its parents

#### Example: f(x, y) = sin(x + y²)
```
x ──┐
    ├─ + ─── sin ─── f
y ─ ² ┘
```

**Forward computation**:
1. v₁ = y²
2. v₂ = x + v₁  
3. f = sin(v₂)

#### Jacobian of Computational Graph
For graph computing f: ℝⁿ → ℝᵐ, the Jacobian is:
Jf = ∏ᵢ J_opᵢ

where J_opᵢ is the Jacobian of operation i.

### 3. Automatic Differentiation

#### Forward-Mode AD

**Key idea**: Propagate derivatives along with values.

**Dual numbers**: Represent x + εx' where ε² = 0
- x: primal value
- x': derivative (tangent)

**Forward pass computation**:
For each operation f(u) = v:
- v.val = f(u.val)
- v.grad = f'(u.val) × u.grad

#### Algorithm 3.1 (Forward-Mode AD)
```
function forward_ad(inputs, seed_direction):
    # Initialize dual numbers
    for i, x in enumerate(inputs):
        x.val = input_values[i]
        x.grad = seed_direction[i]
    
    # Forward pass
    for operation in topological_order:
        compute_primal_and_tangent(operation)
    
    return output.val, output.grad
```

#### Reverse-Mode AD (Backpropagation)

**Key idea**: Propagate derivatives backward from outputs.

**Adjoint variables**: ∂y/∂vᵢ for output y and intermediate vᵢ

**Backward pass computation**:
For each operation vᵢ = f(v₁, ..., vₖ):
∂y/∂vⱼ += (∂y/∂vᵢ)(∂vᵢ/∂vⱼ) for j = 1, ..., k

#### Algorithm 3.2 (Reverse-Mode AD)
```
function reverse_ad(computational_graph):
    # Forward pass: compute values
    forward_pass(graph)
    
    # Initialize output gradient
    output.grad = 1.0
    
    # Backward pass: compute gradients
    for operation in reverse_topological_order:
        compute_adjoints(operation)
    
    return input_gradients
```

### 4. Backpropagation for Neural Networks

#### Neural Network as Computational Graph
For L-layer network:
- **Vertices**: {x, z¹, a¹, z², a², ..., zᴸ, aᴸ, L}
- **Operations**: Linear transformations and activations

#### Forward Pass
For ℓ = 1, ..., L:
1. zℓ = Wℓaℓ⁻¹ + bℓ (linear transformation)
2. aℓ = σ(zℓ) (activation function)

#### Loss Function
L = ℓ(aᴸ, y) + λR(θ)

where ℓ is loss function, R is regularizer, θ = {Wℓ, bℓ}.

#### Backward Pass Derivation

**Step 1: Output layer gradients**
∂L/∂aᴸ = ∇_aᴸ ℓ(aᴸ, y)

**Step 2: Pre-activation gradients**
∂L/∂zᴸ = (∂L/∂aᴸ) ⊙ σ'(zᴸ)

where ⊙ denotes element-wise product.

**Step 3: Weight and bias gradients**
∂L/∂Wᴸ = ∂L/∂zᴸ (aᴸ⁻¹)ᵀ
∂L/∂bᴸ = ∂L/∂zᴸ

**Step 4: Hidden layer gradients**
For ℓ = L-1, ..., 1:
∂L/∂aℓ = (Wℓ⁺¹)ᵀ ∂L/∂zℓ⁺¹
∂L/∂zℓ = (∂L/∂aℓ) ⊙ σ'(zℓ)
∂L/∂Wℓ = ∂L/∂zℓ (aℓ⁻¹)ᵀ
∂L/∂bℓ = ∂L/∂zℓ

#### Theorem 4.1 (Backpropagation Correctness)
The backpropagation algorithm correctly computes ∇_θ L for any differentiable loss function and activation functions.

**Proof**: Direct application of multivariate chain rule to the computational graph. □

### 5. Matrix Calculus for Backpropagation

#### Jacobian-Vector Products
Key insight: We need (∂y/∂x)ᵀv, not full Jacobian ∂y/∂x.

For y = f(x) where f: ℝⁿ → ℝᵐ:
- **Forward mode**: Computes Jv for vector v
- **Reverse mode**: Computes Jᵀv for vector v

#### Matrix Derivatives
For matrix-valued functions, we use:

**Matrix-by-scalar**:
∂(AB)/∂c = (∂A/∂c)B + A(∂B/∂c)

**Matrix-by-matrix** (vectorized):
∂tr(AᵀB)/∂A = B
∂tr(AB)/∂A = Bᵀ

#### Common Neural Network Operations

**Linear layer**: z = Wx + b
- ∂z/∂W = xᵀ ⊗ I (Kronecker product)
- ∂z/∂x = Wᵀ
- ∂z/∂b = I

**Element-wise activation**: a = σ(z)
- ∂a/∂z = diag(σ'(z))

### 6. Computational Complexity Analysis

#### Forward-Mode Complexity
- **Time**: O(n × cost of forward pass)
- **Space**: O(n) for storing dual numbers
- **Efficient for**: n inputs, 1 output (n ≪ m)

#### Reverse-Mode Complexity  
- **Time**: O(m × cost of forward pass)
- **Space**: O(size of computational graph)
- **Efficient for**: 1 input, m outputs (n ≫ m)

#### Theorem 6.1 (Complexity Optimality)
For computing all partial derivatives:
- Forward mode: O(n) passes for n inputs
- Reverse mode: O(m) passes for m outputs

**Neural network case**: Typically n ≫ m, so reverse mode (backprop) is optimal.

### 7. Memory Management in Backpropagation

#### Storage Requirements
**Forward pass**: Store all intermediate activations aℓ
**Backward pass**: Access activations in reverse order

**Memory**: O(L × batch_size × max_layer_width)

#### Memory Optimization Techniques

**Gradient Checkpointing**:
- Store only subset of activations
- Recompute others during backward pass
- Trade computation for memory

**Algorithm 7.1 (Gradient Checkpointing)**:
```
1. During forward pass: store activations at checkpoints
2. During backward pass: 
   - For non-checkpoint layers: recompute forward
   - For checkpoint layers: use stored activations
```

**In-place Operations**:
- Modify tensors without creating copies
- Requires careful gradient computation

### 8. Numerical Considerations

#### Gradient Checking
Compare analytical gradients with numerical approximation:

**Finite differences**:
(f(x + h) - f(x - h)) / (2h)

Choose h ≈ √ε where ε is machine precision.

#### Numerical Stability Issues

**Vanishing gradients**: ∂L/∂x → 0
- Cause: Product of many small derivatives
- Solutions: Skip connections, normalization, better activations

**Exploding gradients**: ∂L/∂x → ∞
- Cause: Product of many large derivatives  
- Solutions: Gradient clipping, proper initialization

**Catastrophic cancellation**: Loss of precision in subtraction
- Occurs in finite difference approximations
- Mitigated by higher-order methods

### 9. Advanced Backpropagation Techniques

#### Second-Order Methods
**Hessian computation**: Second derivatives for optimization
- **Forward-over-reverse**: ∇²f using forward mode on reverse mode
- **Reverse-over-reverse**: More efficient for neural networks

**Gauss-Newton approximation**:
H ≈ JᵀJ where J is Jacobian of residuals

#### Jacobian-Free Methods
Compute Hv without forming full Hessian H:
Hv ≈ (∇f(x + εv) - ∇f(x))/ε

#### Differentiation through Optimization
For f(x) = argmin_y g(x, y):
∂f/∂x = -(∂²g/∂y²)⁻¹(∂²g/∂x∂y)

### 10. Backpropagation through Different Architectures

#### Convolutional Layers
**Forward**: Convolution operation z = W * x
**Backward**: 
- ∂L/∂x: Convolution with flipped kernel
- ∂L/∂W: Convolution between input and upstream gradient

#### Recurrent Layers
**Backpropagation Through Time (BPTT)**:
- Unroll network through time
- Apply standard backpropagation
- **Issue**: Vanishing gradients over long sequences

#### Attention Mechanisms
**Scaled dot-product attention**:
Attention(Q, K, V) = softmax(QKᵀ/√d)V

**Gradients**:
- ∂L/∂Q: Through softmax and dot products
- ∂L/∂K: Similar to Q but different chain rule path
- ∂L/∂V: Direct through weighted sum

### 11. Modern Automatic Differentiation Systems

#### Tape-Based AD
**Dynamic computational graphs**:
- Build graph during forward pass
- Store operations on "tape"
- Replay in reverse for gradients

#### Static Graph Compilation
**Define-then-run**:
- Graph structure known before execution
- Enables optimization (operator fusion, memory planning)
- Used in TensorFlow 1.x, XLA

#### Source-to-Source Transformation
**Differentiate code directly**:
- Transform source code to compute derivatives
- Examples: Tapenade, ADIFOR
- Preserves program structure

### 12. Implementation Considerations

#### Operator Overloading
```python
class Tensor:
    def __init__(self, data, grad_fn=None):
        self.data = data
        self.grad = None
        self.grad_fn = grad_fn
    
    def __add__(self, other):
        result = Tensor(self.data + other.data)
        result.grad_fn = AddBackward(self, other)
        return result
```

#### Function Objects for Backward Pass
```python
class AddBackward:
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
    
    def apply(self, grad_output):
        return grad_output, grad_output
```

#### Gradient Accumulation
For parameter sharing or multiple loss terms:
```python
if param.grad is None:
    param.grad = new_grad
else:
    param.grad += new_grad
```

## Practical Applications

### When to Use Forward vs Reverse Mode
- **Forward mode**: Computing Jacobian-vector products, few inputs
- **Reverse mode**: Training neural networks, many parameters
- **Mixed mode**: Large networks with structured sparsity

### Debugging Gradients
1. **Gradient checking**: Compare with finite differences
2. **Gradient flow analysis**: Check for vanishing/exploding gradients
3. **Unit tests**: Test individual operations
4. **Visualization**: Plot gradient magnitudes across layers

### Memory-Efficient Training
1. **Gradient checkpointing**: Trade compute for memory
2. **Mixed precision**: Use lower precision for some computations
3. **Gradient compression**: For distributed training

## Implementation Details

See `exercise.py` for implementations of:
1. Basic automatic differentiation framework
2. Backpropagation for simple neural networks
3. Gradient checking utilities
4. Memory-efficient backpropagation variants
5. Computational graph visualization
6. Performance profiling tools

## Experiments

1. **Forward vs Reverse Mode**: Compare efficiency for different network sizes
2. **Gradient Checking**: Verify correctness of implementations
3. **Memory Profiling**: Analyze memory usage during training
4. **Numerical Precision**: Study effects of different precisions
5. **Gradient Flow**: Visualize gradients in deep networks

## Research Connections

### Foundational Papers
1. Rumelhart, Hinton & Williams (1986) - "Learning Representations by Back-Propagating Errors"
2. Speelpenning (1980) - "Compiling Fast Partial Derivatives"
3. Griewank & Walther (2008) - "Evaluating Derivatives: Principles and Techniques"

### Modern Developments
1. Chen et al. (2018) - "Neural Ordinary Differential Equations"
2. Baydin et al. (2017) - "Automatic Differentiation in Machine Learning"
3. Innes (2018) - "Fashionable Modelling with Flux"

### Advanced Topics
1. **Differentiable programming**: AD for entire programs
2. **Higher-order derivatives**: Computing Hessians efficiently
3. **Sparse AD**: Exploiting sparsity in gradients
4. **Probabilistic AD**: Uncertainty in gradients

## Resources

### Primary Sources
1. **Griewank & Walther - Evaluating Derivatives**
   - Comprehensive AD theory
2. **Goodfellow, Bengio & Courville - Deep Learning (Ch 6)**
   - Backpropagation in context
3. **Magnus & Neudecker - Matrix Differential Calculus**
   - Matrix derivatives reference

### Software Implementations
1. **PyTorch**: Dynamic tape-based AD
2. **TensorFlow**: Static and dynamic graphs
3. **JAX**: Functional transformations including AD
4. **Autograd**: Lightweight Python AD library

### Video Resources
1. **Stanford CS231n - Backpropagation**
   - Karpathy's clear explanations
2. **MIT 18.337 - Automatic Differentiation**
   - Mathematical foundations
3. **Fast.ai - Practical Deep Learning**
   - Implementation focus

## Socratic Questions

### Understanding
1. Why is reverse-mode AD more efficient than forward-mode for training neural networks?
2. How does the computational graph change for dynamic networks (RNNs, adaptive computation)?
3. What's the relationship between backpropagation and the adjoint method in optimal control?

### Extension
1. How would you compute third-order derivatives efficiently?
2. Can you design an AD system that automatically chooses forward vs reverse mode?
3. What happens to gradients when we have discrete operations (argmax, sampling)?

### Research
1. How can we make AD more memory-efficient for very deep networks?
2. What are the fundamental limits of automatic differentiation accuracy?
3. How does AD interact with different number systems (fixed-point, logarithmic)?

## Exercises

### Theoretical
1. Derive backpropagation equations for LSTM cells
2. Prove that reverse-mode AD computes exact gradients (up to numerical precision)
3. Analyze memory complexity of gradient checkpointing strategies

### Implementation
1. Build minimal AD framework supporting basic operations
2. Implement gradient checking with multiple finite difference schemes
3. Create computational graph visualization tools
4. Code memory-efficient backpropagation for very deep networks

### Research
1. Compare numerical stability of different AD implementations
2. Study gradient flow in networks with different initialization schemes
3. Investigate the effect of mixed precision on gradient accuracy

## Advanced Topics

### Differentiable Programming
- **Differentiable simulators**: Physics engines with gradients
- **Differentiable rendering**: Computer graphics with AD
- **Differentiable optimization**: Gradients through optimization layers

### Probabilistic Differentiation
- **Stochastic gradients**: When gradients themselves are random
- **Gradient estimation**: For non-differentiable functions
- **Uncertainty quantification**: Propagating input uncertainty through AD

### Hardware Considerations
- **GPU memory hierarchy**: Optimizing for different memory types
- **Quantized AD**: Lower precision automatic differentiation
- **Specialized hardware**: AD on TPUs, neuromorphic chips