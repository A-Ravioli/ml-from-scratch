# Neural Network Theory for Machine Learning

## Prerequisites
- Linear algebra (matrix operations, eigenvalues)
- Calculus (chain rule, partial derivatives)
- Probability theory (distributions, expectations)
- Convex optimization basics

## Learning Objectives
- Understand universal approximation theorems
- Master backpropagation algorithm and its theory
- Analyze neural network optimization landscapes
- Connect theory to practical deep learning

## Mathematical Foundations

### 1. Neural Network Architecture

#### Feedforward Network
Consider L-layer network with activation functions σ:

**Layer computations**:
- z^(ℓ) = W^(ℓ)a^(ℓ-1) + b^(ℓ)
- a^(ℓ) = σ(z^(ℓ))

where:
- W^(ℓ) ∈ ℝ^(n_ℓ × n_(ℓ-1)) are weight matrices
- b^(ℓ) ∈ ℝ^(n_ℓ) are bias vectors
- a^(0) = x is the input

#### Function Representation
The network computes:
f(x; θ) = W^(L)σ(W^(L-1)σ(...σ(W^(1)x + b^(1))...) + b^(L-1)) + b^(L)

where θ = {W^(ℓ), b^(ℓ)}_{ℓ=1}^L are parameters.

### 2. Universal Approximation Theory

#### Theorem 2.1 (Universal Approximation - Cybenko 1989)
Let σ be a continuous, bounded, non-constant activation function. Then finite sums of the form:
f(x) = ∑_{j=1}^N v_j σ(w_j^T x + b_j)

are dense in C(K) for any compact set K ⊂ ℝ^d.

**Interpretation**: Single hidden layer networks can approximate any continuous function on compact sets.

#### Theorem 2.2 (Hornik et al. 1989)
The same result holds for any non-polynomial activation function.

#### Modern Extensions
- **ReLU networks**: Telgarsky (2016) - depth-width tradeoffs
- **Deep networks**: Poggio et al. (2017) - compositional functions
- **Approximation rates**: Yarotsky (2017) - explicit bounds

### 3. Expressivity and Capacity

#### Definition 3.1 (VC Dimension of Neural Networks)
For neural networks with W parameters and piecewise linear activations:
VCdim ≤ O(W log W)

#### Theorem 3.1 (Bartlett et al. 2019)
For ReLU networks with W parameters and L layers:
VCdim = Θ(WL log(W))

#### Function Counting
Number of distinct functions a network can compute:
- Linear regions of ReLU network: ≤ ∏_{ℓ=1}^{L-1} (∑_{k=0}^{n_ℓ} (W_ℓ choose k))
- Grows exponentially with depth and width

### 4. Backpropagation Algorithm

#### Loss Function
For training set {(x_i, y_i)}_{i=1}^m:
L(θ) = (1/m) ∑_{i=1}^m ℓ(f(x_i; θ), y_i) + λR(θ)

where ℓ is loss function and R is regularizer.

#### Forward Pass
For each layer ℓ = 1, ..., L:
1. z^(ℓ) = W^(ℓ)a^(ℓ-1) + b^(ℓ)
2. a^(ℓ) = σ(z^(ℓ))

#### Backward Pass
Define error terms:
δ^(L) = ∇_{a^(L)} ℓ ⊙ σ'(z^(L))

For ℓ = L-1, ..., 1:
δ^(ℓ) = (W^(ℓ+1))^T δ^(ℓ+1) ⊙ σ'(z^(ℓ))

#### Gradients
∇_{W^(ℓ)} L = δ^(ℓ) (a^(ℓ-1))^T
∇_{b^(ℓ)} L = δ^(ℓ)

#### Theorem 4.1 (Computational Complexity)
Backpropagation computes gradients in O(W) time, where W is the number of parameters.

**Proof**: Each parameter appears in exactly one forward and one backward computation. □

### 5. Optimization Landscape

#### Non-Convexity
Neural network loss functions are generally non-convex due to:
1. Parameter sharing (weight tying)
2. Composition of non-linear functions
3. Discrete structure (architecture choices)

#### Critical Points
∇L(θ) = 0 can be:
- Global minimum
- Local minimum  
- Saddle point
- Local maximum (rare in high dimensions)

#### Theorem 5.1 (Dauphin et al. 2014)
In high dimensions, critical points are overwhelmingly saddle points rather than local minima.

#### Mode Connectivity
Recent work shows that different local minima are often connected by paths of low loss.

### 6. Initialization Theory

#### Vanishing/Exploding Gradients
Consider gradient magnitudes through layers:
||∇_{W^(1)} L|| ∝ ∏_{ℓ=2}^L ||W^(ℓ)|| ||σ'(z^(ℓ))||

If product → 0: vanishing gradients
If product → ∞: exploding gradients

#### Xavier/Glorot Initialization
Initialize weights from:
W_{ij} ~ N(0, 2/(n_{in} + n_{out}))

**Motivation**: Preserve variance of activations and gradients.

#### He Initialization (for ReLU)
W_{ij} ~ N(0, 2/n_{in})

**Derivation**: For ReLU, Var(σ(z)) = (1/2)Var(z), so need factor of 2.

#### Theorem 6.1 (Signal Propagation)
With proper initialization, activations and gradients maintain reasonable magnitudes through deep networks.

### 7. Neural Tangent Kernel Theory

#### Definition 7.1 (Neural Tangent Kernel)
For infinite-width neural networks, the NTK is:
K(x, x') = E[∇_θ f(x; θ) · ∇_θ f(x'; θ)]

where expectation is over random initialization.

#### Theorem 7.1 (Jacot et al. 2018)
In the infinite width limit, neural network training is equivalent to kernel regression with the NTK.

#### Implications
- Training dynamics become linear
- Global convergence guarantees
- Connection to Gaussian processes

### 8. Generalization Theory

#### Classical Bounds
Using VC dimension d and sample size m:
P[R(f) ≤ R̂(f) + √(d log(m/d) + log(1/δ))/m] ≥ 1 - δ

#### Issues with Deep Learning
- VC bounds are often vacuous (d ≫ m)
- Networks interpolate training data (R̂(f) = 0)
- Yet they generalize well empirically

#### Modern Approaches

##### Implicit Regularization
Gradient descent biases toward "simple" solutions:
- Low norm solutions
- High margin classifiers
- Sparse representations

##### PAC-Bayes Bounds
P[R(f) ≤ R̂(f) + √(KL(Q||P) + log(m/δ))/(2m)] ≥ 1 - δ

where Q is posterior, P is prior over parameters.

##### Compression Bounds
If network can be compressed to C bits:
Generalization error ≤ √(C/m)

### 9. Approximation vs Optimization

#### Statistical Learning Decomposition
Total error = Approximation + Estimation + Optimization

1. **Approximation**: How well can optimal function in class approximate target?
2. **Estimation**: Sample complexity for finding good function
3. **Optimization**: How close does algorithm get to optimal in class?

#### Deep Learning Tradeoffs
- **More parameters**: Better approximation, worse optimization
- **More data**: Better estimation
- **Better algorithms**: Better optimization

### 10. Specific Architectures

#### Convolutional Networks
- **Translation invariance**: Shift input → shift output
- **Local connectivity**: Reduces parameters, encodes structure
- **Weight sharing**: Statistical efficiency

#### Recurrent Networks
- **Memory**: Hidden state carries information
- **Variable length**: Handle sequences naturally
- **Challenges**: Vanishing gradients, limited memory

#### Attention Mechanisms
- **Global connectivity**: All positions can interact
- **Parallelizable**: Unlike RNNs
- **Interpretable**: Attention weights show importance

### 11. Optimization Algorithms

#### Stochastic Gradient Descent
θ_{t+1} = θ_t - η_t ∇L_B(θ_t)

where B is mini-batch.

#### Momentum
v_{t+1} = β v_t + ∇L_B(θ_t)
θ_{t+1} = θ_t - η v_{t+1}

**Intuition**: Accumulates gradients to accelerate progress.

#### Adam
m_t = β_1 m_{t-1} + (1-β_1) g_t
v_t = β_2 v_{t-1} + (1-β_2) g_t^2
θ_t = θ_{t-1} - η m̂_t / (√v̂_t + ε)

where m̂_t, v̂_t are bias-corrected estimates.

## Implementation Details

See `exercise.py` for implementations of:
1. Feedforward networks from scratch
2. Backpropagation algorithm
3. Various initialization schemes
4. Optimization algorithms
5. Activation functions and their derivatives
6. Gradient checking utilities

## Experiments

1. **Universal Approximation**: Approximate various functions with single hidden layer
2. **Initialization Effects**: Compare different initialization schemes
3. **Optimization Landscapes**: Visualize loss surfaces
4. **Generalization**: Study overfitting vs network size

## Research Connections

### Foundational Papers
1. Rumelhart, Hinton & Williams (1986) - "Learning Representations by Back-Propagating Errors"
2. Cybenko (1989) - "Approximation by Superpositions of a Sigmoidal Function"
3. LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition"

### Modern Theory
1. Zhang et al. (2017) - "Understanding Deep Learning Requires Rethinking Generalization"
2. Jacot et al. (2018) - "Neural Tangent Kernel"
3. Neyshabur et al. (2017) - "Exploring Generalization in Deep Learning"

## Resources

### Primary Sources
1. **Goodfellow, Bengio & Courville - Deep Learning**
   - Comprehensive modern treatment
2. **Bishop - Pattern Recognition and Machine Learning**
   - Neural networks from statistical perspective
3. **Hastie, Tibshirani & Friedman - Elements of Statistical Learning**
   - Classical statistical learning view

### Video Resources
1. **Stanford CS231n - Convolutional Neural Networks**
   - Andrej Karpathy's course
2. **MIT 6.034 - Artificial Intelligence**
   - Neural network fundamentals
3. **Fast.ai - Deep Learning for Coders**
   - Practical implementation focus

### Advanced Reading
1. **Poggio & Smale - Mathematical Foundations of Learning**
   - Theoretical foundations
2. **Bartlett & Mendelson - Rademacher and Gaussian Complexities**
   - Generalization theory
3. **Roberts et al. - Principles of Deep Learning Theory**
   - Modern theoretical perspective

## Socratic Questions

### Understanding
1. Why do deep networks work better than shallow ones for many tasks?
2. How does backpropagation relate to the chain rule?
3. What makes neural network optimization non-convex?

### Extension
1. Can we prove global convergence for neural network training?
2. How do architectural choices affect expressivity?
3. What's the role of depth vs width in approximation power?

### Research
1. Why do overparameterized networks generalize well?
2. How can we design better optimization algorithms?
3. What are the fundamental limits of neural network learning?

## Exercises

### Theoretical
1. Prove that ReLU networks can represent any piecewise linear function
2. Derive the backpropagation equations from first principles
3. Analyze the effect of initialization on gradient flow

### Implementation
1. Build a neural network framework from scratch
2. Implement various optimizers and compare convergence
3. Create visualization tools for network behavior

### Research
1. Study the relationship between network width and approximation quality
2. Investigate the lottery ticket hypothesis empirically
3. Explore connections between neural networks and kernel methods