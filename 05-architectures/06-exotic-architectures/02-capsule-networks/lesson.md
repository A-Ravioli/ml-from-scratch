# Capsule Networks

## Prerequisites
- Convolutional neural networks and pooling operations
- Vector operations and transformations
- EM algorithm and iterative optimization
- Part-whole hierarchies in computer vision

## Learning Objectives
- Master the capsule concept and vector-based representations
- Understand dynamic routing algorithms and agreement mechanisms
- Implement routing-by-agreement and EM routing procedures
- Analyze equivariance properties and viewpoint invariance
- Connect to modern attention mechanisms and transformer architectures

## Mathematical Foundations

### 1. Motivation and Core Concepts

#### Limitations of CNNs
- **Pooling destroys information**: Max pooling discards precise locations
- **Lack of equivariance**: Not equivariant to viewpoint changes
- **Part-whole relationships**: Cannot explicitly model hierarchical structures
- **Internal representations**: Scalar activations lack rich structure

#### The Capsule Hypothesis
"A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity"

**Key Properties**:
- **Length**: Represents probability of entity existence
- **Orientation**: Represents instantiation parameters (pose, deformation)
- **Equivariance**: Changes in input cause predictable changes in capsule vectors

### 2. Vector Capsules

#### Capsule Definition
Capsule `i` produces vector `v_i ∈ ℝ^d`:
```
||v_i|| = probability of entity i being present
direction of v_i = instantiation parameters
```

#### Squashing Function
Ensures length is in [0, 1):
```
v_j = (||s_j||² / (1 + ||s_j||²)) * (s_j / ||s_j||)
```

where `s_j` is the total input to capsule j.

### 3. Dynamic Routing Algorithm

#### Routing-by-Agreement (Sabour et al., 2017)

**Prediction Vectors**:
Capsule i predicts what capsule j should output:
```
û_{j|i} = W_{ij} v_i
```

**Coupling Coefficients**:
```
c_{ij} = exp(b_{ij}) / Σ_k exp(b_{ik})
```

where `b_{ij}` are log priors (routing logits).

**Weighted Sum**:
```
s_j = Σ_i c_{ij} û_{j|i}
```

**Update Rule**:
```
b_{ij} ← b_{ij} + û_{j|i} · v_j
```

#### Dynamic Routing Procedure
```
1. Initialize: b_ij = 0 for all i, j
2. For r iterations:
   a. c_ij = softmax(b_ij) over j
   b. s_j = Σ_i c_ij û_{j|i}  
   c. v_j = squash(s_j)
   d. b_ij ← b_ij + û_{j|i} · v_j
3. Return: final v_j values
```

### 4. CapsNet Architecture

#### Primary Capsules Layer
**Convolutional Capsules**:
- Apply 9×9 conv with 256 channels
- Reshape to 8D vectors (32 capsules per location)
- Each location has 6×6 grid of capsule vectors

#### DigitCaps Layer  
**Fully Connected Capsules**:
- 10 capsules (one per digit class)
- 16D vectors
- Connected to all primary capsules via routing

#### Loss Function
**Margin Loss**:
```
L_k = T_k max(0, m⁺ - ||v_k||)² + λ(1-T_k) max(0, ||v_k|| - m⁻)²
```

where:
- `T_k = 1` if digit k is present
- `m⁺ = 0.9`, `m⁻ = 0.1`, `λ = 0.5`

**Reconstruction Loss**:
```
L_recon = ||image - reconstruction||²
```

**Total Loss**:
```
L = L_margin + α L_recon
```

### 5. EM Routing (Hinton et al., 2018)

#### Matrix Capsules
Each capsule outputs a matrix `M_i ∈ ℝ^{4×4}`:
- Represents viewpoint transformation
- More expressive than vector capsules

#### EM Routing Algorithm

**E-step**: Assign parts to wholes
```
r_{ij} = responsibility of part i for whole j
```

**M-step**: Update capsule parameters
```
μ_j, Σ_j = update using weighted assignments
```

#### Gaussian Clusters
Each higher-level capsule represents a Gaussian cluster:
```
p(v_j | capsule j active) = N(v_j; μ_j, Σ_j)
```

#### Iterative Updates
```
1. Initialize: activation probabilities a_j
2. E-step: compute assignment probabilities
   r_{ij} ∝ p(pose_{ij} | μ_j, Σ_j)
3. M-step: update Gaussian parameters
   μ_j, Σ_j = weighted_average(poses, r_{ij})
4. Update activations: a_j based on cluster tightness
```

### 6. Stacked Capsule Autoencoders

#### Set-to-Set Functions
**Object Capsule Encoder**:
```
O = SetTransformer(I)  # Image parts → Object capsules
```

**Image Decoder**:
```
Î = CNN_Decoder(O)  # Object capsules → Reconstructed image
```

#### Unsupervised Learning
**Loss Function**:
```
L = ||I - Î||² + β KL(posterior || prior)
```

**Constellation Loss**:
Encourages consistent object-part relationships across viewpoints.

### 7. Mathematical Properties

#### Equivariance Analysis

**Translation Equivariance**:
If input translates by `t`, capsule vectors should translate predictably.

**Rotation Equivariance**:
```
R(v_i) = R_matrix · v_i
```

**Viewpoint Consistency**:
Capsules should maintain part-whole agreements across viewpoints.

#### Parse Tree Representation
Capsules naturally represent parse trees:
- Leaf capsules: primitive features
- Internal capsules: parts and wholes
- Root capsule: complete object

### 8. Attention Connection

#### Routing as Attention
Dynamic routing ≈ attention mechanism:
```
attention_weight_{ij} = c_{ij}
context_vector_j = Σ_i c_{ij} û_{j|i}
```

#### Transformer Comparison
```
Transformer: attention_ij = softmax(Q_i K_j^T / √d)
Capsule: c_ij = softmax(b_ij)
```

Both learn to route information based on agreement/relevance.

### 9. Training Procedures

#### Curriculum Learning
1. Start with simple viewpoints
2. Gradually increase pose variation
3. Introduce occlusion and noise

#### Regularization Techniques
**Activity Regularization**:
Encourage sparse capsule activations:
```
L_activity = Σ_i ||v_i|| * sparsity_weight
```

**Spread Loss** (alternative to margin loss):
```
L_spread = Σ_{i≠target} max(0, margin - (||v_target|| - ||v_i||))²
```

### 10. Applications

#### Computer Vision
- **Object Recognition**: Part-whole hierarchies
- **Pose Estimation**: Viewpoint-aware representations
- **3D Understanding**: Depth and orientation modeling

#### Natural Language Processing
- **Parsing**: Syntactic structure representation
- **Compositional Semantics**: Part-meaning relationships
- **Entity Recognition**: Hierarchical entity properties

#### Graph Learning
- **Molecular Property Prediction**: Functional group capsules
- **Social Network Analysis**: Community structure modeling
- **Knowledge Graphs**: Relation-aware representations

### 11. Computational Challenges

#### Scalability Issues
- **Routing iterations**: Multiple forward passes required
- **Memory consumption**: O(n²) coupling coefficients
- **Computational cost**: Higher than standard CNNs

#### Solutions
**Efficient Routing**:
```
Sparse routing: limit connections between layers
Fast routing: reduce number of iterations
```

**Approximate Methods**:
- Stochastic routing
- Hierarchical softmax for routing
- Learned routing schedules

### 12. Modern Developments

#### Capsule Transformers
Combine capsules with transformer architectures:
```
CapsuleAttention(Q, K, V) = route(Q_caps, K_caps, V_caps)
```

#### Graph Capsules
Apply capsules to graph-structured data:
```
Graph routing: route between connected nodes
Hierarchical graphs: multi-level capsule hierarchies
```

#### Generative Capsules
Use capsules for generation:
```
VAE-Caps: capsule-based variational autoencoders
GAN-Caps: capsule discriminators and generators
```

## Implementation Details

See `exercise.py` for implementations of:
1. Vector capsules with squashing function
2. Dynamic routing-by-agreement algorithm
3. CapsNet architecture for MNIST/CIFAR
4. EM routing with matrix capsules
5. Reconstruction decoder and loss functions
6. Equivariance analysis and testing
7. Training procedures and optimization
8. Computational efficiency improvements