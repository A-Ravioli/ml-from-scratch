# Semi-Supervised Learning for Machine Learning

## Prerequisites
- Graph theory basics (adjacency matrices, graph Laplacians)
- Linear algebra (eigendecompositions, matrix operations)
- Information theory (entropy, mutual information)
- Basic machine learning (supervised learning, clustering)

## Learning Objectives
- Understand when and why semi-supervised learning works
- Master label propagation and graph-based methods
- Implement self-training and co-training algorithms
- Learn generative and discriminative semi-supervised approaches
- Connect to modern deep semi-supervised learning

## Mathematical Foundations

### 1. Semi-Supervised Learning Setup

#### Problem Definition
Given:
- Labeled data: D_l = {(x₁, y₁), ..., (x_l, y_l)}
- Unlabeled data: D_u = {x_{l+1}, ..., x_{l+u}}
- Usually u ≫ l (much more unlabeled than labeled data)

Goal: Learn classifier that performs better than using D_l alone.

#### Fundamental Assumptions

**Smoothness Assumption**: If two points are close in input space, their outputs should be similar.

**Cluster Assumption**: Decision boundary should lie in low-density regions.

**Manifold Assumption**: Data lies on a low-dimensional manifold in high-dimensional space.

### 2. Graph-Based Methods

#### Graph Construction
Build graph G = (V, E) where:
- V: All data points (labeled + unlabeled)
- E: Edges between similar points

**k-NN Graph**: Connect each point to k nearest neighbors
**ε-neighborhood**: Connect points within distance ε
**Fully connected**: Weight by similarity function

#### Weight Matrix
W_{ij} = similarity(x_i, x_j)

Common choices:
- **Gaussian**: W_{ij} = exp(-||x_i - x_j||²/2σ²)
- **Cosine**: W_{ij} = (x_i · x_j)/(||x_i|| ||x_j||)

#### Graph Laplacian
**Unnormalized**: L = D - W
**Normalized**: L_rw = D⁻¹L = I - D⁻¹W
**Symmetric**: L_sym = D^{-1/2}LD^{-1/2}

where D is degree matrix: D_{ii} = ∑_j W_{ij}

### 3. Label Propagation

#### Algorithm 3.1 (Basic Label Propagation)
1. Initialize: Y₀ with known labels, 0 for unlabeled
2. Iterate: Y_{t+1} = αWY_t + (1-α)Y₀
3. Converge: Y* = (I - αW)⁻¹Y₀

**Interpretation**: Labels diffuse through graph according to edge weights.

#### Harmonic Function Approach
Minimize energy function:
E(f) = ½ ∑_{i,j} W_{ij}(f_i - f_j)²

**Solution**: f_u = -L_{uu}⁻¹L_{ul}f_l

where L_{uu}, L_{ul} are blocks of graph Laplacian.

### 4. Random Walk Interpretation

#### Absorbing Random Walk
- Labeled nodes: absorbing states
- Unlabeled nodes: transient states
- Transition probabilities: P_{ij} = W_{ij}/D_{ii}

**Class probability**: Probability of absorption at each class.

### 5. Self-Training (Self-Learning)

#### Algorithm 5.1 (Self-Training)
1. Train classifier on labeled data D_l
2. Predict on unlabeled data D_u
3. Select most confident predictions
4. Add to labeled set: D_l ← D_l ∪ {confident predictions}
5. Repeat until convergence or all data labeled

#### Confidence Measures
- **Prediction probability**: max_c P(y=c|x)
- **Entropy**: -∑_c P(y=c|x)log P(y=c|x)
- **Margin**: P(y=ĉ|x) - max_{c≠ĉ} P(y=c|x)

#### Theoretical Issues
**Problem**: Can reinforce initial mistakes
**Solution**: Use multiple classifiers (co-training)

### 6. Co-Training

#### Setup Requirements
1. **Feature splits**: X = X⁽¹⁾ ⊕ X⁽²⁾
2. **Sufficiency**: Each view sufficient for learning
3. **Independence**: Views conditionally independent given class

#### Algorithm 6.1 (Co-Training)
1. Train h₁ on D_l using features X⁽¹⁾
2. Train h₂ on D_l using features X⁽²⁾
3. Use h₁ to label examples for h₂ (most confident)
4. Use h₂ to label examples for h₁ (most confident)
5. Retrain both classifiers
6. Repeat until convergence

### 7. Generative Semi-Supervised Learning

#### EM Algorithm for Semi-Supervised Learning
**Model**: p(x,y|θ) = p(x|y,θ)p(y|θ)

**E-step**: Compute posterior probabilities for unlabeled data
q_{ij} = P(y_j = i|x_j, θ^{(t)})

**M-step**: Update parameters using labeled + weighted unlabeled data
θ^{(t+1)} = argmax_θ [∑_l log p(x_l,y_l|θ) + ∑_u ∑_i q_{ij} log p(x_j,y=i|θ)]

### 8. Discriminative Semi-Supervised Learning

#### Entropy Minimization
**Idea**: Minimize entropy of predictions on unlabeled data
L = L_supervised + λH(p_u)

where H(p_u) = -∑_{x∈D_u} ∑_c p(y=c|x) log p(y=c|x)

#### Pseudo-Labeling
Hard version of entropy minimization:
1. Generate pseudo-labels: ŷ = argmax_c p(y=c|x)
2. Train on labeled + pseudo-labeled data

### 9. Consistency Regularization

#### Temporal Ensembling
**Idea**: Predictions should be consistent across epochs
L = L_supervised + λ||p_t - ẑ_t||²

where ẑ_t is exponential moving average of predictions.

#### Mean Teacher
- **Student network**: Regular network being trained
- **Teacher network**: Exponential moving average of student weights
- **Loss**: Consistency between student and teacher predictions

### 10. Semi-Supervised Learning Theory

#### Probably Approximately Correct (PAC) Analysis
**Theorem 10.1**: Under cluster assumption, sample complexity can be significantly reduced.

#### Manifold Learning Connection
If data lies on d-dimensional manifold in D-dimensional space, effective sample complexity depends on d, not D.

#### When Semi-Supervised Learning Helps
1. **Smoothness**: Decision function should be smooth
2. **Low density separation**: Classes separated by low-density regions
3. **Manifold structure**: Data has underlying geometric structure

### 11. Modern Deep Semi-Supervised Learning

#### Variational Autoencoders (VAEs)
Learn latent representation z:
- Encoder: q(z|x)
- Decoder: p(x|z)
- Classifier: p(y|x) or p(y|z)

#### Generative Adversarial Networks (GANs)
- Generator: Creates realistic samples
- Discriminator: Distinguishes real vs fake + classifies labeled data

#### Consistency Regularization Methods
- **Pi-Model**: Consistency under stochastic augmentation
- **Temporal Ensembling**: Consistency across training epochs
- **MixMatch**: Combines multiple techniques

## Implementation Details

Key algorithms to implement:
1. Graph construction and Laplacian computation
2. Label propagation with different graph types
3. Self-training with confidence thresholding
4. Co-training with feature splits
5. EM algorithm for mixture models
6. Consistency regularization methods

## Applications and Use Cases

### When Semi-Supervised Learning Works
1. **Limited labeled data**: Expensive to obtain labels
2. **Natural feature splits**: Multiple views of data
3. **Manifold structure**: Data lies on low-dimensional manifold
4. **Cluster structure**: Classes form tight clusters

### Application Domains
1. **Text classification**: Web page categorization
2. **Image recognition**: Medical imaging with few annotations
3. **Speech recognition**: Unlabeled audio abundant
4. **Bioinformatics**: Protein function prediction
5. **Social networks**: Node classification

## Research Connections

### Seminal Papers
1. Zhu & Ghahramani (2002) - "Learning from Labeled and Unlabeled Data with Label Propagation"
2. Blum & Mitchell (1998) - "Combining Labeled and Unlabeled Data with Co-Training"
3. Chapelle, Schölkopf & Zien (2006) - "Semi-Supervised Learning"
4. Zhou et al. (2004) - "Learning with Local and Global Consistency"

### Modern Developments
1. **Deep semi-supervised learning**: VAEs, GANs for SSL
2. **Self-supervised learning**: Pretext tasks for representation learning
3. **Few-shot learning**: Learning from very few examples
4. **Domain adaptation**: Transfer learning with unlabeled target data

## Resources

### Primary Sources
1. **Chapelle, Schölkopf & Zien - Semi-Supervised Learning**
   - Comprehensive SSL overview
2. **Zhu - Semi-Supervised Learning Literature Survey**
3. **Zhou - Semi-Supervised Learning**

### Video Resources
1. **Stanford CS229 - Semi-Supervised Learning**
2. **MIT 6.867 - Machine Learning**

## Exercises

### Implementation
1. Implement label propagation on synthetic data
2. Build self-training algorithm with different base classifiers
3. Create co-training framework with natural feature splits
4. Visualize graph-based methods on 2D data

### Research
1. Study when semi-supervised learning helps vs hurts
2. Compare different graph construction methods
3. Investigate consistency regularization empirically

## Advanced Topics

### Multi-View Learning
- **Canonical Correlation Analysis**: Find correlated projections
- **Deep CCA**: Neural network extensions
- **Multi-view clustering**: Unsupervised multi-view methods

### Active Learning Connection
- **Semi-supervised active learning**: Choose which points to label
- **Uncertainty sampling**: Query most uncertain points
- **Query by committee**: Use ensemble disagreement

### Weakly Supervised Learning
- **Noisy labels**: Learning with incorrect labels
- **Partial labels**: Only some labels provided
- **Label proportions**: Only class distributions known