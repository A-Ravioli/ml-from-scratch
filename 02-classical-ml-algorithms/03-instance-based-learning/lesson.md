# Instance-Based Learning for Machine Learning

## Prerequisites
- Linear algebra (distance metrics, norms)
- Probability theory (density estimation, curse of dimensionality)
- Basic statistics (histograms, kernel methods)
- Computational complexity analysis

## Learning Objectives
- Master k-nearest neighbors and its variants
- Understand curse of dimensionality and distance concentration
- Implement efficient nearest neighbor search algorithms
- Learn kernel density estimation and bandwidth selection
- Connect to non-parametric statistics and local methods

## Mathematical Foundations

### 1. Distance Metrics and Similarity

#### Definition 1.1 (Metric Space)
A metric d: X × X → ℝ must satisfy:
1. d(x, y) ≥ 0 (non-negativity)
2. d(x, y) = 0 ⟺ x = y (identity)
3. d(x, y) = d(y, x) (symmetry)
4. d(x, z) ≤ d(x, y) + d(y, z) (triangle inequality)

#### Common Distance Metrics

**Minkowski Distance Family**:
L_p(x, y) = (∑_{i=1}^d |x_i - y_i|^p)^{1/p}

- **Manhattan (L₁)**: L₁(x, y) = ∑|x_i - y_i|
- **Euclidean (L₂)**: L₂(x, y) = √(∑(x_i - y_i)²)
- **Chebyshev (L∞)**: L∞(x, y) = max_i |x_i - y_i|

**Mahalanobis Distance**:
d_M(x, y) = √((x - y)ᵀ Σ⁻¹ (x - y))

where Σ is the covariance matrix. Accounts for feature correlation and scaling.

**Cosine Distance**:
d_cos(x, y) = 1 - (x · y)/(||x|| ||y||)

Measures angle between vectors, invariant to scaling.

### 2. k-Nearest Neighbors (k-NN)

#### Algorithm 2.1 (k-NN Classification)
Given query point x and training set D = {(x₁, y₁), ..., (xₙ, yₙ)}:

1. Compute distances: dᵢ = d(x, xᵢ) for i = 1, ..., n
2. Find k nearest neighbors: N_k(x) = {indices of k smallest distances}
3. Predict: ŷ = mode({yᵢ : i ∈ N_k(x)})

#### Algorithm 2.2 (k-NN Regression)
Same as classification, but predict:
ŷ = (1/k) ∑_{i ∈ N_k(x)} yᵢ

#### Weighted k-NN
Instead of uniform voting, weight by inverse distance:
w_i = 1/d(x, xᵢ) (or w_i = exp(-d(x, xᵢ)²/σ²))

**Classification**: ŷ = argmax_c ∑_{i ∈ N_k(x)} w_i 𝟙[yᵢ = c]
**Regression**: ŷ = ∑_{i ∈ N_k(x)} w_i yᵢ / ∑_{i ∈ N_k(x)} w_i

### 3. Theoretical Analysis of k-NN

#### Theorem 3.1 (Universal Consistency)
Let R* be the Bayes risk and R_n be the k-NN risk. If:
- k → ∞ as n → ∞
- k/n → 0 as n → ∞

Then R_n → R* almost surely.

**Proof sketch**: 
- As n → ∞, k-NN becomes local averaging
- With k/n → 0, neighborhoods shrink
- Local averaging approaches conditional expectation

#### Theorem 3.2 (Rate of Convergence)
Under smoothness assumptions, the excess risk satisfies:
E[R_n - R*] = O(n^{-4/(4+d)})

where d is the dimension. This shows the **curse of dimensionality**.

#### Bias-Variance Decomposition
For k-NN regression:
- **Bias**: Increases with k (more smoothing)
- **Variance**: Decreases with k (more averaging)
- **Optimal k**: Balances bias-variance tradeoff

### 4. Curse of Dimensionality

#### Distance Concentration Phenomenon
**Theorem 4.1**: For i.i.d. points in high dimensions with bounded moments:
d_max(x) - d_min(x) / d_min(x) → 0 in probability as d → ∞

**Implication**: All points become approximately equidistant, making nearest neighbors meaningless.

#### Volume Concentration
In d-dimensional unit hypercube, most volume is near the boundary:
Volume_shell(r) / Volume_total = 1 - (1-r)^d

For r = 0.1 and d = 10: ~65% of volume is in outer 10% shell.

#### Practical Implications
1. **Sample complexity**: Need exponentially more data as d increases
2. **Distance metrics**: Euclidean distance becomes less discriminative
3. **Feature selection**: Crucial to remove irrelevant dimensions
4. **Dimensionality reduction**: PCA, t-SNE before k-NN

### 5. Efficient Nearest Neighbor Search

#### Brute Force: O(nd) per query
Simply compute distance to all training points.

#### k-d Trees
**Idea**: Recursively partition space with axis-aligned hyperplanes.

#### Algorithm 5.1 (k-d Tree Construction)
```
function BuildKDTree(points, depth=0):
    if points.empty():
        return null
    
    axis = depth % d  // cycle through dimensions
    points.sort(key=lambda p: p[axis])
    median = len(points) // 2
    
    return Node(
        point=points[median],
        axis=axis,
        left=BuildKDTree(points[:median], depth+1),
        right=BuildKDTree(points[median+1:], depth+1)
    )
```

**Search complexity**: O(log n) average, O(n) worst case
**Works well**: Low dimensions (d ≤ 10)
**Fails**: High dimensions due to curse of dimensionality

#### Ball Trees
**Idea**: Recursively partition points into hyperspheres.

Better for high dimensions than k-d trees, but still struggles with d > 20.

#### Locality-Sensitive Hashing (LSH)
**Idea**: Hash similar points to same buckets with high probability.

**Family of hash functions**: H is (r₁, r₂, p₁, p₂)-sensitive if:
- If d(x, y) ≤ r₁: Pr[h(x) = h(y)] ≥ p₁
- If d(x, y) ≥ r₂: Pr[h(x) = h(y)] ≤ p₂

**Example for Euclidean space**:
h(x) = ⌊(a · x + b) / w⌋

where a ~ N(0, I), b ~ U[0, w], w is bucket width.

### 6. Kernel Density Estimation

#### Motivation
Estimate probability density p(x) from samples {x₁, ..., xₙ}.

#### Definition 6.1 (Kernel Density Estimator)
p̂(x) = (1/n) ∑_{i=1}^n (1/h^d) K((x - xᵢ)/h)

where:
- K is the kernel function
- h is the bandwidth parameter
- d is the dimension

#### Common Kernels
**Gaussian**: K(u) = (2π)^{-d/2} exp(-||u||²/2)
**Epanechnikov**: K(u) = (3/4)(1 - ||u||²) if ||u|| ≤ 1, else 0
**Uniform**: K(u) = 1 if ||u|| ≤ 1, else 0

#### Bandwidth Selection
**Too small h**: Undersmoothing, spiky estimates
**Too large h**: Oversmoothing, loss of detail

#### Silverman's Rule of Thumb
For Gaussian kernel in 1D:
h = 1.06 σ̂ n^{-1/5}

where σ̂ is sample standard deviation.

#### Cross-Validation Bandwidth
Choose h minimizing leave-one-out log-likelihood:
h* = argmax_h ∑_{i=1}^n log p̂_{-i}(xᵢ)

where p̂_{-i} is KDE excluding point i.

### 7. Theoretical Properties of KDE

#### Theorem 7.1 (Bias and Variance)
Under regularity conditions:

**Bias**: E[p̂(x)] - p(x) = (h²/2) ∇²p(x) ∫ u² K(u) du + O(h⁴)

**Variance**: Var[p̂(x)] = (1/nh^d) p(x) ∫ K²(u) du + O(1/n)

#### Optimal Bandwidth
Minimizing Mean Integrated Squared Error (MISE):
h_opt = C n^{-1/(d+4)}

where C depends on kernel and true density.

#### Curse of Dimensionality for KDE
Convergence rate: O(n^{-4/(d+4)})

For d = 1: O(n^{-4/5})
For d = 10: O(n^{-4/14}) ≈ O(n^{-2/7})

### 8. Local Regression Methods

#### k-NN Regression Extension
Instead of averaging, fit local models.

#### Algorithm 8.1 (Local Linear Regression)
For query point x:
1. Find k nearest neighbors N_k(x)
2. Fit weighted linear regression:
   min_β ∑_{i ∈ N_k(x)} w_i (yᵢ - β₀ - β₁ᵀ(xᵢ - x))²
3. Predict: ŷ = β₀

#### LOWESS (Locally Weighted Scatterplot Smoothing)
Combines local regression with robust weights to handle outliers.

### 9. Lazy vs Eager Learning

#### Instance-Based = Lazy Learning
- **No training phase**: Store all training data
- **All computation at query time**: Find neighbors, make prediction
- **Memory**: O(n) storage
- **Query time**: Expensive without indexing

#### Advantages
- **Adapts to local patterns**: Different behavior in different regions
- **Handles complex decision boundaries**: No parametric assumptions
- **Incorporates new data easily**: Just add to training set

#### Disadvantages
- **Computational cost**: Expensive at query time
- **Storage requirements**: Must store entire training set
- **Curse of dimensionality**: Performance degrades in high dimensions
- **Sensitive to irrelevant features**: Equal weighting of all dimensions

### 10. Feature Selection and Weighting

#### Relief Algorithm
**Idea**: Weight features by their ability to distinguish near hits/misses.

#### Algorithm 10.1 (Relief)
```
for each feature f:
    W[f] = 0

for i = 1 to m:  // sample m instances
    randomly select instance R
    find nearest hit H (same class)
    find nearest miss M (different class)
    
    for each feature f:
        W[f] = W[f] - diff(f, R, H)/m + diff(f, R, M)/m
```

where diff(f, x, y) = |x_f - y_f| for continuous features.

#### Learning Distance Metrics
**Large Margin Nearest Neighbor (LMNN)**:
Learn Mahalanobis distance matrix M that:
- Pulls same-class neighbors closer
- Pushes different-class points apart

### 11. Variations and Extensions

#### Radius-Based Neighbors
Instead of k nearest, use all points within radius r:
N_r(x) = {i : d(x, xᵢ) ≤ r}

**Advantage**: Adapts to local density
**Disadvantage**: May have empty neighborhoods

#### Locally Adaptive k-NN
Choose k based on local density or class distribution.

#### Condensed Nearest Neighbor
Reduce training set size while maintaining classification accuracy.

#### Edited Nearest Neighbor
Remove noisy training points that are misclassified by their neighbors.

### 12. Computational Considerations

#### Approximation Methods
**Best Bin First**: Approximate k-d tree search
**Randomized k-d Trees**: Multiple random trees
**Hierarchical k-means**: Tree of k-means clusters

#### Parallel Algorithms
- **Data parallelism**: Distribute training points
- **Query parallelism**: Process multiple queries simultaneously
- **GPU acceleration**: Massive parallel distance computations

#### Memory Hierarchy
- **Cache-friendly layouts**: Z-order curves, cache-oblivious structures
- **External memory**: Algorithms for data larger than RAM

## Applications and Use Cases

### When to Use Instance-Based Methods
1. **Complex decision boundaries**: Non-linear, irregular shapes
2. **Local patterns**: Different behavior in different regions
3. **Small datasets**: When parametric models might overfit
4. **Incremental learning**: Easy to add new training examples

### When NOT to Use
1. **High dimensions**: Curse of dimensionality
2. **Large datasets**: Memory and computational constraints
3. **Real-time applications**: Query-time computation expensive
4. **Many irrelevant features**: All features weighted equally

### Application Domains
1. **Recommendation systems**: User-item collaborative filtering
2. **Image classification**: Template matching, face recognition
3. **Time series**: Finding similar patterns
4. **Anomaly detection**: Outliers have few neighbors
5. **Information retrieval**: Document similarity

## Implementation Details

See `exercise.py` for implementations of:
1. k-NN classifier and regressor with multiple distance metrics
2. Weighted k-NN variants
3. k-d tree construction and search
4. Kernel density estimation with bandwidth selection
5. Local regression methods
6. Feature weighting algorithms

## Experiments

1. **Curse of Dimensionality**: Performance vs dimension
2. **k Selection**: Cross-validation for optimal k
3. **Distance Metrics**: Comparison on different datasets
4. **Indexing Structures**: Runtime comparison of search methods
5. **Bandwidth Selection**: KDE performance with different bandwidths

## Research Connections

### Seminal Papers
1. Cover & Hart (1967) - "Nearest Neighbor Pattern Classification"
   - First theoretical analysis of k-NN
2. Fix & Hodges (1951) - "Discriminatory Analysis"
   - Original k-NN algorithm
3. Bentley (1975) - "Multidimensional Binary Search Trees"
   - k-d trees for efficient search
4. Indyk & Motwani (1998) - "Approximate Nearest Neighbors"
   - Locality-sensitive hashing
5. Silverman (1986) - "Density Estimation for Statistics and Data Analysis"
   - Comprehensive KDE theory

### Modern Developments
1. **Deep metric learning**: Learn embeddings for better k-NN
2. **Approximate methods**: FAISS, Annoy for large-scale NN search
3. **Kernel machines**: Connection to support vector machines
4. **Graph-based methods**: k-NN graphs for semi-supervised learning

## Resources

### Primary Sources
1. **Hastie, Tibshirani & Friedman - Elements of Statistical Learning (Ch 13)**
   - Comprehensive coverage of prototype methods
2. **Duda, Hart & Stork - Pattern Classification (Ch 4)**
   - Classical treatment of nearest neighbor methods
3. **Silverman - Density Estimation for Statistics and Data Analysis**
   - Definitive reference for kernel density estimation

### Video Resources
1. **MIT 6.034 - Nearest Neighbors**
   - Patrick Winston's AI course
2. **Stanford CS229 - Locally Weighted Regression**
   - Andrew Ng's machine learning course
3. **Caltech CS156 - Nonparametric Methods**
   - Yaser Abu-Mostafa's learning theory

### Advanced Reading
1. **Györfi et al. - A Distribution-Free Theory of Nonparametric Regression**
   - Theoretical foundations
2. **Devroye, Györfi & Lugosi - A Probabilistic Theory of Pattern Recognition**
   - Mathematical analysis of k-NN
3. **Scott - Multivariate Density Estimation**
   - Advanced KDE theory and practice

## Socratic Questions

### Understanding
1. Why does k-NN become less effective as dimensionality increases?
2. How does the choice of distance metric affect k-NN performance?
3. What's the relationship between kernel density estimation and k-NN?

### Extension
1. How would you modify k-NN for imbalanced datasets?
2. Can you design a distance metric that adapts to local density?
3. What happens to k-NN in the limit as k → n?

### Research
1. How do modern deep learning embeddings improve k-NN?
2. Can you combine parametric and non-parametric methods effectively?
3. What's the optimal data structure for nearest neighbor search in 1000 dimensions?

## Exercises

### Theoretical
1. Prove that 1-NN has at most twice the Bayes error for large n
2. Derive the bias-variance decomposition for k-NN regression
3. Show that kernel density estimation converges to true density

### Implementation
1. Implement k-NN with multiple distance metrics and weighting schemes
2. Build k-d tree from scratch with visualization
3. Code kernel density estimation with automatic bandwidth selection
4. Create local regression methods (LOWESS)

### Research
1. Study curse of dimensionality empirically across datasets
2. Compare indexing structures for different data distributions
3. Investigate feature selection methods for high-dimensional k-NN

## Advanced Topics

### Conformal Prediction
Use k-NN to provide prediction intervals with guaranteed coverage.

### Manifold Learning
Assume data lies on low-dimensional manifold, use geodesic distances.

### Active Learning
Choose which points to label to maximize k-NN performance.

### Multi-task Learning
Share neighborhood structure across related tasks.

### Online Learning
Maintain approximate nearest neighbors as data streams arrive.