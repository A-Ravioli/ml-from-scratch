# Tree-Based Methods for Machine Learning

## Prerequisites
- Basic probability theory (entropy, conditional probability)
- Information theory fundamentals
- Optimization theory (greedy algorithms)
- Understanding of overfitting and model complexity

## Learning Objectives
- Master decision tree construction and pruning algorithms
- Understand ensemble methods: bagging, random forests, boosting
- Connect information theory to feature selection
- Implement efficient tree algorithms from scratch
- Analyze bias-variance tradeoffs in tree ensembles

## Mathematical Foundations

### 1. Decision Trees: Information-Theoretic Foundation

#### Definition 1.1 (Entropy)
For a discrete random variable Y with probability mass function p(y):
H(Y) = -∑_{y} p(y) log₂ p(y)

**Interpretation**: Average information content (bits) needed to encode Y.

#### Definition 1.2 (Conditional Entropy)
H(Y|X) = ∑_{x} p(x) H(Y|X=x) = -∑_{x,y} p(x,y) log₂ p(y|x)

#### Definition 1.3 (Information Gain)
IG(Y,X) = H(Y) - H(Y|X)

**Physical meaning**: Reduction in uncertainty about Y after observing X.

#### Theorem 1.1 (Properties of Entropy)
1. H(Y) ≥ 0 with equality iff Y is deterministic
2. H(Y) ≤ log₂ |Y| with equality iff Y is uniform
3. IG(Y,X) ≥ 0 with equality iff X and Y are independent

### 2. CART Algorithm (Classification and Regression Trees)

#### Binary Tree Construction
At each node, find the split that maximizes information gain:
(X_j, t) = argmax_{j,t} IG(Y, X_j ≤ t)

For regression, replace entropy with variance:
IG_regression(Y, X_j ≤ t) = Var(Y) - [|L|Var(Y_L) + |R|Var(Y_R)]/|Y|

#### Algorithm 2.1 (CART Construction)
```
function BuildTree(D, depth=0):
    if StoppingCriterion(D, depth):
        return Leaf(MajorityClass(D))
    
    (j*, t*) = argmax_{j,t} InformationGain(D, j, t)
    
    D_left = {(x,y) ∈ D : x_j ≤ t*}
    D_right = {(x,y) ∈ D : x_j > t*}
    
    return Node(
        feature=j*,
        threshold=t*,
        left=BuildTree(D_left, depth+1),
        right=BuildTree(D_right, depth+1)
    )
```

#### Stopping Criteria
1. **Maximum depth**: Prevent trees from growing too deep
2. **Minimum samples**: Require minimum samples per leaf
3. **Minimum improvement**: Stop if information gain < threshold
4. **Pure nodes**: All samples have same label

### 3. Tree Pruning Theory

#### Problem: Overfitting
Deep trees memorize training data but generalize poorly.

#### Cost-Complexity Pruning (Weakest Link)
Define cost-complexity measure:
R_α(T) = R(T) + α|T_leaves|

where R(T) is misclassification rate and α controls complexity penalty.

#### Theorem 3.1 (Optimal Subtree Sequence)
For any α ≥ 0, there exists a unique minimal subtree T(α) that minimizes R_α(T).

**Proof sketch**: 
- For α = 0, T(0) is the full tree
- As α increases, internal nodes become "weakest links"
- Pruning weakest links gives nested sequence of optimal subtrees

#### Algorithm 3.1 (Minimal Cost-Complexity Pruning)
1. Grow full tree T₀
2. For each internal node t, compute:
   α(t) = [R(prune(t)) - R(t)] / [|leaves(t)| - 1]
3. Prune node with smallest α(t)
4. Repeat until root only

### 4. Random Forests: Bootstrap Aggregating

#### Bootstrap Sampling
Given dataset D = {(x₁,y₁), ..., (xₙ,yₙ)}, create B bootstrap samples:
D*ᵦ = {(x*₁,y*₁), ..., (x*ₙ,y*ₙ)} where each (x*ᵢ,y*ᵢ) is sampled with replacement from D

#### Algorithm 4.1 (Random Forest)
```
function RandomForest(D, B, m):
    Trees = []
    for b = 1 to B:
        D*_b = BootstrapSample(D)
        T_b = BuildRandomTree(D*_b, m)  // m features per split
        Trees.append(T_b)
    return Trees

function Predict(x, Trees):
    votes = [T.predict(x) for T in Trees]
    return MajorityVote(votes)  // or average for regression
```

#### Feature Randomness
At each split, randomly select m ≪ d features and find best split among them.
**Typical choices**: m = ⌊√d⌋ for classification, m = ⌊d/3⌋ for regression.

#### Theorem 4.1 (Out-of-Bag Error)
Each bootstrap sample excludes ~36.8% of original data. These "out-of-bag" (OOB) samples provide unbiased estimate of generalization error.

### 5. Boosting: Adaptive Reweighting

#### AdaBoost Algorithm
**Intuition**: Focus on misclassified examples by reweighting training data.

#### Algorithm 5.1 (AdaBoost.M1)
```
Input: Training set D = {(x₁,y₁),...,(xₙ,yₙ)}, yᵢ ∈ {-1,+1}

Initialize: w₁⁽ⁱ⁾ = 1/n for i = 1,...,n

For t = 1 to T:
    1. Train weak learner h_t using weights w_t
    2. Compute weighted error: ε_t = ∑ᵢ w_t⁽ⁱ⁾ 𝟙[h_t(xᵢ) ≠ yᵢ]
    3. Compute coefficient: α_t = ½ ln((1-ε_t)/ε_t)
    4. Update weights: w_{t+1}⁽ⁱ⁾ = w_t⁽ⁱ⁾ exp(-α_t yᵢ h_t(xᵢ)) / Z_t
    
Final classifier: H(x) = sign(∑_{t=1}^T α_t h_t(x))
```

#### Theorem 5.1 (AdaBoost Training Error)
The training error of AdaBoost decreases exponentially:
Pr[H(xᵢ) ≠ yᵢ] ≤ ∏_{t=1}^T 2√(ε_t(1-ε_t))

**Proof**: Use the fact that Z_t = 2√(ε_t(1-ε_t)) and bound indicator function.

#### Theorem 5.2 (Margin Theory)
AdaBoost maximizes the minimum margin:
margin(xᵢ) = yᵢ ∑_{t=1}^T α_t h_t(xᵢ) / ∑_{t=1}^T α_t

Generalization depends on margin distribution, not just training error.

### 6. Gradient Boosting

#### Functional Gradient Descent
Treat ensemble as function: F(x) = ∑_{t=1}^T f_t(x)

**Forward stagewise fitting**: At each step, add function that minimizes loss:
f_t = argmin_f ∑ᵢ L(yᵢ, F_{t-1}(xᵢ) + f(xᵢ))

#### Gradient Boosting Algorithm
**Key insight**: Fit f_t to negative gradient of loss function.

#### Algorithm 6.1 (Gradient Boosting)
```
Initialize: F₀(x) = argmin_γ ∑ᵢ L(yᵢ, γ)

For t = 1 to T:
    1. Compute negative gradients:
       rᵢₜ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=F_{t-1}}
    
    2. Fit regression tree h_t to {(xᵢ, rᵢₜ)}
    
    3. Find optimal step size:
       γ_t = argmin_γ ∑ᵢ L(yᵢ, F_{t-1}(xᵢ) + γh_t(xᵢ))
    
    4. Update: F_t(x) = F_{t-1}(x) + γ_t h_t(x)
```

#### Loss Functions
1. **Squared loss**: L(y,F) = ½(y-F)², gradient = F-y
2. **Logistic loss**: L(y,F) = log(1+exp(-yF)), gradient = -y/(1+exp(yF))
3. **Huber loss**: Robust to outliers

### 7. XGBoost: Extreme Gradient Boosting

#### Second-Order Taylor Approximation
L_t = ∑ᵢ L(yᵢ, F_{t-1}(xᵢ) + f_t(xᵢ))
    ≈ ∑ᵢ [L(yᵢ, F_{t-1}(xᵢ)) + gᵢf_t(xᵢ) + ½hᵢf_t²(xᵢ)]

where gᵢ = ∂L/∂F|_{F_{t-1}} and hᵢ = ∂²L/∂F²|_{F_{t-1}}

#### Regularized Objective
Obj_t = ∑ᵢ [gᵢf_t(xᵢ) + ½hᵢf_t²(xᵢ)] + Ω(f_t)

where Ω(f_t) = γT + ½λ∑_{j=1}^T w_j² is regularization term.

#### Optimal Leaf Weights
For tree structure T, optimal leaf weights:
w_j* = -∑_{i∈I_j} gᵢ / (∑_{i∈I_j} hᵢ + λ)

where I_j is set of samples in leaf j.

### 8. Feature Importance

#### Impurity-Based Importance
For each feature j:
Importance_j = ∑_{nodes using feature j} (weighted impurity decrease)

#### Permutation Importance
1. Compute baseline error on validation set
2. For each feature j:
   - Randomly permute feature j values
   - Compute new error
   - Importance = increase in error

#### SHAP Values (SHapley Additive exPlanations)
For prediction f(x), SHAP value φⱼ satisfies:
f(x) = E[f(X)] + ∑_{j=1}^d φⱼ

where φⱼ = ∑_{S⊆N\{j}} [|S|!(d-|S|-1)!/d!][f(S∪{j}) - f(S)]

### 9. Theoretical Analysis

#### Bias-Variance Decomposition for Trees
**Single tree**: High variance, low bias (overfits)
**Bagging**: Reduces variance while maintaining bias
**Boosting**: Reduces bias and variance but can overfit

#### Theorem 9.1 (Bagging Variance Reduction)
If base learners have variance σ² and correlation ρ:
Var(bagged predictor) = ρσ² + (1-ρ)σ²/B

As B → ∞: Var → ρσ²

#### Consistency Results
**Theorem 9.2**: Under regularity conditions, random forests are consistent:
E[L(Y, F_n(X))] → E[L(Y, f*(X))]

where f* is Bayes optimal predictor.

### 10. Computational Complexity

#### Tree Construction
- **Naive**: O(nd² log n) for each tree
- **Optimized**: O(nd log n) with pre-sorting
- **Histogram method**: O(nd) using discrete bins

#### Memory Efficiency
- **Sparse features**: Skip zero-valued features
- **Block structure**: Cache-friendly data layout
- **Approximate splitting**: Use quantiles instead of exact splits

## Applications and Use Cases

### When to Use Trees
1. **Mixed data types**: Handles categorical and numerical naturally
2. **Non-linear relationships**: Captures interactions automatically
3. **Interpretability**: Decision paths are human-readable
4. **Missing values**: Can handle without imputation

### When NOT to Use Trees
1. **Linear relationships**: Inefficient representation
2. **High-dimensional sparse data**: Text, images better suited for other methods
3. **Smooth functions**: Neural networks often better

### Feature Engineering for Trees
1. **Binning continuous variables**: Can improve splits
2. **Creating interaction features**: Helps capture complex relationships
3. **Temporal features**: Lags, moving averages for time series

## Implementation Details

See `exercise.py` for implementations of:
1. Decision tree with various splitting criteria
2. Pruning algorithms (cost-complexity, reduced error)
3. Random forest with feature randomness
4. AdaBoost with exponential loss
5. Gradient boosting with different loss functions
6. Feature importance calculation methods

## Experiments

1. **Bias-Variance Analysis**: Empirical decomposition for single tree vs ensemble
2. **Hyperparameter Sensitivity**: Tree depth, number of trees, learning rate
3. **Feature Importance**: Compare different importance measures
4. **Robustness**: Performance with noise, outliers, missing data

## Research Connections

### Seminal Papers
1. Breiman et al. (1984) - "Classification and Regression Trees (CART)"
2. Quinlan (1986) - "Induction of Decision Trees"
3. Breiman (1996) - "Bagging Predictors"
4. Breiman (2001) - "Random Forests"
5. Freund & Schapire (1997) - "A Decision-Theoretic Generalization of On-line Learning"
6. Friedman (2001) - "Greedy Function Approximation: A Gradient Boosting Machine"

### Modern Extensions
1. **Extremely Randomized Trees**: Random thresholds
2. **Rotation Forests**: Principal component preprocessing
3. **Oblique trees**: Linear combination splits
4. **Multi-output trees**: Structured prediction

### Theoretical Advances
1. **PAC-Bayesian analysis**: Generalization bounds for ensembles
2. **Stability analysis**: How small data changes affect predictions
3. **Approximation theory**: Representation power of tree ensembles

## Resources

### Primary Sources
1. **Hastie, Tibshirani & Friedman - Elements of Statistical Learning (Ch 9, 10, 15)**
   - Comprehensive coverage of tree methods
2. **James et al. - Introduction to Statistical Learning (Ch 8)**
   - Accessible introduction with R examples
3. **Murphy - Machine Learning: A Probabilistic Perspective (Ch 16)**
   - Probabilistic view of tree methods

### Video Resources
1. **MIT 6.034 - Decision Trees**
   - Patrick Winston's classic AI course
2. **Stanford CS229 - Tree Methods**
   - Andrew Ng's machine learning course
3. **Caltech CS156 - The VC Dimension**
   - Learning theory foundations

### Advanced Reading
1. **Biau & Scornet (2016) - "A Random Forest Guided Tour"**
   - Modern theoretical perspective
2. **Schapire (2013) - "Explaining AdaBoost"**
   - Comprehensive boosting survey
3. **Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"**
   - Engineering perspective on gradient boosting

## Socratic Questions

### Understanding
1. Why does information gain never decrease when adding a split?
2. How does feature randomness in Random Forests reduce overfitting?
3. What's the difference between bias reduction in boosting vs variance reduction in bagging?

### Extension
1. How would you modify CART for multi-output problems?
2. Can you design a tree algorithm that handles streaming data?
3. What happens to tree methods in very high dimensions?

### Research
1. How do modern neural networks relate to deep decision trees?
2. Can you combine differentiable trees with gradient descent?
3. What's the theoretical limit of ensemble method performance?

## Exercises

### Theoretical
1. Prove that information gain is always non-negative
2. Derive the optimal leaf weights for XGBoost
3. Show that AdaBoost minimizes exponential loss

### Implementation
1. Build decision tree from scratch with multiple splitting criteria
2. Implement Random Forest with out-of-bag error estimation
3. Code gradient boosting for regression and classification
4. Create visualization tools for tree structure and feature importance

### Research
1. Compare different pruning strategies empirically
2. Study the effect of correlation between base learners in ensembles
3. Investigate tree methods for structured output problems

## Advanced Topics

### Probabilistic Trees
- **Bayesian trees**: MCMC over tree structures
- **Soft trees**: Probabilistic splits instead of hard decisions
- **Tree-structured Gaussian processes**: Non-parametric Bayesian approach

### Deep Trees
- **Adaptive neural trees**: Learned splitting functions
- **Forest embeddings**: Trees as feature extractors for neural networks
- **Differentiable trees**: End-to-end gradient descent

### Specialized Domains
- **Survival trees**: Time-to-event data
- **Multi-task trees**: Shared structure across related problems
- **Federated trees**: Privacy-preserving distributed learning