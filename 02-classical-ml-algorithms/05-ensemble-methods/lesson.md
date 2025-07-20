# Ensemble Methods for Machine Learning

## Prerequisites
- Understanding of bias-variance tradeoff
- Basic optimization theory
- Statistical learning theory (overfitting, generalization)
- Knowledge of base learning algorithms (trees, linear models)

## Learning Objectives
- Master the theory behind ensemble learning
- Understand when and why ensembles work
- Implement bagging, boosting, and stacking from scratch
- Learn advanced ensemble techniques and their trade-offs
- Connect ensemble methods to modern deep learning

## Mathematical Foundations

### 1. Ensemble Learning Theory

#### Definition 1.1 (Ensemble)
An ensemble is a collection of models whose predictions are combined:
F(x) = ∑ᵢ₌₁ᴹ wᵢfᵢ(x)

where fᵢ are base learners and wᵢ are combination weights.

#### Why Ensembles Work
**Condorcet's Jury Theorem**: If individual voters are more likely to be correct than incorrect and vote independently, then the majority vote is more likely to be correct than any individual.

**Requirements**:
1. **Accuracy**: Base learners better than random
2. **Diversity**: Base learners make different errors

#### Bias-Variance Decomposition for Ensembles
For regression, if base learners have bias B and variance V:
- **Bagging**: Bias = B, Variance = V/M (if uncorrelated)
- **Boosting**: Can reduce both bias and variance

### 2. Bootstrap Aggregating (Bagging)

#### Algorithm 2.1 (Bagging)
1. For b = 1 to B:
   - Draw bootstrap sample Dᵦ from training set D
   - Train model fᵦ on Dᵦ
2. Combine: F(x) = (1/B) ∑ᵦ fᵦ(x) (regression) or majority vote (classification)

#### Theoretical Analysis
**Theorem 2.1**: If base learners are unbiased and uncorrelated:
Var(ensemble) = Var(individual)/B

**In practice**: Learners are correlated, so variance reduction is less than 1/B.

#### Out-of-Bag (OOB) Error Estimation
Each bootstrap sample excludes ~36.8% of original data.
Use excluded samples as validation set for unbiased error estimate.

### 3. Random Forests Extension

#### Randomization Sources
1. **Bootstrap sampling**: Different training sets
2. **Feature randomness**: Random subset of features per split
3. **Threshold randomness**: Random thresholds (Extremely Randomized Trees)

#### Theoretical Properties
**Theorem 3.1 (Consistency)**: Under mild conditions, Random Forest error converges to Bayes error as number of trees → ∞.

### 4. Boosting Algorithms

#### AdaBoost (Adaptive Boosting)
**Key idea**: Focus on misclassified examples by reweighting.

**Algorithm 4.1 (AdaBoost)**:
1. Initialize weights: wᵢ⁽¹⁾ = 1/n
2. For t = 1 to T:
   - Train weak learner hₜ with weights wᵢ⁽ᵗ⁾
   - Compute error: εₜ = ∑ᵢ wᵢ⁽ᵗ⁾𝟙[hₜ(xᵢ) ≠ yᵢ]
   - Compute coefficient: αₜ = ½ln((1-εₜ)/εₜ)
   - Update weights: wᵢ⁽ᵗ⁺¹⁾ ∝ wᵢ⁽ᵗ⁾exp(-αₜyᵢhₜ(xᵢ))
3. Final classifier: H(x) = sign(∑ₜ αₜhₜ(x))

#### Theoretical Guarantees
**Theorem 4.1**: AdaBoost training error decreases exponentially:
∏ₜ 2√(εₜ(1-εₜ))

#### Gradient Boosting Framework
**Insight**: Boosting can be viewed as gradient descent in function space.

**Algorithm 4.2 (Gradient Boosting)**:
1. Initialize: F₀(x) = argmin_γ ∑ᵢ L(yᵢ, γ)
2. For m = 1 to M:
   - Compute negative gradients: rᵢₘ = -[∂L(yᵢ,F(xᵢ))/∂F(xᵢ)]_{F=F_{m-1}}
   - Train regressor hₘ to predict residuals rᵢₘ
   - Find optimal step: γₘ = argmin_γ ∑ᵢ L(yᵢ, F_{m-1}(xᵢ) + γhₘ(xᵢ))
   - Update: Fₘ(x) = F_{m-1}(x) + γₘhₘ(x)

### 5. Stacking (Stacked Generalization)

#### Two-Level Learning
**Level 0**: Train base learners on training data
**Level 1**: Train meta-learner on base learner predictions

#### Algorithm 5.1 (Stacking)
1. Split training data: 70% train, 30% holdout
2. Train base learners on train set
3. Get predictions on holdout set
4. Train meta-learner: meta-features → true labels
5. For test data: base predictions → meta-learner → final prediction

#### Theoretical Justification
Stacking learns optimal combination weights rather than using simple averaging.

### 6. Voting Methods

#### Hard Voting (Majority Vote)
ŷ = mode{f₁(x), f₂(x), ..., fₘ(x)}

#### Soft Voting (Weighted Average)
For classification with probabilities:
P(y=c|x) = (1/M) ∑ₘ Pₘ(y=c|x)

#### Optimal Voting Weights
**Theorem 6.1**: Optimal weights minimize expected loss:
w* = argmin_w E[L(y, ∑ᵢ wᵢfᵢ(x))]

For squared loss: wᵢ ∝ 1/MSE(fᵢ)

### 7. Diversity Measures

#### Disagreement Measures
**Q-statistic**: Correlation between classifier errors
**Correlation coefficient**: Linear correlation between outputs
**Entropy**: Information-theoretic diversity

#### Ambiguity Decomposition
**Theorem 7.1 (Krogh-Vedelsby)**: For regression ensembles:
Ensemble Error = Average Individual Error - Average Ambiguity

where Ambiguity measures disagreement among predictors.

### 8. Advanced Ensemble Techniques

#### Bayesian Model Averaging (BMA)
Weight models by posterior probability:
P(y|x,D) = ∑ₖ P(y|x,Mₖ,D)P(Mₖ|D)

#### Dynamic Ensemble Selection
Choose different ensemble members for different regions of input space.

#### Mixture of Experts
**Gating network**: Learns which expert to trust for each input
**Expert networks**: Specialized on different parts of input space

### 9. Multi-Class Ensemble Extensions

#### Error-Correcting Output Codes (ECOC)
1. Create binary encoding of classes
2. Train binary classifier for each bit position
3. Decode predictions to recover class

#### One-vs-All and One-vs-One
**One-vs-All**: K binary classifiers (class i vs rest)
**One-vs-One**: K(K-1)/2 binary classifiers (class i vs class j)

### 10. Ensemble Pruning

#### Motivation
Not all ensemble members contribute equally; some may hurt performance.

#### Pruning Strategies
1. **Forward selection**: Greedily add best performers
2. **Backward elimination**: Remove worst performers
3. **Genetic algorithms**: Evolutionary selection
4. **Clustering**: Keep representatives from each cluster

## Implementation Details

Key algorithms to implement:
1. Bootstrap sampling and OOB error estimation
2. AdaBoost with multiple weak learners
3. Gradient boosting with different loss functions
4. Stacking with cross-validation
5. Ensemble diversity measures
6. Dynamic model selection

## Applications and Use Cases

### When Ensembles Excel
1. **Noisy data**: Multiple models reduce overfitting
2. **Limited training data**: Variance reduction helps
3. **Complex decision boundaries**: Different models capture different patterns
4. **High-stakes decisions**: Uncertainty quantification important

### Computational Considerations
- **Parallel training**: Base learners often independent
- **Memory requirements**: Store multiple models
- **Prediction time**: Linear in number of models

## Research Connections

### Seminal Papers
1. Breiman (1996) - "Bagging Predictors"
2. Freund & Schapire (1997) - "A Decision-Theoretic Generalization of On-line Learning"
3. Wolpert (1992) - "Stacked Generalization"
4. Dietterich (2000) - "Ensemble Methods in Machine Learning"

### Modern Developments
1. **Deep ensembles**: Ensembles of neural networks
2. **Snapshot ensembles**: Multiple models from single training run
3. **Multi-task ensembles**: Shared representations across tasks

## Resources

### Primary Sources
1. **Zhou - Ensemble Methods: Foundations and Algorithms**
   - Comprehensive ensemble theory
2. **Hastie, Tibshirani & Friedman - Elements of Statistical Learning (Ch 8, 10, 15)**
3. **Rokach - Pattern Classification Using Ensemble Methods**

### Video Resources
1. **MIT 15.097 - Prediction Methods**
2. **Stanford CS229 - Ensemble Methods**

## Exercises

### Implementation
1. Implement bagging with different base learners
2. Code AdaBoost and visualize weight evolution
3. Build stacking framework with cross-validation
4. Create ensemble diversity visualization tools

### Research
1. Study bias-variance tradeoff empirically
2. Compare ensemble methods on various datasets
3. Investigate optimal ensemble size

## Advanced Topics

### Deep Learning Connections
- **Dropout**: Ensemble of sub-networks
- **BatchNorm**: Ensemble over mini-batches
- **Model averaging**: Weight averaging vs prediction averaging

### Modern Ensemble Learning
- **AutoML ensembles**: Automated ensemble construction
- **Continual learning**: Ensembles for non-stationary environments
- **Federated ensembles**: Distributed ensemble learning