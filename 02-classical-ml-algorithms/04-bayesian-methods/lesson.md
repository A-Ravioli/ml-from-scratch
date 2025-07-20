# Bayesian Methods for Machine Learning

## Prerequisites
- Probability theory (Bayes' theorem, conditional probability)
- Statistics (likelihood, prior/posterior distributions)
- Linear algebra (matrix operations, eigendecompositions)
- Calculus (optimization, derivatives)
- Basic knowledge of Gaussian distributions

## Learning Objectives
- Master Naive Bayes classification and its variants
- Understand Bayesian linear regression and model comparison
- Implement Gaussian Processes from first principles
- Learn variational inference and MCMC basics
- Connect Bayesian thinking to regularization and uncertainty quantification

## Mathematical Foundations

### 1. Bayes' Theorem and Bayesian Inference

#### Bayes' Theorem
For events A and B:
P(A|B) = P(B|A)P(A) / P(B)

#### Bayesian Learning Framework
**Posterior = Likelihood × Prior / Evidence**

P(θ|D) = P(D|θ)P(θ) / P(D)

where:
- θ: parameters
- D: observed data
- P(θ): prior belief about parameters
- P(D|θ): likelihood of data given parameters
- P(θ|D): posterior belief after seeing data
- P(D): marginal likelihood (evidence)

#### Maximum A Posteriori (MAP) Estimation
θ_MAP = argmax_θ P(θ|D) = argmax_θ P(D|θ)P(θ)

#### Bayesian Prediction
For new data x*:
P(y*|x*, D) = ∫ P(y*|x*, θ)P(θ|D) dθ

This integrates over parameter uncertainty.

### 2. Naive Bayes Classification

#### Assumption: Conditional Independence
P(x₁, x₂, ..., xₐ|y) = ∏ᵢ₌₁ᵈ P(xᵢ|y)

Features are independent given the class label.

#### Classification Rule
ŷ = argmax_c P(y=c|x) = argmax_c P(x|y=c)P(y=c)

Taking logarithms:
ŷ = argmax_c [log P(y=c) + ∑ᵢ₌₁ᵈ log P(xᵢ|y=c)]

#### Gaussian Naive Bayes
Assume P(xᵢ|y=c) ~ N(μᵢc, σᵢc²)

**Parameters to estimate**:
- Class priors: P(y=c) = n_c / n
- Feature means: μᵢc = mean of feature i in class c
- Feature variances: σᵢc² = variance of feature i in class c

#### Multinomial Naive Bayes
For discrete features (e.g., word counts):
P(xᵢ|y=c) = Multinomial(θᵢc)

With Laplace smoothing:
θᵢc = (count(xᵢ, c) + α) / (∑ⱼ count(xⱼ, c) + α|V|)

#### Bernoulli Naive Bayes
For binary features:
P(xᵢ|y=c) = θᵢc^xᵢ (1-θᵢc)^(1-xᵢ)

### 3. Bayesian Linear Regression

#### Model Setup
y = Xβ + ε, where ε ~ N(0, σ²I)

#### Prior Distribution
β ~ N(μ₀, Σ₀)

Common choice: μ₀ = 0, Σ₀ = τ²I (isotropic prior)

#### Likelihood
P(y|X, β, σ²) = N(Xβ, σ²I)

#### Posterior Distribution
P(β|X, y, σ²) = N(μₙ, Σₙ)

where:
Σₙ = (Σ₀⁻¹ + σ⁻²XᵀX)⁻¹
μₙ = Σₙ(Σ₀⁻¹μ₀ + σ⁻²Xᵀy)

#### Predictive Distribution
For new input x*:
P(y*|x*, X, y) = N(x*ᵀμₙ, x*ᵀΣₙx* + σ²)

**Key insight**: Prediction includes both parameter uncertainty (x*ᵀΣₙx*) and noise (σ²).

#### Connection to Ridge Regression
MAP estimate: β_MAP = argmax_β P(β|X, y)

With isotropic prior β ~ N(0, τ²I):
β_MAP = (XᵀX + σ²/τ² I)⁻¹Xᵀy

This is Ridge regression with λ = σ²/τ²!

### 4. Model Comparison and Evidence

#### Marginal Likelihood (Evidence)
P(D) = ∫ P(D|θ)P(θ) dθ

Measures how well model explains data, averaged over all parameter values.

#### Bayes Factors
For comparing models M₁ and M₂:
BF₁₂ = P(D|M₁) / P(D|M₂)

- BF > 3: Moderate evidence for M₁
- BF > 10: Strong evidence for M₁
- BF > 100: Decisive evidence for M₁

#### Automatic Relevance Determination (ARD)
Use hierarchical priors to automatically determine feature relevance:

βⱼ ~ N(0, αⱼ⁻¹)
αⱼ ~ Gamma(a, b)

Features with large αⱼ are effectively turned off.

### 5. Gaussian Processes

#### Definition
A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution.

f(x) ~ GP(m(x), k(x, x'))

where:
- m(x) = E[f(x)] is the mean function
- k(x, x') = Cov[f(x), f(x')] is the covariance function (kernel)

#### Common Kernels

**Squared Exponential (RBF)**:
k(x, x') = σ²exp(-||x - x'||²/(2ℓ²))

**Matérn 3/2**:
k(x, x') = σ²(1 + √3r/ℓ)exp(-√3r/ℓ), r = ||x - x'||

**Periodic**:
k(x, x') = σ²exp(-2sin²(π|x-x'|/p)/ℓ²)

**Linear**:
k(x, x') = σ²xᵀx'

#### GP Regression

**Model**:
f(x) ~ GP(0, k(x, x'))
y = f(x) + ε, ε ~ N(0, σₙ²)

**Training data**: D = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ

**Posterior predictive distribution**:
P(f*|x*, D) = N(μ*, σ²*)

where:
μ* = k*ᵀ(K + σₙ²I)⁻¹y
σ²* = k** - k*ᵀ(K + σₙ²I)⁻¹k*

- k* = [k(x*, x₁), ..., k(x*, xₙ)]ᵀ
- K = [k(xᵢ, xⱼ)]ᵢⱼ (Gram matrix)
- k** = k(x*, x*)

#### Hyperparameter Learning
Maximize marginal likelihood:
log P(y|X, θ) = -½yᵀ(K + σₙ²I)⁻¹y - ½log|K + σₙ²I| - n/2 log(2π)

where θ contains kernel hyperparameters.

#### GP Classification
For classification, use sigmoid link function:
P(y=1|f) = σ(f) = 1/(1 + exp(-f))

Posterior is non-Gaussian; use approximations:
- Laplace approximation
- Variational inference
- Expectation propagation

### 6. Variational Inference

#### Problem: Intractable Posteriors
For complex models, P(θ|D) cannot be computed analytically.

#### Variational Approximation
Approximate P(θ|D) with simpler distribution q(θ).

**Goal**: Minimize KL divergence:
KL(q||p) = ∫ q(θ) log(q(θ)/P(θ|D)) dθ

#### Evidence Lower Bound (ELBO)
log P(D) = ELBO + KL(q||p)

where:
ELBO = E_q[log P(D, θ)] - E_q[log q(θ)]

**Optimization**: Maximize ELBO ⟺ Minimize KL divergence

#### Mean Field Approximation
Assume q(θ) = ∏ᵢ qᵢ(θᵢ) (factorized approximation)

**Coordinate ascent**: Update each factor in turn:
log q*ⱼ(θⱼ) = E_{q₋ⱼ}[log P(D, θ)] + const

#### Variational Bayes for Linear Regression
**Model**: y = Xβ + ε, β ~ N(0, α⁻¹I), α ~ Gamma(a, b)

**Variational distributions**:
q(β) = N(μ_β, Σ_β)
q(α) = Gamma(a_α, b_α)

**Updates** (coordinate ascent):
- Update q(β) given q(α)
- Update q(α) given q(β)
- Iterate until convergence

### 7. Markov Chain Monte Carlo (MCMC)

#### Motivation
When variational approximation is insufficient, use sampling.

#### Metropolis-Hastings Algorithm
1. Propose new state: θ* ~ q(θ*|θ⁽ᵗ⁾)
2. Compute acceptance probability:
   α = min(1, [P(θ*)π(θ*|θ⁽ᵗ⁾)] / [P(θ⁽ᵗ⁾)π(θ⁽ᵗ⁾|θ*)])
3. Accept with probability α

#### Gibbs Sampling
For multivariate θ = (θ₁, ..., θₖ):
1. Sample θ₁⁽ᵗ⁺¹⁾ ~ P(θ₁|θ₂⁽ᵗ⁾, ..., θₖ⁽ᵗ⁾, D)
2. Sample θ₂⁽ᵗ⁺¹⁾ ~ P(θ₂|θ₁⁽ᵗ⁺¹⁾, θ₃⁽ᵗ⁾, ..., θₖ⁽ᵗ⁾, D)
3. Continue for all parameters

#### Hamiltonian Monte Carlo (HMC)
Use gradient information to make smarter proposals:
1. Introduce momentum variables p
2. Simulate Hamiltonian dynamics
3. Propose based on trajectory

More efficient than random walk methods.

### 8. Bayesian Neural Networks

#### Uncertainty in Neural Networks
Standard NNs give point estimates. Bayesian NNs quantify uncertainty.

#### Weight Uncertainty
Place priors on network weights:
w ~ N(0, σ²I)

**Challenge**: Posterior is intractable for realistic networks.

#### Approximation Methods
**Variational dropout**: Treat dropout as approximate inference
**Monte Carlo dropout**: Use dropout at test time
**Bayes by Backprop**: Variational inference for weights

#### Predictive Uncertainty
**Epistemic uncertainty**: Due to model/parameter uncertainty
**Aleatoric uncertainty**: Inherent noise in data

### 9. Bayesian Optimization

#### Problem
Optimize expensive black-box function f(x):
x* = argmax_x f(x)

Examples: Hyperparameter tuning, experimental design

#### Gaussian Process Surrogate
Model f(x) with GP: f ~ GP(μ, k)

**Posterior mean**: Best current estimate of f
**Posterior variance**: Uncertainty about f

#### Acquisition Functions
Balance exploration vs exploitation:

**Expected Improvement**:
EI(x) = E[max(f(x) - f⁺, 0)]

**Upper Confidence Bound**:
UCB(x) = μ(x) + βσ(x)

**Probability of Improvement**:
PI(x) = P(f(x) > f⁺)

#### Algorithm
1. Fit GP to observed data
2. Optimize acquisition function to choose next x
3. Evaluate f(x), add to dataset
4. Repeat until budget exhausted

### 10. Hierarchical Bayesian Models

#### Motivation
Share information across related problems/groups.

#### Model Structure
**Level 1**: yᵢⱼ ~ P(yᵢⱼ|θᵢ) (individual observations)
**Level 2**: θᵢ ~ P(θᵢ|φ) (group-level parameters)
**Level 3**: φ ~ P(φ) (population-level hyperparameters)

#### Example: School Test Scores
- yᵢⱼ: score of student j in school i
- θᵢ: average ability in school i
- φ: population parameters

**Shrinkage effect**: Individual estimates pulled toward population mean.

#### Empirical Bayes
Use data to estimate hyperparameters:
φ̂ = argmax_φ ∏ᵢ P(yᵢ|φ)

Then use P(θᵢ|yᵢ, φ̂) for inference.

## Computational Considerations

### Numerical Stability
- **Log-space computations**: Avoid underflow
- **Cholesky decomposition**: For positive definite matrices
- **Woodbury identity**: Efficient matrix inversions

### Scalability
- **Inducing points**: Sparse GP approximations
- **Variational inference**: Scalable to large datasets
- **Stochastic optimization**: Mini-batch methods

### Software Tools
- **Stan**: Probabilistic programming language
- **PyMC3**: Python MCMC library
- **GPy/GPyTorch**: Gaussian process libraries
- **Edward/TensorFlow Probability**: Deep probabilistic programming

## Applications and Use Cases

### When to Use Bayesian Methods
1. **Uncertainty quantification**: Need confidence intervals
2. **Small datasets**: Priors help with limited data
3. **Sequential decisions**: Online learning, active learning
4. **Model selection**: Compare different model structures
5. **Interpretability**: Prior knowledge incorporation

### Application Domains
1. **Medical diagnosis**: Uncertainty crucial for treatment decisions
2. **A/B testing**: Bayesian bandit algorithms
3. **Recommender systems**: Collaborative filtering with uncertainty
4. **Time series**: State-space models, changepoint detection
5. **Computer vision**: Bayesian deep learning for safety-critical applications

## Implementation Details

See `exercise.py` for implementations of:
1. Naive Bayes (Gaussian, Multinomial, Bernoulli)
2. Bayesian linear regression with uncertainty quantification
3. Gaussian Processes with multiple kernels
4. Variational inference for simple models
5. MCMC samplers (Metropolis-Hastings, Gibbs)
6. Bayesian model comparison

## Experiments

1. **Naive Bayes Variants**: Compare on text and continuous data
2. **Uncertainty Calibration**: How well do confidence intervals match reality?
3. **GP Kernel Comparison**: Effect of different kernels on regression
4. **MCMC Diagnostics**: Convergence assessment and effective sample size
5. **Bayesian vs Frequentist**: Compare approaches on same problems

## Research Connections

### Seminal Papers
1. Laplace (1774) - "Memoir on the Probability of Causes of Events"
   - Early Bayesian inference
2. Good (1965) - "The Estimation of Probabilities"
   - Modern Bayesian statistics
3. Neal (1996) - "Bayesian Learning for Neural Networks"
   - Bayesian deep learning foundations
4. Williams & Rasmussen (2006) - "Gaussian Processes for Machine Learning"
   - Comprehensive GP treatment
5. Blei, Kucukelbir & McAuliffe (2017) - "Variational Inference: A Review"
   - Modern variational methods

### Modern Developments
1. **Variational autoencoders**: Deep generative models
2. **Bayesian deep learning**: Uncertainty in neural networks
3. **Neural processes**: Combining GPs and neural networks
4. **Normalizing flows**: Flexible posterior approximations
5. **Differentiable programming**: End-to-end Bayesian learning

## Resources

### Primary Sources
1. **Bishop - Pattern Recognition and Machine Learning (Ch 3, 4, 9)**
   - Excellent Bayesian foundations
2. **Murphy - Machine Learning: A Probabilistic Perspective**
   - Comprehensive probabilistic ML
3. **Rasmussen & Williams - Gaussian Processes for Machine Learning**
   - Definitive GP reference
4. **Gelman et al. - Bayesian Data Analysis**
   - Applied Bayesian statistics

### Video Resources
1. **MIT 6.034 - Probabilistic Inference**
   - Patrick Winston's AI course
2. **Stanford CS228 - Probabilistic Graphical Models**
   - Daphne Koller's course
3. **David MacKay - Information Theory, Inference, and Learning**
   - Classic lecture series

### Advanced Reading
1. **Neal - MCMC using Hamiltonian dynamics**
   - Advanced sampling methods
2. **Ghahramani - Probabilistic machine learning and artificial intelligence**
   - Modern probabilistic ML survey
3. **Blei - Variational Inference: A Review for Statisticians**
   - Comprehensive VI overview

## Socratic Questions

### Understanding
1. Why does Naive Bayes work well despite the independence assumption often being violated?
2. How does Bayesian regularization differ from L1/L2 penalties?
3. What's the relationship between GP regression and kernel ridge regression?

### Extension
1. How would you design priors for deep neural networks?
2. Can you extend Gaussian processes to discrete outputs?
3. What happens to Bayesian inference in the infinite data limit?

### Research
1. How can we make Bayesian deep learning more computationally efficient?
2. What are the fundamental limits of variational approximations?
3. How do we choose between different MCMC algorithms for a given problem?

## Exercises

### Theoretical
1. Derive the posterior for Bayesian linear regression
2. Show that MAP estimation with Gaussian prior equals Ridge regression
3. Prove that GP regression reduces to linear regression with finite basis functions

### Implementation
1. Build Naive Bayes from scratch with Laplace smoothing
2. Implement GP regression with multiple kernels and hyperparameter optimization
3. Code variational inference for Bayesian linear regression
4. Write Metropolis-Hastings sampler for simple posterior

### Research
1. Compare Bayesian and frequentist confidence intervals on real data
2. Study how GP kernel choice affects extrapolation behavior
3. Investigate computational scaling of different inference methods

## Advanced Topics

### Nonparametric Bayesian Methods
- **Dirichlet processes**: Infinite mixture models
- **Gaussian process latent variable models**: Dimensionality reduction
- **Indian buffet process**: Infinite feature models

### Approximate Inference
- **Expectation propagation**: Alternative to variational inference
- **Integrated nested Laplace approximation (INLA)**: Fast approximate inference
- **Automatic differentiation variational inference**: Scalable VI

### Bayesian Deep Learning
- **Weight uncertainty**: Distributions over network parameters
- **Functional uncertainty**: Distributions over function space
- **Calibrated uncertainty**: Matching confidence to accuracy

### Modern Applications
- **Federated learning**: Bayesian approaches to distributed learning
- **Meta-learning**: Few-shot learning with Bayesian optimization
- **Causal inference**: Bayesian approaches to causality