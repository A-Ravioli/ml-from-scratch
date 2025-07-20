# Linear Models for Machine Learning

## Prerequisites
- Linear algebra (matrix decompositions, eigenvalues)
- Probability theory (Gaussian distributions, MLE)
- Convex optimization basics

## Learning Objectives
- Master linear regression from probabilistic and optimization perspectives
- Understand regularization (Ridge, Lasso, Elastic Net)
- Connect to Bayesian inference and MAP estimation
- Implement efficient algorithms for large-scale problems

## Mathematical Foundations

### 1. Linear Regression

#### Model Setup
Given training data {(xᵢ, yᵢ)}ᵢ₌₁ⁿ where xᵢ ∈ ℝᵈ, yᵢ ∈ ℝ:

**Linear model**: yᵢ = xᵢᵀβ + εᵢ

Matrix form: y = Xβ + ε where X ∈ ℝⁿˣᵈ, β ∈ ℝᵈ, ε ∈ ℝⁿ

#### Assumptions
1. **Linearity**: E[y|x] = xᵀβ
2. **Independence**: εᵢ are independent
3. **Homoscedasticity**: Var(εᵢ) = σ² (constant variance)
4. **Normality**: εᵢ ~ N(0, σ²) (for inference)

### 2. Ordinary Least Squares (OLS)

#### Optimization Problem
minimize ||y - Xβ||²₂ = ∑ᵢ₌₁ⁿ (yᵢ - xᵢᵀβ)²

#### Theorem 2.1 (Normal Equations)
The optimal solution satisfies:
XᵀXβ* = Xᵀy

If XᵀX is invertible:
β* = (XᵀX)⁻¹Xᵀy

**Proof**: 
∇_β ||y - Xβ||²₂ = ∇_β (yᵀy - 2βᵀXᵀy + βᵀXᵀXβ) = -2Xᵀy + 2XᵀXβ = 0 □

#### Geometric Interpretation
- Projection of y onto column space of X
- Residual e = y - Xβ* ⟂ col(X)
- β* minimizes distance from y to col(X)

### 3. Probabilistic Interpretation

#### Maximum Likelihood Estimation
Assume εᵢ ~ N(0, σ²) independently. Then:
yᵢ ~ N(xᵢᵀβ, σ²)

**Likelihood**:
L(β, σ²) = ∏ᵢ₌₁ⁿ (1/√(2πσ²)) exp(-(yᵢ - xᵢᵀβ)²/(2σ²))

**Log-likelihood**:
ℓ(β, σ²) = -n/2 log(2πσ²) - 1/(2σ²) ∑ᵢ₌₁ⁿ (yᵢ - xᵢᵀβ)²

#### Theorem 3.1 (MLE = OLS)
The MLE for β is identical to the OLS estimator:
β̂_MLE = argmax_β ℓ(β, σ²) = (XᵀX)⁻¹Xᵀy

### 4. Statistical Properties

#### Theorem 4.1 (Gauss-Markov Theorem)
Under assumptions 1-3, the OLS estimator β̂ is BLUE (Best Linear Unbiased Estimator):
1. **Unbiased**: E[β̂] = β
2. **Minimum variance**: among all linear unbiased estimators

**Proof Sketch**:
- E[β̂] = E[(XᵀX)⁻¹Xᵀy] = E[(XᵀX)⁻¹XᵀXβ + (XᵀX)⁻¹Xᵀε] = β
- For any other linear unbiased estimator Ay with E[Ay] = β for all β
- Var(Ay) - Var(β̂) is positive semidefinite □

#### Covariance Matrix
Var(β̂) = σ²(XᵀX)⁻¹

#### Confidence Intervals
Under normality: β̂ⱼ ~ N(βⱼ, σ²[(XᵀX)⁻¹]ⱼⱼ)

95% CI: β̂ⱼ ± 1.96√(σ̂²[(XᵀX)⁻¹]ⱼⱼ)

### 5. Ridge Regression

#### Motivation
When XᵀX is ill-conditioned or singular, OLS:
- Has high variance
- Is numerically unstable
- Overfits with many features

#### Ridge Optimization
minimize ||y - Xβ||²₂ + λ||β||²₂

where λ ≥ 0 is the regularization parameter.

#### Theorem 5.1 (Ridge Solution)
β̂_ridge = (XᵀX + λI)⁻¹Xᵀy

**Proof**: 
∇_β [||y - Xβ||²₂ + λ||β||²₂] = -2Xᵀy + 2XᵀXβ + 2λβ = 0 □

#### Properties
1. **Always exists**: XᵀX + λI is always invertible for λ > 0
2. **Shrinkage**: ||β̂_ridge||₂ ≤ ||β̂_OLS||₂
3. **Bias-variance tradeoff**: Introduces bias but reduces variance

### 6. Lasso Regression

#### Optimization Problem
minimize ||y - Xβ||²₂ + λ||β||₁

where ||β||₁ = ∑ⱼ |βⱼ| is the ℓ₁ norm.

#### Key Properties
1. **Sparse solutions**: Can set coefficients exactly to zero
2. **Feature selection**: Automatically selects relevant features
3. **Non-differentiable**: Requires specialized optimization

#### Subgradient Conditions
At optimal β*, for each j:
- If β*ⱼ > 0: (Xᵀ(y - Xβ*))ⱼ = λ
- If β*ⱼ < 0: (Xᵀ(y - Xβ*))ⱼ = -λ  
- If β*ⱼ = 0: |(Xᵀ(y - Xβ*))ⱼ| ≤ λ

#### Coordinate Descent Algorithm
For each j in turn:
β̂ⱼ ← S(∑ᵢ xᵢⱼ(yᵢ - ∑ₖ≠ⱼ xᵢₖβₖ)/∑ᵢ x²ᵢⱼ, λ/∑ᵢ x²ᵢⱼ)

where S(z, γ) = sign(z)max(|z| - γ, 0) is soft thresholding.

### 7. Elastic Net

#### Motivation
Combines Ridge and Lasso:
- Ridge: good when many features are relevant
- Lasso: good for feature selection but unstable with correlated features

#### Optimization Problem
minimize ||y - Xβ||²₂ + λ₁||β||₁ + λ₂||β||²₂

Equivalent form:
minimize ||y - Xβ||²₂ + λ[(1-α)||β||²₂ + α||β||₁]

where α ∈ [0,1] controls the mixing.

### 8. Bayesian Linear Regression

#### Prior Distribution
β ~ N(0, τ²I) (isotropic Gaussian prior)

#### Posterior Distribution
Given data (X, y), the posterior is:
β|X, y ~ N(μ_post, Σ_post)

where:
Σ_post = (σ⁻²XᵀX + τ⁻²I)⁻¹
μ_post = σ⁻²Σ_post Xᵀy

#### Connection to Ridge
The MAP estimate β̂_MAP = μ_post equals the Ridge solution with λ = σ²/τ².

#### Predictive Distribution
For new point x*:
y*|x*, X, y ~ N(x*ᵀμ_post, σ² + x*ᵀΣ_post x*)

### 9. Generalized Linear Models

#### Exponential Family
y ~ f(y; θ, φ) = exp((yθ - b(θ))/a(φ) + c(y, φ))

#### GLM Components
1. **Linear predictor**: η = Xβ
2. **Link function**: g(μ) = η where μ = E[y]
3. **Variance function**: Var(y) = φV(μ)

#### Examples
1. **Linear regression**: Normal distribution, identity link
2. **Logistic regression**: Binomial distribution, logit link
3. **Poisson regression**: Poisson distribution, log link

### 10. Computational Aspects

#### Normal Equations: O(d³ + nd²)
1. Compute XᵀX (O(nd²))
2. Compute Xᵀy (O(nd))
3. Solve linear system (O(d³))

Good when d ≪ n.

#### SVD Approach: O(nd²)
X = UΣVᵀ, then β̂ = VΣ⁻¹Uᵀy

More numerically stable.

#### Gradient Descent: O(ndt)
For large n, iterate:
β ← β - α/n Xᵀ(Xβ - y)

#### Stochastic Gradient Descent
For very large n, use mini-batches:
β ← β - α/|B| ∑ᵢ∈B xᵢ(xᵢᵀβ - yᵢ)

## Applications and Examples

### Feature Engineering
1. **Polynomial features**: [x, x², x³, ...]
2. **Interaction terms**: [x₁, x₂, x₁x₂, ...]
3. **Basis functions**: [φ₁(x), φ₂(x), ...]

### Regularization Selection
- **Cross-validation**: Choose λ minimizing CV error
- **Information criteria**: AIC, BIC
- **Analytical**: Stein's unbiased risk estimate

### High-Dimensional Regime
When d > n or d ≈ n:
- OLS fails (XᵀX singular)
- Ridge always works
- Lasso provides sparsity
- Theory: sparse recovery, compressed sensing

## Implementation Details

See `exercise.py` for implementations of:
1. OLS via normal equations and SVD
2. Ridge regression with regularization path
3. Lasso via coordinate descent
4. Elastic Net optimization
5. Bayesian linear regression
6. Cross-validation for hyperparameter tuning

## Experiments

1. **Bias-Variance Decomposition**: Empirical verification
2. **Regularization Paths**: λ vs coefficient values
3. **Feature Selection**: Lasso vs forward selection
4. **Computational Comparison**: Normal equations vs iterative methods

## Research Connections

### Seminal Papers
1. Gauss (1809) - Method of least squares
2. Tikhonov (1943) - Ridge regression (Tikhonov regularization)
3. Tibshirani (1996) - "Regression Shrinkage and Selection via the Lasso"
4. Zou & Hastie (2005) - "Regularization and Variable Selection via Elastic Net"

### Modern Extensions
1. **Sparse regression**: Group Lasso, fused Lasso
2. **Non-convex penalties**: SCAD, MCP
3. **High-dimensional theory**: Sure screening, β-min conditions
4. **Computational advances**: LARS, coordinate descent

## Resources

### Primary Sources
1. **Hastie, Tibshirani & Friedman - Elements of Statistical Learning**
   - Comprehensive treatment of regularization
2. **Bishop - Pattern Recognition and Machine Learning**
   - Bayesian perspective
3. **Bühlmann & van de Geer - Statistics for High-Dimensional Data**
   - Modern high-dimensional theory

### Video Resources
1. **Stanford CS229 - Linear Regression**
   - Andrew Ng's classic lectures
2. **MIT 18.650 - Statistics for Applications**
   - Rigorous statistical treatment
3. **Caltech CS156 - Linear Models**
   - Yaser Abu-Mostafa

### Advanced Reading
1. **Wainwright - High-Dimensional Statistics**
   - Modern theory
2. **Hastie et al. - Statistical Learning with Sparsity**
   - Lasso and extensions
3. **Rasmussen & Williams - Gaussian Processes**
   - Bayesian nonparametric extensions

## Socratic Questions

### Understanding
1. Why does regularization help with overfitting?
2. When would you choose Lasso over Ridge?
3. How does the Bayesian view connect to frequentist regularization?

### Extension
1. What happens to linear regression in the limit d → ∞?
2. How would you extend linear models to non-Euclidean spaces?
3. Can you derive the dual formulation of Ridge regression?

### Research
1. How do modern deep learning optimizers relate to classical regression?
2. What's the role of implicit regularization in overparameterized models?
3. How can we handle non-linear relationships while maintaining interpretability?

## Exercises

### Theoretical
1. Prove the Gauss-Markov theorem in detail
2. Derive the posterior distribution for Bayesian linear regression
3. Show that Ridge regression is the limit of Elastic Net as α → 0

### Implementation
1. Implement all regression variants from scratch
2. Build cross-validation framework for hyperparameter tuning
3. Create visualization tools for regularization paths

### Research
1. Study sparse recovery guarantees for Lasso
2. Implement and compare different optimization algorithms
3. Explore connections to compressed sensing theory