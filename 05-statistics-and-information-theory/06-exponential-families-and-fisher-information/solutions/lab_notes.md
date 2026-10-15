# exponential families & fisher information · lab notes

## experiment 1: who reaches the bound?

**what people usually predict:** the MLE is "efficient", so the MLE $\hat\sigma^2$ sits on the bound; $s^2$, being unbiased, is close to it; nothing can be below a lower bound.

**what happens (variance / bound and MSE / bound, 20,000 datasets):**

| n | poisson x̄ | s² (n−1) | MLE σ̂² (n) | S²ₙ₊₁ (n+1) |
|---|---|---|---|---|
| 3 | 1.00 / 1.00 | 1.49 / 1.49 | 0.66 / 0.83 | 0.37 / 0.75 |
| 10 | 1.00 / 1.00 | 1.11 / 1.11 | 0.90 / 0.95 | 0.74 / 0.91 |
| 30 | 1.01 / 1.01 | 1.04 / 1.04 | 0.97 / 0.99 | 0.91 / 0.97 |
| 100 | 0.99 / 0.99 | 1.01 / 1.01 | 0.99 / 1.00 | 0.97 / 0.99 |

$\bar x$ sits exactly on the bound for the Poisson rate at every $n$, even $n = 3$. the unbiased $s^2$ is above it, by the factor $\frac{n}{n-1}$ (1.5 at $n = 3$). the MLE and $S^2_{n+1}$ are *below* the lower bound, in variance and even in MSE. all four converge to the bound as $n$ grows.

**why:** the Poisson rate $\lambda$ is the mean parameter $\mathbb{E}[T(X)]$ of its exponential family, and $\bar x = \bar T$ has covariance exactly $I(\eta)/n$, which is the bound for the mean parameter: efficient at every $n$ (the equality case of Cauchy–Schwarz, where the estimator is an affine function of the score). $\sigma^2$ is not a mean parameter (the mean parameters are $\mathbb{E}X$ and $\mathbb{E}X^2$), and the best unbiased estimator, $s^2$, has variance $\frac{2\sigma^4}{n-1}$ against the bound $\frac{2\sigma^4}{n}$. the estimators below the bound are **biased**: the theorem's proof uses $\partial_\theta\mathbb{E}\hat\theta = 1$, which fails for them. shrinking toward zero cuts the variance by more than it adds in squared bias (topic 01), so even their MSE beats the "limit". the bound is a statement about unbiased estimators, not a law about estimation. asymptotically everything sensible lines up on it, because the bias of $\hat\sigma^2$ and $S^2_{n+1}$ vanishes like $1/n$.

## experiment 2: the MLE in natural parameters

**what people usually predict:** the MLE is unbiased and near the bound, maybe 10–20% above it at $n = 10$.

**what happens (600 fits per $n$, true $\eta = (0.5, -0.25)$):**

| n | bias of η̂ (η₁ / η₂) | var(η̂₁) / bound | var(η̂₂) / bound |
|---|---|---|---|
| 10 | +0.205 / −0.112 | 3.37 | 5.00 |
| 200 | +0.008 / −0.005 | 1.10 | 1.04 |

at $n = 10$ the natural-parameter MLE is badly biased (by 40% of $\eta_1$ and 45% of $\eta_2$) and its variance is 3–5 times the bound. the scatter plot shows why: a long tail of fits with very negative $\eta_2$, from datasets whose sample variance happened to be small. at $n = 200$ the cloud sits inside the bound's ellipse, the bias has nearly vanished, and the variances are within about 10% of the bound (within the noise of 600 fits).

**why:** your fitter returns the exact MLE, and in mean parameters that's $\bar T$, unbiased. but $\eta_2 = -\frac{1}{2\sigma^2}$ is a nonlinear function of the mean parameters, and $\hat\eta_2 = -\frac{1}{2\hat\sigma^2}$. since $1/x$ is convex, Jensen's inequality makes $\mathbb{E}[1/\hat\sigma^2] > 1/\mathbb{E}\hat\sigma^2$: biased. worse, $\hat\sigma^2$ with 9 degrees of freedom is occasionally tiny, and $1/\hat\sigma^2$ then explodes: a heavy right tail, hence a huge variance. "asymptotically efficient" promises that $\sqrt n(\hat\eta - \eta)$ approaches $\mathcal{N}(0, I^{-1})$; it says nothing about $n = 10$, and which parameterization you use matters enormously at small $n$. it's the same lesson as topic 02's on paper 4 (the MAP moves with the parameterization) and topic 01's (unbiasedness doesn't survive nonlinear maps). when you report an uncertainty for a parameter from a small sample, a bootstrap (topic 03) on the scale you care about is often more honest than $I^{-1}/n$.

## experiment 3: Rao–Blackwell, and a bound nobody reaches

**what people usually predict:** conditioning on the sufficient statistic helps a little; the improved estimator, being "the best", reaches the bound.

**what happens (λ = 2, n = 10, 50,000 datasets; target $e^{-2} = 0.1353$, bound $3.66 \times 10^{-3}$):**

| estimator | bias | variance / bound | MSE / bound |
|---|---|---|---|
| naive 1[x₁ = 0] | +0.003 (noise) | 32.5 | 32.5 |
| Rao–Blackwell (1 − 1/n)^S | −0.0003 (noise) | 1.10 | 1.10 |
| plug-in MLE e^(−x̄) | +0.013 | 1.20 | 1.25 |

Rao–Blackwellization cuts the variance by a factor of about 30 while staying unbiased. but even the best unbiased estimator sits 10% above the bound, and nothing reaches it. the plug-in MLE is biased upward and slightly worse here.

**why:** for any estimator $\delta$, $E[\delta \mid S]$ has the same mean (tower property) and variance $\operatorname{Var}(\delta) - \mathbb{E}[\operatorname{Var}(\delta \mid S)]$, never larger: that's the Rao–Blackwell theorem. conditioning on the sufficient statistic averages away the part of $\delta$'s randomness that carries no information about $\lambda$. given $S = s$, the counts are multinomial with equal cell probabilities, so $P(x_1 = 0 \mid S = s) = (1 - 1/n)^s$. since $S$ is also *complete*, this is the unique best unbiased estimator (Lehmann–Scheffé). and yet it can't reach the bound: equality in Cramér–Rao requires the estimator to be an affine function of the score, $S - n\lambda$, and an unbiased affine function of $S$ can only estimate an affine function of $\lambda$, the mean parameter. $e^{-\lambda}$ isn't one, so for it the bound is unattainable at any finite $n$; the gap shrinks like $1/n$. in exponential families, the parameters with efficient estimators are exactly the mean parameters (and their affine functions). the "1.10" at $\lambda = 2$ grows with $\lambda$: drag the slider to 4 and the gap widens, because $e^{-\lambda}$ curves more relative to the noise in $S$.
