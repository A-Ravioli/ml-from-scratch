# generalized linear models

## why this matters

you've now fitted three models that look different and are secretly the same. least squares predicts a real number, logistic regression a probability, and in both the gradient was "prediction minus target, times the input", and the solver a sequence of weighted least-squares problems. that's not a coincidence. both are **generalized linear models**: a linear predictor $\eta = x^\top w + b$, a distribution for $y$ from an exponential family, and a link function connecting the two. pick the distribution that matches your data and you get the right loss, the right gradient and the right solver for free.

the case that makes this matter in practice is **counts**: rentals per day, claims per policy, clicks per ad, mutations per gene. least squares on counts predicts negative values, assumes the noise is the same size for a station that sees 2 bikes a day and one that sees 200, and models effects as additive when they're usually multiplicative. Poisson regression fixes all three with one change of distribution. and when the counts are noisier than the Poisson distribution allows, which is most of the time, the GLM framework also tells you how to detect it and what it does to your confidence intervals. you'll build one `GLM` class with Gaussian, Bernoulli and Poisson families, fitted by a single IRLS loop.

## warm-up: predict first

1. a Poisson model has a coefficient of $-0.8$ on "it rained". what does rain do to the expected count?
2. for Poisson data, how does the variance of $y$ relate to its mean? what does least squares assume instead?
3. you fit least squares to bike-rental counts. can a prediction be negative? where would that happen?
4. a GLM with the Gaussian family and the identity link is which model you already know? with the Bernoulli family and the logit link?
5. the counts' variance is four times their mean. are the Poisson GLM's coefficients still right? are its standard errors?

## prerequisites check

- [ ] you know the Poisson and Bernoulli distributions, their means and variances (→ ch 04 · topic 02)
- [ ] you know maximum likelihood and the Fisher information as the inverse variance of the MLE (→ ch 05 · topics 02–03)
- [ ] you've built IRLS for logistic regression and know why it's Newton's method (→ topic 04)
- [ ] you've built `LinearRegression` and `Ridge` (→ topics 01–02)
- [ ] you can read a log-likelihood ratio and know what a deviance-like quantity measures (→ ch 05 · topic 04)

## core ideas

### idea 1: exponential families and canonical links

the three distributions you've used share a form. in its simplest version (no extra scale parameter), an **exponential family** has densities

$$p(y \mid \theta) = h(y)\exp\big(y\theta - b(\theta)\big),$$

with a natural parameter $\theta$ and a log-normalizer $b(\theta)$. two facts make it useful: the mean is $\mu = b'(\theta)$ and the variance is $b''(\theta)$. since the variance is a function of the mean, write it as the **variance function** $V(\mu)$:

| family | $y$ | $b(\theta)$ | mean $\mu = b'(\theta)$ | $V(\mu)$ | canonical link $g(\mu) = \theta$ |
|---|---|---|---|---|---|
| Gaussian ($\sigma = 1$) | real | $\theta^2/2$ | $\theta$ | $1$ | identity: $\mu$ |
| Bernoulli | 0/1 | $\log(1 + e^\theta)$ | $\sigma(\theta)$ | $\mu(1 - \mu)$ | logit: $\log\frac{\mu}{1 - \mu}$ |
| Poisson | 0, 1, 2, ... | $e^\theta$ | $e^\theta$ | $\mu$ | log: $\log\mu$ |

a GLM sets the natural parameter equal to the linear predictor, $\theta = \eta = x^\top w + b$. equivalently, the mean is $\mu = g^{-1}(\eta)$, where $g$ is the **canonical link**: identity for the Gaussian (least squares), logit for the Bernoulli (logistic regression) and log for the Poisson (warm-up 4). the inverse link maps the whole real line to the allowed means: anything for the Gaussian, $(0, 1)$ for the Bernoulli, $(0, \infty)$ for the Poisson, so predictions are always valid (warm-up 3). and the variance function says how noisy $y$ is at each mean: constant for least squares, largest at $\mu = 1/2$ for coin flips, proportional to the mean for counts (warm-up 2). a canonical link has a tidy property you can check numerically: $d\mu/d\eta = b''(\theta) = V(\mu)$.

**the log link makes effects multiplicative.** with $\mu = \exp(b + w_1x_1 + w_2x_2)$, increasing $x_2$ by one multiplies $\mu$ by $e^{w_2}$, whatever $x_1$ is. a coefficient of $-0.8$ on rain means rainy days have $e^{-0.8} \approx 0.45$ times the rentals (warm-up 1): a fixed percentage, not a fixed number.

**the deviance** measures misfit in each family's own currency: twice the log-likelihood gap between a "saturated" model that predicts every $y_i$ exactly and the model:

$$d(y, \mu) = 2\big[\log p(y \mid \mu = y) - \log p(y \mid \mu)\big].$$

for the Gaussian it's $(y - \mu)^2$, for the Poisson $2[y\log(y/\mu) - (y - \mu)]$ and for the Bernoulli $-2\log$ of the probability given to the observed outcome (with $0\log 0 = 0$ throughout). it's never negative, it's zero when $\mu = y$, and minimizing its sum is maximum likelihood.

> **▶ build: stage 1.** implement the classes `Gaussian`, `Bernoulli` and `Poisson`, each with `inverse_link`, `link`, `variance` and `deviance` (per sample). handle $0\log 0$ and use a stable sigmoid.

### idea 2: one loss, one gradient, one update

every GLM minimizes the same objective: half the mean deviance (the mean negative log-likelihood, up to a constant), plus a ridge penalty that skips the intercept,

$$L(\theta) = \frac{1}{2n}\sum_i d(y_i, \mu_i) + \frac12\sum_j\lambda_j\theta_j^2, \qquad \mu = g^{-1}(A\theta).$$

with a canonical link, the gradient collapses to the formula you've now seen three times (on paper 1):

$$\nabla L(\theta) = \frac1nA^\top(\mu - y) + \lambda \odot \theta.$$

and since $d\mu/d\eta = V(\mu)$, the Hessian is $\frac1nA^\top WA + \operatorname{diag}(\lambda)$ with weights $W = \operatorname{diag}(V(\mu))$: positive semidefinite, so the loss is convex for all three families.

Newton's step then rearranges, exactly as in topic 04, into a weighted ridge regression:

$$\big(A^\top WA + n\,\operatorname{diag}(\lambda)\big)\,\theta_{\text{new}} = A^\top Wz, \qquad z = \eta + \frac{y - \mu}{V(\mu)}.$$

the **working response** $z$ is a linearization of $g(y)$ around the current fit: the current linear predictor plus the residual translated into $\eta$ units. the weights are the inverse variance on that scale, so noisy observations (large-$\mu$ counts) don't get more say just because their raw residuals are bigger. this is **iteratively reweighted least squares**: refit a weighted least-squares problem, update $\mu$, $W$ and $z$, repeat.

one numerical detail matters. $V(\mu)$ can be tiny (a Bernoulli probability of $10^{-12}$, a Poisson mean near 0), and dividing by it gives enormous working responses. but the solve only needs $Wz = W\eta + (y - \mu)$, where the division cancels. compute that instead.

> **🔮 predict:** for the Gaussian family, how many IRLS steps does it take to reach the exact answer from any starting point?

> **▶ build: stage 2.** implement `glm_loss(theta, A, y, family, penalty)`, `glm_gradient(...)` and `irls_step(...)`, which returns the solution of the weighted ridge system (falling back to `np.linalg.lstsq` if the solve fails).

### idea 3: the estimator

`GLM(family="poisson", *, alpha=0.0, max_iters=100, tol=1e-08, fit_intercept=True)` is a `Regressor` (so `score` is $R^2$) whose `predict` returns the fitted mean $\mu$. `fit`:

- looks up the family with the given `get_family`, and checks that $y$ is valid for it: 0/1 for the Bernoulli, non-negative for the Poisson. a negative count isn't a modelling choice, it's a data error.
- starts at the model that predicts the mean everywhere: weights 0, intercept $g(\bar y)$ (clipped so that $\bar y = 0$ doesn't give $\log 0$). that's a sensible fit already, and close enough for Newton.
- repeats IRLS steps, each moving from $\theta$ towards `irls_step`'s answer $\theta^\star$, trying $\theta + t(\theta^\star - \theta)$ with $t = 1, \tfrac12, \tfrac14, \dots$ until the loss doesn't increase (up to rounding, $f + 10^{-12}(1 + |f|)$, as in topic 04; at most 30 halvings, then it stops). far from the optimum a full Newton step on a Poisson model can overshoot badly, since $e^\eta$ explodes; the halving guards against that, and near the optimum it's never needed.
- stops when $\|\nabla L\| \le$ `tol` or after `max_iters` steps, recording `n_iter_` and `converged_`.

with the Gaussian family this reproduces `Ridge` (with $\alpha_{\text{ridge}} = n\alpha$, because $L$ uses a mean and ridge a sum), with the Bernoulli family it reproduces `LogisticRegression`, and with the Poisson family it matches scikit-learn's `PoissonRegressor`, whose loss is the same half mean deviance. one loop, three models.

> **▶ build: stage 3.** implement `GLM.fit` and `GLM.predict`.

### idea 4: is the fit any good, and is the variance right?

**deviance explained.** the natural analogue of $R^2$ for a GLM compares the model's total deviance with that of the model that predicts $\bar y$ everywhere:

$$D^2 = 1 - \frac{\sum_i d(y_i, \hat\mu_i)}{\sum_i d(y_i, \bar y)}.$$

for the Gaussian family it *is* $R^2$; for counts it's a better yardstick than $R^2$, because it weighs each residual by how surprising it is under the model (a miss of 5 when you predicted 2 is far worse than a miss of 5 when you predicted 200).

**overdispersion.** the Poisson family makes a strong claim: the variance equals the mean. real counts usually vary more, because the rate itself varies between observations in ways your features don't capture (two stations with the same features, but one near a university). the **Pearson dispersion**

$$\hat\phi = \frac{1}{n - p}\sum_i\frac{(y_i - \hat\mu_i)^2}{V(\hat\mu_i)}$$

estimates the factor by which the variance exceeds $V(\mu)$: about 1 if the family is right, much more under overdispersion. (for the Gaussian family with $\sigma \ne 1$ it estimates $\sigma^2$: topic 01's unbiased noise variance.)

what does overdispersion break? not the coefficients: the gradient condition $A^\top(y - \mu) = 0$ only involves the *mean*, so if the mean model is right, the estimates are still consistent (warm-up 5). what breaks is everything that relies on the variance. the standard errors from the inverse Fisher information, $\operatorname{Cov}(\hat\theta) \approx (A^\top WA)^{-1}$, assume $\operatorname{Var}(y) = V(\mu)$; if the truth is $\phi V(\mu)$, they're too small by a factor of $\sqrt\phi$, and "95%" intervals cover far less often. the simplest fix (*quasi-Poisson*) multiplies the standard errors by $\sqrt{\hat\phi}$; others fit a negative binomial model, use the bootstrap, or use "sandwich" standard errors. the lab measures how bad the naive intervals get.

> **🔮 predict:** counts with variance about 4 times their mean, fitted by a Poisson GLM. what fraction of the textbook 95% intervals for a slope will contain the true slope?

> **▶ build: stage 4.** implement `d2_score(y, mu, family)` and `pearson_dispersion(y, mu, family, n_params)`.

## worked example

two counts with one feature: $x = (0, 1)$, $y = (1, 3)$, a Poisson GLM with an intercept: $A = \begin{pmatrix}1 & 0\\1 & 1\end{pmatrix}$, $\theta = (b, w)$. two points and two parameters, so the fit will be exact: $\mu = (1, 3)$, i.e. $b = \log 1 = 0$ and $w = \log 3 \approx 1.0986$.

**the start.** $\bar y = 2$, so $\theta_0 = (\log 2, 0)$ and $\mu = (2, 2)$. the deviances are $d(1, 2) = 2[\log\tfrac12 + 1] \approx 0.614$ and $d(3, 2) = 2[3\log\tfrac32 - 1] \approx 0.433$, so $L = \frac12 \cdot \frac{0.614 + 0.433}{2} \approx 0.262$. the gradient is $\frac12A^\top(\mu - y) = \frac12A^\top(1, -1) = (0, -0.5)$: raise the slope.

**one IRLS step.** the weights are $V(\mu) = \mu = (2, 2)$ and the working responses $z = \eta + (y - \mu)/\mu = \log 2 + (-\tfrac12, \tfrac12) \approx (0.193, 1.193)$. with equal weights, weighted least squares through two points is the line through both: $b = 0.193$, $w = 1.193 - 0.193 = 1$. so $\theta_1 = (\log 2 - \tfrac12, 1) \approx (0.193, 1.0)$, and the loss drops to 0.0169.

**the rest.** the next steps give $(0.0175, 1.0854)$, $(0.00015, 1.09847)$, then $(0, 1.098612)$: gradient norms $0.30, 0.017, 9 \times 10^{-5}, 6 \times 10^{-9}, 6 \times 10^{-16}$, quadratic convergence again.

the tests check the first step and the final fit.

## on paper

1. **the gradient.** for a canonical link, show that the derivative of $\frac12 d(y, \mu)$ with respect to $\eta$ is $\mu - y$. (use $\log p(y \mid \theta) = y\theta - b(\theta) + \log h(y)$ with $\theta = \eta$.)

<details><summary>solution</summary>

$\frac12d(y, \mu) = \log p(y \mid \text{saturated}) - \log p(y \mid \theta)$, and the first term doesn't depend on $\eta$. so its derivative is $-\frac{d}{d\eta}[y\eta - b(\eta)] = b'(\eta) - y = \mu - y$. by the chain rule with $\eta_i = a_i^\top\theta$, $\nabla\frac{1}{2n}\sum_id_i = \frac1nA^\top(\mu - y)$.

</details>

2. **IRLS is Newton.** with $W = \operatorname{diag}(V(\mu))$, show that $\theta - H^{-1}\nabla L$ solves $(A^\top WA + n\Lambda)\theta_{\text{new}} = A^\top W z$ for the working response $z = \eta + (y - \mu)/V(\mu)$.

<details><summary>solution</summary>

$H = \frac1n(A^\top WA + n\Lambda)$ because $d\mu_i/d\eta_i = V(\mu_i)$. multiply $\theta_{\text{new}} = \theta - H^{-1}\nabla L$ by $nH$: $(A^\top WA + n\Lambda)\theta_{\text{new}} = A^\top WA\theta + n\Lambda\theta - A^\top(\mu - y) - n\Lambda\theta = A^\top(W\eta + y - \mu) = A^\top Wz$.

</details>

3. **why the Gaussian GLM is ridge with $n\alpha$.** write out $L$ for the Gaussian family and compare it with ridge's objective.

<details><summary>solution</summary>

$L = \frac{1}{2n}\sum_i(y_i - \eta_i)^2 + \frac\alpha2\|w\|^2$. multiplying by $2n$ doesn't move the minimizer and gives $\|y - A\theta\|^2 + n\alpha\|w\|^2$: ridge with $\alpha_{\text{ridge}} = n\alpha$. and because the gradient is linear in $\theta$, one IRLS (Newton) step from anywhere lands on the minimum.

</details>

4. **overdispersion and the standard errors.** suppose $\operatorname{Var}(y_i) = \phi V(\mu_i)$ but you fit a Poisson GLM. using the estimating equation $A^\top(y - \mu(\hat\theta)) = 0$ and a first-order expansion, show that $\operatorname{Cov}(\hat\theta) \approx \phi(A^\top WA)^{-1}$.

<details><summary>solution</summary>

expand around the true $\theta_0$: $0 = A^\top(y - \mu(\hat\theta)) \approx A^\top(y - \mu_0) - A^\top W A(\hat\theta - \theta_0)$, using $\partial\mu/\partial\theta = WA$. so $\hat\theta - \theta_0 \approx (A^\top WA)^{-1}A^\top(y - \mu_0)$, with covariance $(A^\top WA)^{-1}A^\top\operatorname{Cov}(y)A(A^\top WA)^{-1} = (A^\top WA)^{-1}A^\top(\phi W)A(A^\top WA)^{-1} = \phi(A^\top WA)^{-1}$. the Poisson standard errors assume $\phi = 1$, so they are too small by $\sqrt\phi$. (if $\operatorname{Var}(y)$ isn't proportional to $V(\mu)$ at all, the "sandwich" form in the middle step is the general answer.)

</details>

## experiment

`lab.py` runs two experiments on your code:

1. daily bike-rental counts: least squares against a Poisson GLM, with negative predictions, test error, deviance and residuals.
2. 300 simulated datasets: how often the textbook 95% interval for a Poisson slope covers the truth, with and without overdispersion.

## check your understanding

1. why is the log link a good default for counts even apart from the Poisson distribution?

<details><summary>answer</summary>

it keeps predictions positive, and it makes effects multiplicative, which is how most count processes work: a 10% rise in traffic raises accidents by some percentage at every station, rather than by the same number at a quiet and a busy one. it also makes coefficients easy to read as rate ratios.

</details>

2. a Poisson GLM's coefficient for "weekend" is 0.3. state its meaning in one sentence.

<details><summary>answer</summary>

holding the other features fixed, weekend days have about $e^{0.3} \approx 1.35$ times the expected count of weekdays: 35% more.

</details>

3. your counts are recorded over different lengths of time (some stations report weekly, some daily). how would you put that into a Poisson GLM?

<details><summary>answer</summary>

model the rate: $\mu_i = t_i\exp(x_i^\top w + b)$, i.e. $\log\mu_i = \log t_i + x_i^\top w + b$. the log exposure enters the linear predictor with a coefficient fixed at 1 (an *offset*). without offset support, adding $\log t_i$ as a feature and checking that its coefficient is near 1 is a reasonable approximation.

</details>

4. why does the IRLS weight for a Poisson observation equal its fitted mean?

<details><summary>answer</summary>

the weight is $V(\mu)$ for a canonical link. on the $\eta = \log\mu$ scale, a count's working residual $(y - \mu)/\mu$ has variance $\mu/\mu^2 = 1/\mu$, so the inverse-variance weight is $\mu$: residuals of big counts are more precise on the log scale.

</details>

5. the Pearson dispersion of your Poisson fit is 1.05. are you done?

<details><summary>answer</summary>

the variance assumption looks fine on average, but that doesn't validate the mean model: check residuals against each feature and against $\hat\mu$ for patterns, compare deviance on held-out data with alternatives, and look for zero inflation (more zeros than the Poisson allows can coexist with a dispersion near 1).

</details>

6. you fit a Bernoulli GLM to 0/1 data and get a Pearson dispersion of 3. what does that mean?

<details><summary>answer</summary>

for ungrouped 0/1 data, very little: each $y_i$ is 0 or 1 and the variance is fully determined by the mean, so there's no room for overdispersion and the Pearson statistic is not informative (its distribution isn't close to $\chi^2$). overdispersion is meaningful for binomial *counts* (successes out of $m$ trials) and Poisson counts.

</details>

## common bugs

- **$0\log 0$ computed as `nan`.** `y * np.log(y / mu)` with `y = 0`. symptom: `nan` deviances, losses and $D^2$ for any data with zeros.
- **dividing by $V(\mu)$ in the working response.** symptom: overflow or `inf` in `irls_step` when a Bernoulli probability saturates or a Poisson mean is near 0.
- **starting from $\theta = 0$ with the Poisson family and large counts.** $\mu = 1$ everywhere, the first Newton step is huge, and without step halving it overshoots into $e^{\eta}$ overflow. symptom: `inf` losses on the first step. start at $g(\bar y)$.
- **predicting $\eta$ instead of $\mu$.** returning `X @ coef_ + intercept_` from `predict`. symptom: negative "counts", and $R^2$ far below what the coefficients suggest.
- **comparing a Poisson model and least squares by $R^2$ alone.** it ignores that the noise grows with the mean. symptom: conclusions that flip when you use the deviance. compare on held-out deviance or log-likelihood.
- **trusting Poisson standard errors without checking the dispersion.** symptom: everything is "significant", and intervals that look precise but don't cover.

## going deeper

- Nelder & Wedderburn (1972), "Generalized linear models", *Journal of the Royal Statistical Society A* 135(3): 370–384. the paper that unified them, with IRLS. [doi:10.2307/2344614](https://doi.org/10.2307/2344614)
- McCullagh & Nelder (1989), *Generalized Linear Models*, 2nd ed., Chapman & Hall/CRC, chapters 2, 4 and 6. the standard reference: families, deviance, quasi-likelihood and overdispersion.
- Wedderburn (1974), "Quasi-likelihood functions, generalized linear models, and the Gauss–Newton method", *Biometrika* 61(3): 439–447. why only the mean and variance functions matter, and the dispersion correction. [doi:10.1093/biomet/61.3.439](https://doi.org/10.1093/biomet/61.3.439)
- Cameron & Trivedi (2013), *Regression Analysis of Count Data*, 2nd ed., Cambridge University Press, chapters 2–3: Poisson regression, overdispersion and its tests. [doi:10.1017/CBO9781139013567](https://doi.org/10.1017/CBO9781139013567)
- Agresti (2015), *Foundations of Linear and Generalized Linear Models*, Wiley, chapters 4 and 7.

## connections

- ← builds on: exponential-family distributions and maximum likelihood (ch 04–05), Newton's method (ch 06), least squares, ridge and logistic regression (topics 01, 02, 04), $R^2$ and the `Regressor` protocol (ch 07).
- → used in: the capstone's evaluation (deviance-style log losses), Bayesian GLMs and the Laplace approximation (ch 12), exponential families in variational inference (ch 12), the cross-entropy and Poisson losses of neural networks (ch 15), and gradient boosting with arbitrary losses, which reuses this gradient-and-Hessian recipe (ch 10).
