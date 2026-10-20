# bayesian linear regression

## why this matters

you fit a regression to sensor readings taken between 0 °C and 30 °C, and someone asks for a prediction at 45 °C. least squares gives you a number, with the same confidence it had in the middle of the data. it has no way to say "I've never seen anything like this". worse, it can't tell you which of two models (a line, a cubic) the data actually support, except by holding out data you'd rather train on.

the Bayesian version of linear regression fixes both problems with one idea: treat the weights as unknown, with a probability distribution, rather than as one best guess. for a linear model with Gaussian noise every step is exact, in closed form, with nothing but matrix algebra. that makes it the cleanest place to learn the three objects the rest of this chapter computes, approximately, for harder models: the **posterior** (what you believe about the parameters after seeing data), the **predictive distribution** (what you expect the next observation to be, uncertainty included), and the **evidence** (how well a whole model, not a parameter setting, explains the data). Gaussian processes (ch 09), Bayesian optimization, Kalman filters and the last layer of many "uncertainty-aware" neural networks are this model in disguise.

## warm-up: predict first

1. you fit ridge regression with penalty $\lambda$. there's a Bayesian model whose most probable weights are exactly the ridge weights. what is its prior?
2. you see one data point $(x_1, y_1)$ for the line $y = w_0 + w_1 x$. what does the set of weight pairs consistent with it look like?
3. a model with Gaussian-bump features centered inside $[-1, 1]$, trained on data in $[-1, 1]$. is its predictive uncertainty at $x = 3$ larger or smaller than inside the data?
4. data come from a cubic. which polynomial degree has the highest marginal likelihood: 3, or 9 (which fits the training data better)?
5. you process a dataset all at once, or one point at a time, using each posterior as the next prior. do you get the same posterior?

## prerequisites check

- [ ] you can condition and marginalize a multivariate Gaussian, and know its density (→ ch 04 · topic 04)
- [ ] you know least squares and the normal equations, and why $\Phi^\top\Phi$ can be singular (→ ch 02 · topic 04)
- [ ] you know Bayes' rule and conjugate priors (→ ch 04 · topic 01, ch 05 · topic 02)
- [ ] you know ridge regression as penalized least squares (→ ch 06, ch 07 · topic 04)

## core ideas

### idea 1: linear in the weights, not in the inputs

"linear regression" means the prediction is linear in the **weights**: $f(x) = w^\top\phi(x)$, where $\phi$ maps an input to a vector of $d$ **basis functions** (features). the inputs can enter in any nonlinear way you like. two families do most of the work in this topic:

- **polynomials**: $\phi(x) = (1, x, x^2, \dots, x^D)$. global: every feature grows without bound as $|x| \to \infty$.
- **Gaussian bumps**: a constant, then $\phi_j(x) = \exp\big(-(x - c_j)^2 / (2s^2)\big)$ for centers $c_1, \dots, c_k$ and a width $s$. local: each one only "fires" near its center, and they all vanish far from every center.

stack the feature vectors of $n$ inputs as rows and you get the **design matrix** $\Phi$ of shape $(n, d)$. the model is $y = \Phi w + \varepsilon$. note that the model here has no separate intercept: the constant feature plays that role, and it gets the same prior as every other weight.

which family you choose will matter more than you'd think: it decides what the model believes *far from the data*, where the data have no say. the lab makes this vivid.

> **spec.** `polynomial_basis(x, degree)` returns `(n, degree + 1)` with columns $x^0, \dots, x^{\text{degree}}$. `gaussian_basis(x, centers, width)` returns `(n, 1 + k)`: a column of ones, then one bump per center. both accept lists and flatten `x`.

> **▶ build: stage 1.** implement `polynomial_basis` and `gaussian_basis`.

### idea 2: the posterior is Gaussian

put a prior on the weights and a likelihood on the data:

$$w \sim \mathcal N(0, \alpha^{-1} I), \qquad y \mid w \sim \mathcal N(\Phi w, \beta^{-1} I).$$

$\alpha$ is the prior **precision** (inverse variance): large $\alpha$ says "weights are small". $\beta$ is the noise precision. Bayes' rule says $p(w \mid y) \propto p(y \mid w)\,p(w)$; take logs and keep the terms that depend on $w$:

$$\log p(w \mid y) = -\frac{\beta}{2}\|y - \Phi w\|^2 - \frac{\alpha}{2}\|w\|^2 + \text{const}.$$

that's a quadratic in $w$, so the posterior is Gaussian. matching it to $-\frac12(w - m_N)^\top S_N^{-1}(w - m_N)$ (on paper 1):

$$S_N^{-1} = A = \alpha I + \beta\Phi^\top\Phi, \qquad m_N = \beta S_N\Phi^\top y.$$

three things to notice. first, $A$ is always invertible (the $\alpha I$ sees to that), even with fewer points than weights: the prior fills in what the data leave undetermined. second, the posterior mean is exactly the **ridge** solution with $\lambda = \alpha/\beta$ (warm-up 1): ridge regression is the most probable weight vector under a Gaussian prior. the Bayesian treatment keeps the whole Gaussian instead of only its peak. third, the posterior precision is a *sum*: prior precision plus data precision, one term $\beta\phi_i\phi_i^\top$ per data point.

that last fact means you can learn **sequentially**. with a general Gaussian prior $\mathcal N(m_0, S_0)$, the same completing-the-square gives

$$S_N^{-1} = S_0^{-1} + \beta\Phi^\top\Phi, \qquad m_N = S_N\big(S_0^{-1}m_0 + \beta\Phi^\top y\big).$$

feed today's posterior in as tomorrow's prior, batch after batch, and you end at exactly the posterior you'd get from all the data at once (warm-up 5). no data are stored, nothing is approximated. this is the measurement update of the Kalman filter.

> **spec.** `posterior(Phi, y, alpha, beta)` returns `(mean (d,), cov (d, d))`; `sequential_posterior(prior_mean, prior_cov, Phi, y, beta)` updates any Gaussian prior with a batch. symmetrize the covariance (`(S + S.T) / 2`) against rounding. `np.linalg` is fine here: you built its pieces in chapter 02.

> **🔮 predict:** one data point $(x_1, y_1)$ with the basis $(1, x)$. the posterior precision gains the rank-one term $\beta\phi_1\phi_1^\top$. in which direction of weight space does the posterior shrink, and in which does it stay as wide as the prior?

> **▶ build: stage 2.** implement `posterior` and `sequential_posterior`.

### idea 3: predicting with uncertainty

the quantity you actually care about is the next observation $y_*$ at a new input $x_*$. it's $y_* = \phi_*^\top w + \varepsilon$, where $w \sim \mathcal N(m_N, S_N)$ under the posterior and $\varepsilon \sim \mathcal N(0, \beta^{-1})$ independently. a linear function of Gaussians plus independent Gaussian noise is Gaussian, with

$$p(y_* \mid y) = \mathcal N\big(\phi_*^\top m_N,\ \sigma_*^2\big), \qquad \sigma_*^2 = \underbrace{\frac{1}{\beta}}_{\text{noise}} + \underbrace{\phi_*^\top S_N\,\phi_*}_{\text{weight uncertainty}}.$$

this is the **posterior predictive distribution**: the model's prediction averaged over every weight vector it still considers plausible, rather than plugged in at one. the two terms are different kinds of uncertainty. the first is irreducible noise (sometimes called *aleatoric*): more data won't shrink it. the second is ignorance about the weights (*epistemic*): it shrinks like $1/n$ where you have data.

where does the second term grow? wherever $\phi_*$ points in a direction of weight space the data haven't pinned down. for polynomials, $\|\phi(x)\|$ explodes as $|x| \to \infty$, so the error bars fan out away from the data: the model knows it's extrapolating. for Gaussian bumps, $\phi(x) \to (1, 0, \dots, 0)$ far from every center, and $\phi_*^\top S_N\phi_*$ collapses to the posterior variance of the constant weight, which the data pinned down well. the model becomes *more* confident far from the data than in the gaps between them (warm-up 3). nothing is wrong with the algebra: the model really believes the function is flat out there, because its basis can't express anything else. predictive uncertainty is only ever uncertainty *within the model*.

> **spec.** `predictive(Phi, mean, cov, beta)` returns `(mean (n,), var (n,))` at the rows of `Phi`. compute only the diagonal of $\Phi S\Phi^\top$ (`np.einsum("ij,jk,ik->i", ...)` or `np.sum((Phi @ S) * Phi, axis=1)`), never the full $n \times n$ matrix.

> **▶ build: stage 3.** implement `predictive`.

### idea 4: the evidence

how good is a *model* (a basis, plus values of $\alpha$ and $\beta$), as opposed to a weight vector? ask how probable it made the data, with the weights integrated out:

$$p(y \mid \alpha, \beta) = \int p(y \mid w, \beta)\,p(w \mid \alpha)\,dw.$$

this is the **evidence** (or marginal likelihood). you can read off its value from the model directly: $y = \Phi w + \varepsilon$ is a linear function of Gaussians, so $y \sim \mathcal N(0,\ \beta^{-1}I + \alpha^{-1}\Phi\Phi^\top)$. that covariance is $n \times n$, which is fine for 50 points and hopeless for 50,000. completing the square inside the integral instead (on paper 2) gives a form that needs only $d \times d$ matrices:

$$\log p(y \mid \alpha, \beta) = \frac{d}{2}\log\alpha + \frac{n}{2}\log\beta - E(m_N) - \frac12\log|A| - \frac{n}{2}\log 2\pi,$$

$$E(m_N) = \frac{\beta}{2}\|y - \Phi m_N\|^2 + \frac{\alpha}{2}m_N^\top m_N.$$

why does this compare models fairly, when the training error doesn't? the evidence is a probability distribution over *datasets*, so it has to sum to one. a flexible model (degree 9) can explain many different datasets, so it spreads its probability thinly and gives any particular one a small value. a model that's too rigid (degree 1) can't explain the data at all. the one that wins is the simplest model that explains the data: **Occam's razor**, automatically, from the sum rule. in the formula you can see the trade-off directly: $-E$ rewards fit, and $\frac{d}{2}\log\alpha - \frac12\log|A|$ (the "Occam factor") penalizes the shrinkage of the posterior volume relative to the prior.

> **spec.** `log_evidence(Phi, y, alpha, beta)` returns a Python float, computed with the $d \times d$ formula (`np.linalg.slogdet` for $\log|A|$). the checks compare it with the $n \times n$ Gaussian density and also watch your memory.

> **🔮 predict:** 30 points from a cubic. the training error keeps falling as the polynomial degree goes from 3 to 9. what does the evidence do?

> **▶ build: stage 4.** implement `log_evidence`.

### idea 5: the estimator, and learning α and β from the evidence

wrap it all in an `mlfs` estimator. `BayesianLinearRegression(alpha=1.0, beta=1.0)` is a `Regressor`: `fit` computes the posterior, `predict` returns the predictive mean (or mean and standard deviation), `score` is $R^2$ (inherited), `sample_weights` draws from the posterior.

where do $\alpha$ and $\beta$ come from? you could cross-validate them, but the evidence offers something better: maximize it. this is **type-II maximum likelihood** (or *empirical Bayes*): maximum likelihood one level up, for the hyperparameters, with the weights integrated out. setting the derivatives of the log evidence to zero (on paper 3) gives the fixed-point equations of MacKay (1992):

$$\gamma = \sum_i\frac{\lambda_i}{\alpha + \lambda_i}, \qquad \alpha \leftarrow \frac{\gamma}{m_N^\top m_N}, \qquad \beta \leftarrow \frac{n - \gamma}{\|y - \Phi m_N\|^2},$$

where $\lambda_i$ are the eigenvalues of $\beta\Phi^\top\Phi$. the quantity $\gamma \in [0, d]$ has a lovely meaning: it's the **effective number of parameters**. a direction with $\lambda_i \gg \alpha$ is determined by the data and counts as 1; a direction with $\lambda_i \ll \alpha$ is determined by the prior and counts as 0. the $\beta$ update is then the familiar "residual sum of squares over $n$ minus the number of fitted parameters", with $\gamma$ in place of $d$. iterate: compute the posterior, update $\gamma$, $\alpha$, $\beta$, repeat until nothing changes. each step costs one $d \times d$ solve; the eigenvalues of $\Phi^\top\Phi$ only need computing once (multiply them by the current $\beta$).

> **spec.** `BayesianLinearRegression(*, alpha=1.0, beta=1.0, optimize=False, max_iters=100)` stores its arguments unchanged. `fit(X, y)` (X is the design matrix) sets `alpha_`, `beta_` (the constructor's values, or the evidence maximizers if `optimize=True`), `n_iter_` (fixed-point updates done, 0 without `optimize`), `mean_`, `cov_` (the posterior under `alpha_`, `beta_`), and returns `self`. with `optimize=True`, start from `alpha`, `beta`, apply the updates above, and stop when both change by less than a relative $10^{-10}$ or after `max_iters` updates (guard against division by zero: clip $\alpha, \beta$ to $[10^{-12}, 10^{12}]$). `predict(X, *, return_std=False)` returns the mean, or `(mean, std)` with the noise included. `log_evidence(X, y)` uses `alpha_`, `beta_`. `sample_weights(n=1, *, rng=None)` returns `(n, d)` posterior draws, `mean_ + Z L^T` with `L` the Cholesky factor of `cov_` and `Z` standard normals from `as_rng(rng)`.

> **▶ build: stage 5.** implement `BayesianLinearRegression`.

## worked example

one weight, no intercept: $\Phi = \begin{pmatrix}1\\2\end{pmatrix}$, $y = (1, 3)$, $\alpha = \beta = 1$.

**posterior.** $A = \alpha + \beta(1^2 + 2^2) = 6$, so $S_N = 1/6$ and $m_N = \beta S_N(1\cdot1 + 2\cdot3) = 7/6$. (least squares would give $7/5$: the prior shrinks it towards 0.)

**predictive at $\phi_* = 3$.** mean $3\cdot\frac76 = 3.5$; variance $\frac1\beta + 9\cdot\frac16 = 1 + 1.5 = 2.5$. the weight uncertainty contributes more than the noise here, because $\phi_* = 3$ is outside the data.

**evidence.** directly: $y \sim \mathcal N(0, I + \Phi\Phi^\top)$ with covariance $\begin{pmatrix}2 & 2\\2 & 5\end{pmatrix}$, determinant 6, inverse $\frac16\begin{pmatrix}5 & -2\\-2 & 2\end{pmatrix}$. the quadratic form is $\frac16(5 - 12 + 18) = \frac{11}{6}$, so $\log p(y) = -\frac{11}{12} - \frac12\log 6 - \log 2\pi \approx -3.6504$. with the $d \times d$ formula: the residuals are $(1 - \frac76, 3 - \frac{14}{6}) = (-\frac16, \frac46)$, so $E = \frac12\cdot\frac{17}{36} + \frac12\cdot\frac{49}{36} = \frac{11}{12}$, and $-\frac12\log|A| = -\frac12\log 6$: the same number. your `posterior`, `predictive` and `log_evidence` must reproduce all three.

## on paper

1. **complete the square.** show that $-\frac\beta2\|y - \Phi w\|^2 - \frac\alpha2\|w\|^2 = -\frac12(w - m_N)^\top A(w - m_N) - E(m_N)$ with $A$, $m_N$ and $E$ as in the lesson.

<details><summary>solution</summary>

expand the left side: $-\frac12\big[w^\top(\alpha I + \beta\Phi^\top\Phi)w - 2\beta w^\top\Phi^\top y + \beta y^\top y\big] = -\frac12\big[w^\top Aw - 2w^\top Am_N + \beta y^\top y\big]$ using $Am_N = \beta\Phi^\top y$. adding and subtracting $m_N^\top Am_N$ gives $-\frac12(w - m_N)^\top A(w - m_N) - \frac12(\beta y^\top y - m_N^\top Am_N)$. finally $\beta y^\top y - m_N^\top A m_N = \beta y^\top y - \beta m_N^\top\Phi^\top y$, and expanding $2E(m_N) = \beta\|y - \Phi m_N\|^2 + \alpha\|m_N\|^2 = \beta y^\top y - 2\beta m_N^\top\Phi^\top y + m_N^\top Am_N$ shows they're equal.

</details>

2. **the evidence in $d \times d$ form.** use on paper 1 and the Gaussian integral $\int e^{-\frac12(w - m)^\top A(w - m)}dw = (2\pi)^{d/2}|A|^{-1/2}$ to derive the lesson's formula for $\log p(y \mid \alpha, \beta)$.

<details><summary>solution</summary>

$p(y \mid w)p(w) = \big(\frac{\beta}{2\pi}\big)^{n/2}\big(\frac{\alpha}{2\pi}\big)^{d/2}\exp\big(-\frac\beta2\|y - \Phi w\|^2 - \frac\alpha2\|w\|^2\big)$. by on paper 1 the exponent is $-\frac12(w - m_N)^\top A(w - m_N) - E(m_N)$; integrating over $w$ gives $(2\pi)^{d/2}|A|^{-1/2}e^{-E(m_N)}$. the $(2\pi)^{d/2}$ cancels the one from the prior. take logs: $\frac n2\log\beta + \frac d2\log\alpha - E(m_N) - \frac12\log|A| - \frac n2\log 2\pi$.

</details>

3. **the α update.** differentiate the log evidence with respect to $\alpha$ and derive $\alpha = \gamma / m_N^\top m_N$. (hint: $\frac{d}{d\alpha}\log|A| = \sum_i\frac{1}{\alpha + \lambda_i}$, and $m_N$ is a stationary point of $E$, so you can ignore its dependence on $\alpha$.)

<details><summary>solution</summary>

$\frac{\partial}{\partial\alpha}$: $\frac{d}{2\alpha} - \frac12m_N^\top m_N - \frac12\sum_i\frac{1}{\alpha + \lambda_i} = 0$. multiply by $2\alpha$: $\alpha m_N^\top m_N = d - \sum_i\frac{\alpha}{\alpha + \lambda_i} = \sum_i\frac{\lambda_i}{\alpha + \lambda_i} = \gamma$. (the $\beta$ update follows the same way, using $\frac{d}{d\beta}\lambda_i = \lambda_i/\beta$.) it's a fixed point, not a closed form, because $\gamma$ and $m_N$ depend on $\alpha$.

</details>

4. **ridge is MAP.** show that the posterior mode $m_N$ minimizes $\|y - \Phi w\|^2 + \lambda\|w\|^2$ for $\lambda = \alpha/\beta$. what prior gives the lasso?

<details><summary>solution</summary>

the mode maximizes $\log p(w \mid y)$, i.e. minimizes $\frac\beta2\|y - \Phi w\|^2 + \frac\alpha2\|w\|^2 = \frac\beta2\big(\|y - \Phi w\|^2 + \frac\alpha\beta\|w\|^2\big)$. a Laplace prior $p(w) \propto e^{-\lambda'\|w\|_1}$ gives the lasso, but then the posterior isn't Gaussian and its mean isn't sparse: only the mode is.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how the predictive uncertainty behaves far from the data, for a polynomial basis and for a Gaussian-bump basis.
2. which polynomial degree the evidence prefers, compared with the training and test error.
3. what the posterior over a line's two weights looks like after 0, 1, 2 and 20 points.

## check your understanding

1. why is $A = \alpha I + \beta\Phi^\top\Phi$ invertible even when $n < d$?

<details><summary>answer</summary>

$\Phi^\top\Phi$ is positive semidefinite, so every eigenvalue of $A$ is at least $\alpha > 0$. the prior supplies information in the directions the data don't.

</details>

2. as $n \to \infty$ with fixed $\alpha$, what happens to $m_N$ and $S_N$?

<details><summary>answer</summary>

$\Phi^\top\Phi$ grows like $n$, so $S_N \approx (\beta\Phi^\top\Phi)^{-1} \to 0$ like $1/n$, and $m_N$ approaches the least-squares solution: the prior's influence fades. the predictive variance tends to the noise $1/\beta$.

</details>

3. (warm-up 3.) how would you make the Gaussian-bump model's uncertainty grow away from the data?

<details><summary>answer</summary>

give it basis functions that don't vanish there (more bumps covering the region, or a polynomial part), or take the limit of infinitely many bumps everywhere: that's a Gaussian process with a squared-exponential kernel (ch 09), whose predictive variance returns to the prior variance away from the data.

</details>

4. you multiply all the targets by 10. what happens to the evidence-optimal $\beta$ and $\alpha$?

<details><summary>answer</summary>

the noise variance scales by 100, so $\beta$ drops by 100; the weights scale by 10, so $\alpha$ (the inverse weight variance) drops by 100 too. $\gamma = \sum_i\lambda_i/(\alpha + \lambda_i)$ doesn't change: $\lambda_i/\alpha$ depends only on the ratio $\beta/\alpha$, which stays the same.

</details>

5. why can't you compare two models by the maximum likelihood $\max_w p(y \mid w)$ instead of the evidence?

<details><summary>answer</summary>

the maximum likelihood never decreases when you add features, so it always picks the biggest model. the evidence averages the likelihood over the prior, and a big model wastes prior mass on weights that explain the data badly.

</details>

6. what does $\gamma$ equal when $\alpha \to 0$, and when $\alpha \to \infty$?

<details><summary>answer</summary>

$\alpha \to 0$ (flat prior): every direction with $\lambda_i > 0$ counts fully, $\gamma = \operatorname{rank}\Phi$. $\alpha \to \infty$: the prior pins every weight to 0, $\gamma = 0$.

</details>

7. the evidence is computed from the training data only. why isn't that "testing on the training set"?

<details><summary>answer</summary>

by the chain rule $p(y_1, \dots, y_n) = \prod_i p(y_i \mid y_1, \dots, y_{i-1})$: the evidence is the product of one-step-ahead predictive probabilities, each made *before* seeing that point. it's a kind of sequential cross-validation. (it is sensitive to the prior, though, which cross-validation isn't.)

</details>

## common bugs

- **forgetting the noise in the predictive variance.** symptom: error bars that shrink to zero inside the data. $\sigma_*^2 = 1/\beta + \phi^\top S\phi$.
- **mixing up precision and variance.** symptom: answers off by factors of $\alpha^2$ or $\beta^2$. $\alpha$ and $\beta$ are *inverse* variances: the prior covariance is $I/\alpha$.
- **building the $n \times n$ covariance.** symptom: the evidence or predictive variance runs out of memory on a large dataset. use the $d \times d$ formulas and only the diagonal of $\Phi S\Phi^\top$.
- **refitting the prior in `sequential_posterior`.** symptom: sequential and batch answers disagree. the prior term is $S_0^{-1}m_0$, not 0, once $m_0 \ne 0$.
- **dividing by zero in the fixed point.** symptom: `ZeroDivisionError` or NaN on tiny datasets, where the evidence wants $\alpha \to \infty$ (all weights zero) or $\beta \to \infty$ (exact interpolation). clip.

## going deeper

- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, §3.3–3.5 (Bayesian linear regression, the evidence approximation, effective number of parameters). [microsoft.com/en-us/research/publication/pattern-recognition-machine-learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- MacKay (1992), "Bayesian interpolation", *Neural Computation* 4(3). the fixed-point updates and the Occam factor. [doi:10.1162/neco.1992.4.3.415](https://doi.org/10.1162/neco.1992.4.3.415)
- Murphy (2023), *Probabilistic Machine Learning: Advanced Topics*, MIT Press, ch. 15 (Bayesian linear models). [probml.github.io/pml-book/book2.html](https://probml.github.io/pml-book/book2.html)
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press, ch. 2: the weight-space view, and how it becomes a Gaussian process. [gaussianprocess.org/gpml](http://gaussianprocess.org/gpml/)
- Tipping (2001), "Sparse Bayesian learning and the relevance vector machine", *JMLR* 1. one $\alpha$ per weight, with the same evidence updates, prunes features automatically. [jmlr.org/papers/v1/tipping01a.html](https://www.jmlr.org/papers/v1/tipping01a.html)

## connections

- ← builds on: Gaussian conditioning and marginalization (ch 04 · topic 04), least squares and the normal equations (ch 02 · topic 04), Bayes' rule and conjugacy (ch 05), ridge regression and the `Regressor` protocol (ch 07).
- → used in: Gaussian process regression, the infinite-feature limit of this model (ch 09), the conjugate posteriors that MCMC and VI are checked against (topics 05–07), Bayesian optimization, Kalman filtering (sequential updates), and Laplace / last-layer approximations for neural network uncertainty.
