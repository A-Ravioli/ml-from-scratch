# gaussian processes

## why this matters

kernel ridge regression gives you a prediction. it doesn't tell you how much to trust it, and it doesn't tell you which kernel width or regularization to use, except by cross-validation. a **Gaussian process** (GP) is the Bayesian version of the same model, and it answers both. its predictions come with error bars that widen away from the data, which is exactly what you need to decide *where to measure next*: that's Bayesian optimization, this chapter's capstone. and its hyperparameters can be learned by maximizing a single smooth function of the training data, the **marginal likelihood**, with a gradient optimizer, instead of a grid search.

a GP is a probability distribution over functions. you'll see that its posterior mean is *exactly* the kernel ridge predictor of topic 05, with the regularization strength equal to the noise variance, so everything you built there carries over. what's new is the covariance: a GP knows what it doesn't know. GPs are the standard tool for small, expensive datasets (experiment design, hyperparameter tuning, geostatistics, where they're called kriging, calibration of simulators), and the infinite-width limit of neural networks is a GP too (chapter 19).

## warm-up: predict first

1. before seeing any data, a GP with an RBF kernel of length scale 1 assigns each $f(x)$ a distribution. what is it, and how correlated are $f(0)$ and $f(3)$?
2. you observe $f$ at a few points without noise. what's the posterior standard deviation *at* those points? far away from them?
3. the GP posterior mean and kernel ridge regression use the same kernel. when are they identical?
4. as you shrink the length scale towards 0, the training data fit gets better and better. why doesn't the marginal likelihood keep increasing?
5. to choose the next point to evaluate, should you pick the one with the highest predicted mean, the highest uncertainty, or something else?

## prerequisites check

- [ ] you can condition a multivariate Gaussian on some of its coordinates (→ ch 04 · topic 04)
- [ ] you can solve linear systems with a Cholesky factor and compute $\log\det$ from it (→ ch 02 · topic 03)
- [ ] you know matrix calculus identities for $\partial\log\det A$ and $\partial A^{-1}$ (→ ch 02 · topic 07)
- [ ] you've used L-BFGS from `mlfs.optim` (→ ch 06 · topic 05)
- [ ] you've built kernel ridge regression (→ topic 05) and know what the RBF kernel's length scale does (→ topic 03)

## core ideas

### idea 1: a distribution over functions

a **Gaussian process** with mean function $m$ and covariance function (kernel) $k$ is a random function $f$ such that, for *any* finite set of inputs $x_1, \dots, x_n$, the vector $(f(x_1), \dots, f(x_n))$ is multivariate Gaussian with mean $m(x_i)$ and covariance $K_{ij} = k(x_i, x_j)$. that's the whole definition, and it's consistent because the marginals of a Gaussian are Gaussian. it's why kernels must be PSD (topic 03): $K$ is a covariance matrix.

you'll use a zero mean on centred targets (subtract the training mean of $y$, add it back to predictions) and three **stationary** kernels, functions of $r = \|x - y\|/\ell$:

$$k_{\text{RBF}} = \sigma_f^2\, e^{-r^2/2}, \qquad k_{3/2} = \sigma_f^2\,(1 + \sqrt3\, r)\,e^{-\sqrt3\, r}, \qquad k_{5/2} = \sigma_f^2\left(1 + \sqrt5\, r + \tfrac53 r^2\right)e^{-\sqrt5\, r}.$$

two hyperparameters control them. the **variance** $\sigma_f^2$ is the prior variance of each $f(x)$ (warm-up 1: $f(x) \sim \mathcal N(0, \sigma_f^2)$), so it sets the typical amplitude. the **length scale** $\ell$ sets how far you must move before $f$ changes appreciably: the correlation between $f(0)$ and $f(3)$ under the RBF kernel with $\ell = 1$ is $e^{-4.5} \approx 0.011$. (with $\gamma = 1/(2\ell^2)$ this is topic 03's RBF kernel, times $\sigma_f^2$.) the kernels differ in smoothness: samples from the RBF prior are infinitely differentiable, Matérn-5/2 samples twice, Matérn-3/2 samples once. many real functions are rougher than the RBF assumes, which is why Matérn-5/2 is the usual default in Bayesian optimization.

> **▶ build: stage 1.** implement `stationary_kernel(X, Y=None, *, kind, lengthscale, variance)` for `"rbf"`, `"matern32"` and `"matern52"`, with distances from your `pairwise_distances` (pass `Y=None` through). an unknown kind raises `ValueError`. `kernel_lengthscale_grad`, the derivative you'll need in stage 4, is given.

### idea 2: the posterior

observations are noisy: $y_i = f(x_i) + \varepsilon_i$ with $\varepsilon_i \sim \mathcal N(0, \sigma_n^2)$ independent. then the training targets and the function values $f_* = f(X_*)$ at test points are jointly Gaussian,

$$\begin{pmatrix} y \\ f_* \end{pmatrix} \sim \mathcal N\left(0,\ \begin{pmatrix} K + \sigma_n^2 I & K_* \\ K_*^\top & K_{**} \end{pmatrix}\right),$$

with $K = k(X, X)$, $K_* = k(X, X_*)$ and $K_{**} = k(X_*, X_*)$. conditioning on $y$ (the formula from chapter 04) gives the posterior

$$\mathbb E[f_* \mid y] = K_*^\top (K + \sigma_n^2 I)^{-1} y, \qquad \mathrm{Cov}[f_* \mid y] = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*.$$

look at the mean: it's $\sum_i \beta_i k(x_i, x_*)$ with $\beta = (K + \sigma_n^2 I)^{-1}y$, which is kernel ridge regression with $\alpha = \sigma_n^2$ (warm-up 3). the regularization strength of topic 05 *is* the noise variance. the covariance is new. it doesn't depend on $y$ at all, only on where the data are: it's the prior covariance minus what the observations explain. near a training point the variance drops (to about $\sigma_n^2$ for small noise, and to 0 without noise, warm-up 2), and far from all of them it returns to the prior $\sigma_f^2$.

the stable way to compute it: one Cholesky factorization $LL^\top = K + \sigma_n^2 I$ (plus a tiny jitter, $10^{-8}$, so it never fails on a numerically singular $K$), then $\alpha = L^{-\top}L^{-1}y$, the mean $K_*^\top\alpha$, and with $V = L^{-1}K_*$ the covariance $K_{**} - V^\top V$, whose diagonal gives the standard deviations without forming the full matrix: $\sigma_*^2 = \sigma_f^2 - \sum_i V_{i*}^2$. the cost is the same as KRR: $O(n^3)$ once, then $O(n)$ per mean and $O(n^2)$ per variance.

this is the uncertainty of the *function*. a new noisy observation at $x_*$ has variance $\sigma_*^2 + \sigma_n^2$; add the noise when you want predictive intervals for data.

> **▶ build: stage 2.** implement `fit` (store `X_train_`, `y_mean_` and the centred `y_train_`; $\theta_0 = \log(\ell, \sigma_f^2, \sigma_n^2)$ from `__init__`; if `optimize`, replace it by `optimize_theta(theta0)`, which you'll write in stage 5; finish with the given `set_theta`, which computes `L_` and `alpha_`) and `predict(X, *, return_std=False, return_cov=False)`. an unfitted GP predicts its prior: zeros, and $\sqrt{\sigma_f^2}$ or $K_{**}$. the checks use `optimize=False` until stage 5.

> **🔮 predict:** ten noisy observations on $[0, 5]$ and an RBF kernel with $\ell = 0.5$. sketch the posterior standard deviation over $[-2, 7]$. where is it largest? is it ever larger than the prior's?

### idea 3: sampling functions

the posterior (or the prior) at $m$ points is a Gaussian $\mathcal N(\mu, \Sigma)$, so a sampled function is $\mu + Az$ with $AA^\top = \Sigma$ and $z \sim \mathcal N(0, I_m)$. draw it on a dense grid and plot it, and you see what the GP believes: wiggly or smooth, large or small, pinned near the data, free far away. the matrix root needs care, because a posterior covariance on a dense grid is numerically singular (nearby points are almost perfectly correlated) and can have eigenvalues of $-10^{-15}$. a Cholesky factorization would fail; an eigendecomposition $\Sigma = U\,\mathrm{diag}(w)\,U^\top$ with $A = U\,\mathrm{diag}(\sqrt{\max(w, 0)})$ always works.

> **▶ build: stage 3.** implement `sample_y(X, n_samples=1, *, rng=None)`: the mean and covariance from your `predict(X, return_cov=True)` (so an unfitted GP samples its prior), an eigendecomposition, and `standard_normal((n_samples, m))` from `as_rng(rng)`. return $(n_\text{samples}, m)$.

### idea 4: the marginal likelihood

how well do hyperparameters $\theta = (\ell, \sigma_f^2, \sigma_n^2)$ explain the data? integrate the function out: under the GP, $y \sim \mathcal N(0, K_y)$ with $K_y = K_\theta + \sigma_n^2 I$, so the **log marginal likelihood** (or log evidence) is

$$\log p(y \mid X, \theta) = \underbrace{-\tfrac12 y^\top K_y^{-1} y}_{\text{data fit}}\ \underbrace{-\ \tfrac12\log\det K_y}_{\text{complexity penalty}}\ -\ \tfrac n2\log 2\pi.$$

with the Cholesky factor, $y^\top K_y^{-1}y = y^\top\alpha$ and $\frac12\log\det K_y = \sum_i \log L_{ii}$. the two terms pull in opposite directions (warm-up 4). a short length scale makes $K$ close to diagonal, so the model can fit anything and the data-fit term improves; but a flexible model spreads its probability over many possible datasets, so it can't assign much to the one you observed, and $\log\det K_y$ grows. the marginal likelihood is maximized by the *simplest* model that explains the data: an automatic Occam's razor, with no validation set.

it's differentiable in $\theta$. with $\alpha = K_y^{-1}y$ and the identities $\partial\log\det A = \mathrm{tr}(A^{-1}\partial A)$ and $\partial A^{-1} = -A^{-1}(\partial A)A^{-1}$ (on paper 2),

$$\frac{\partial \log p(y \mid \theta)}{\partial\theta_j} = \frac12\mathrm{tr}\!\left((\alpha\alpha^\top - K_y^{-1})\,\frac{\partial K_y}{\partial\theta_j}\right).$$

work in **log** hyperparameters, $\theta = \log(\ell, \sigma_f^2, \sigma_n^2)$: they're positive, they matter on a multiplicative scale, and the derivatives are simple. $\partial K_y/\partial\log\sigma_f^2 = K$, $\partial K_y/\partial\log\sigma_n^2 = \sigma_n^2 I$, and $\partial K_y/\partial\log\ell$ is `kernel_lengthscale_grad` (for the RBF kernel, $K \odot r^2$). $\mathrm{tr}(AB) = \sum_{ij}A_{ij}B_{ij}$ for symmetric matrices, so no matrix product is needed.

> **▶ build: stage 4.** implement `log_marginal_likelihood(theta=None)` (`theta=None` means the fitted `theta_`; if the Cholesky raises `LinAlgError`, return `-np.inf`) and `lml_gradient(theta)`, with the same $K_y = K + (\sigma_n^2 + 10^{-8})I$ as `set_theta`. the check compares your gradient with finite differences.

### idea 5: learning the hyperparameters

choosing $\theta$ by maximizing the marginal likelihood is called **type-II maximum likelihood** (or empirical Bayes): the function is integrated out, the three hyperparameters are optimized. minimize $-\log p(y \mid \theta)$ with your L-BFGS from chapter 06, using $-\nabla$ as the gradient. it costs one $O(n^3)$ factorization per evaluation, which is fine for a few thousand points.

the objective isn't convex, and it often has more than one local maximum. the classic case: a short length scale with small noise ("the wiggles are signal") against a long length scale with large noise ("the wiggles are noise"). both explain the data; the evidence decides how much better one is, and a gradient method finds whichever basin it starts in. the standard remedy is **restarts**: optimize from the initial $\theta_0$ and from a few random starting points, and keep the best. with only three hyperparameters, overfitting them is rarely a problem, unlike optimizing $n$ function values. trial steps of the line search can reach extreme values; let those evaluate to $+\infty$ (a failed Cholesky) quietly, and the line search simply backtracks.

> **▶ build: stage 5.** implement `optimize_theta(theta0)`: minimize $-$LML with `lbfgs(f, g, start, max_iters=200)` from `theta0` and from `n_restarts` starts `theta0 + uniform(-3, 3, size=3)` drawn from one `as_rng(self.rng)`, and return the $\theta$ with the highest LML (never worse than `theta0` itself). fit must not change the hyperparameters passed to `__init__`: the learned ones go in `theta_`, `lengthscale_`, `variance_`, `noise_`.

> **🔮 predict:** you fit a GP with `optimize=True` to 25 noisy points from $\sin 3x$. starting from a length scale of 5 and noise 0.5, will L-BFGS find the "wiggles are signal" explanation by itself?

### idea 6: where to look next: expected improvement

suppose $f$ is expensive, like the validation score of a model trained with hyperparameters $x$, and you want its maximum in as few evaluations as possible. fit a GP to the evaluations so far, and use its posterior to decide where to evaluate next. an **acquisition function** scores candidates by how promising they are. picking the highest mean exploits (and gets stuck near the current best); picking the highest uncertainty explores (and wastes evaluations on hopeless regions) (warm-up 5). **expected improvement** balances both: with $f_{\text{best}}$ the best value observed so far, it's the expected amount by which $f(x)$ beats it,

$$\mathrm{EI}(x) = \mathbb E\big[\max(f(x) - f_{\text{best}} - \xi,\ 0)\big] = (\mu - f_{\text{best}} - \xi)\,\Phi(z) + \sigma\,\varphi(z), \qquad z = \frac{\mu - f_{\text{best}} - \xi}{\sigma},$$

for $f(x) \sim \mathcal N(\mu, \sigma^2)$, with $\Phi$ and $\varphi$ the standard normal CDF and density (on paper 4). it's large where the mean is high *or* the uncertainty is large, and it's always nonnegative. at $\sigma = 0$ the formula divides by zero, and the limit as $\sigma \to 0$ is $\max(\mu - f_{\text{best}} - \xi, 0)$; the contract's convention is simply $\mathrm{EI} = 0$ there. the two agree where it matters: a noise-free GP has $\sigma = 0$ only at evaluated points, where $\mu$ is an observed value, so $\mu \le f_{\text{best}}$. the small margin $\xi \ge 0$ demands a minimum improvement and nudges the search towards exploration. the standard normal CDF is $\Phi(z) = \frac12\big(1 + \mathrm{erf}(z/\sqrt2)\big) = \frac12\mathrm{erfc}(-z/\sqrt2)$; use the second form, because for very negative $z$ the first subtracts two nearly equal numbers and EI comes out with no correct digits, or even negative.

> **▶ build: stage 6.** implement `expected_improvement(mu, sigma, best, *, xi=0.01)` for maximization: arrays broadcast, a scalar in gives a float out, 0 wherever `sigma == 0` (without dividing by zero), and never negative (clip at 0). `math.erfc` vectorized with `np.vectorize` gives $\Phi$ accurately in the far tail.

## worked example

a GP with an RBF kernel, $\ell = 1$, $\sigma_f^2 = 1$, $\sigma_n^2 = 0.1$, fitted to $x = (0, 1)$, $y = (1, 3)$.

- centre: $\bar y = 2$, so the GP models $y_c = (-1, 1)$.
- $K + \sigma_n^2 I = \begin{pmatrix}1.1 & e^{-1/2}\\ e^{-1/2} & 1.1\end{pmatrix}$ with $e^{-1/2} = 0.606531$. the vector $(-1, 1)$ is an eigenvector with eigenvalue $1.1 - 0.606531 = 0.493469$, so $\alpha = (-1, 1)/0.493469 = (-2.026472, 2.026472)$.
- at $x_* = 0.5$: $k_* = (e^{-1/8}, e^{-1/8}) = 0.882497\,(1, 1)$, so $k_*^\top\alpha = 0$ and the mean is $\bar y = 2$, halfway, by symmetry. at $x_* = 0$: $k_* = (1, 0.606531)$, the mean is $2 + (-2.026472 + 0.606531 \times 2.026472) = 1.202658$, pulled from the observed 1 towards the mean 2 by the noise.
- variance at $x_* = 0.5$: $k_*$ is along $(1, 1)$, the eigenvector with eigenvalue $1.706531$, so $k_*^\top(K + \sigma_n^2 I)^{-1}k_* = 2 \times 0.882497^2/1.706531 = 0.912729$, and $\sigma_*^2 = 1 - 0.912729 = 0.087271$, $\sigma_* = 0.2954$: much less than the prior's 1, because both observations are close.
- **expected improvement** with $\mu = f_{\text{best}} + \xi$ and $\sigma = 1$: $z = 0$, so $\mathrm{EI} = 0 \cdot \frac12 + 1 \cdot \varphi(0) = 0.39894$.

the tests check $\alpha$, both means, the standard deviation and the EI value.

## on paper

1. **GP regression is kernel ridge regression.** show that the posterior mean $K_*^\top(K + \sigma_n^2 I)^{-1}y$ equals the KRR prediction with $\alpha = \sigma_n^2$. then show that it's also the MAP estimate of $f$ at the training points under the prior $f \sim \mathcal N(0, K)$.

<details><summary>solution</summary>

the KRR prediction is $\sum_i\beta_i k(x_i, x_*) = K_*^\top\beta$ with $\beta = (K + \alpha I)^{-1}y$: the same expression with $\alpha = \sigma_n^2$. MAP: maximize $\log p(y \mid f) + \log p(f) = -\frac{1}{2\sigma_n^2}\|y - f\|^2 - \frac12 f^\top K^{-1}f + \text{const}$. the gradient $\frac{1}{\sigma_n^2}(y - f) - K^{-1}f = 0$ gives $f = K(K + \sigma_n^2 I)^{-1}y$, the posterior mean at the training points. with $f = K\beta$, the penalty $f^\top K^{-1}f = \beta^\top K\beta = \|f\|^2_{\mathcal H}$: the GP prior *is* the RKHS-norm penalty.

</details>

2. **the gradient of the evidence.** using $\partial\log\det A = \mathrm{tr}(A^{-1}\partial A)$ and $\partial A^{-1} = -A^{-1}(\partial A)A^{-1}$, derive $\partial_j\log p(y \mid \theta) = \frac12\mathrm{tr}\big((\alpha\alpha^\top - K_y^{-1})\partial_j K_y\big)$.

<details><summary>solution</summary>

$\partial_j\left(-\frac12 y^\top K_y^{-1}y\right) = \frac12 y^\top K_y^{-1}(\partial_j K_y)K_y^{-1}y = \frac12\alpha^\top(\partial_j K_y)\alpha = \frac12\mathrm{tr}(\alpha\alpha^\top\partial_j K_y)$. $\partial_j\left(-\frac12\log\det K_y\right) = -\frac12\mathrm{tr}(K_y^{-1}\partial_j K_y)$. adding them gives the formula. for log-parameters: $\partial K_y/\partial\log\sigma_f^2 = \sigma_f^2\,\partial K/\partial\sigma_f^2 = K$, and $\partial K_y/\partial\log\sigma_n^2 = \sigma_n^2 I$.

</details>

3. **the length scale derivative.** for the RBF kernel $k = \sigma_f^2 e^{-d^2/(2\ell^2)}$ (with $d = \|x - y\|$), show $\partial k/\partial\log\ell = k \cdot d^2/\ell^2$. for the Matérn-3/2 kernel with $s = \sqrt3\,d/\ell$, show $\partial k/\partial\log\ell = \sigma_f^2 s^2 e^{-s}$.

<details><summary>solution</summary>

RBF: $\partial_\ell k = k \cdot d^2/\ell^3$, and $\partial_{\log\ell} = \ell\,\partial_\ell$, so $k\,d^2/\ell^2$. Matérn-3/2: $k = \sigma_f^2(1 + s)e^{-s}$, $\partial_s k = \sigma_f^2(e^{-s} - (1 + s)e^{-s}) = -\sigma_f^2 s e^{-s}$, and $\partial_{\log\ell}s = -s$, so $\partial_{\log\ell}k = \sigma_f^2 s^2 e^{-s}$.

</details>

4. **expected improvement.** for $f \sim \mathcal N(\mu, \sigma^2)$ and $\sigma > 0$, compute $\mathbb E[\max(f - c, 0)]$ with $c = f_{\text{best}} + \xi$.

<details><summary>solution</summary>

write $f = \mu + \sigma u$ with $u \sim \mathcal N(0, 1)$; the integrand is positive for $u > -z$ with $z = (\mu - c)/\sigma$. $\mathbb E = \int_{-z}^\infty(\mu - c + \sigma u)\varphi(u)\,du = (\mu - c)\Phi(z) + \sigma\int_{-z}^\infty u\varphi(u)\,du = (\mu - c)\Phi(z) + \sigma\varphi(z)$, using $\int_a^\infty u\varphi(u)\,du = \varphi(a)$ and $\varphi(-z) = \varphi(z)$.

</details>

## experiment

`lab.py` uses your `GaussianProcessRegressor`:

1. prior and posterior samples under the RBF, Matérn-5/2 and Matérn-3/2 kernels, with a length-scale slider.
2. the log marginal likelihood as a landscape over length scale and noise: two explanations of the same data.
3. on real data (California housing): learned hyperparameters, calibration of the predictive intervals, and the GP mean against kernel ridge regression.

## check your understanding

1. why does the posterior covariance not depend on the observed $y$?

<details><summary>answer</summary>

conditioning a jointly Gaussian vector on some coordinates gives a covariance that depends only on the joint covariance, not on the observed values. so the GP's uncertainty depends on *where* you measured, not on *what* you saw. (the hyperparameters, once learned from $y$, do bring $y$ in indirectly.)

</details>

2. what does a learned noise variance close to 0 combined with a tiny length scale suggest?

<details><summary>answer</summary>

the model is explaining every wiggle as signal, which for noisy data usually means a poor local optimum or an overly flexible kernel. compare its log marginal likelihood with a fit from a longer length scale; check predictions on held-out data.

</details>

3. a GP and KRR with the same kernel make identical mean predictions. what does the GP add?

<details><summary>answer</summary>

the posterior variance (error bars, which drive Bayesian optimization and active learning), samples of plausible functions, and a principled way to set the kernel's hyperparameters and the noise level (the marginal likelihood) without cross-validation.

</details>

4. why optimize the logarithms of the hyperparameters?

<details><summary>answer</summary>

they must be positive, and the log maps them to all of $\mathbb R$, so an unconstrained optimizer can't step into invalid values. they also act multiplicatively (doubling a length scale matters the same at 0.1 as at 10), so the log makes the landscape better scaled.

</details>

5. in Bayesian optimization, why is EI zero at an already-evaluated point when there's no noise?

<details><summary>answer</summary>

the posterior there has $\sigma = 0$ and mean equal to the observed value, which is at most $f_{\text{best}}$. evaluating it again can't improve on what you know, so EI correctly gives it no value. with noisy observations, $\sigma$ stays positive and re-evaluating can be worthwhile.

</details>

6. GP regression costs $O(n^3)$. which ideas from topic 05 apply?

<details><summary>answer</summary>

the same ones: Nyström or inducing-point approximations (sparse GPs, with $m$ inducing inputs and $O(nm^2)$ cost) and random features (a Bayesian linear model on random Fourier features approximates the GP). for $n$ in the tens of thousands, iterative solvers (conjugate gradients) on GPUs are another route.

</details>

## common bugs

- **forgetting to centre $y$** (or to add the mean back). symptom: predictions sag towards 0 away from the data, and the learned variance is inflated to cover the offset.
- **the noise as a standard deviation in one place and a variance in another.** symptom: a factor that's the square of what you expect, and a learned noise that disagrees with sklearn's by exactly that.
- **`np.linalg.inv` instead of Cholesky solves.** slower and less stable; the log determinant from `np.linalg.det` overflows or underflows for $n$ in the hundreds. use $2\sum\log L_{ii}$.
- **gradient with respect to the raw hyperparameter while optimizing its log.** symptom: the finite-difference check fails by a factor of $\ell$ or $\sigma^2$.
- **Cholesky of a posterior covariance on a dense grid.** symptom: `LinAlgError` when sampling. use an eigendecomposition with clipped eigenvalues.
- **EI dividing by $\sigma = 0$.** symptom: `nan` (and a warning) at evaluated points, which can make `argmax` pick them.

## going deeper

- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press, chapters 2, 4 and 5 (regression, covariance functions, model selection and the marginal likelihood). free online. [gaussianprocess.org/gpml](http://gaussianprocess.org/gpml/)
- Jones, Schonlau & Welch (1998), "Efficient global optimization of expensive black-box functions", *Journal of Global Optimization* 13: 455–492. expected improvement. [doi:10.1023/A:1008306431147](https://doi.org/10.1023/A:1008306431147)
- Snoek, Larochelle & Adams (2012), "Practical Bayesian optimization of machine learning algorithms", *NeurIPS*. Matérn-5/2 GPs and EI for hyperparameter tuning. [arXiv:1206.2944](https://arxiv.org/abs/1206.2944)
- Shahriari, Swersky, Wang, Adams & de Freitas (2016), "Taking the human out of the loop: a review of Bayesian optimization", *Proceedings of the IEEE* 104(1): 148–175. [doi:10.1109/JPROC.2015.2494218](https://doi.org/10.1109/JPROC.2015.2494218)
- Kanagawa, Hennig, Sejdinovic & Sriperumbudur (2018), "Gaussian processes and kernel methods: a review on connections and equivalences". [arXiv:1807.02582](https://arxiv.org/abs/1807.02582)
- Neal (1996), *Bayesian Learning for Neural Networks*, Springer, chapter 2: infinitely wide networks are GPs. [doi:10.1007/978-1-4612-0745-0](https://doi.org/10.1007/978-1-4612-0745-0)

## connections

- ← builds on: conditioning Gaussians (ch 04), Cholesky solves and log determinants (ch 02), matrix calculus (ch 02), L-BFGS (ch 06), kernels (topic 03) and kernel ridge regression (topic 05).
- → used in: Bayesian optimization (this chapter's capstone), Bayesian linear regression and the evidence (ch 12), the NNGP and neural tangent kernel limits of wide networks (ch 19), and uncertainty estimates and calibration in safety-critical prediction (ch 28).
