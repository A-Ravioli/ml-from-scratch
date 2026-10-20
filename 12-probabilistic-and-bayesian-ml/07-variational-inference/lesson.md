# variational inference

## why this matters

MCMC (topics 05–06) is asymptotically exact: run it long enough and it gets the posterior right. but "long enough" can mean hours for a model with a million data points, every iteration touches all of them, and you can never be quite sure it has converged. sometimes you'd happily trade exactness for speed and a clear stopping criterion: fitting a topic model to a million documents, an approximate posterior inside a variational autoencoder (ch 21), a Bayesian neural network.

**variational inference** turns inference into optimization. pick a family of simple distributions $q$ (independent Gaussians, say), and find the member closest to the true posterior. "closest" means smallest KL divergence, and the trick is that you can minimize it without knowing the posterior's normalizing constant, by maximizing a quantity you *can* compute: the **evidence lower bound**, or ELBO. you then have all the tools of optimization at your disposal: coordinate ascent with closed-form updates for conjugate models, and stochastic gradients with the reparameterization trick for everything else.

the price is bias, and it has a direction. this topic builds both algorithms and then shows you, precisely, how the answer is wrong: VI with the usual KL underestimates posterior variance, sometimes badly, and ignores all but one mode. knowing *why* (the direction of the KL) is what lets you decide when VI is good enough.

## warm-up: predict first

1. you fit a Gaussian $q$ to a posterior by maximizing the ELBO. can the ELBO ever be larger than $\log p(x)$?
2. the posterior over two parameters is a Gaussian with correlation 0.9 and unit marginal variances. the best approximation with *independent* coordinates: what variances does it report?
3. the posterior has two well-separated modes. where does the best single Gaussian go?
4. coordinate ascent on the ELBO for a Gaussian mixture: does the ELBO ever go down? does it always find the same answer?
5. to get gradients of $\mathbb E_{q_\theta}[f(z)]$ with respect to $\theta$, can you just differentiate inside the expectation?

## prerequisites check

- [ ] you know KL divergence, its asymmetry, and Gaussian entropy (→ ch 05 · topic 05)
- [ ] you can use `mlfs.optim.Adam` with `Param` objects (→ ch 06 · topic 04)
- [ ] you know Monte Carlo estimates and the Gaussian reparameterization $z = \mu + \sigma\varepsilon$ (→ ch 04)
- [ ] you know the EM algorithm in the form of Baum–Welch (→ topic 04) and Gaussian conjugacy (→ topic 01)

## core ideas

### idea 1: the ELBO

for a model $p(x, z)$ with data $x$ and latent variables or parameters $z$, write the log evidence as an expectation under any distribution $q(z)$:

$$\log p(x) = \underbrace{\mathbb E_q[\log p(x, z) - \log q(z)]}_{\mathcal L(q),\ \text{the ELBO}} + \underbrace{\mathrm{KL}\big(q(z)\,\|\,p(z \mid x)\big)}_{\ge 0}$$

(on paper 1). the left side doesn't depend on $q$, so **maximizing the ELBO is the same as minimizing the KL** to the true posterior, and the ELBO is always a lower bound on $\log p(x)$ (warm-up 1), touching it exactly when $q$ is the posterior. crucially, the ELBO only needs the *unnormalized* joint $p(x, z)$, the same thing MCMC needs. written as $\mathcal L = \mathbb E_q[\log p(x, z)] + H(q)$, it's a tug of war: the first term pulls $q$ towards high-density regions, the entropy $H(q)$ keeps it from collapsing onto the mode.

for a Gaussian $q = \mathcal N(\mu, \operatorname{diag}(\sigma^2))$ the entropy is known exactly, $H(q) = \sum_i\log\sigma_i + \frac d2(1 + \log 2\pi)$, and the first term can be estimated by Monte Carlo: draw $\varepsilon_s \sim \mathcal N(0, I)$, set $z_s = \mu + \sigma \odot \varepsilon_s$, and average $\log p(x, z_s)$. parameterize by $\log\sigma$ so it can't go negative.

> **spec.** `elbo_diag(log_prob, mean, log_std, eps)` takes the unnormalized log density `log_prob` (a function of one vector `(d,)`), the variational parameters (both `(d,)`) and standard normal draws `eps` of shape `(S, d)`, and returns the Monte Carlo ELBO estimate $\frac1S\sum_s\log p(z_s) + H(q)$ as a float.

> **▶ build: stage 1.** implement `elbo_diag`.

### idea 2: mean field and coordinate ascent

the classic family is **mean field**: $q(z) = \prod_jq_j(z_j)$, independent factors. there's a beautiful general result for it (on paper 2): holding all factors but one fixed, the best $q_j$ is

$$\log q_j^*(z_j) = \mathbb E_{q_{-j}}[\log p(x, z)] + \text{const},$$

the expected log joint, averaging over everything else. in conjugate models this has a standard form, so you can cycle through the factors, setting each to its optimum: **coordinate ascent variational inference** (CAVI). each update maximizes the ELBO over one factor exactly, so the ELBO never decreases (warm-up 4), and you stop when it stops changing. but it's a non-convex problem, so the fixed point depends on where you start.

the running example (Blei, Kucukelbir & McAuliffe, 2017): a 1-D Bayesian Gaussian mixture with $K$ unit-variance components and equal weights,

$$\mu_k \sim \mathcal N(0, \sigma^2), \qquad c_i \sim \text{Uniform}\{1..K\}, \qquad x_i \mid c_i, \mu \sim \mathcal N(\mu_{c_i}, 1),$$

with the mean-field family $q(\mu_k) = \mathcal N(m_k, s_k^2)$ and $q(c_i) = \text{Categorical}(\varphi_i)$. the general result gives two updates (on paper 3):

$$\varphi_{ik} \propto \exp\Big(m_kx_i - \tfrac12(s_k^2 + m_k^2)\Big), \qquad s_k^2 = \frac{1}{1/\sigma^2 + \sum_i\varphi_{ik}}, \qquad m_k = s_k^2\sum_i\varphi_{ik}x_i.$$

read them: a point is assigned softly to the components whose means are near it (with $\mathbb E[\mu_k^2] = m_k^2 + s_k^2$, so an uncertain mean pulls less); each component's mean is a shrunk, weighted average of its points, with a posterior variance that falls like one over its soft count. compare EM for a Gaussian mixture: the same assignment step, but point estimates for the means. CAVI keeps a distribution over them. with $K = 1$ there's nothing to approximate (every $\varphi_{i1} = 1$), $q(\mu)$ is the exact conjugate posterior, and the ELBO equals $\log p(x)$ exactly, a great check.

> **spec.** `cavi_gmm(x, n_components, *, prior_var=10.0, max_iters=100, tol=1e-8, rng=None)`: initialize $m$ with `n_components` distinct data points chosen by `as_rng(rng).choice(n, K, replace=False)`, and $s^2 = 1$. each iteration: update $\varphi$ (normalize with `logsumexp`), then $s^2$ and $m$, then compute the ELBO with the provided `gmm_elbo(x, m, s2, phi, prior_var)` and append it; stop after `max_iters` iterations or when the ELBO changed by less than `tol`. return `dict(m=(K,), s2=(K,), phi=(n, K), elbo=list)`.

> **🔮 predict:** with $K = 1$, $x = (1, 2, 3)$ and $\sigma^2 = 1$, what are $m$ and $s^2$ after one iteration, and what's the ELBO?

> **▶ build: stage 2.** implement `cavi_gmm`.

### idea 3: the reparameterization gradient

CAVI needs conjugacy, and someone has to derive the updates. for a generic model, maximize the ELBO by gradient ascent on the variational parameters $\theta = (\mu, \log\sigma)$. the obstacle (warm-up 5): the expectation in $\mathcal L(\theta) = \mathbb E_{q_\theta}[\log p(z)] + H(q_\theta)$ is over a distribution that depends on $\theta$, so you can't just move the gradient inside.

the **reparameterization trick** removes the dependence: write $z = \mu + \sigma \odot \varepsilon$ with $\varepsilon \sim \mathcal N(0, I)$, a distribution that doesn't depend on $\theta$. now the expectation is over $\varepsilon$, and the gradient goes inside:

$$\nabla_\mu\mathcal L = \mathbb E_\varepsilon[\nabla\log p(z)], \qquad \nabla_{\log\sigma}\mathcal L = \mathbb E_\varepsilon[\nabla\log p(z) \odot \varepsilon \odot \sigma] + 1,$$

where the $+1$ is the derivative of the entropy $\sum_i\log\sigma_i$, and the chain rule gives $\partial z_i/\partial\log\sigma_i = \varepsilon_i\sigma_i$. estimate both expectations with the same $S$ draws used for the ELBO. this estimator has much lower variance than the general-purpose "score function" (REINFORCE) estimator $\mathbb E[\log p(z)\,\nabla_\theta\log q_\theta(z)]$, because it uses the gradient of the model, not just its values. it's the same trick that trains variational autoencoders (ch 21).

> **spec.** `elbo_grad_diag(grad_log_prob, mean, log_std, eps)` returns `(g_mean, g_log_std)`, each `(d,)`: the exact gradient of `elbo_diag` (for these draws `eps`) with respect to `mean` and `log_std`. the checks compare it with finite differences.

> **▶ build: stage 3.** implement `elbo_grad_diag`.

### idea 4: stochastic optimization of the ELBO

put it together: at every iteration draw fresh $\varepsilon$, compute the gradient estimate, and take an optimizer step. the gradient is noisy (it's a Monte Carlo estimate), so this is stochastic optimization, and the adaptive step sizes of Adam (chapter 06) handle the noise well. with a dataset of $N$ points you can also subsample a minibatch to estimate $\log p(x, z)$, scaling the likelihood by $N/B$: that's **stochastic variational inference**, which is how VI scales to millions of data points.

two practical consequences of the noise. the ELBO trace is jagged (average it over iterations to see the trend), and with a constant learning rate the parameters never settle exactly: they jitter around the optimum with an amplitude that scales with the learning rate. smaller learning rates (or a decaying schedule) give more precise answers in more iterations.

> **spec.** `reparam_vi(log_prob, dim, *, grad_log_prob=None, family='diagonal', n_samples=16, lr=0.01, n_iters=2000, rng=None)`: start from mean 0 and $\log\sigma = 0$, as `mlfs.optim.Param` objects driven by `Adam(params, lr=lr)`. each iteration: `eps = rng.standard_normal((n_samples, dim))`, append the ELBO estimate for these draws to the history, set each `param.grad` to the *negative* ELBO gradient (Adam minimizes), `step()`. `grad_log_prob=None` means `mlfs.autodiff.grad(log_prob, z)` per draw. return `(MultivariateNormal(mean, cov), history)` with the final parameters, where for `'diagonal'` `cov = diag(exp(2 log_std))`. an unknown `family` raises `ValueError`.

> **▶ build: stage 4.** implement `reparam_vi` (diagonal family).

### idea 5: full covariance, and what the KL direction does

mean field can't represent correlations. a **full-covariance** Gaussian can: $q = \mathcal N(\mu, LL^\top)$ with $L$ lower triangular, sampled as $z = \mu + L\varepsilon$. its entropy is $\sum_i\log L_{ii} + \frac d2(1 + \log 2\pi)$, and the reparameterization gradient is

$$\nabla_\mu\mathcal L = \mathbb E[\nabla\log p(z)], \qquad \nabla_L\mathcal L = \operatorname{tril}\big(\mathbb E[\nabla\log p(z)\,\varepsilon^\top]\big) + \operatorname{diag}(1/L_{ii}).$$

keep the diagonal positive by optimizing $\log L_{ii}$ (the chain rule multiplies those gradient entries by $L_{ii}$). the cost is $d^2$ parameters instead of $2d$.

now the bias. the KL you minimize, $\mathrm{KL}(q\|p) = \mathbb E_q[\log q - \log p]$, is an expectation under $q$: it's huge wherever $q$ puts mass and $p$ doesn't, and it doesn't care where $p$ has mass that $q$ ignores. so $q$ is **mode-seeking**: it squeezes inside the posterior. for a correlated Gaussian target and a mean-field $q$, the optimum (on paper 4) has the right mean and variances $1/\Lambda_{ii}$, the inverse diagonal of the *precision*, which are the conditional variances, smaller than the marginal ones: at correlation 0.9, $\sqrt{1 - 0.81} = 0.44$ instead of 1 (warm-up 2). for a posterior with well-separated modes, the best $q$ sits on one of them (warm-up 3); when the modes are close enough that the valley between them is shallow, a broad $q$ covering both can still win, and the lab maps out where the switch happens. the other direction, $\mathrm{KL}(p\|q)$, is mass-covering (it's what maximum likelihood minimizes, and what expectation propagation targets), but it needs expectations under $p$, which is the thing you can't compute.

> **spec.** `elbo_grad_full(grad_log_prob, mean, L, eps)` returns `(g_mean (d,), g_L (d, d))` for $z_s = \mu + L\varepsilon_s$: the gradient above, with zeros in the upper triangle. in `reparam_vi`, `family='full'` keeps an unconstrained `(d, d)` `Param` $R$ with $L = \operatorname{tril}(R, -1) + \operatorname{diag}(e^{\operatorname{diag}R})$, starting from $R = 0$ ($L = I$), and returns `cov = L L^T`.

> **🔮 predict:** a posterior with two well-separated modes of weights 0.4 and 0.6. a single Gaussian fitted by maximizing the ELBO: where does it go, and what is its $-\mathrm{KL}(q\|p)$ roughly?

> **▶ build: stage 5.** implement `elbo_grad_full` and the `'full'` branch.

## worked example

**the ELBO of $q$ against itself.** if the target is $q$ itself (normalized), the KL is 0 and so is the ELBO's expectation. for a finite sample, $\log q(z_s) = -\frac12\|\varepsilon_s\|^2 - \sum_i\log\sigma_i - \frac d2\log 2\pi$, so the estimate is $\frac1S\sum_s\log q(z_s) + H(q) = \frac12\big(d - \frac1S\sum_s\|\varepsilon_s\|^2\big)$: exactly zero on average, and for $d = 3$ with draws whose mean squared norm is 3.2, $-0.1$.

**CAVI with one component.** $x = (1, 2, 3)$, $\sigma^2 = 1$. every $\varphi_{i1} = 1$, so $s^2 = 1/(1 + 3) = 0.25$ and $m = 0.25\cdot6 = 1.5$: the exact posterior of a Gaussian mean. the ELBO is then $\log p(x)$, with $x \sim \mathcal N(0, I + \mathbf 1\mathbf 1^\top)$: the covariance has determinant 4 and inverse $I - \frac14\mathbf 1\mathbf 1^\top$, so the quadratic form is $14 - 36/4 = 5$ and $\log p(x) = -\frac52 - \frac12\log 4 - \frac32\log 2\pi = -5.9499$.

**mean field on a correlated Gaussian.** target $\mathcal N(0, \Sigma)$ with $\Sigma = \begin{pmatrix}1 & 0.8\\0.8 & 1\end{pmatrix}$. its precision is $\Lambda = \frac{1}{0.36}\begin{pmatrix}1 & -0.8\\-0.8 & 1\end{pmatrix}$, so the mean-field optimum has standard deviations $1/\sqrt{\Lambda_{ii}} = 0.6$ each, against the true marginal 1. the gap is $\mathrm{KL}(q^*\|p) = \frac12\log(\Lambda_{11}\Lambda_{22}|\Sigma|) = \frac12\log(1/0.36) = 0.51$ nats. your `cavi_gmm` and `reparam_vi` must reproduce these.

## on paper

1. **the ELBO decomposition.** show $\log p(x) = \mathcal L(q) + \mathrm{KL}(q(z)\|p(z \mid x))$.

<details><summary>solution</summary>

$\mathrm{KL}(q\|p(\cdot \mid x)) = \mathbb E_q[\log q(z) - \log p(z \mid x)] = \mathbb E_q[\log q(z) - \log p(x, z)] + \log p(x)$, using $p(z \mid x) = p(x, z)/p(x)$. rearrange: $\log p(x) = \mathbb E_q[\log p(x, z) - \log q(z)] + \mathrm{KL} = \mathcal L(q) + \mathrm{KL}$.

</details>

2. **the optimal mean-field factor.** holding $q_{-j}$ fixed, show that the $q_j$ maximizing $\mathcal L$ satisfies $\log q_j^* = \mathbb E_{-j}[\log p(x, z)] + \text{const}$.

<details><summary>solution</summary>

as a function of $q_j$: $\mathcal L = \mathbb E_{q_j}\big[\mathbb E_{-j}[\log p(x, z)]\big] - \mathbb E_{q_j}[\log q_j] + \text{const}$. define $\tilde p(z_j) \propto \exp\mathbb E_{-j}[\log p(x, z)]$. then $\mathcal L = -\mathrm{KL}(q_j\|\tilde p) + \text{const}$, maximized by $q_j = \tilde p$.

</details>

3. **the CAVI updates for the mixture.** derive the $\varphi$ and $q(\mu_k)$ updates from on paper 2.

<details><summary>solution</summary>

$\log p(x, c, \mu) = \sum_k\log\mathcal N(\mu_k; 0, \sigma^2) + \sum_i\big[-\log K + \sum_k[c_i = k]\log\mathcal N(x_i; \mu_k, 1)\big]$. for $c_i$: $\mathbb E[\log\mathcal N(x_i; \mu_k, 1)] = -\frac12(x_i^2 - 2x_im_k + m_k^2 + s_k^2) + \text{const}$; dropping the $k$-independent terms gives $\varphi_{ik} \propto \exp(m_kx_i - \frac12(m_k^2 + s_k^2))$. for $\mu_k$: $\log q(\mu_k) = -\frac{\mu_k^2}{2\sigma^2} + \sum_i\varphi_{ik}(x_i\mu_k - \frac12\mu_k^2) + \text{const}$, a Gaussian with precision $1/\sigma^2 + \sum_i\varphi_{ik}$ and mean $\sum_i\varphi_{ik}x_i$ divided by that precision.

</details>

4. **mean field on a Gaussian.** the target is $\mathcal N(\mu, \Lambda^{-1})$; $q = \prod_i\mathcal N(m_i, s_i^2)$. use on paper 2 to find the optimum.

<details><summary>solution</summary>

$\log p = -\frac12(z - \mu)^\top\Lambda(z - \mu)$. as a function of $z_i$, with the others averaged: $-\frac12\Lambda_{ii}z_i^2 + z_i\big(\Lambda_{ii}\mu_i - \sum_{j \ne i}\Lambda_{ij}(m_j - \mu_j)\big) + \text{const}$. so $q_i$ is Gaussian with variance $1/\Lambda_{ii}$, and the means satisfy a linear system whose solution is $m = \mu$. the variance is the *conditional* variance of $z_i$ given the others, which is never larger than the marginal $(\Lambda^{-1})_{ii}$.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how the standard deviations reported by mean-field VI compare with the truth on a correlated Gaussian, as the correlation grows.
2. where a single Gaussian fitted by VI (reverse KL) lands on a bimodal target, compared with moment matching (forward KL).
3. how often CAVI for a Gaussian mixture finds the right clusters from random starts.

## check your understanding

1. the ELBO went up by 3 nats between two models. does that mean the evidence went up by 3 nats?

<details><summary>answer</summary>

no: $\log p(x) = \mathcal L + \mathrm{KL}$, and the KL can differ between the models. comparing ELBOs compares lower bounds; it's a reasonable heuristic for model selection but not the evidence itself.

</details>

2. why does CAVI need conjugacy, and what replaces it in black-box VI?

<details><summary>answer</summary>

the optimal factor $\exp\mathbb E_{-j}[\log p]$ must be a distribution you can recognize and compute expectations under; conjugate-exponential models guarantee that. black-box VI fixes the family (say, Gaussians) and follows Monte Carlo gradients of the ELBO, which only need $\log p$ and its gradient.

</details>

3. why parameterize the standard deviation by its log?

<details><summary>answer</summary>

it must stay positive, and gradient steps on an unconstrained parameter can't cross zero; it also makes the entropy linear in the parameter (its gradient is exactly 1) and makes steps relative rather than absolute.

</details>

4. mean-field VI reports a posterior sd of 0.44 where MCMC reports 1. which would you trust, and how could you tell without MCMC?

<details><summary>answer</summary>

MCMC (if it has converged). signs of trouble: strong posterior correlations (check a full-covariance fit, or the Hessian at the mode), or calibration checks on simulated data. a full-covariance $q$ fixes this particular case; multimodality needs other remedies.

</details>

5. what happens to the variance of the reparameterization gradient if you use $S = 1$ sample per iteration?

<details><summary>answer</summary>

it's still unbiased, just noisier (variance $\propto 1/S$). in practice $S = 1$ often works with a small learning rate, especially with minibatches, since the minibatch noise dominates anyway.

</details>

6. CAVI is "EM with distributions". what exactly is the E-step and what's the M-step in that analogy, and what does VI add?

<details><summary>answer</summary>

the $\varphi$ update is the E-step (soft assignments given the current means). EM's M-step sets the means to point estimates (weighted averages); CAVI instead sets $q(\mu_k)$ to a Gaussian with a mean (a shrunk weighted average) *and* a variance, and the assignment step then accounts for that uncertainty through $\mathbb E[\mu_k^2] = m_k^2 + s_k^2$. EM is the limit of CAVI where the $q(\mu)$ factors are point masses.

</details>

## common bugs

- **forgetting the entropy.** symptom: $q$ collapses to a point at the mode ($\sigma \to 0$). the ELBO is $\mathbb E[\log p] + H(q)$, and the $+1$ in the $\log\sigma$ gradient comes from $H$.
- **ascending with a minimizer.** symptom: the ELBO plummets. Adam minimizes; give it the negative gradient.
- **the gradient with respect to $\sigma$ instead of $\log\sigma$.** symptom: finite differences disagree by a factor of $\sigma$. $\partial z/\partial\log\sigma = \varepsilon\sigma$.
- **dropping $s_k^2$ in the assignment update.** symptom: CAVI converges, but to the wrong fixed point, overconfident for small clusters. $\mathbb E[\mu_k^2] = m_k^2 + s_k^2$.
- **unnormalized responsibilities.** symptom: rows of $\varphi$ that don't sum to 1, or overflow for far-away points. subtract the row's `logsumexp`.
- **judging convergence from one noisy ELBO value.** symptom: stopping at a random point. average the trace over a window, and lower the learning rate for precise answers.

## going deeper

- Blei, Kucukelbir & McAuliffe (2017), "Variational inference: a review for statisticians", *Journal of the American Statistical Association* 112(518). CAVI and the Gaussian mixture example. [arXiv:1601.00670](https://arxiv.org/abs/1601.00670)
- Jordan, Ghahramani, Jaakkola & Saul (1999), "An introduction to variational methods for graphical models", *Machine Learning* 37. [doi:10.1023/A:1007665907178](https://doi.org/10.1023/A:1007665907178)
- Kingma & Welling (2014), "Auto-encoding variational Bayes", *ICLR*. the reparameterization trick. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Titsias & Lázaro-Gredilla (2014), "Doubly stochastic variational Bayes for non-conjugate inference", *ICML*. reparameterized VI with full-covariance Gaussians.
- Kucukelbir, Tran, Ranganath, Gelman & Blei (2017), "Automatic differentiation variational inference", *JMLR* 18. [arXiv:1603.00788](https://arxiv.org/abs/1603.00788)
- Hoffman, Blei, Wang & Paisley (2013), "Stochastic variational inference", *JMLR* 14. [arXiv:1206.7051](https://arxiv.org/abs/1206.7051)
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, ch. 10 (and §10.1.2 for the shrinking variances).

## connections

- ← builds on: KL divergence and entropy (ch 05 · topic 05), Adam and `Param` (ch 06 · topic 04), forward-mode autodiff (ch 03), the multivariate Gaussian and its Cholesky factor (ch 02, ch 04), EM in the form of Baum–Welch (topic 04), conjugate Gaussian posteriors (topic 01).
- → used in: the capstone's VI backend, variational autoencoders (ch 21), diffusion models' ELBO (ch 22), variational dropout and Bayesian deep learning, and the evidence bounds behind many latent-variable models.
