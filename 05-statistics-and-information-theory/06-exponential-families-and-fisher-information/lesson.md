# exponential families & fisher information

## why this matters

in topic 02 you derived maximum-likelihood estimates one distribution at a time: set the derivative to zero, solve, repeat for the next family. it worked, but it hid a pattern. the Bernoulli, the Poisson, the exponential, the Gaussian, the gamma, the beta, the categorical, the Dirichlet: all of them can be written in one common form, the **exponential family**, and once you write them that way, a single piece of code fits all of them, computes their means and variances, and tells you how precisely their parameters can possibly be estimated.

this matters well beyond tidiness. generalized linear models (logistic regression, Poisson regression; ch 08) are exponential families with a linear model for the natural parameter, which is why they all share one fitting algorithm. the softmax output layer of every classifier is a categorical exponential family. variational inference (ch 12) has closed-form updates exactly when the model is built from exponential families. and the **Fisher information**, which falls straight out of this form, measures how much a single observation tells you about a parameter. it gives the **Cramér–Rao bound**, a hard floor on the variance of any unbiased estimator, and it reappears as the geometry behind natural gradients and the curvature of the KL divergence.

you'll build one class that represents any exponential family from three functions, get its moments and Fisher information by differentiating with *your* forward-mode autodiff from ch 03, write a generic maximum-likelihood fitter, and then, in the lab, catch estimators sitting exactly on the Cramér–Rao bound, and others sneaking below it.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. what do the Bernoulli, Poisson and Gaussian distributions have in common, mathematically?
2. to fit a Poisson by maximum likelihood, you only need the sum of the counts, not the counts themselves. is there a similar summary for a Gaussian? for a Cauchy?
3. if you know the normalizing constant of a distribution as a function of its parameters, can you get its mean without integrating?
4. is there a limit to how precisely you can estimate a parameter from $n$ observations, whatever estimator you use?
5. topic 01 found an estimator for the uniform's end point with MSE of order $1/n^2$. does that contradict question 4's limit, if there is one?

## prerequisites check

- [ ] you know maximum likelihood and the MLEs of the Bernoulli, Poisson, exponential and Gaussian (→ topic 02)
- [ ] you know bias, variance and the bias–variance decomposition (→ topic 01)
- [ ] you can differentiate with `mlfs.autodiff.grad` and `numerical_jacobian` (→ ch 03 · topics 02 and 04)
- [ ] you know Newton's method and Hessians (→ ch 03 · topic 03)
- [ ] you know covariance matrices and how a linear map transforms them (→ ch 04 · topic 03)

## core ideas

### idea 1: one form for many distributions

a family of distributions is an **exponential family** if its densities (or pmfs) can be written

$$p(x \mid \eta) = h(x)\exp\big(\eta^\top T(x) - A(\eta)\big).$$

four ingredients:

- $T(x) \in \mathbb{R}^k$, the **sufficient statistics**: the only features of $x$ the parameters interact with;
- $\eta \in \mathbb{R}^k$, the **natural parameters**;
- $h(x) \ge 0$, the **base measure**, which doesn't involve $\eta$;
- $A(\eta)$, the **log-partition function**, which makes everything normalize: $A(\eta) = \log\int h(x)e^{\eta^\top T(x)}\,dx$.

that's warm-up 1: the log-density is *linear in the parameters* $\eta$, apart from the normalizer $A(\eta)$. the parameters you're used to (a probability $p$, a mean $\mu$, a variance $\sigma^2$) are usually not the natural ones; they're related to $\eta$ by a fixed transformation.

the form explains warm-up 2. for $n$ i.i.d. observations the log-likelihood is

$$\ell(\eta) = \sum_i\log h(x_i) + \eta^\top\sum_i T(x_i) - nA(\eta),$$

so the data enter only through $\sum_i T(x_i)$. two datasets with the same total sufficient statistic have log-likelihoods that differ by a constant, and so give the same MLE and the same posterior. compressing your data to $\sum_i T(x_i)$ loses nothing about $\eta$. for the Gaussian, $T(x) = (x, x^2)$, so $(\sum x_i, \sum x_i^2)$ is all you need. the Cauchy isn't an exponential family, and no fixed-size summary is sufficient for it: you have to keep all $n$ points. (the Fisher–Neyman factorization theorem makes "sufficient" precise in general.)

> **▶ build: stage 1.** implement `ExponentialFamily.__init__` (store `T`, `A`, `log_h`, `eta0`, `dim`) and `log_prob(x, eta)` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: four families in canonical form

rewriting a familiar distribution in canonical form is a matter of taking logs and collecting the terms that multiply both $x$ and a parameter.

**Bernoulli.** $p^x(1 - p)^{1-x} = \exp\big(x\log\frac{p}{1-p} + \log(1 - p)\big)$. so $T(x) = x$, the natural parameter is the **log-odds** $\eta = \log\frac{p}{1-p}$, and $A(\eta) = -\log(1 - p) = \log(1 + e^\eta)$, the softplus. inverting, $p = \sigma(\eta)$: the logistic sigmoid appears on its own.

**Poisson.** $\frac{\lambda^x e^{-\lambda}}{x!} = \frac1{x!}\exp(x\log\lambda - \lambda)$: $T(x) = x$, $\eta = \log\lambda$, $A(\eta) = e^\eta$, $h(x) = 1/x!$.

**exponential.** $re^{-rx} = \exp(-rx + \log r)$ for $x \ge 0$: $T(x) = x$, $\eta = -r < 0$, $A(\eta) = -\log(-\eta)$.

**Gaussian.** $\frac{1}{\sqrt{2\pi\sigma^2}}\exp\big(-\frac{(x - \mu)^2}{2\sigma^2}\big) = \frac{1}{\sqrt{2\pi}}\exp\Big(\frac{\mu}{\sigma^2}x - \frac{1}{2\sigma^2}x^2 - \frac{\mu^2}{2\sigma^2} - \frac12\log\sigma^2\Big)$. so $T(x) = (x, x^2)$, $\eta = \big(\frac{\mu}{\sigma^2}, -\frac{1}{2\sigma^2}\big)$, and in terms of $\eta$ (on paper 1), $A(\eta) = -\frac{\eta_1^2}{4\eta_2} - \frac12\log(-2\eta_2)$, with $\eta_2 < 0$.

the set of $\eta$ where $A$ is finite is the **natural parameter space**. for the exponential it's $\eta < 0$; for the Gaussian, $\eta_2 < 0$. step outside and the "density" doesn't normalize; your fitter will have to stay inside.

> **▶ build: stage 2.** implement `bernoulli_family()`, `poisson_family()`, `exponential_family()` and `gaussian_family()`, each returning an `ExponentialFamily`. write each $A$ with operations your `Dual` numbers support, so stage 3 can differentiate it.

### idea 3: everything comes from $A$

here's warm-up 3, the remarkable property of the log-partition function. differentiate $A(\eta) = \log\int h(x)e^{\eta^\top T(x)}\,dx$:

$$\nabla A(\eta) = \frac{\int T(x)\,h(x)e^{\eta^\top T(x)}\,dx}{\int h(x)e^{\eta^\top T(x)}\,dx} = \mathbb{E}_\eta[T(X)].$$

the gradient of $A$ is the mean of the sufficient statistics, called the **mean parameters** $\mu(\eta)$. differentiate again (on paper 2) and you get the covariance:

$$\nabla^2 A(\eta) = \operatorname{Cov}_\eta[T(X)].$$

a covariance matrix is positive semidefinite, so $A$ is **convex**, and the log-likelihood $\eta^\top\sum T(x_i) - nA(\eta)$ is concave: one maximum, no local traps. check the Bernoulli: $A' = \sigma(\eta) = p$, the mean, and $A'' = \sigma(\eta)(1 - \sigma(\eta)) = p(1 - p)$, the variance.

now the **Fisher information**. the **score** is the gradient of the log-likelihood of one observation, $s(x; \eta) = \nabla_\eta\log p(x \mid \eta)$. it has mean zero at the true parameter, and its covariance is the Fisher information:

$$I(\eta) = \operatorname{Cov}_\eta[s(X; \eta)] = -\mathbb{E}_\eta\big[\nabla^2_\eta\log p(X \mid \eta)\big].$$

intuitively, it's the expected curvature of the log-likelihood: a sharply peaked log-likelihood pins the parameter down, a flat one doesn't. for an exponential family, $s(x; \eta) = T(x) - \nabla A(\eta)$ and $\nabla^2_\eta\log p = -\nabla^2 A(\eta)$, so

$$I(\eta) = \nabla^2 A(\eta) = \operatorname{Cov}_\eta[T(X)].$$

one more connection, to topic 05: for nearby parameters, $\mathrm{KL}(p_\eta \,\|\, p_{\eta + \delta}) \approx \frac12\delta^\top I(\eta)\delta$. the Fisher information is the curvature of the KL divergence, a local measure of how distinguishable nearby distributions are. it's the metric behind natural-gradient methods.

so: one gradient of $A$ gives the means, one Hessian gives the Fisher information. you'll compute the gradient exactly with your forward-mode `grad`, and the Hessian by differentiating that gradient numerically.

> **▶ build: stage 3.** implement `mean_params(eta)` with `mlfs.autodiff.grad` and `fisher_information(eta)` with `mlfs.autodiff.numerical_jacobian` (then symmetrize).

### idea 4: maximum likelihood is moment matching

set the gradient of the average log-likelihood $\frac1n\ell(\eta) = \eta^\top\bar T - A(\eta) + \text{const}$ to zero, where $\bar T = \frac1n\sum_i T(x_i)$:

$$\nabla A(\hat\eta) = \bar T \quad\Longleftrightarrow\quad \mathbb{E}_{\hat\eta}[T(X)] = \frac1n\sum_i T(x_i).$$

the MLE is the parameter whose expected sufficient statistics equal the observed ones: **moment matching**. for the Gaussian it says the model's $\mathbb{E}X$ and $\mathbb{E}X^2$ must equal the sample's, which gives $\hat\mu = \bar x$ and $\hat\sigma^2 = \frac1n\sum(x_i - \bar x)^2$, as in topic 02. in mean parameters the MLE is simply $\hat\mu = \bar T$.

to get $\hat\eta$ you must invert $\nabla A$, which rarely has a closed form. **Newton's method** does it. the gradient is $g = \bar T - \nabla A(\eta)$ and the Hessian is $-\nabla^2 A(\eta) = -I(\eta)$, so the Newton step is

$$\eta \leftarrow \eta + I(\eta)^{-1}\big(\bar T - \nabla A(\eta)\big).$$

for exponential families, Newton's method and **Fisher scoring** (replace the observed Hessian with minus the Fisher information) are the same algorithm. it's the algorithm behind fitting logistic regression ("iteratively reweighted least squares", ch 08).

two practical details. **damping**: a full Newton step can jump out of the natural parameter space (a Gaussian start at $\mathcal{N}(0, 1)$ with data near 30 asks for a step that makes $\eta_2$ positive). so halve the step until the log-likelihood is finite and no smaller. and **existence**: the MLE may not exist. for Bernoulli data that are all ones, the log-likelihood keeps increasing as $\eta \to \infty$ ($\hat p = 1$ is at $\eta = +\infty$), so the iteration must be capped. that's topic 02's overconfident MLE again, seen from the natural parameter.

> **🔮 predict:** you fit a Gaussian with your Newton fitter to 600 datasets of $n = 10$ points. the MLE of the *mean* parameters is unbiased. is the MLE of the natural parameter $\eta_2 = -1/(2\sigma^2)$ unbiased too?

> **▶ build: stage 4.** implement `fit(x, *, eta0=None, max_iters=100, tol=1e-10)`, damped Newton, stopping when $\lVert g\rVert \le$ `tol` or after `max_iters` iterations.

### idea 5: the Cramér–Rao bound

warm-up 4: yes, there's a limit. for a scalar parameter $\theta$ and $n$ i.i.d. observations, any **unbiased** estimator $\hat\theta$ satisfies

$$\operatorname{Var}(\hat\theta) \ge \frac{1}{n\,I(\theta)}.$$

the proof (on paper 3) is one application of Cauchy–Schwarz: an unbiased estimator must be correlated with the score, and the score's variance is $nI(\theta)$. for $k$ parameters the bound is a matrix inequality, $\operatorname{Cov}(\hat\theta) \succeq \frac1n I(\theta)^{-1}$: the inverse matrix, not the inverse of each diagonal entry. an unbiased estimator that attains the bound is **efficient**.

the bound is about a particular parameterization. for a function $g(\theta)$, the bound becomes $\frac{\nabla g^\top I(\theta)^{-1}\nabla g}{n}$ (the delta method). in an exponential family, the mean parameters have an efficient estimator, $\bar T$, at every $n$: its covariance is $\operatorname{Cov}(T)/n = I(\eta)/n$, and the bound for $\mu = \nabla A(\eta)$ is $\nabla^2 A\, I^{-1}\nabla^2 A/n = I/n$. exactly equal. for any other parameterization, efficiency only holds asymptotically. the MLE is **asymptotically efficient**: under regularity conditions, $\sqrt n(\hat\theta_{\text{MLE}} - \theta) \to \mathcal{N}(0, I(\theta)^{-1})$, so for large $n$ no unbiased estimator does better.

read the fine print, because the lab will. the bound is for **unbiased** estimators. biased ones can go below it, as topic 01's shrinkage estimators do. and it needs **regularity**: the support of $p(x \mid \theta)$ must not depend on $\theta$, so that you can differentiate under the integral. the uniform on $[0, \theta]$ violates that, which is how topic 01's max-based estimator reached an MSE of order $1/n^2$ (warm-up 5): no contradiction, the theorem doesn't apply.

> **🔮 predict:** the Gaussian variance MLE $\hat\sigma^2$ divides by $n$, and the bound for $\sigma^2$ is $2\sigma^4/n$. at $n = 10$, is the variance of $\hat\sigma^2$ above the bound, at it, or below it?

> **▶ build: stage 5.** implement `score(x, eta)`, returning $T(x_i) - \nabla A(\eta)$ for each observation, and `cramer_rao_bound(eta, n)`, the matrix $I(\eta)^{-1}/n$.

## worked example

**the Bernoulli, end to end.** take $p = 0.2$. the natural parameter is $\eta = \log(0.2/0.8) = -1.386$, and $A(\eta) = \log(1 + e^{-1.386}) = \log 1.25 = 0.223$. so $\log p(1 \mid \eta) = -1.386 - 0.223 = -1.609 = \log 0.2$ and $\log p(0 \mid \eta) = -0.223 = \log 0.8$. the mean parameter is $A'(\eta) = \sigma(-1.386) = 0.2$, and the Fisher information is $A''(\eta) = 0.2 \cdot 0.8 = 0.16$. with $n = 50$ observations, no unbiased estimator of the log-odds can have variance below $1/(50 \cdot 0.16) = 0.125$.

**a Gaussian in natural parameters.** $\mathcal{N}(1.5, 2)$ has $\eta = (0.75, -0.25)$ and mean parameters $\mathbb{E}[(X, X^2)] = (1.5, 1.5^2 + 2) = (1.5, 4.25)$. $\mathcal{N}(1, 2)$ has Fisher information $\operatorname{Cov}(X, X^2) = \begin{pmatrix}\sigma^2 & 2\mu\sigma^2\\ 2\mu\sigma^2 & 4\mu^2\sigma^2 + 2\sigma^4\end{pmatrix} = \begin{pmatrix}2 & 4\\ 4 & 16\end{pmatrix}$.

**a Poisson score.** at $\lambda = 2$ ($\eta = \log 2$), the score of an observation $x$ is $x - 2$: for $x = (0, 2, 5)$ it's $(-2, 0, 3)$. it averages to zero over the Poisson(2) distribution, and its variance is $\lambda = 2 = I(\eta)$.

the tests check these, with the Gaussian Fisher information at $\mathcal{N}(1, 2)$ and $\mathcal{N}(0.5, 2)$.

## on paper

1. **the Gaussian log-partition function.** starting from $A = \frac{\mu^2}{2\sigma^2} + \frac12\log\sigma^2$ (up to the constant absorbed into $h$), express $A$ in terms of $\eta_1 = \mu/\sigma^2$ and $\eta_2 = -1/(2\sigma^2)$.

<details><summary>solution</summary>

$\sigma^2 = -\frac{1}{2\eta_2}$ and $\mu = \eta_1\sigma^2 = -\frac{\eta_1}{2\eta_2}$. then $\frac{\mu^2}{2\sigma^2} = \frac{\eta_1^2/(4\eta_2^2)}{-1/\eta_2} = -\frac{\eta_1^2}{4\eta_2}$ and $\frac12\log\sigma^2 = -\frac12\log(-2\eta_2)$. so $A(\eta) = -\frac{\eta_1^2}{4\eta_2} - \frac12\log(-2\eta_2)$. check: $\partial A/\partial\eta_1 = -\frac{\eta_1}{2\eta_2} = \mu$. ✓

</details>

2. **the second derivative of $A$.** show that $\nabla^2 A(\eta) = \operatorname{Cov}_\eta[T(X)]$ by differentiating $\nabla A(\eta) = \frac{\int T h e^{\eta^\top T}}{\int h e^{\eta^\top T}}$.

<details><summary>solution</summary>

write $Z(\eta) = \int h e^{\eta^\top T}$, so $\nabla A = \frac{\int T h e^{\eta^\top T}}{Z}$. by the quotient rule, $\nabla^2 A = \frac{\int TT^\top h e^{\eta^\top T}}{Z} - \frac{\int T h e^{\eta^\top T}}{Z}\frac{\int T^\top h e^{\eta^\top T}}{Z} = \mathbb{E}[TT^\top] - \mathbb{E}[T]\mathbb{E}[T]^\top = \operatorname{Cov}[T]$.

</details>

3. **Cramér–Rao.** for a scalar parameter, let $\hat\theta$ be unbiased and $S = \sum_i s(X_i; \theta)$ the total score. using $\mathbb{E}S = 0$, $\operatorname{Var}S = nI(\theta)$ and $\operatorname{Cov}(\hat\theta, S) = 1$, prove $\operatorname{Var}\hat\theta \ge 1/(nI)$.

<details><summary>solution</summary>

Cauchy–Schwarz for covariances: $\operatorname{Cov}(\hat\theta, S)^2 \le \operatorname{Var}(\hat\theta)\operatorname{Var}(S)$, so $1 \le \operatorname{Var}(\hat\theta)\,nI(\theta)$. the covariance is 1 because $\operatorname{Cov}(\hat\theta, S) = \mathbb{E}[\hat\theta S] = \int\hat\theta\,\frac{\partial_\theta p}{p}\,p = \partial_\theta\int\hat\theta p = \partial_\theta\mathbb{E}\hat\theta = \partial_\theta\theta = 1$. two steps need care: swapping $\partial_\theta$ and $\int$ (regularity: the support can't move with $\theta$), and $\mathbb{E}\hat\theta = \theta$ (unbiasedness). equality holds iff $\hat\theta$ is an affine function of $S$, which for exponential families happens exactly for the mean parameters.

</details>

4. **the CR bound for a Gaussian variance.** with $\mu$ unknown, the bound for $\sigma^2$ is $2\sigma^4/n$. compare with the variance of $s^2$, $2\sigma^4/(n - 1)$, and of the MLE $\hat\sigma^2 = \frac{n-1}{n}s^2$.

<details><summary>solution</summary>

$s^2$ is unbiased with variance $\frac{2\sigma^4}{n-1} > \frac{2\sigma^4}{n}$: not efficient, but its efficiency $\frac{n-1}{n} \to 1$. $\hat\sigma^2$ has variance $\big(\frac{n-1}{n}\big)^2\frac{2\sigma^4}{n-1} = \frac{2(n-1)\sigma^4}{n^2} < \frac{2\sigma^4}{n}$: *below* the bound, which is allowed because $\hat\sigma^2$ is biased. its MSE, $\frac{(2n - 1)\sigma^4}{n^2}$, is below the bound too.

</details>

## experiment

`lab.py` runs three experiments with your code:

1. four estimators measured against the Cramér–Rao bound, for growing $n$.
2. your Newton fitter on many small and large Gaussian datasets, against the bound's ellipse.
3. estimating $P(X = 0)$ for a Poisson: Rao–Blackwellization, and a bound no unbiased estimator reaches.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. write the categorical distribution over $K$ classes as an exponential family. where does the softmax come from?

<details><summary>answer</summary>

with $T(x)$ the one-hot vector of $x$ and $\eta_k = \log p_k$ (up to a shared constant), $p(x) = \exp(\eta^\top T(x) - A(\eta))$ with $A(\eta) = \log\sum_k e^{\eta_k}$, the logsumexp. then $\nabla A(\eta) = \operatorname{softmax}(\eta)$: the mean parameters (class probabilities) are the softmax of the natural parameters (the logits). a classifier's output layer produces natural parameters.

</details>

2. why is the log-likelihood of an exponential family concave in $\eta$, and why does that matter?

<details><summary>answer</summary>

it's linear in $\eta$ minus $nA(\eta)$, and $A$ is convex because its Hessian is a covariance matrix. so any stationary point is the global maximum and Newton's method can't get trapped. (in other parameterizations, like the Gaussian's $(\mu, \sigma^2)$, the log-likelihood need not be concave.)

</details>

3. the Fisher information of a Bernoulli in terms of $p$ is $\frac{1}{p(1-p)}$, but in terms of $\eta$ it's $p(1 - p)$. which is right?

<details><summary>answer</summary>

both. Fisher information depends on the parameterization: $I_\eta = I_p\,(dp/d\eta)^2 = \frac{1}{p(1-p)}\big(p(1-p)\big)^2 = p(1-p)$. near $p = 0$, a single observation tells you a lot about $p$ (on its own scale) but very little about the log-odds, which can be anywhere in $(-\infty, -5)$ without changing much.

</details>

4. your Newton fitter takes a full step and $A$ returns `nan`. what happened, and what's the fix?

<details><summary>answer</summary>

the step left the natural parameter space (for the Gaussian, $\eta_2 \ge 0$), where $A$ is undefined. halve the step until $A$ is finite and the log-likelihood doesn't decrease: a backtracking line search.

</details>

5. an estimator has variance below the Cramér–Rao bound. name two ways that can happen.

<details><summary>answer</summary>

it's biased (shrinkage estimators, the variance MLE, James–Stein), or the model isn't regular (the support depends on the parameter, like the uniform's end point).

</details>

6. what does "asymptotically efficient" promise about the MLE at $n = 10$?

<details><summary>answer</summary>

nothing quantitative: it's a statement about the limit. at small $n$ the MLE can be biased, and its variance can be far above the bound (the lab measures several times the bound for a Gaussian's natural parameters at $n = 10$).

</details>

## common bugs

- **natural vs. mean vs. standard parameters.** passing $(\mu, \sigma^2)$ where $\eta$ is expected. symptom: log densities that are wildly off, or `nan` because $\eta_2 > 0$.
- **$+A(\eta)$ instead of $-A(\eta)$** in `log_prob`. symptom: densities that don't sum to 1.
- **forgetting $\log x!$** in the Poisson base measure. the MLE is unaffected, but log densities (and the tests against scipy) are wrong.
- **an undamped Newton step.** symptom: `nan` parameters whenever the start is far from the answer.
- **a `while` loop on the tolerance alone.** for data where the MLE doesn't exist (all-ones Bernoulli), it never ends. always cap the iterations.
- **inverting the Fisher information entrywise.** the multi-parameter bound is the *matrix* inverse; $1/I_{11}$ is only right when the parameters are orthogonal ($I_{12} = 0$).

## going deeper

- Wainwright & Jordan (2008), "Graphical models, exponential families, and variational inference", *Foundations and Trends in Machine Learning* 1(1–2): 1–305, chapter 3: exponential families, mean parameters and the convex geometry of $A$. [doi:10.1561/2200000001](https://doi.org/10.1561/2200000001)
- Lehmann & Casella (1998), *Theory of Point Estimation*, 2nd ed., Springer, chapters 1.5 and 2.5–2.6: exponential families, sufficiency, Rao–Blackwell and the information inequality.
- Wasserman (2004), *All of Statistics*, Springer, chapter 9: Fisher information, the delta method, and the asymptotic efficiency of the MLE.
- Amari (1998), "Natural gradient works efficiently in learning", *Neural Computation* 10(2): 251–276. Fisher information as a geometry for optimization. [doi:10.1162/089976698300017746](https://doi.org/10.1162/089976698300017746)
- Murphy (2022), *Probabilistic Machine Learning: An Introduction*, MIT Press, section 3.4: the exponential family from the ML side. free at [probml.github.io/pml-book](https://probml.github.io/pml-book/book1.html).

## connections

- ← builds on: maximum likelihood (topic 02), bias and efficiency (topic 01), KL divergence (topic 05), forward-mode autodiff and numerical Jacobians (ch 03), Newton's method (ch 03), linear solves (ch 02).
- → used in: Newton and quasi-Newton optimization (ch 06), generalized linear models and logistic regression by IRLS (ch 08), softmax outputs and cross-entropy (ch 08, 14), conjugate priors and variational inference (ch 12), the Laplace approximation (ch 12), natural gradients (ch 15), and the policy gradient's score function (ch 24).
