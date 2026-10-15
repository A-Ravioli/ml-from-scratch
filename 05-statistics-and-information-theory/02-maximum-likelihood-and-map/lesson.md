# maximum likelihood & MAP

## why this matters

topic 01 compared estimators once you had them. but where do estimators come from? for the uniform's end point you could guess three, but for a logistic regression with 50 weights, or a language model with a billion, nobody is going to guess. you need a *principle* that turns "a model with parameters" plus "data" into "parameter values", automatically.

this topic gives you the two principles that almost all of machine learning runs on. **maximum likelihood** picks the parameters under which the observed data were most probable. training a classifier by minimizing cross-entropy loss, fitting a Gaussian mixture, training a language model on next-token prediction: all maximum likelihood. **Bayesian inference** treats the parameters themselves as uncertain, puts a prior distribution on them, and updates it with the data using Bayes' rule. its simplest summary, the **MAP** estimate, turns out to be maximum likelihood plus a penalty, which is where weight decay and L2 regularization come from.

you'll also see the two principles disagree in an instructive way. after three heads in three flips, maximum likelihood says the coin *always* lands heads, and the lab shows that this kind of overconfidence costs an infinite amount of log loss. a prior fixes it with a single pseudo-count.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. a coin lands heads 3 times in 3 flips. what's the maximum-likelihood estimate of $P(\text{heads})$? would you bet at those odds?
2. why do people maximize the *log*-likelihood rather than the likelihood itself? is the answer the same?
3. you start with a prior, observe 20 flips, update, then observe 30 more and update again. is the result the same as updating once on all 50?
4. a prior says the mean is $0 \pm 1$; 4 measurements with noise variance 4 average 6. where does the posterior mean land: near 0, near 6, or in between? where exactly?
5. two analysts start with very different priors and see the same 1000 flips. do they end up agreeing?

## prerequisites check

- [ ] you know Bayes' rule, $P(A \mid B) = P(B \mid A)P(A)/P(B)$ (→ ch 04 · topic 01)
- [ ] you know the Bernoulli, Poisson, exponential, normal and Beta distributions (→ ch 04 · topic 02)
- [ ] you can find the maximum of a function by setting its derivative to zero (→ ch 01 · topic 02, ch 03 · topic 02)
- [ ] you can compute `logsumexp` stably (→ ch 00 · topic 03)
- [ ] you know what bias and MSE are (→ topic 01)

## core ideas

### idea 1: the likelihood

a **statistical model** is a family of distributions $p(x \mid \theta)$ indexed by a parameter $\theta$: the Bernoulli family indexed by $p$, the normal family by $(\mu, \sigma^2)$. once you've observed data $x_1, \dots, x_n$, flip your point of view: hold the data fixed and ask how probable they were under each $\theta$. that function of $\theta$ is the **likelihood**:

$$L(\theta) = p(x_1, \dots, x_n \mid \theta) = \prod_{i=1}^n p(x_i \mid \theta) \quad \text{(for i.i.d. data)}.$$

the likelihood is *not* a probability distribution over $\theta$: it needn't integrate to 1 over $\theta$, and it says nothing about which $\theta$ is likely before seeing data. it's a score: how well each candidate explains what happened.

you almost always work with its logarithm, the **log-likelihood** $\ell(\theta) = \sum_i \log p(x_i \mid \theta)$. three reasons (warm-up 2). a product of $n$ numbers below 1 underflows: 1000 coin flips at probability 0.5 each have likelihood $2^{-1000} \approx 10^{-301}$, at the edge of what a float64 can hold. sums are easier to differentiate than products. and since log is increasing, the maximizer doesn't change.

for the three families you'll write:

$$\ell_{\text{Bern}}(p) = k\log p + (n - k)\log(1 - p), \qquad k = \textstyle\sum_i x_i,$$

$$\ell_{\mathcal{N}}(\mu, \sigma^2) = -\frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2 - \frac n2\log(2\pi\sigma^2),$$

$$\ell_{\text{Pois}}(\lambda) = \Big(\sum_i x_i\Big)\log\lambda - n\lambda - \sum_i \log(x_i!).$$

notice the Bernoulli log-likelihood depends on the data only through $k$, and the normal one only through $\sum x_i$ and $\sum x_i^2$. such summaries are called **sufficient statistics**, and topic 06 builds a whole theory on them. one edge case matters: at $p = 1$ with all-ones data, the term $(n - k)\log(1 - p)$ is $0 \cdot \log 0$, which you must treat as 0 (the event "no tails" has probability 1), not as `nan`.

> **▶ build: stage 1.** implement `bernoulli_loglik(x, p)`, `normal_loglik(x, mu, sigma2)` and `poisson_loglik(x, lam)` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: maximum likelihood

the **maximum-likelihood estimate** (MLE) is the parameter that maximizes the likelihood:

$$\hat\theta_{\text{MLE}} = \arg\max_\theta\, \ell(\theta).$$

for the families here, set the derivative to zero. for the Bernoulli, $\ell'(p) = k/p - (n - k)/(1 - p) = 0$ gives $\hat p = k/n$: the fraction of ones. for the normal, $\partial\ell/\partial\mu = \sum(x_i - \mu)/\sigma^2 = 0$ gives $\hat\mu = \bar x$, and then $\partial\ell/\partial\sigma^2 = 0$ gives $\hat\sigma^2 = \frac1n\sum(x_i - \bar x)^2$: the divide-by-$n$ variance from topic 01, biased but MLE. the Poisson gives $\hat\lambda = \bar x$, and the exponential, with $\ell(r) = n\log r - r\sum x_i$, gives $\hat r = 1/\bar x$ (on paper 1).

why is this a good idea? one answer comes in topic 06: under regularity conditions the MLE is consistent and, for large $n$, has the smallest variance any unbiased estimator can have. another is more ML-flavored. divide the log-likelihood by $n$:

$$\frac1n\ell(\theta) = \frac1n\sum_i\log p(x_i \mid \theta) \xrightarrow{n\to\infty} \mathbb{E}_{x \sim p_{\text{data}}}\big[\log p(x \mid \theta)\big].$$

maximizing that expectation is the same as minimizing the **cross-entropy** between the data distribution and the model, or equivalently the **KL divergence** from data to model (topic 05). so "maximize likelihood" and "minimize cross-entropy loss" are the same instruction. every classifier you'll train in this course does exactly this.

the MLE has a weakness, and warm-up 1 exposes it: with 3 heads in 3 flips, $\hat p = 1$. the model now claims tails is impossible. it fits the data perfectly, which is exactly the problem: with little data, maximum likelihood **overfits**.

> **🔮 predict:** you'll use $\hat p$ to predict the next flip and pay $-\log q$ when the outcome you gave probability $q$ happens. for a coin with true $p = 0.05$, what's your *expected* payment using the MLE after 20 flips? (the lab measures it, along with a fix.)

> **▶ build: stage 2.** implement `mle_bernoulli(x)`, `mle_normal(x)` (returning `(mu, sigma2)`), `mle_poisson(x)` and `mle_exponential(x)` (the rate).

### idea 3: Bayes' rule for parameters

the Bayesian move is to treat $\theta$ as a random variable. before seeing data, your beliefs are a **prior** $p(\theta)$. after, they're the **posterior**:

$$p(\theta \mid x) = \frac{p(x \mid \theta)\,p(\theta)}{p(x)} \propto \text{likelihood} \times \text{prior}.$$

the denominator $p(x) = \int p(x \mid \theta)p(\theta)\,d\theta$ (the **evidence**) doesn't depend on $\theta$; it just normalizes.

for a coin, a convenient prior is the **Beta** distribution, $p(\theta) \propto \theta^{a-1}(1 - \theta)^{b-1}$. multiply by the likelihood $\theta^k(1 - \theta)^{n-k}$ and you get

$$p(\theta \mid x) \propto \theta^{a + k - 1}(1 - \theta)^{b + n - k - 1}: \qquad \text{Beta}(a, b) \to \text{Beta}(a + k,\; b + n - k).$$

the posterior is in the same family as the prior. a prior with that property is **conjugate** to the likelihood, and it makes updating a matter of bookkeeping: add the heads to $a$ and the tails to $b$. so $a$ and $b$ act like **pseudo-counts**, heads and tails you pretend to have seen already, and $m = a + b$ measures how strongly the prior is held. the order of the data doesn't matter, and updating in batches gives the same answer as updating all at once, because each posterior just becomes the next prior (warm-up 3).

a posterior is a whole distribution. to get a single number you summarize it:

- the **posterior mean** $\frac{a}{a + b}$, which is also the **posterior predictive** probability that the next flip is heads, $\int \theta\,p(\theta \mid x)\,d\theta$.
- the **MAP** (maximum a posteriori) estimate, the posterior's mode, $\frac{a - 1}{a + b - 2}$ (for $a, b \ge 1$ with $a + b > 2$; otherwise the density is flat or piles up at an end point).

with a uniform prior Beta(1, 1), three heads in three flips give the posterior Beta(4, 1). its mode is 1, the MLE again, but its mean is $4/5$. the posterior mean $\frac{k + 1}{n + 2}$ is **Laplace's rule of succession**, which never predicts probability 0 or 1. that answers warm-up 1: you shouldn't bet at the MLE's odds.

> **▶ build: stage 3.** implement `beta_bernoulli_posterior(x, a, b)`, `beta_mean(a, b)` and `beta_mode(a, b)` (raising `ValueError` outside its domain).

### idea 4: shrinkage, the Bayesian way

the same story for a continuous parameter. you measure a quantity $\mu$ with Gaussian noise of known variance $\sigma^2$, $x_i \sim \mathcal{N}(\mu, \sigma^2)$, and your prior is $\mu \sim \mathcal{N}(\mu_0, \tau^2)$. the log-posterior is a sum of quadratics in $\mu$, hence a quadratic, so the posterior is Gaussian. completing the square (on paper 2) gives

$$\frac{1}{\tau_n^2} = \frac{1}{\tau^2} + \frac{n}{\sigma^2}, \qquad \mu_n = \tau_n^2\Big(\frac{\mu_0}{\tau^2} + \frac{\sum_i x_i}{\sigma^2}\Big).$$

read it in terms of **precisions** (inverse variances). the posterior precision is the prior's precision plus $n$ times one measurement's. the posterior mean is a precision-weighted average of the prior mean and the sample mean:

$$\mu_n = w\,\mu_0 + (1 - w)\,\bar x, \qquad w = \frac{1/\tau^2}{1/\tau^2 + n/\sigma^2}.$$

that's warm-up 4: with a prior of variance 1 and four measurements of variance 4, the prior and the data have equal precision (1 each), so the posterior mean lands exactly halfway, at 3, with variance $1/2$.

it's topic 01's shrinkage again, now with a reason: the estimate is pulled toward the prior mean, more strongly when data are scarce or noisy. as $n \to \infty$ or $\tau^2 \to \infty$ (a flat prior), $w \to 0$ and you get the MLE $\bar x$ with variance $\sigma^2/n$.

and here's the bridge to machine learning. MAP maximizes $\log p(x \mid \theta) + \log p(\theta)$. with a Gaussian prior $\mathcal{N}(0, \tau^2)$ on each weight of a model, $\log p(\theta) = -\lVert\theta\rVert^2/(2\tau^2) + \text{const}$, so

$$\hat\theta_{\text{MAP}} = \arg\max_\theta\, \Big[\ell(\theta) - \frac{1}{2\tau^2}\lVert\theta\rVert^2\Big]:$$

maximum likelihood with an L2 penalty. ridge regression (ch 08) and weight decay (ch 15) are MAP estimates in disguise, and the regularization strength is a prior variance.

> **▶ build: stage 4.** implement `normal_posterior(x, sigma2, mu0, tau2)`, returning the posterior mean and variance.

### idea 5: posteriors on a grid

conjugate priors are convenient but rare. change the prior to a mixture, or the likelihood to anything nonstandard, and there's no formula. with one parameter you don't need one: evaluate the log-likelihood and log-prior on a fine grid $\theta_1 < \dots < \theta_m$, add them, and normalize.

$$\pi_j = \frac{\exp(\ell(\theta_j) + \log p(\theta_j))}{\sum_{j'}\exp(\ell(\theta_{j'}) + \log p(\theta_{j'}))}.$$

the $\pi_j$ are probabilities on the grid points; divided by the grid spacing they approximate the posterior density. the catch is numerical: with $n = 10{,}000$ data points the log-likelihoods are around $-10^4$, `exp` of which is exactly 0 in float64, and $0/0$ is `nan`. the fix is the one from ch 00: subtract the maximum before exponentiating, which is what `logsumexp` does. ($\pi_j = \exp(s_j - \operatorname{logsumexp}(s))$.)

from the grid posterior you can read off anything: the mean $\sum_j\theta_j\pi_j$, the MAP $\theta_{\arg\max\pi}$, and a **credible interval**. the equal-tailed 95% interval runs from the point where the cumulative posterior probability reaches 2.5% to where it reaches 97.5%. unlike the confidence intervals of topic 03, a credible interval means exactly what people want intervals to mean: given the model, the prior and the data, $\theta$ is in it with probability 0.95.

grids are hopeless in high dimensions ($100$ points per axis in 10 dimensions is $10^{20}$ points). that's what Markov chain Monte Carlo and variational inference are for (ch 12). but for one or two parameters, a grid is exact enough, transparent, and makes a great check on fancier methods.

> **🔮 predict:** two analysts start from a uniform prior and from a confident, wrong Beta(40, 10) prior (mean 0.8) on a coin whose true bias is 0.3. after 1000 flips, how far apart are their posterior means?

> **▶ build: stage 5.** implement `grid_posterior(log_lik, log_prior, grid)` (normalized in log space) and `credible_interval(grid, post, level=0.95)`.

## worked example

**a coin.** the flips $(1, 0, 1, 1, 0, 1, 1, 1)$: $k = 6$ heads in $n = 8$. the MLE is $\hat p = 0.75$. with a Beta(2, 2) prior (two pseudo-heads, two pseudo-tails), the posterior is Beta(2 + 6, 2 + 2) = Beta(8, 4), with mean $8/12 = 0.667$ and mode (MAP) $7/10 = 0.7$. both are pulled from 0.75 toward the prior's 0.5. a grid posterior with a Beta(2, 2) log-prior reproduces the same mean and peak.

**a log-likelihood.** for the first four flips $(1, 0, 1, 1)$ at $p = 0.75$: $3\log 0.75 + \log 0.25 = 3(-0.2877) - 1.3863 = -2.249$.

**a measurement.** the data $(4, 6, 5, 9)$. the MLE is $\hat\mu = 6$, $\hat\sigma^2 = (4 + 0 + 1 + 9)/4 = 3.5$. now suppose the noise variance is known, $\sigma^2 = 4$, and the prior is $\mathcal{N}(0, 1)$. the posterior precision is $1/1 + 4/4 = 2$, so the variance is $0.5$; the mean is $0.5\,(0/1 + 24/4) = 3$.

**a credible interval.** on the grid $(0, 1, 2, 3, 4)$ with probabilities $(0.1, 0.2, 0.4, 0.2, 0.1)$, the cumulative sums are $(0.1, 0.3, 0.7, 0.9, 1.0)$. the 60% interval needs tails of 0.2: the cumulative probability first reaches 0.2 at 1 and first reaches 0.8 at 3, so it's $(1, 3)$.

the tests check all of these.

## on paper

1. **the exponential MLE.** for $x_1, \dots, x_n$ i.i.d. with density $re^{-rx}$, derive $\hat r$. is it unbiased? (hint: $\sum x_i$ is Gamma$(n, r)$, and $\mathbb{E}[1/G] = r/(n - 1)$ for $G \sim$ Gamma$(n, r)$.)

<details><summary>solution</summary>

$\ell(r) = n\log r - r\sum x_i$, so $\ell'(r) = n/r - \sum x_i = 0$ gives $\hat r = n/\sum x_i = 1/\bar x$, and $\ell''(r) = -n/r^2 < 0$ confirms a maximum. $\mathbb{E}\hat r = n\,\mathbb{E}[1/\sum x_i] = \frac{n}{n - 1}r$: biased upward, though consistent. the MLE is **invariant** under reparameterization (the MLE of $1/r$ is $1/\hat r = \bar x$, which *is* unbiased for the mean), but unbiasedness isn't: a nonlinear function of an unbiased estimator is usually biased.

</details>

2. **normal–normal.** show that with prior $\mathcal{N}(\mu_0, \tau^2)$ and data $x_i \sim \mathcal{N}(\mu, \sigma^2)$, the log-posterior is a quadratic in $\mu$, and read off the posterior mean and variance.

<details><summary>solution</summary>

up to constants, $\log p(\mu \mid x) = -\frac{(\mu - \mu_0)^2}{2\tau^2} - \sum_i\frac{(x_i - \mu)^2}{2\sigma^2}$. the coefficient of $\mu^2$ is $-\frac12\big(\frac1{\tau^2} + \frac n{\sigma^2}\big)$, and the coefficient of $\mu$ is $\frac{\mu_0}{\tau^2} + \frac{\sum x_i}{\sigma^2}$. a Gaussian $\mathcal{N}(m, v)$ has log-density $-\frac{\mu^2}{2v} + \frac{m\mu}{v} + \text{const}$, so matching coefficients gives $1/v = 1/\tau^2 + n/\sigma^2$ and $m = v\big(\frac{\mu_0}{\tau^2} + \frac{\sum x_i}{\sigma^2}\big)$.

</details>

3. **the MAP of a Beta.** differentiate $\log[\theta^{a-1}(1 - \theta)^{b-1}]$ to find the mode. what happens when $a = b = 1$, and when $a < 1$?

<details><summary>solution</summary>

$\frac{a - 1}{\theta} - \frac{b - 1}{1 - \theta} = 0$ gives $\theta = \frac{a - 1}{a + b - 2}$. with $a = b = 1$ the density is flat and every $\theta$ is a mode. with $a < 1$ the density goes to infinity at $\theta = 0$, so there's no maximum inside, and the formula can give a "mode" outside $[0, 1]$. that's why `beta_mode` refuses those cases.

</details>

4. **MAP depends on the parameterization.** a uniform prior on $p$ isn't uniform on the log-odds $\eta = \log\frac{p}{1 - p}$. show that the density of $\eta$ is $\sigma(\eta)(1 - \sigma(\eta))$, and that for $k$ heads in $n$ flips, the MAP of $\eta$ maps back to $p = \frac{k + 1}{n + 2}$, not $\frac{k}{n}$.

<details><summary>solution</summary>

$p = \sigma(\eta)$ and $dp/d\eta = \sigma(\eta)(1 - \sigma(\eta))$, so a uniform density on $p$ becomes that density on $\eta$ (change of variables, ch 03). the posterior on $\eta$ is $\propto p^k(1 - p)^{n-k}\cdot p(1 - p) = p^{k+1}(1 - p)^{n - k + 1}$ with $p = \sigma(\eta)$. its log-derivative in $\eta$ is $(k + 1)(1 - p) - (n - k + 1)p = 0$, so $p = \frac{k + 1}{n + 2}$. the MAP on $p$ was $k/n$. MAP estimates are modes of densities, and densities change under reparameterization; the MLE (no prior) and the posterior mean of a fixed quantity don't have this problem.

</details>

## experiment

`lab.py` runs three experiments with your code:

1. how much a (wrong) prior is worth, as a function of prior strength and data size.
2. predicting a rare event: the expected log loss of the MLE against Laplace's rule.
3. three very different priors on the same coin, updated on more and more data.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. why is the likelihood not a probability distribution over $\theta$?

<details><summary>answer</summary>

it's $p(x \mid \theta)$ viewed as a function of $\theta$ with $x$ fixed. it's a density in $x$, not in $\theta$, so it needn't integrate to 1 over $\theta$ (for a coin, $\int_0^1 p^k(1-p)^{n-k}\,dp = \frac{k!(n-k)!}{(n+1)!}$). multiplying by a prior and normalizing is what turns it into a distribution over $\theta$.

</details>

2. a Beta(a, b) prior is "worth" how many observations?

<details><summary>answer</summary>

$a + b$ of them: the posterior mean $\frac{a + k}{a + b + n}$ weighs the prior mean like $a + b$ extra flips. after $n \gg a + b$ flips the prior's influence fades; while $n$ is comparable to $a + b$, it dominates.

</details>

3. why does MAP with a Gaussian prior correspond to L2 regularization, and what does a Laplace prior $p(\theta) \propto e^{-|\theta|/b}$ correspond to?

<details><summary>answer</summary>

MAP maximizes $\ell(\theta) + \log p(\theta)$. a Gaussian log-prior is $-\theta^2/(2\tau^2)$, an L2 penalty. a Laplace log-prior is $-|\theta|/b$, an L1 penalty: that's the lasso (ch 08), which drives some weights exactly to zero.

</details>

4. with a flat prior, is the MAP always the MLE?

<details><summary>answer</summary>

yes for the parameterization in which the prior is flat, since then the log-posterior is the log-likelihood plus a constant. but "flat" depends on the parameterization (on paper 4): flat in $p$ is not flat in the log-odds, and the MAP moves when you change coordinates.

</details>

5. why do two analysts with different priors eventually agree?

<details><summary>answer</summary>

the log-posterior is $\ell(\theta) + \log p(\theta)$. the log-likelihood is a sum of $n$ terms and grows like $n$; the log-prior stays fixed. so as $n$ grows the likelihood dominates, the posterior concentrates around the MLE with width $\propto 1/\sqrt n$, and its shape becomes Gaussian (the Bernstein–von Mises theorem). the exception: a prior that gives zero probability to the truth can never recover, since $\log 0 = -\infty$ whatever the data say.

</details>

6. why compute the grid posterior in log space?

<details><summary>answer</summary>

likelihoods of even moderately large datasets underflow to exactly zero. in log space they're ordinary numbers; subtracting the maximum (logsumexp) before exponentiating keeps the largest term at $e^0 = 1$, so the normalization is exact and nothing becomes $0/0$.

</details>

7. a language model assigns probability 0 to a word that then appears in the test set. what's its test perplexity, and how does this lesson suggest fixing it?

<details><summary>answer</summary>

infinite: the log loss of that word is $-\log 0$. it's the MLE's overfitting on a huge scale: counts of 0 become probabilities of 0. smoothing (adding pseudo-counts, as in the ch 04 capstone's add-$k$) is exactly a Dirichlet prior's posterior mean, the many-category version of Laplace's rule.

</details>

## common bugs

- **`0 * log 0` as `nan`.** `k * np.log(p)` with $k = 0$ and $p = 0$ is `0 * -inf = nan`. skip the term when its count is zero.
- **the normal log-likelihood with $\sigma$ in place of $\sigma^2$.** the normalizer is $-\frac n2\log(2\pi\sigma^2)$. writing `log(2 * pi * sigma)` instead halves the penalty for a large variance, and the maximizing $\sigma^2$ doubles.
- **forgetting $\log x!$ in the Poisson log-likelihood.** it doesn't change the MLE, but it makes log-likelihoods wrong, and comparisons between models (or with scipy) fail.
- **swapping heads and tails in the Beta update.** heads go to $a$, tails to $b$. symptom: posteriors pulled the wrong way.
- **returning the precision instead of the variance** from `normal_posterior`.
- **exponentiating log-posteriors directly.** symptom: a grid posterior full of `nan` as soon as $n$ gets large.

## going deeper

- Wasserman (2004), *All of Statistics*, Springer, chapter 9 (maximum likelihood) and chapter 11 (Bayesian inference).
- Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), *Bayesian Data Analysis*, 3rd ed., CRC Press, chapters 2–3: single-parameter models, conjugate priors, and the normal model. free from the authors at [stat.columbia.edu/~gelman/book](http://www.stat.columbia.edu/~gelman/book/).
- MacKay (2003), *Information Theory, Inference, and Learning Algorithms*, Cambridge University Press, chapters 3 and 21–24: inference as a natural extension of probability, with coins. free at [inference.org.uk/itila](http://www.inference.org.uk/itila/).
- Murphy (2022), *Probabilistic Machine Learning: An Introduction*, MIT Press, chapter 4: MLE, MAP, regularization and Bayesian statistics from the ML side. free at [probml.github.io/pml-book](https://probml.github.io/pml-book/book1.html).
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, sections 2.1–2.3: the beta–binomial and Gaussian conjugate analyses, and sequential updating.

## connections

- ← builds on: Bayes' rule (ch 04 · topic 01), the standard distributions (ch 04 · topic 02), derivatives and maxima (ch 01, ch 03), `logsumexp` (ch 00 · topic 03), bias and shrinkage (topic 01).
- → used in: likelihood-based intervals and tests (topics 03–04), cross-entropy and KL (topic 05), exponential families and the MLE's optimality (topic 06), logistic regression and ridge/lasso as MAP (ch 08), naive Bayes and Gaussian discriminant analysis (ch 08), Bayesian ML, MCMC and variational inference (ch 12), and training any neural network with a cross-entropy loss (ch 14 onward).
