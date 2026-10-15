# estimation

## why this matters

a website shows a new checkout page to 40 visitors and 7 of them buy something. what's the conversion rate? "7/40 = 17.5%" is the obvious answer, and it's an **estimate**: a number computed from data, standing in for a number you can't see. every quantity you'll report in this course is one: a test accuracy, a learned weight, a gradient from a minibatch, the variance used to normalize a feature.

the question this topic asks is: *how do you know an estimate is any good?* you can't check it against the truth, because you don't know the truth. what you can do is study the **recipe** that produced it. imagine running the same recipe on thousands of fresh datasets drawn from the same source. do its answers center on the truth? how much do they scatter? does the scatter shrink as the datasets grow? those three questions are bias, variance and consistency, and together they decide which recipe to trust.

the answers are not always the ones you were taught. "use the unbiased estimator" sounds like a law of nature. you'll find that dividing a sum of squares by $n + 1$ beats the textbook's $n - 1$, and in the lab you'll meet an estimator that beats the obvious one for every possible truth, by shrinking unrelated quantities toward an arbitrary point. the tool for all of this is a simulator you build: an **estimator-evaluation harness**.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. an estimator is unbiased, and another is biased. can the biased one be closer to the truth on average?
2. to estimate a variance, you divide the sum of squared deviations by $n - 1$. for Gaussian data, is that the divisor with the smallest mean squared error?
3. you want the upper end $\theta$ of a Uniform$(0, \theta)$ distribution. is twice the sample mean a sensible estimate? is the sample maximum?
4. if an estimator's error shrinks like $1/\sqrt n$ (its MSE like $1/n$), is that the fastest possible?
5. "twice the first observation" is an unbiased estimate of $\theta$ for Uniform$(0, \theta)$, at every sample size. is it a good estimator?

## prerequisites check

- [ ] you know expectation and variance, and that $\operatorname{Var}(\bar X_n) = \sigma^2/n$ (→ ch 04 · topic 03)
- [ ] you can pass a `rng` around and write a `sampler(size, rng)` (→ ch 04 · topics 02 and 05)
- [ ] you know the law of large numbers and convergence in probability (→ ch 04 · topic 05)
- [ ] you can vectorize along an axis with NumPy (`axis=-1`, `keepdims=True`) (→ ch 00 · topic 02)
- [ ] you can fit a straight line with least squares (→ ch 02 · topic 04)

## core ideas

### idea 1: an estimator is a random variable

an **estimator** $\hat\theta$ is a function of the data, $\hat\theta = g(X_1, \dots, X_n)$, meant to approximate a fixed unknown number $\theta$. the data are random, so $\hat\theta$ is random too: it has a distribution, called its **sampling distribution**. everything you can say about an estimator's quality is a property of that distribution.

three numbers summarize it.

$$\operatorname{bias}(\hat\theta) = \mathbb{E}[\hat\theta] - \theta, \qquad \operatorname{Var}(\hat\theta) = \mathbb{E}\big[(\hat\theta - \mathbb{E}\hat\theta)^2\big], \qquad \operatorname{MSE}(\hat\theta) = \mathbb{E}\big[(\hat\theta - \theta)^2\big].$$

bias is systematic error: where the estimates center, relative to the truth. variance is random error: how much they scatter around their own center. MSE is the expected squared distance to the truth, and it's the sum of the other two:

$$\operatorname{MSE}(\hat\theta) = \operatorname{bias}(\hat\theta)^2 + \operatorname{Var}(\hat\theta).$$

(on paper 1 proves it: add and subtract $\mathbb{E}\hat\theta$ inside the square, and the cross term has expectation zero.) this is the **bias–variance decomposition**, and it answers warm-up 1. an unbiased estimator pays all its MSE in variance. a biased one can pay a little bias and save a lot of variance. nothing about "unbiased" guarantees "closest on average".

think of darts. bias is how far the center of your cluster is from the bullseye; variance is how spread out the cluster is. a tight cluster slightly off-center can beat a loose cluster centered exactly on the bullseye.

when you simulate, you replace expectations with averages over $R$ simulated estimates $\hat\theta_1, \dots, \hat\theta_R$. use the same conventions on both sides, dividing by $R$ (not $R - 1$) in the variance, and the decomposition holds *exactly* for the simulated numbers too, not just approximately. that makes it a free correctness check on your code.

> **▶ build: stage 1.** implement `bias_variance_mse(estimates, theta)` in `exercise.py` (`marimo edit exercise.py`), returning a dict with `"bias"`, `"variance"` (ddof 0) and `"mse"`.

### idea 2: the evaluation harness

to measure an estimator's bias, variance and MSE you need its sampling distribution, and you get that by simulation: pick a distribution with a known parameter $\theta$, draw $R$ independent datasets of size $n$, run the estimator on each, and summarize. that's a Monte Carlo estimate of the estimator's properties, so it has its own error bar.

doing it well is a matter of layout. draw all $R$ datasets at once, as one array of shape $(R, n)$, one dataset per row. then an estimator is a function that maps that array to $R$ estimates, one per row, like `lambda x: x.mean(axis=-1)`. no Python loop over datasets, so $R = 100{,}000$ costs a fraction of a second.

how precise is a simulated MSE? it's an average of the $R$ squared errors $(\hat\theta_r - \theta)^2$, so by the CLT (ch 04 · topic 05) its standard error is

$$\operatorname{se}(\widehat{\operatorname{MSE}}) = \frac{\operatorname{sd}\big((\hat\theta_r - \theta)^2\big)}{\sqrt R}.$$

report it. when two estimators' simulated MSEs differ by less than a couple of these, the simulation hasn't told you which is better. squared errors are skewed, so this error bar is itself only approximate, but it's far better than none.

> **🔮 predict:** you simulate the sample mean of $n = 25$ standard normals with $R = 40{,}000$ repetitions. the true MSE is $1/25 = 0.04$. roughly how big is the Monte Carlo standard error of your simulated MSE: $10^{-2}$, $10^{-3}$, or $10^{-4}$? (the squared error of a normal with variance $s^2$ has standard deviation $\sqrt2\, s^2$.)

> **▶ build: stage 2.** implement `simulate(estimator, sampler, n, n_reps, *, rng=None)`, which draws `sampler((n_reps, n), rng)` and applies the estimator, and `evaluate(...)`, which returns `bias_variance_mse` of the simulated estimates plus `"mse_se"`.

### idea 3: three ways to estimate one number

here's a problem with more than one reasonable answer. data come from Uniform$(0, \theta)$, for example waiting times up to an unknown maximum, and you want $\theta$.

**method of moments.** the mean of Uniform$(0, \theta)$ is $\theta/2$. set the sample mean equal to it and solve: $\hat\theta_{\text{mom}} = 2\bar X$. it's unbiased, with variance $4 \cdot \frac{\theta^2}{12n} = \frac{\theta^2}{3n}$.

**maximum likelihood.** the density of each draw is $1/\theta$ on $[0, \theta]$, so the likelihood of the data is $\theta^{-n}$ if every $X_i \le \theta$ and 0 otherwise. that's largest for the smallest $\theta$ that's still allowed: $\hat\theta_{\max} = \max_i X_i$. (topic 02 is all about this principle.) it can never overshoot, so it's biased low. on paper 2 derives

$$\mathbb{E}[\max_i X_i] = \frac{n}{n + 1}\theta, \qquad \operatorname{MSE}(\hat\theta_{\max}) = \frac{2\theta^2}{(n + 1)(n + 2)}.$$

**bias-corrected max.** multiply by $(n + 1)/n$ to remove the bias: $\hat\theta_{\text{ub}} = \frac{n + 1}{n}\max_i X_i$, with MSE $\frac{\theta^2}{n(n + 2)}$.

that answers warm-up 3. $2\bar X$ is sensible but wasteful: it can even return an estimate *smaller* than a value you've already seen, which is impossible. the max-based estimators have MSE of order $1/n^2$ against $1/n$ for the mean. at $n = 100$ that's $\theta^2/10{,}200$ against $\theta^2/300$: 34 times smaller.

> **▶ build: stage 3.** implement `theta_mom(x)`, `theta_max(x)` and `theta_unbiased_max(x)`, each along the last axis, so that one call evaluates a whole `(n_reps, n)` simulation.

### idea 4: unbiased isn't optimal

the sample variance

$$s^2 = \frac{1}{n - 1}\sum_{i=1}^n (X_i - \bar X)^2$$

is unbiased: dividing by $n - 1$ instead of $n$ corrects for measuring deviations from $\bar X$ instead of from the unknown $\mu$ (ch 04 · topic 03). but is $n - 1$ the best divisor? consider the whole family

$$S_c^2 = \frac{1}{c}\sum_{i=1}^n (X_i - \bar X)^2 = \frac{SS}{c}.$$

for Gaussian data, $SS/\sigma^2$ has a chi-squared distribution with $n - 1$ degrees of freedom, with mean $n - 1$ and variance $2(n - 1)$. so

$$\operatorname{bias}(S_c^2) = \sigma^2\Big(\frac{n - 1}{c} - 1\Big), \qquad \operatorname{Var}(S_c^2) = \frac{2(n - 1)\sigma^4}{c^2}.$$

at $c = n - 1$ the bias is zero and the MSE is $\frac{2\sigma^4}{n - 1}$. but making $c$ a little larger shrinks the variance at a rate that beats the bias you add: minimizing $\operatorname{bias}^2 + \operatorname{Var}$ over $c$ (on paper 3) gives

$$c^\star = n + 1, \qquad \operatorname{MSE}(S^2_{n+1}) = \frac{2\sigma^4}{n + 1}.$$

that's warm-up 2: for Gaussian data the unbiased divisor is the *worst* of the three usual choices ($n - 1$, $n$, $n + 1$). the general lesson is **shrinkage**. an estimator with high variance can often be improved by pulling it toward some fixed point, here toward 0. the pull adds bias, but cuts the variance by more. regularization in machine learning (ridge regression in ch 08, weight decay in ch 15) is the same trade, made on purpose.

two caveats keep this honest. the optimal divisor depends on the distribution: the lab measures it for light- and heavy-tailed data. and MSE is one loss among many. if you'll average variance estimates from many small groups, unbiasedness matters, because biases add up while variances average out.

> **🔮 predict:** for $n = 10$ Gaussian draws, how much smaller is the MSE of $S^2_{11}$ than that of the unbiased $S^2_9$? a few percent, about 20%, or more than half?

> **▶ build: stage 4.** implement `scaled_variance(x, c)` (row-wise, deviations from each row's own mean) and `scaled_variance_mse(n, c, sigma2=1.0)`, the exact Gaussian MSE above.

### idea 5: consistency and rates

a good estimator should get better with more data. formally, $\hat\theta_n$ is **consistent** if it converges to $\theta$ in probability as $n \to \infty$: $P(|\hat\theta_n - \theta| > \varepsilon) \to 0$ for every $\varepsilon > 0$. a convenient sufficient condition comes straight from Markov's inequality applied to the squared error:

$$P(|\hat\theta_n - \theta| > \varepsilon) \le \frac{\operatorname{MSE}(\hat\theta_n)}{\varepsilon^2},$$

so if the MSE goes to zero, the estimator is consistent. that requires *both* the bias and the variance to vanish.

consistency and unbiasedness are different properties, and neither implies the other. $\max_i X_i$ is biased at every $n$ but consistent. "twice the first observation", $2X_1$, is unbiased at every $n$, but it ignores everything except one draw, so its MSE is $\theta^2/3$ forever: inconsistent, and useless (warm-up 5).

consistency says the error goes to zero; the **rate** says how fast. if $\operatorname{MSE} \approx C n^{-r}$, then $\log \operatorname{MSE} \approx \log C - r \log n$: a straight line on a log–log plot with slope $-r$. you measure $r$ by simulating the MSE at several $n$ and fitting that line by least squares. for an average of i.i.d. terms $r = 1$, the familiar $\sigma^2/n$. most estimators you'll meet have $r = 1$, and topic 06 shows that for "regular" problems it's the best possible rate. the uniform's max achieves $r = 2$ because the problem is *not* regular: the edge of the support depends on $\theta$, and a single draw near the edge pins $\theta$ down sharply (warm-up 4).

> **▶ build: stage 5.** implement `mse_curve(estimator, sampler, theta, ns, n_reps, *, rng=None)`, the simulated MSE for each $n$, using one generator for all of them, and `convergence_rate(ns, values)`, the least-squares slope in log–log coordinates.

## worked example

**the decomposition by hand.** four simulated estimates $(1, 2, 3, 6)$ of $\theta = 2$. their mean is 3, so the bias is $3 - 2 = 1$. their deviations from 3 are $(-2, -1, 0, 3)$, so the variance is $(4 + 1 + 0 + 9)/4 = 3.5$. their errors are $(-1, 0, 1, 4)$, so the MSE is $(1 + 0 + 1 + 16)/4 = 4.5 = 1^2 + 3.5$.

**three estimates of a uniform's end.** the data $(1, 4, 2, 5)$, from Uniform$(0, \theta)$. the mean is 3, so $\hat\theta_{\text{mom}} = 6$. the max is 5, so $\hat\theta_{\max} = 5$ and $\hat\theta_{\text{ub}} = \frac54 \cdot 5 = 6.25$. at $n = 5$ and $\theta = 3$, the exact MSEs are $\theta^2/15 = 0.6$ for the moment estimator, $2\theta^2/42 = 0.429$ for the max, and $\theta^2/35 = 0.257$ for the corrected max.

**dividing by different $c$.** the data $(2, 4, 4, 4, 5, 5, 7, 9)$ have mean 5 and squared deviations $(9, 1, 1, 1, 0, 0, 4, 16)$, summing to $SS = 32$. so $S_7^2 = 4.571$, $S_8^2 = 4$ and $S_9^2 = 3.556$. for $n = 10$ Gaussian draws with $\sigma^2 = 2$, the exact MSEs are $2 \cdot 4/9 = 0.889$ at $c = 9$, $19 \cdot 4/100 = 0.76$ at $c = 10$, and $8/11 = 0.727$ at $c = 11$.

the tests check all of these numbers.

## on paper

1. **the decomposition.** prove $\operatorname{MSE} = \operatorname{bias}^2 + \operatorname{Var}$, and show it also holds exactly for simulated estimates with ddof 0.

<details><summary>solution</summary>

write $m = \mathbb{E}\hat\theta$. then $\hat\theta - \theta = (\hat\theta - m) + (m - \theta)$, so $(\hat\theta - \theta)^2 = (\hat\theta - m)^2 + 2(\hat\theta - m)(m - \theta) + (m - \theta)^2$. take expectations: the cross term is $2(m - \theta)\,\mathbb{E}[\hat\theta - m] = 0$, leaving $\operatorname{Var} + \operatorname{bias}^2$. the simulated version is the same algebra with $\mathbb{E}$ replaced by the average over the $R$ estimates and $m$ by their mean: the average of $\hat\theta_r - m$ is exactly zero. with ddof 1 in the variance, the identity is off by a factor $(R - 1)/R$.

</details>

2. **the max of uniforms.** for $X_1, \dots, X_n$ i.i.d. Uniform$(0, \theta)$, find the CDF of $M = \max_i X_i$, then $\mathbb{E}M$, $\mathbb{E}M^2$ and $\operatorname{MSE}(M)$.

<details><summary>solution</summary>

$P(M \le t) = P(\text{all } X_i \le t) = (t/\theta)^n$ for $0 \le t \le \theta$, so the density is $n t^{n-1}/\theta^n$. then $\mathbb{E}M = \int_0^\theta t \cdot n t^{n-1}/\theta^n\,dt = \frac{n}{n+1}\theta$ and $\mathbb{E}M^2 = \frac{n}{n+2}\theta^2$. so $\operatorname{MSE} = \mathbb{E}M^2 - 2\theta\,\mathbb{E}M + \theta^2 = \theta^2\big(\frac{n}{n+2} - \frac{2n}{n+1} + 1\big) = \frac{2\theta^2}{(n+1)(n+2)}$. for the corrected max, the variance is $\frac{(n+1)^2}{n^2}\operatorname{Var}(M)$ with $\operatorname{Var}(M) = \frac{n\theta^2}{(n+1)^2(n+2)}$, giving $\frac{\theta^2}{n(n+2)}$.

</details>

3. **the best divisor.** with $\operatorname{MSE}(c) = \sigma^4\big[\big(\frac{n-1}{c} - 1\big)^2 + \frac{2(n-1)}{c^2}\big]$, find the minimizing $c$ and the minimum.

<details><summary>solution</summary>

substitute $u = 1/c$: $\operatorname{MSE}/\sigma^4 = ((n-1)u - 1)^2 + 2(n-1)u^2$, a quadratic in $u$. its derivative is $2(n-1)((n-1)u - 1) + 4(n-1)u = 2(n-1)\big((n+1)u - 1\big)$, zero at $u = 1/(n+1)$. so $c^\star = n + 1$. plugging in: bias $= \sigma^2\big(\frac{n-1}{n+1} - 1\big) = -\frac{2\sigma^2}{n+1}$, variance $= \frac{2(n-1)\sigma^4}{(n+1)^2}$, total $\frac{4 + 2(n-1)}{(n+1)^2}\sigma^4 = \frac{2\sigma^4}{n+1}$. compared with $\frac{2\sigma^4}{n-1}$ for $s^2$, a saving of a factor $\frac{n-1}{n+1}$: 18% at $n = 10$.

</details>

4. **unbiased but inconsistent.** show that $2X_1$ is unbiased for the Uniform$(0, \theta)$ end point, compute its MSE, and show that $P(|2X_1 - \theta| > \theta/2)$ doesn't go to 0.

<details><summary>solution</summary>

$\mathbb{E}[2X_1] = 2 \cdot \theta/2 = \theta$, and $\operatorname{Var}(2X_1) = 4\theta^2/12 = \theta^2/3$, for every $n$. $|2X_1 - \theta| > \theta/2$ means $X_1 < \theta/4$ or $X_1 > 3\theta/4$, which has probability $1/2$ whatever $n$ is.

</details>

## experiment

`lab.py` runs three experiments with your harness:

1. dividing by $n - 1$, $n$ or $n + 1$, for data with light, normal and heavy tails.
2. Stein's paradox: estimating many unrelated means at once.
3. consistency and rates for five estimators of a uniform's end, including one that beats $1/n$.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. an estimator has bias 0.1 and variance 0.02. another has bias 0 and variance 0.05. which has the smaller MSE?

<details><summary>answer</summary>

the first: $0.01 + 0.02 = 0.03 < 0.05$.

</details>

2. why does the harness draw one `(n_reps, n)` array instead of looping over repetitions?

<details><summary>answer</summary>

speed, and simplicity. one vectorized call to the sampler and one to the estimator replace $R$ Python-level iterations, so $R = 10^5$ is cheap. it also forces the estimator to be written row-wise, which is how you'd apply it to many datasets in practice.

</details>

3. your simulation says estimator A has MSE 0.0102 ± 0.0003 and B has 0.0100 ± 0.0003 (Monte Carlo standard errors). which is better?

<details><summary>answer</summary>

you can't tell. the difference, 0.0002, is smaller than either standard error. you need more repetitions, or a paired comparison: run both estimators on the *same* simulated datasets, so the noise they share cancels in the difference. (the same idea makes paired tests more powerful in topic 04 and the capstone.)

</details>

4. is the MLE always unbiased? always consistent?

<details><summary>answer</summary>

neither is guaranteed. the variance MLE (divide by $n$) and the uniform's max are biased. the MLE is consistent under regularity conditions (topic 06), which cover most models you'll fit, but there are classic counterexamples where it isn't, for example when the number of parameters grows with $n$ (the Neyman–Scott problem).

</details>

5. why might you prefer the unbiased $s^2$ even though $S^2_{n+1}$ has smaller MSE?

<details><summary>answer</summary>

if you'll combine many estimates, by averaging the variances of many groups, or plugging them into a formula that sums them, the biases accumulate while the random errors average out, and unbiasedness wins. also, $n + 1$ is optimal only for Gaussian data; for heavy-tailed data a larger divisor is better, and for light tails a smaller one.

</details>

6. how would you check that an estimator is consistent using only your harness?

<details><summary>answer</summary>

compute its MSE (or $P(|\hat\theta - \theta| > \varepsilon)$) over a range of growing $n$ and check that it heads to zero. `mse_curve` plus `convergence_rate` gives you both whether and how fast. a flat curve (rate near 0) means inconsistent; a curve that flattens at a positive value means a bias that doesn't vanish.

</details>

7. ridge regression adds $\lambda\lVert w\rVert^2$ to the least-squares loss, giving biased weights. why would anyone want that?

<details><summary>answer</summary>

for exactly the reason $S^2_{n+1}$ beats $s^2$: shrinking the weights toward zero adds bias but can reduce their variance by more, especially when features are correlated or $n$ is small relative to the number of weights. lower MSE of the weights usually means better predictions on new data (ch 07 and ch 08).

</details>

## common bugs

- **ddof 1 in the simulated variance.** symptom: `mse` and `bias**2 + variance` disagree by a factor $(R - 1)/R$. use ddof 0 when you summarize simulations.
- **variance around $\theta$ instead of around the estimates' own mean.** symptom: "variance" equals the MSE and bias seems to vanish.
- **datasets in columns.** drawing `(n, n_reps)` and averaging over `axis=-1` gives `n` estimates, each from `n_reps` points. symptom: the wrong shape, or estimates that look far too precise.
- **re-seeding inside the loop over $n$.** creating a new generator from the same seed for every $n$ reuses the same random numbers, so results at different $n$ are correlated. normalize `rng` once.
- **`x.mean()` without an axis** in a row-wise estimator. symptom: every row gets the same value, the global mean.
- **comparing estimators within their Monte Carlo error.** symptom: rankings that flip when you change the seed.

## going deeper

- Wasserman (2004), *All of Statistics*, Springer, chapter 6 (bias, variance, MSE, consistency) and chapter 9 (method of moments and MLE).
- Lehmann & Casella (1998), *Theory of Point Estimation*, 2nd ed., Springer, chapters 2 and 5: unbiasedness, risk, admissibility and the James–Stein estimator, rigorously. [doi:10.1007/b98854](https://doi.org/10.1007/b98854)
- James & Stein (1961), "Estimation with quadratic loss", *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, 361–379. the estimator in the lab. [projecteuclid.org/euclid.bsmsp/1200512173](https://projecteuclid.org/euclid.bsmsp/1200512173)
- Efron & Morris (1977), "Stein's paradox in statistics", *Scientific American* 236(5): 119–127. the classic readable account, with baseball batting averages. [doi:10.1038/scientificamerican0577-119](https://doi.org/10.1038/scientificamerican0577-119)
- Efron & Hastie (2016), *Computer Age Statistical Inference*, Cambridge University Press, chapter 7: James–Stein and ridge regression as the same idea. free from the authors at [hastie.su.domains/CASI](https://hastie.su.domains/CASI/).

## connections

- ← builds on: expectation and variance (ch 04 · topic 03), samplers (ch 04 · topic 02), the LLN and convergence in probability (ch 04 · topic 05), least-squares line fits (ch 02 · topic 04).
- → used in: maximum likelihood and MAP, where priors act as shrinkage (topic 02), sampling distributions and confidence intervals (topic 03), the Cramér–Rao bound on variance (topic 06), the bias–variance trade-off in model selection (ch 07), ridge regression (ch 08), and empirical Bayes in Bayesian ML (ch 12).
