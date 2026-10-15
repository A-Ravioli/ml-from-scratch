# confidence intervals & the bootstrap

## why this matters

your new model scores 87.3% accuracy on a test set of 500 examples, and the old one scored 85.9%. is the new one better? the honest answer starts with another question: if you had drawn a *different* 500 test examples, how much would those numbers have moved? if the answer is "about ±3 points", the 1.4-point gap means nothing. if it's "±0.3", it's real.

a **confidence interval** is how you say that in one line: "87.3%, 95% CI [84.3, 90.2]". it turns a single number into a statement about how well the data pin down the truth. topic 01 measured an estimator's spread by simulating fresh datasets, which works when you invent the distribution. with real data you get one dataset, and no way to draw another.

this topic gives you two ways around that. **normal theory** uses the central limit theorem: many estimators are approximately Gaussian, so mean ± 1.96 standard errors does the job. the **bootstrap** is a computational trick of startling generality: resample your own data, with replacement, thousands of times, and treat the spread of the recomputed statistic as if it came from fresh datasets. it works for medians, correlations, AUCs, differences between two models, anything you can compute. you'll build both, and a simulator that checks how often their "95%" intervals actually contain the truth. the answer is often not 95%.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. you compute a 95% confidence interval [2.1, 3.4] for a mean. is there a 95% probability that the true mean is in [2.1, 3.4]?
2. with $n = 5$ normal observations, how often does $\bar x \pm 1.96\,s/\sqrt n$ contain the true mean?
3. a bootstrap sample is $n$ draws *with replacement* from your $n$ data points. about what fraction of the original points appear in it at least once?
4. can the bootstrap give a valid interval for any statistic, given enough resamples?
5. two models are scored on the same test examples. should you resample each model's scores separately, or resample examples?

## prerequisites check

- [ ] you know the CLT and the standard error $\sigma/\sqrt n$ (→ ch 04 · topic 05)
- [ ] you know $\Phi$ and $\Phi^{-1}$, and that $\Phi^{-1}(0.975) = 1.96$ (→ ch 04 · topic 02)
- [ ] you know what a sampling distribution, bias and variance are (→ topic 01)
- [ ] you can simulate many datasets at once with a `sampler(size, rng)` (→ topic 01)
- [ ] you can compute quantiles with `np.quantile` (→ ch 00 · topic 02)

## core ideas

### idea 1: what an interval promises

recall from topic 01 that an estimate $\hat\theta$ has a **sampling distribution**: the distribution of values it would take over repeated datasets. its standard deviation is the **standard error**. for the sample mean of $n$ i.i.d. draws, the standard error is $\sigma/\sqrt n$, and by the CLT the sampling distribution is approximately normal:

$$\frac{\bar X - \mu}{\sigma/\sqrt n} \approx \mathcal{N}(0, 1) \quad\Longrightarrow\quad P\Big(\bar X - 1.96\frac{\sigma}{\sqrt n} \le \mu \le \bar X + 1.96\frac{\sigma}{\sqrt n}\Big) \approx 0.95.$$

replace the unknown $\sigma$ by the sample standard deviation $s$ (ddof 1) and you get the **normal interval** $\bar x \pm z_{1-\alpha/2}\, s/\sqrt n$, with $z_{1-\alpha/2} = \Phi^{-1}(1 - \alpha/2)$: 1.645 for 90%, 1.96 for 95%, 2.576 for 99%.

now the subtle part (warm-up 1). in the probability statement above, $\mu$ is fixed and the **interval is random**: it moves from dataset to dataset. "95% confidence" is a property of the *procedure*: if you repeated the experiment many times, 95% of the intervals you'd compute would contain $\mu$. once you've computed one interval, [2.1, 3.4], it either contains $\mu$ or it doesn't; there's nothing random left. saying "there's a 95% probability that $\mu$ is in [2.1, 3.4]" requires treating $\mu$ as random, which is the Bayesian credible interval of topic 02. in practice the two often nearly coincide, but they answer different questions, and the difference matters when they don't.

replacing $\sigma$ by $s$ isn't free. with few points, $s$ is itself noisy, and it's sometimes much too small, which makes the interval too short. for normal data the exact fix is to use a Student $t$ quantile with $n - 1$ degrees of freedom instead of 1.96 (2.776 for $n = 5$). you'll keep 1.96 and *measure* the damage.

> **▶ build: stage 1.** implement `normal_ci(x, alpha=0.05)` in `exercise.py` (`marimo edit exercise.py`), using `NormalDist().inv_cdf` for the quantile.

### idea 2: checking the promise

an interval method's **nominal coverage** is the $1 - \alpha$ it claims. its **actual coverage** is the probability that it contains the truth, which depends on the distribution of the data and on $n$. the two agree only when the method's assumptions hold. the only general way to find the actual coverage is to measure it: simulate many datasets from a distribution with known $\theta$, compute an interval from each, and count.

$$\widehat{\text{coverage}} = \frac{1}{R}\sum_{r=1}^R \mathbb{1}\big[\text{lo}_r \le \theta \le \text{hi}_r\big].$$

it's a proportion, so its Monte Carlo standard error is $\sqrt{c(1 - c)/R}$: about 0.007 for $R = 1000$ and $c = 0.95$. also record the mean **width** $\text{hi} - \text{lo}$. coverage alone can be gamed: the interval $(-\infty, \infty)$ has perfect coverage and is useless. you want the shortest intervals that still keep their promise.

warm-up 2 is a good first measurement. with $n = 5$ normal points, $\bar x \pm 1.96\,s/\sqrt5$ has actual coverage $P(|T_4| \le 1.96) = 0.879$, where $T_4$ has a $t$ distribution with 4 degrees of freedom. it misses one time in eight, not one in twenty.

> **🔮 predict:** for very right-skewed data (lognormal) and $n = 10$, the normal interval's coverage is well below 95%. are its misses spread evenly between "interval too low" and "interval too high"? (the lab measures it.)

> **▶ build: stage 2.** implement `coverage(ci_fn, sampler, theta, n, n_reps, *, rng=None)`, returning the actual coverage and the mean width.

### idea 3: the bootstrap

suppose your statistic is a median, or a correlation, or the AUC of a classifier. there's no tidy formula for its standard error. here's the bootstrap's idea (Efron, 1979). the sampling distribution is what you'd see if you could draw fresh datasets from the population $F$. you can't. but your data define an **empirical distribution** $\hat F$, which puts probability $1/n$ on each observed point, and it's the best available stand-in for $F$. drawing a dataset of size $n$ from $\hat F$ means drawing $n$ points from your data **with replacement**. so:

1. draw a **bootstrap sample** $x^*_1, \dots, x^*_n$ with replacement from $x_1, \dots, x_n$.
2. compute the statistic on it: a **bootstrap replicate** $\hat\theta^*$.
3. repeat $B$ times (a few thousand).

the spread of the $\hat\theta^*_b$ around $\hat\theta$ estimates the spread of $\hat\theta$ around $\theta$. the simplest interval, the **percentile interval**, takes the $\alpha/2$ and $1 - \alpha/2$ quantiles of the replicates.

a bootstrap sample is not a shuffle. each point appears 0, 1, 2, … times; the chance that a given point is left out is $(1 - 1/n)^n \to e^{-1} \approx 0.368$, so a bootstrap sample contains about 63.2% of the distinct original points (warm-up 3). that's precisely what creates the variation between replicates. resampling *without* replacement just permutes the data, and every replicate of the mean equals $\bar x$.

when each observation is a *tuple* (an example's label and prediction; a model A score and a model B score on the same example), resample the **rows**: one set of indices per replicate, applied to every array. that keeps each observation's parts together, and preserves the dependence between them. it's the answer to warm-up 5, and the lab shows how much it matters.

two honest caveats. the bootstrap estimates the variance of $\bar x$ as $\hat\sigma^2/n$ with the ddof-0 $\hat\sigma$, so it's slightly too narrow at small $n$. and it inherits every weakness of $\hat F$ as a stand-in for $F$: if the sample hasn't seen the tail, neither have the resamples.

> **▶ build: stage 3.** implement `bootstrap_replicates(data, statistic, n_boot, *, rng=None)` (one array or a tuple of arrays, resampled jointly) and `bootstrap_ci(..., method="percentile")`, part of `mlfs.stats`.

### idea 4: BCa, a better bootstrap interval

the percentile interval is right when the bootstrap distribution is centered on $\hat\theta$ and its shape doesn't depend on $\theta$. often neither holds. the variance of skewed data, for instance, is estimated with a sampling distribution that's skewed, and whose spread grows with the true variance. the **bias-corrected and accelerated** (BCa) interval (Efron, 1987) keeps the replicates but reads off *different* quantiles, adjusted by two numbers.

**bias correction $z_0$.** if $\hat\theta$ were median-unbiased, half the replicates would fall below it. measure the actual fraction and convert it to a normal score:

$$z_0 = \Phi^{-1}(p_0), \qquad p_0 = \frac{\#\{\hat\theta^*_b < \hat\theta\} + \#\{\hat\theta^*_b \le \hat\theta\}}{2B}.$$

(counting ties as half matters for discrete statistics like the median; clip $p_0$ away from 0 and 1 so $z_0$ stays finite.)

**acceleration $a$.** how fast the standard error changes with $\theta$, which shows up as skewness. it's estimated with the **jackknife**: recompute the statistic $n$ times, leaving out one observation each time, giving $\hat\theta_{(1)}, \dots, \hat\theta_{(n)}$. with $d_i = \bar\theta_{(\cdot)} - \hat\theta_{(i)}$,

$$a = \frac{\sum_i d_i^3}{6\big(\sum_i d_i^2\big)^{3/2}}.$$

**the adjusted levels.** for each nominal tail $z = \Phi^{-1}(\alpha/2)$ and $z = \Phi^{-1}(1 - \alpha/2)$, use the quantile of the replicates at

$$\text{level} = \Phi\Big(z_0 + \frac{z_0 + z}{1 - a(z_0 + z)}\Big).$$

with $z_0 = a = 0$ the levels are $\alpha/2$ and $1 - \alpha/2$: plain percentile. a positive $z_0$ (most replicates below $\hat\theta$) shifts both levels up; a positive $a$ stretches the upper one. the result is "second-order accurate": for smooth statistics, the error in the coverage of each *one-sided* endpoint (the chance of missing below, or above) shrinks like $1/n$ instead of the percentile method's $1/\sqrt n$. (for two-sided intervals the leading errors of the two ends partly cancel, so the comparison is subtler.)

no bootstrap rescues everything (warm-up 4). it needs the statistic to be a smooth function of the data's distribution. the max isn't: the lab shows a statistic whose bootstrap intervals have coverage exactly zero.

> **🔮 predict:** for the sample variance of exponential data, will BCa move the percentile interval up, down, or just widen it?

> **▶ build: stage 4.** implement `jackknife(data, statistic)`, `bca_interval(theta_hat, replicates, jack, alpha)`, and the `method="bca"` branch of `bootstrap_ci`.

## worked example

**a normal interval.** the data $(4, 6, 5, 9)$: mean 6, squared deviations summing to 14, so $s = \sqrt{14/3} = 2.160$ and the standard error is $2.160/\sqrt4 = 1.080$. the 95% interval is $6 \pm 1.96 \cdot 1.080 = (3.883, 8.117)$. (with $n = 4$ and Gaussian data, its actual coverage is only about 86%.)

**a coverage check.** an interval that is exactly $(0, 0)$ contains $\theta = 0$: coverage is a closed-interval count, $\text{lo} \le \theta \le \text{hi}$.

**bootstrap samples.** with 100 distinct points, a bootstrap sample contains on average $100(1 - 0.99^{100}) = 63.4$ distinct values. if your replicates of "number of distinct values" are all 100, you've been permuting, not resampling.

**a jackknife.** for $(1, 2, 3, 6)$ and the mean, leaving out each point gives $(11/3, 10/3, 3, 2) = (3.667, 3.333, 3.000, 2.000)$. their mean is 3; the deviations $d_i$ are $(-0.667, -0.333, 0, 1)$, so $\sum d^3 = -0.296 - 0.037 + 0 + 1 = 0.667$ and $\sum d^2 = 1.556$, giving $a = 0.667/(6 \cdot 1.556^{1.5}) = 0.057$: a small positive acceleration from the one large value.

**BCa reduces to percentile.** if replicates $(1, 2, 2, 2, 3)$ (repeated) are tied with $\hat\theta = 2$, then $p_0 = (1 + 4)/10 = 0.5$ and $z_0 = 0$; with constant jackknife values $a = 0$; and BCa returns exactly the percentile interval.

the tests check the normal interval, the closed-interval count, the distinct-value count, the jackknife values and the tie case.

## on paper

1. **the bootstrap of the mean, exactly.** for a bootstrap sample of size $n$ from data $x_1, \dots, x_n$, show that $\mathbb{E}^*[\bar x^*] = \bar x$ and $\operatorname{Var}^*(\bar x^*) = \hat\sigma^2/n$ with $\hat\sigma^2 = \frac1n\sum(x_i - \bar x)^2$. ($\mathbb{E}^*$ is over the resampling, with the data fixed.)

<details><summary>solution</summary>

each $x^*_j$ is a draw from $\hat F$: it equals $x_i$ with probability $1/n$, so $\mathbb{E}^*x^*_j = \bar x$ and $\operatorname{Var}^*x^*_j = \frac1n\sum(x_i - \bar x)^2 = \hat\sigma^2$. the $x^*_j$ are independent, so $\bar x^*$ has mean $\bar x$ and variance $\hat\sigma^2/n$. compared with the usual estimate $s^2/n$ it's smaller by $(n - 1)/n$, which is why percentile intervals for the mean are slightly too narrow at small $n$.

</details>

2. **the probability of being left out.** show that a given point is missing from a bootstrap sample with probability $(1 - 1/n)^n$, and find the limit.

<details><summary>solution</summary>

each of the $n$ draws misses the point with probability $1 - 1/n$, independently, so all miss with probability $(1 - 1/n)^n$. since $\log(1 - 1/n)^n = n\log(1 - 1/n) = n(-1/n - 1/(2n^2) - \dots) \to -1$, the limit is $e^{-1} \approx 0.368$. so about 63.2% of the points appear at least once. (this is also where random forests' "out-of-bag" examples come from, ch 10.)

</details>

3. **the paired variance.** two scores $A$ and $B$ per example, with variances $\sigma_A^2, \sigma_B^2$ and correlation $\rho$. find $\operatorname{Var}(\bar A - \bar B)$ for $n$ paired examples, and the variance an "unpaired" analysis assumes.

<details><summary>solution</summary>

$\operatorname{Var}(\bar A - \bar B) = \frac1n\big(\sigma_A^2 + \sigma_B^2 - 2\rho\sigma_A\sigma_B\big)$. treating the two samples as independent drops the covariance term and gives $\frac1n(\sigma_A^2 + \sigma_B^2)$. with $\sigma_A = \sigma_B = 1$ and $\rho = 0.9$, that's $2/n$ against $0.2/n$: a standard error $\sqrt{10} \approx 3.2$ times too big. pairing is free power.

</details>

4. **BCa for a symmetric case.** suppose the replicates are symmetric about $\hat\theta$ and the jackknife values are symmetric about their mean. what are $z_0$ and $a$, and what does BCa return?

<details><summary>solution</summary>

symmetry about $\hat\theta$ puts half the replicates below it (ties counted half), so $p_0 = 1/2$ and $z_0 = 0$. symmetric $d_i$ have $\sum d_i^3 = 0$, so $a = 0$. then both levels are the nominal $\alpha/2$ and $1 - \alpha/2$, and BCa is the percentile interval. BCa only changes things when there's bias or skew to correct.

</details>

## experiment

`lab.py` runs three experiments with your code:

1. the actual coverage of the normal, percentile and BCa intervals for the mean of skewed data.
2. bootstrap intervals for the maximum.
3. paired versus unpaired bootstrap intervals for the difference between two models.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. you run 20 independent experiments and compute a 95% interval in each. how many do you expect to miss the truth?

<details><summary>answer</summary>

about one ($20 \times 0.05$), if the method's actual coverage equals its nominal coverage. that's what the 95% promises: a long-run frequency of the procedure, not a statement about any single interval.

</details>

2. how does the width of a normal interval for a mean scale with $n$? how many more data do you need to halve it?

<details><summary>answer</summary>

the width is $2z\,s/\sqrt n$, proportional to $1/\sqrt n$. halving it takes four times the data. it's why going from "±1%" to "±0.1%" on a test accuracy takes 100 times more test examples.

</details>

3. why is the bootstrap estimate of a standard error noisy when $B$ is small, and why doesn't increasing $B$ fix small $n$?

<details><summary>answer</summary>

with $B$ replicates the Monte Carlo error of a quantile shrinks like $1/\sqrt B$, so a few thousand make it negligible. but that's error relative to the *ideal* bootstrap (infinite $B$), which is itself an approximation built from $n$ points. its error depends on $n$, and no number of resamples can add information the sample doesn't contain.

</details>

4. your test set has 300 examples and you compute accuracy. what statistic and data do you pass to `bootstrap_ci`?

<details><summary>answer</summary>

the per-example correctness vector $c \in \{0, 1\}^{300}$ (or the tuple `(y_true, y_pred)`), and `np.mean` (or `lambda y, p: np.mean(y == p)`). resampling examples is the right unit because the examples are what was sampled. for a metric that isn't a mean of per-example terms, like AUC or F1, pass the tuple and recompute the metric on each resample.

</details>

5. why does the percentile bootstrap undercover for the mean at small $n$ even for normal data?

<details><summary>answer</summary>

two reasons. it estimates the standard error with the ddof-0 $\hat\sigma$ (on paper 1), slightly too small. and like the normal interval with 1.96, it ignores the extra uncertainty from estimating $\sigma$ at all, which the $t$ distribution accounts for. at $n = 10$ the two together cost several points of coverage.

</details>

6. the jackknife leaves out one point at a time. why not use it instead of the bootstrap for everything?

<details><summary>answer</summary>

the jackknife gives a standard error for smooth statistics, cheaply ($n$ evaluations). but it fails for non-smooth ones (the jackknife variance of the median is inconsistent), and it doesn't give the shape of the sampling distribution, which intervals for skewed statistics need. BCa uses each tool for what it's good at: the bootstrap for the distribution, the jackknife for the acceleration.

</details>

7. a colleague's BCa interval for a median comes out as a single point. what probably happened?

<details><summary>answer</summary>

the median of discrete data (or small $n$) takes few distinct values, so many replicates tie with $\hat\theta$, and the bootstrap distribution is a few spikes. either ties were counted as "below" (making $p_0$ extreme) or both adjusted levels landed on the same spike. counting ties as half helps; with very few distinct values, no bootstrap interval for a median is reliable.

</details>

## common bugs

- **resampling without replacement** (`rng.permutation`). symptom: every replicate of the mean equals $\bar x$, and the interval has zero width.
- **resampling the arrays of a tuple independently.** symptom: intervals for paired differences that are far too wide; correlations that collapse to 0.
- **using $\alpha$ instead of $\alpha/2$** for the percentile quantiles. symptom: a 90% interval labeled 95%.
- **ddof 0 in `normal_ci`.** symptom: intervals slightly too narrow, coverage a little lower still.
- **the sign of the acceleration.** $d_i = \bar\theta_{(\cdot)} - \hat\theta_{(i)}$, not the reverse. symptom: BCa shifts the interval the wrong way for skewed statistics.
- **building an $(n_{\text{boot}}, n)$ index matrix** for large $n$. with $10^4$ test examples and $10^4$ resamples that's 800 MB. draw one index set per replicate.

## going deeper

- Efron (1979), "Bootstrap methods: another look at the jackknife", *Annals of Statistics* 7(1): 1–26. the paper that started it. [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552)
- Efron (1987), "Better bootstrap confidence intervals", *Journal of the American Statistical Association* 82(397): 171–185. BCa. [doi:10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410)
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall/CRC, chapters 6, 13 and 14: standard errors, percentile and BCa intervals.
- DiCiccio & Efron (1996), "Bootstrap confidence intervals", *Statistical Science* 11(3): 189–228. why BCa is second-order accurate, and where it fails. [doi:10.1214/ss/1032280214](https://doi.org/10.1214/ss/1032280214)
- Wasserman (2004), *All of Statistics*, Springer, chapters 6–8: confidence intervals and the bootstrap, concisely.

## connections

- ← builds on: the CLT and standard errors (ch 04 · topic 05), sampling distributions and simulation harnesses (topic 01), credible intervals (topic 02).
- → used in: permutation tests, which resample under a null instead (topic 04), the capstone's model-comparison toolkit, every evaluation from ch 07 on (`mlfs.stats.bootstrap_ci`), out-of-bag estimates in random forests (ch 10), and multi-seed experiment reporting (`mlfs.research`, ch 30).
