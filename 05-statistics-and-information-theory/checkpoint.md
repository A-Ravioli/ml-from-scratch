# 05 · statistics & information theory · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. an estimator has bias 0.2 and variance 0.01; another is unbiased with variance 0.06. which has the smaller MSE? what general lesson does this illustrate?

<details><summary>answer</summary>

the first: $0.04 + 0.01 = 0.05 < 0.06$. MSE $=$ bias$^2$ $+$ variance, so a little bias can buy a large reduction in variance. "unbiased" is a property, not a guarantee of accuracy (topic 01).

</details>

2. for Gaussian data, which divisor of the sum of squared deviations gives the variance estimate with the smallest MSE, and why isn't it $n - 1$?

<details><summary>answer</summary>

$n + 1$. dividing by more than $n - 1$ shrinks the estimate toward 0: the bias it adds enters squared, so it's second-order, while the variance reduction is first-order. the optimum depends on the kurtosis: heavier tails favor more shrinkage (topic 01).

</details>

3. what's the difference between an unbiased estimator and a consistent one? give an estimator that is one but not the other.

<details><summary>answer</summary>

unbiased: $\mathbb{E}\hat\theta_n = \theta$ at each $n$. consistent: $\hat\theta_n \to \theta$ in probability as $n \to \infty$. $2X_1$ for the Uniform$(0, \theta)$ end point is unbiased but inconsistent; $\max_i X_i$ is biased but consistent (topic 01).

</details>

4. you see 3 heads in 3 flips. give the MLE, the posterior mean under a uniform prior, and the MAP under a uniform prior. which would you use to predict the next flip, and why?

<details><summary>answer</summary>

MLE $= 1$, posterior Beta(4, 1) with mean $4/5$ and mode (MAP) $1$. use the posterior mean (Laplace's rule): it's the posterior predictive probability, and it never assigns probability 0 to an outcome that can happen, which would cost infinite log loss (topic 02).

</details>

5. why is MAP estimation with a Gaussian prior on the weights the same as L2 regularization?

<details><summary>answer</summary>

MAP maximizes $\log p(\text{data} \mid w) + \log p(w)$, and a $\mathcal{N}(0, \tau^2 I)$ prior has $\log p(w) = -\lVert w\rVert^2/(2\tau^2) + $ const. so MAP $=$ maximum likelihood plus an L2 penalty with strength $1/(2\tau^2)$ (topic 02).

</details>

6. what does a 95% confidence interval promise? what does it not promise?

<details><summary>answer</summary>

it promises that the *procedure* covers the true parameter in 95% of repeated experiments. it doesn't say that any particular computed interval contains the parameter with probability 0.95 (that's a Bayesian credible interval), and its actual coverage can fall short of 95% when the method's assumptions fail (topic 03).

</details>

7. what's a bootstrap sample, and what fraction of the original points does one contain, roughly?

<details><summary>answer</summary>

$n$ draws *with replacement* from the $n$ data points. each point is left out with probability $(1 - 1/n)^n \approx e^{-1}$, so a bootstrap sample contains about 63% of the distinct points (topic 03).

</details>

8. name a statistic for which the bootstrap fails, and say why.

<details><summary>answer</summary>

the maximum (of Uniform data, say): every bootstrap replicate is at most the sample max, which is below the true end point, so bootstrap intervals never contain it. the bootstrap needs statistics that depend smoothly on the distribution; the max depends on its extreme edge, where the empirical distribution and the true one differ most (topic 03).

</details>

9. two models are scored on the same 1000 test examples. why should a confidence interval for their accuracy difference resample examples jointly rather than each model's results separately?

<details><summary>answer</summary>

their errors are correlated (hard examples are hard for both), and $\operatorname{Var}(\bar a - \bar b) = (\sigma_a^2 + \sigma_b^2 - 2\operatorname{Cov}(a, b))/n$. joint resampling keeps the covariance; independent resampling drops it and overstates the variance, often several-fold, losing the power to detect real differences (topics 03–04, capstone).

</details>

10. what is a p-value, precisely? why is it *not* the probability that the null hypothesis is true?

<details><summary>answer</summary>

the probability, computed assuming the null is true, of a test statistic at least as extreme as the one observed. it's $P(\text{data this extreme} \mid H_0)$; the probability the null is true is $P(H_0 \mid \text{data})$, which needs a prior over hypotheses and Bayes' rule (topic 04).

</details>

11. in a paired permutation test, what exactly gets permuted, and what null hypothesis justifies it?

<details><summary>answer</summary>

within each pair, the two values are swapped with probability 1/2, independently (for the mean difference: random sign flips of $x_i - y_i$). it's justified by the null that within each pair the two labels are exchangeable (e.g. both models are equally good on every example in distribution) (topic 04).

</details>

12. what do Bonferroni and Benjamini–Hochberg each control, and when would you prefer each?

<details><summary>answer</summary>

Bonferroni controls the family-wise error rate (the probability of any false rejection); BH controls the false discovery rate (the expected fraction of rejections that are false). use Bonferroni (or Holm) when a single false claim is costly, BH for screening many hypotheses where some false leads are acceptable (topic 04).

</details>

13. why is the KL divergence not a distance? give a context where its direction matters.

<details><summary>answer</summary>

it's asymmetric, $\mathrm{KL}(p \,\|\, q) \ne \mathrm{KL}(q \,\|\, p)$, and fails the triangle inequality. fitting $q$ by maximum likelihood minimizes $\mathrm{KL}(p \,\|\, q)$ (mode covering: $q$ spreads over all of $p$), while variational inference minimizes $\mathrm{KL}(q \,\|\, p)$ (mode seeking: $q$ locks onto one mode) (topic 05).

</details>

14. why can no symbol code compress a binary source with $P(1) = 0.1$ below 1 bit per symbol, even though its entropy is 0.47 bits? how do you get closer?

<details><summary>answer</summary>

every codeword needs at least one bit, and there are only two symbols. code blocks of $k$ symbols instead: the per-symbol overhead drops below $1/k$, so the rate approaches the entropy (topic 05).

</details>

15. what does the Cramér–Rao bound say, and name two ways an estimator can have variance below it.

<details><summary>answer</summary>

any unbiased estimator of $\theta$ from $n$ i.i.d. observations has variance at least $1/(nI(\theta))$ (matrix version: $I^{-1}/n$). an estimator can go below it if it's biased (shrinkage, the variance MLE) or if the model isn't regular (the support depends on $\theta$, as for the Uniform$(0, \theta)$ end point) (topic 06).

</details>

## 2. by hand (no computer)

1. **shrinking a mean.** $X_1, \dots, X_{10}$ are i.i.d. $\mathcal{N}(\mu, 1)$. (a) find the MSE of $\bar X$ and of $0.9\bar X$ as functions of $\mu$. (b) for which $\mu$ is the shrunk estimator better? (c) is $0.9\bar X$ consistent?

<details><summary>solution</summary>

(a) $\bar X$: unbiased, variance $1/10$, MSE $0.1$. $0.9\bar X$: bias $-0.1\mu$, variance $0.81/10 = 0.081$, MSE $0.01\mu^2 + 0.081$. (b) $0.01\mu^2 + 0.081 < 0.1$ iff $\mu^2 < 1.9$, i.e. $|\mu| < 1.38$: shrinkage helps when the truth is near the target, and hurts far from it (unlike James–Stein, which shrinks by a data-dependent amount). (c) no: $0.9\bar X \to 0.9\mu$, so it's inconsistent unless $\mu = 0$. a fixed shrinkage factor must fade with $n$ (like $\frac{n}{n + c}$) to keep consistency.

</details>

2. **a posterior and a code.** (a) with a Beta(1, 1) prior and 7 heads in 10 flips, give the posterior, its mean and its mode, and the MLE. (b) build a Huffman code for the four symbols with probabilities $(0.4, 0.3, 0.2, 0.1)$. give the codeword lengths, the expected length, and the entropy in bits ($\log_2 3 \approx 1.585$, $\log_2 5 \approx 2.322$).

<details><summary>solution</summary>

(a) Beta(8, 4): mean $8/12 = 0.667$, mode $7/10 = 0.7$, which equals the MLE $0.7$ (a uniform prior's MAP is the MLE). (b) merge $0.1 + 0.2 = 0.3$; then the two $0.3$s into $0.6$; then $0.4 + 0.6$. lengths $(1, 2, 3, 3)$, e.g. $(0, 10, 110, 111)$. expected length $0.4 + 0.6 + 0.6 + 0.3 = 1.9$ bits. entropy: $0.4\log_2 2.5 + 0.3\log_2(10/3) + 0.2\log_2 5 + 0.1\log_2 10 = 0.529 + 0.521 + 0.464 + 0.332 = 1.846$ bits. so $H \le L = 1.9 < H + 1$, and the code wastes 0.054 bits per symbol.

</details>

3. **a permutation test and a correction.** (a) four paired differences are $d = (2, 3, -1, 4)$. compute the exact two-sided sign-flip p-value for the mean difference (all 16 sign patterns). (b) five independent tests give p-values $(0.25, 0.01, 0.03, 0.04, 0.002)$. which hypotheses do Bonferroni and Benjamini–Hochberg reject at 0.05?

<details><summary>solution</summary>

(a) the observed sum is $2 + 3 - 1 + 4 = 8$. sign patterns give sums $\pm 2 \pm 3 \pm 1 \pm 4$; the ones with $|\text{sum}| \ge 8$ are $\pm 10$ (all signs equal) and $\pm 8$ (only the 1 flipped), 4 of 16, so $p = 0.25$. with only four pairs, nothing can be more significant than $2/16 = 0.125$ two-sided. (b) Bonferroni rejects $p \le 0.01$: the tests with $0.01$ and $0.002$. BH: sorted $(0.002, 0.01, 0.03, 0.04, 0.25)$ against $(0.01, 0.02, 0.03, 0.04, 0.05)$; the largest $k$ that passes is 4, so it rejects every test except the one with $p = 0.25$.

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **the bootstrap and the permutation test**, the core of `mlfs.stats`. work in scratch directories, starting from the *untouched* stub notebooks and the topics' checks, so your own solved `exercise.py` files stay where they are. from the repo root:

```bash
B=05-statistics-and-information-theory/03-confidence-intervals-and-the-bootstrap
P=05-statistics-and-information-theory/04-hypothesis-testing
mkdir -p /tmp/rebuild-0503 /tmp/rebuild-0504
git show HEAD:$B/exercise.py > /tmp/rebuild-0503/exercise.py   # the stubs as shipped
git show HEAD:$P/exercise.py > /tmp/rebuild-0504/exercise.py
cp $B/test_exercise.py /tmp/rebuild-0503/
cp $P/test_exercise.py /tmp/rebuild-0504/
```

(if you've committed your own solutions, regenerate the stubs instead: copy each topic's `solutions/` into its scratch directory and run `python scripts/stubify.py /tmp/rebuild-0503 /tmp/rebuild-0504`.) open each `exercise.py` with `marimo edit`, and implement from memory: in topic 03, `bootstrap_replicates`, `bootstrap_ci` (percentile and BCa), `jackknife` and `bca_interval`; in topic 04, `p_value`, `permutation_null` (unpaired and paired) and `permutation_test`. then run, from the repo root:

```bash
pytest /tmp/rebuild-0503/test_exercise.py -k "stage3 or stage4"
pytest /tmp/rebuild-0504/test_exercise.py -k "stage1 or stage2 or stage3"
```

you pass if every one of those checks is green within the hour. (`rejection_rate` is part of topic 04's stage 3: rebuild it too, it's six lines.)

**harder variant** (for testing out of the chapter with confidence): also rebuild topic 05's information measures and Huffman coder, and topic 06's `ExponentialFamily.fit`, and pass all of their checks within 100 minutes in total.

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each part of a problem separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the topic, and the lessons are organized by the same ideas.
