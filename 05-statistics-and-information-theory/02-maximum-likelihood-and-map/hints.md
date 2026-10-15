# maximum likelihood & MAP · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

write each log-likelihood in terms of the data's sufficient statistics: the number of ones $k$ for the Bernoulli, the sum of squared deviations from `mu` for the normal, the total count for the Poisson. the Bernoulli edge cases come from terms whose *count* is zero.

</details>

<details><summary>hint 2: the approach</summary>

Bernoulli: start from 0.0, add `k * math.log(p)` only if `k > 0`, and `(n - k) * math.log1p(-p)` only if `n - k > 0`. if a needed probability is 0 (`p == 0` with `k > 0`, or `p == 1` with `n - k > 0`), the answer is `-math.inf`. normal: `-0.5 * np.sum((x - mu) ** 2) / sigma2 - 0.5 * n * math.log(2 * math.pi * sigma2)`. Poisson: `sum(x) * log(lam) - n * lam - sum(lgamma(x + 1))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def bernoulli_loglik(x, p):
    x = np.asarray(x)
    k, n = int(np.sum(x)), len(x)
    out = 0.0
    if k > 0:
        out += k * math.log(p) if p > 0 else -math.inf
    if n - k > 0:
        out += (n - k) * math.log1p(-p) if p < 1 else -math.inf
    return float(out)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

each MLE comes from setting the derivative of the log-likelihood to zero (idea 2). all four are one or two lines.

</details>

<details><summary>hint 2: the approach</summary>

Bernoulli and Poisson: the mean of the data. normal: the mean, and the mean squared deviation from it (divide by $n$, not $n - 1$). exponential: the *rate* is one over the mean.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def mle_normal(x):
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    return mu, float(np.mean((x - mu) ** 2))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

a Beta prior's parameters are pseudo-counts: heads add to `a`, tails to `b`. the mean and mode are formulas in `a` and `b`.

</details>

<details><summary>hint 2: the approach</summary>

`k = np.sum(x)`, posterior `(a + k, b + len(x) - k)`. mean `a / (a + b)`. mode `(a - 1) / (a + b - 2)`, but first `raise ValueError` if `a < 1 or b < 1 or a + b <= 2`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def beta_bernoulli_posterior(x, a, b):
    x = np.asarray(x)
    k = float(np.sum(x))
    return float(a + k), float(b + len(x) - k)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

work with precisions (one over variances). they add: the prior contributes `1 / tau2`, and each of the `n` data points contributes `1 / sigma2`.

</details>

<details><summary>hint 2: the approach</summary>

`prec = 1 / tau2 + n / sigma2`. the posterior mean is `(mu0 / tau2 + sum(x) / sigma2) / prec`, and the posterior variance is `1 / prec`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
prec = 1.0 / tau2 + len(x) / sigma2
mean = (mu0 / tau2 + np.sum(x) / sigma2) / prec
return float(mean), float(1.0 / prec)
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

add the two log arrays, then normalize *in log space*: subtracting `logsumexp` of the whole array from every entry gives log-probabilities that sum (after `exp`) to exactly 1. for the interval, cumulative sums give the posterior CDF on the grid.

</details>

<details><summary>hint 2: the approach</summary>

`logp = log_lik(grid) + log_prior(grid)`, then `np.exp(logp - logsumexp(logp))`. for the interval: `cdf = np.cumsum(post)`, `tail = (1 - level) / 2`, and `np.searchsorted(cdf, tail)` is the first index where `cdf >= tail`. do the same with `1 - tail` for the upper end (guarding against an index equal to `len(grid)`).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def credible_interval(grid, post, level=0.95):
    grid = np.asarray(grid, dtype=float)
    cdf = np.cumsum(post)
    tail = (1 - level) / 2
    lo = grid[np.searchsorted(cdf, tail)]
    hi = grid[min(np.searchsorted(cdf, 1 - tail), len(grid) - 1)]
    return float(lo), float(hi)
```

</details>
