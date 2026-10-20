# density estimation · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: the approach</summary>

turn every point into a bin number: `np.floor((x - origin) / bin_width)`. the count for an evaluation point is the number of *training* bin numbers equal to its bin number. if the training bin numbers are sorted, the equal ones form a block, and `np.searchsorted(sorted_bins, b, side="right") - np.searchsorted(sorted_bins, b, side="left")` is its length, for a whole array of `b` at once.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
train_bins = np.sort(np.floor((x_train - origin) / bin_width))
eval_bins = np.floor((x_eval - origin) / bin_width)
counts = np.searchsorted(train_bins, eval_bins, side="right") - np.searchsorted(train_bins, eval_bins, side="left")
return counts / (len(x_train) * bin_width)
```

</details>

## stage 2

<details><summary>hint 1: the approach</summary>

write $u^2 = r^2/h^2$ (`sq_dist / h**2`). each kernel is "log of the shape" plus "log of the constant": gaussian $-u^2/2 - \frac d2\log(2\pi) - d\log h$; Epanechnikov $\log(1 - u^2) + \log\frac{d+2}{2} - \log V_d - d\log h$ where $u^2 < 1$; top-hat $-\log V_d - d\log h$ where $u^2 < 1$. $\log V_d = \frac d2\log\pi - \log\Gamma(\frac d2 + 1)$, with `math.lgamma`.

</details>

<details><summary>hint 2: nearly the answer</summary>

use `np.where(u2 < 1, value, -np.inf)` for the compact kernels. `np.log(1 - u2)` warns (and gives `nan`) where `u2 > 1`, even though `np.where` throws those entries away: compute `np.log(np.clip(1 - u2, 0, None))` inside `np.errstate(divide="ignore")`.

</details>

## stage 3

<details><summary>hint 1: the approach</summary>

`score_samples`: `D2 = pairwise_distances(X, self.X_, squared=True)` is $(m, n)$; `log_kernel(D2, self.bandwidth, d, self.kernel)` gives the log of every kernel value; `logsumexp(..., axis=1)` adds them up per query in log space; subtract `log(n)` for the average.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
logk = log_kernel(D2, self.bandwidth, self.X_.shape[1], self.kernel)
return logsumexp(logk, axis=1) - math.log(len(self.X_))
```

`score` is `float(np.sum(self.score_samples(X)))`. in `fit`, check `bandwidth > 0` and the kernel name, and keep `X` from `check_X_y`.

</details>

## stage 4

<details><summary>hint 1: the approach</summary>

one generator, `gen = as_rng(rng)`, for both steps: `gen.integers(len(self.X_), size=n)` picks the rows, `gen.standard_normal((n, d))` gives the noise. the noise is multiplied by the bandwidth (a standard deviation).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
return self.X_[idx] + self.bandwidth * gen.standard_normal((n, self.X_.shape[1]))
```

raise `NotImplementedError` first if `self.kernel != "gaussian"`.

</details>

## stage 5

<details><summary>hint 1: the approach</summary>

Silverman: `sigma = np.mean(np.std(X, axis=0, ddof=1))`, then the formula with `n, d = X.shape`. `select_bandwidth`: `splits = list(KFold(cv, shuffle=True, rng=as_rng(rng)).split(X))` once; for each bandwidth, the mean over `(tr, te)` in `splits` of `np.mean(KernelDensity(h).fit(X[tr]).score_samples(X[te]))`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
scores = np.array([np.mean([np.mean(KernelDensity(h).fit(X[tr]).score_samples(X[te])) for tr, te in splits])
                   for h in bandwidths])
return float(np.asarray(bandwidths)[np.argmax(scores)]), scores
```

the `list(...)` around the splits matters: `split` is a generator, and you need the same folds for every bandwidth.

</details>
