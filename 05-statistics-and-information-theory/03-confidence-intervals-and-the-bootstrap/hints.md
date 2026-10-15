# confidence intervals & the bootstrap · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

three ingredients: the sample mean, the standard error $s/\sqrt n$ with $s$ computed with `ddof=1`, and the normal quantile $z = \Phi^{-1}(1 - \alpha/2)$.

</details>

<details><summary>hint 2: the approach</summary>

`z = NormalDist().inv_cdf(1 - alpha / 2)`, `half = z * np.std(x, ddof=1) / np.sqrt(len(x))`, and return `(mean - half, mean + half)` as floats.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
z = NormalDist().inv_cdf(1 - alpha / 2)
half = z * np.std(x, ddof=1) / np.sqrt(len(x))
m = float(np.mean(x))
return m - float(half), m + float(half)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

same layout as topic 01's harness: draw all the datasets at once, one per row. interval methods work on one dataset at a time, so loop over the rows.

</details>

<details><summary>hint 2: the approach</summary>

`data = sampler((n_reps, n), as_rng(rng))`. for each `row`, `lo, hi = ci_fn(row)`; count `lo <= theta <= hi` (closed interval) and add up `hi - lo`. divide both totals by `n_reps`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
rng = as_rng(rng)
data = np.asarray(sampler((n_reps, n), rng), dtype=float)
hits, widths = 0, 0.0
for row in data:
    lo, hi = ci_fn(row)
    hits += lo <= theta <= hi
    widths += hi - lo
return float(hits / n_reps), float(widths / n_reps)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

turn `data` into a tuple of arrays first (wrap a single array as `(data,)`), so one code path handles both cases. each replicate needs `n` indices drawn **with replacement**, shared by every array.

</details>

<details><summary>hint 2: the approach</summary>

in a loop over `n_boot`: `i = rng.integers(0, n, size=n)`, then `statistic(*(a[i] for a in arrays))`. for `bootstrap_ci`, validate `method` first, compute the replicates once, and for `"percentile"` return `np.quantile(reps, [alpha / 2, 1 - alpha / 2])`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
rng = as_rng(rng)
arrays = data if isinstance(data, tuple) else (data,)
arrays = tuple(np.asarray(a) for a in arrays)
n = len(arrays[0])
out = np.empty(n_boot)
for b in range(n_boot):
    i = rng.integers(0, n, size=n)
    out[b] = statistic(*(a[i] for a in arrays))
return out
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

the jackknife is a loop over `i` that removes row `i` from every array (`np.delete(a, i, axis=0)`). `bca_interval` is a direct transcription of idea 4's three formulas: $z_0$, then $a$, then the two adjusted levels.

</details>

<details><summary>hint 2: the approach</summary>

`p0 = (np.sum(reps < theta_hat) + np.sum(reps <= theta_hat)) / (2 * B)`, clipped to `[1/(2B), 1 - 1/(2B)]`, and `z0 = NormalDist().inv_cdf(p0)`. `d = np.mean(jack) - jack`, `a = np.sum(d**3) / (6 * np.sum(d**2) ** 1.5)` (or 0 when the denominator is 0). then for each `z` in the two normal quantiles, `NormalDist().cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))`, and `np.quantile(reps, levels)`. in `bootstrap_ci`, `theta_hat = statistic(*data_arrays)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
nd = NormalDist()
levels = [nd.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
          for z in (nd.inv_cdf(alpha / 2), nd.inv_cdf(1 - alpha / 2))]
lo, hi = np.quantile(reps, levels)
return float(lo), float(hi)
```

</details>
