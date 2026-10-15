# estimation · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

compute the mean of the estimates once and reuse it. bias compares that mean with `theta`; variance compares each estimate with that mean; MSE compares each estimate with `theta`.

</details>

<details><summary>hint 2: the approach</summary>

`center = est.mean()`, then `bias = center - theta`, `variance = np.mean((est - center) ** 2)` (that's ddof 0) and `mse = np.mean((est - theta) ** 2)`. wrap each in `float(...)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
est = np.asarray(estimates, dtype=float)
center = est.mean()
return {"bias": float(center - theta),
        "variance": float(np.mean((est - center) ** 2)),
        "mse": float(np.mean((est - theta) ** 2))}
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`simulate` is three lines: normalize the generator, draw all the data, apply the estimator. `evaluate` calls `simulate` and `bias_variance_mse`, then adds one entry.

</details>

<details><summary>hint 2: the approach</summary>

`rng = as_rng(rng)`, then `data = sampler((n_reps, n), rng)`: shape `(n_reps, n)`, one dataset per row. `estimator(data)` returns `(n_reps,)`. in `evaluate`, the squared errors are `(est - theta) ** 2`, and their standard error is `np.std(sq, ddof=1) / np.sqrt(n_reps)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def evaluate(estimator, sampler, theta, n, n_reps, *, rng=None):
    est = simulate(estimator, sampler, n, n_reps, rng=rng)
    out = bias_variance_mse(est, theta)
    sq = (est - theta) ** 2
    out["mse_se"] = float(np.std(sq, ddof=1) / np.sqrt(n_reps))
    return out
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

all three are one-liners with `axis=-1`, so they work on a single dataset `(n,)` and on a stack of datasets `(n_reps, n)`. the sample size is the length of the last axis.

</details>

<details><summary>hint 2: the approach</summary>

`2 * np.mean(x, axis=-1)`, `np.max(x, axis=-1)`, and `(n + 1) / n * np.max(x, axis=-1)` with `n = x.shape[-1]`. convert `x` with `np.asarray(x, dtype=float)` first.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def theta_unbiased_max(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    return (n + 1) / n * np.max(x, axis=-1)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

for `scaled_variance`, each row's deviations must be measured from *that row's* mean. `keepdims=True` keeps the mean's shape broadcastable against `x`. for `scaled_variance_mse`, write the bias and the variance from idea 4 and add `bias**2 + variance`.

</details>

<details><summary>hint 2: the approach</summary>

`dev = x - x.mean(axis=-1, keepdims=True)`, then `np.sum(dev**2, axis=-1) / c`. for the formula: `bias = sigma2 * ((n - 1) / c - 1)` and `var = 2 * sigma2**2 * (n - 1) / c**2`. convert `c` to a float array so that an array of divisors works.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def scaled_variance_mse(n, c, sigma2=1.0):
    c = np.asarray(c, dtype=float)
    bias = sigma2 * ((n - 1) / c - 1.0)
    var = 2.0 * sigma2**2 * (n - 1) / c**2
    out = bias**2 + var
    return float(out) if out.ndim == 0 else out
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`mse_curve` is a loop over `ns` that calls your `evaluate`. normalize `rng` once, before the loop, and pass the same generator to every call. `convergence_rate` is a straight-line fit in log–log coordinates.

</details>

<details><summary>hint 2: the approach</summary>

`np.polyfit(np.log(ns), np.log(values), 1)` returns `(slope, intercept)` of the least-squares line. the slope is the rate. don't use just the first and last points: that's not a least-squares fit, and it's much noisier.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def mse_curve(estimator, sampler, theta, ns, n_reps, *, rng=None):
    rng = as_rng(rng)
    return np.array([evaluate(estimator, sampler, theta, int(n), n_reps, rng=rng)["mse"] for n in ns])
```

</details>
