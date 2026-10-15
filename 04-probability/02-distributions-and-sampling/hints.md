# distributions & sampling · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`rng.random(size)` already does the `size` bookkeeping for you: `size=None` gives one float, an int or tuple gives an array of that shape. so every sampler in this stage is "draw `rng.random(size)`, transform it elementwise". convert to a Python float at the end when `size is None`.

</details>

<details><summary>hint 2: the approach</summary>

`uniform`: `low + (high - low) * u`. `inverse_cdf_sample`: draw `u`, replace exact zeros with `np.where(u > 0, u, 2.0**-54)`, return `inv_cdf(u)`. `exponential`: call `inverse_cdf_sample` with the inverse CDF from the lesson, $-\ln(1 - u)/\lambda$ (`-np.log1p(-u) / rate` is the accurate way to write it for small `u`).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def inverse_cdf_sample(inv_cdf, size=None, *, rng=None):
    rng = np.random.default_rng(rng)
    u = rng.random(size)
    u = np.where(u > 0.0, u, 2.0**-54)
    x = inv_cdf(u)
    return float(x) if size is None else np.asarray(x, dtype=float)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`bernoulli` is a comparison: `rng.random(size) < p` is a boolean array, and `np.asarray(..., dtype=np.int64)` turns it into 0s and 1s (it also works when `size=None` gives a single bool). `categorical` is the discrete inverse CDF: cumulative sums plus `np.searchsorted`.

</details>

<details><summary>hint 2: the approach</summary>

`categorical`: `cdf = np.cumsum(probs)`, `u = rng.random(size) * cdf[-1]`, then `np.searchsorted(cdf, u, side="right")`. the scaling keeps `u` below the last boundary even when the cumsum rounds to `0.9999999999999999`.

`poisson`, Knuth: draw everything as flat arrays of length `n = prod(size)` and reshape at the end. keep a running product per draw and a boolean mask of draws still "active" (product above `exp(-lam)`). while any are active, add 1 to their counts and multiply their products by fresh uniforms. large `lam`: build `k = np.arange(K + 1)`, the log-pmf `k log(lam) - lam - log(k!)` (with `log(k!)` as a cumulative sum of `log(1..k)`), exponentiate, normalize, and call your `categorical`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
threshold = math.exp(-lam)
counts = np.zeros(n, dtype=np.int64)
prod = rng.random(n)
active = prod > threshold
while active.any():
    counts[active] += 1
    prod[active] *= rng.random(int(active.sum()))
    active = prod > threshold
```

the first uniform is drawn before the loop: a draw whose very first uniform is already at or below the threshold has count 0.

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`box_muller` is three lines straight from the formula, and NumPy's `np.log`, `np.sqrt`, `np.cos`, `np.sin` work elementwise, so arrays come for free.

</details>

<details><summary>hint 2: the approach</summary>

in `normal`, let `n = 1 if size is None else int(np.prod(size))` and `m = (n + 1) // 2` pairs. draw `u1 = 1.0 - rng.random(m)` (in $(0, 1]$, so the log is finite) and `u2 = rng.random(m)`, call `box_muller`, `np.concatenate` the two halves, keep the first `n`, then scale, shift, and reshape to `size`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
z1, z2 = box_muller(1.0 - rng.random(m), rng.random(m))
z = np.concatenate([z1, z2])[:n]
x = loc + scale * z
return float(x[0]) if size is None else x.reshape(size)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

one round: propose `m` points with `x = sample_proposal(m, rng)`, compute `log_ratio = log_target(x) - log_M - log_proposal(x)`, draw `m` uniforms, and keep the points with `np.log(u) < log_ratio`. repeat rounds until you have `size` points, then cut to exactly `size`.

</details>

<details><summary>hint 2: the approach</summary>

collect each round's accepted points in a list and `np.concatenate` at the end (this works for 1-D draws and for `(m, d)` draws alike). before accepting, check `np.any(log_ratio > 1e-9)` and raise `ValueError` if so: the envelope is too low. choose `m` from how many draws you still need divided by the acceptance rate you've observed so far (start from a guess like $e^{-\log M}$), plus a few extra.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
accepted, n_got, n_tried = [], 0, 0
rate = min(1.0, math.exp(-log_M))
while n_got < size:
    m = int(math.ceil(1.2 * (size - n_got) / max(rate, 1e-4))) + 16
    x = sample_proposal(m, rng)
    log_ratio = np.asarray(log_target(x)) - log_M - np.asarray(log_proposal(x))
    if np.any(log_ratio > 1e-9):
        raise ValueError("envelope too low: increase log_M")
    keep = np.log(rng.random(m)) < log_ratio
    accepted.append(np.asarray(x)[keep])
    n_got, n_tried = n_got + int(keep.sum()), n_tried + m
    rate = max(n_got, 1) / n_tried
return np.concatenate(accepted)[:size]
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

Marsaglia–Tsang is a rejection loop like stage 4, but with its own proposal: draw a batch of normals `z` (your `normal`) and uniforms `u` (your `uniform`), compute `v = (1 + c z)^3`, and keep the entries that pass both tests. fill an output array until it has `n` entries.

</details>

<details><summary>hint 2: the approach</summary>

`v` can be $\le 0$, and `np.log(v)` would then warn. compute the log on a safe copy, `vs = np.where(v > 0, v, 1.0)`, and combine the masks: `ok = (v > 0) & (np.log(u) < 0.5 * z**2 + d - d * vs + d * np.log(vs))`. the accepted draws are `d * v[ok]`. for `shape < 1`, call your own `gamma(shape + 1, ...)` and multiply by `u ** (1 / shape)` with `u = 1 - uniform(...)`. `beta` is two calls to `gamma` with scale 1 and one division.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
d = shape - 1.0 / 3.0
c = 1.0 / math.sqrt(9.0 * d)
out, filled = np.empty(n), 0
while filled < n:
    m = n - filled + 8
    z = normal(0.0, 1.0, m, rng=rng)
    u = uniform(0.0, 1.0, m, rng=rng)
    v = (1.0 + c * z) ** 3
    vs = np.where(v > 0, v, 1.0)
    ok = (v > 0) & (np.log(u) < 0.5 * z**2 + d - d * vs + d * np.log(vs))
    got = d * v[ok][: n - filled]
    out[filled:filled + len(got)] = got
    filled += len(got)
```

</details>
