# concentration inequalities · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

both bounds are one formula plus a clip. `np.minimum(1.0, ...)` clips elementwise and works whether its argument is a number or an array.

</details>

<details><summary>hint 2: the approach</summary>

convert inputs with `np.asarray(..., dtype=float)` so that lists and arrays behave. Markov is `mean / t`; Chebyshev is `var / t**2`. note that Chebyshev takes the *variance*, not the standard deviation.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def chebyshev_bound(var, t):
    return np.minimum(1.0, np.asarray(var, dtype=float) / np.asarray(t, dtype=float) ** 2)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`hoeffding_bound` is the formula $2\exp(-2nt^2/(b-a)^2)$, clipped to 1, written with `np.exp` so it works on arrays.

</details>

<details><summary>hint 2: the approach</summary>

for the sample size, set the bound equal to $\delta$ and solve: $n = (b - a)^2\log(2/\delta)/(2t^2)$. then `math.ceil` it, since $n$ must be a whole number and the bound only gets smaller as $n$ grows, and return `int(...)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def hoeffding_sample_size(t, delta, a=0.0, b=1.0):
    return int(math.ceil((b - a) ** 2 * math.log(2.0 / delta) / (2.0 * t**2)))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

same structure as `hoeffding_bound`, with a different exponent: $-nt^2/(2\sigma^2 + \tfrac23ct)$.

</details>

<details><summary>hint 2: the approach</summary>

convert `n` and `t` with `np.asarray`, compute `2.0 * np.exp(-n * t**2 / (2.0 * var + 2.0 * c * t / 3.0))`, and clip to 1. watch the parentheses: the whole denominator divides $nt^2$.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
n, t = np.asarray(n, dtype=float), np.asarray(t, dtype=float)
return np.minimum(1.0, 2.0 * np.exp(-n * t**2 / (2.0 * var + 2.0 * c * t / 3.0)))
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

for Chernoff, compute `q = p + t`, then the KL divergence, then `exp(-n * kl)`. the only difficulty is the edges: `q > 1` (return 0), and `q == 1`, where the second term $(1 - q)\log\frac{1-q}{1-p}$ is $0 \cdot \log 0$, which should count as 0.

</details>

<details><summary>hint 2: the approach</summary>

clip `q` into $[0, 1]$ for the computation, and build each KL term with `np.where(condition, formula, 0.0)` inside `with np.errstate(divide="ignore", invalid="ignore"):` (NumPy evaluates both branches of `np.where`, so the warnings are harmless). finish with `np.where(q > 1, 0.0, np.exp(-n * kl))`. for `empirical_tail`, compare a column of deviations against a row of thresholds, `dev[:, None] >= t[None, :]`, and average down the columns.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(samples, dtype=float)
dev = np.abs(x - mu) if two_sided else x - mu
t_arr = np.asarray(t, dtype=float)
out = np.mean(dev[:, None] >= t_arr.reshape(-1)[None, :], axis=0).reshape(t_arr.shape)
return float(out) if t_arr.ndim == 0 else out
```

</details>
