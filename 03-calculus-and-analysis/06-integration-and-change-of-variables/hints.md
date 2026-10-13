# integration & change of variables · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

build the $n$ cell centers along each axis separately (each axis has its own width $h_k$), combine them into every possible point with `np.meshgrid`, and evaluate `f` on all of them in one call.

</details>

<details><summary>hint 2: the approach</summary>

`low = np.atleast_1d(np.asarray(low, dtype=float))` (same for `high`) handles floats for $d = 1$. `h = (high - low) / n`. the centers on axis $k$ are `low[k] + (np.arange(n) + 0.5) * h[k]`. `np.meshgrid(*axes, indexing="ij")` gives $d$ arrays of shape $(n, \dots, n)$; `np.stack(..., axis=-1).reshape(-1, d)` turns them into an $(n^d, d)$ array of points. the answer is `np.sum(f(points)) * np.prod(h)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
h = (high - low) / n
axes = [lo + (np.arange(n) + 0.5) * hk for lo, hk in zip(low, h)]
grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, low.size)
return float(np.sum(f(grid)) * np.prod(h))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

sample all $n$ points at once as an $(n, d)$ array of uniforms, stretched into the box. the estimate is the box volume times the mean of `f`; the standard error is the volume times the sample standard deviation over $\sqrt n$.

</details>

<details><summary>hint 2: the approach</summary>

`rng = np.random.default_rng(rng)` accepts `None`, a seed, or a `Generator`. `pts = low + (high - low) * rng.random((n, d))`. `vals = f(pts)`, `V = np.prod(high - low)`. return `(V * vals.mean(), V * vals.std(ddof=1) / np.sqrt(n))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
rng = np.random.default_rng(rng)
low = np.atleast_1d(np.asarray(low, dtype=float))
high = np.atleast_1d(np.asarray(high, dtype=float))
vol = float(np.prod(high - low))
vals = f(low + (high - low) * rng.random((n, low.size)))
return vol * float(vals.mean()), vol * float(vals.std(ddof=1)) / np.sqrt(n)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

two library calls: one for the Jacobian (yours, from topic 02), one for the log-determinant.

</details>

<details><summary>hint 2: the approach</summary>

`J = numerical_jacobian(T, z)` with `z` as a float array. `np.linalg.slogdet(J)` returns `(sign, logabsdet)`; you want the second entry, as a float.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
J = numerical_jacobian(T, np.atleast_1d(np.asarray(z, dtype=float)))
return float(np.linalg.slogdet(J)[1])
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

for one point it's one line: the log-density of $Z$ at $T^{-1}(x)$ plus the log-Jacobian-determinant of $T^{-1}$ at $x$. for a batch, apply the one-point version to each row.

</details>

<details><summary>hint 2: the approach</summary>

convert `x` to a float array. if `x.ndim == 2`, return `np.array([...])` of the single-point results for each row. otherwise return `log_pz(T_inv(x)) + log_abs_det_jacobian(T_inv, x)`. note that the Jacobian is of `T_inv`, evaluated at `x`, not of $T$.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
if x.ndim == 2:
    return np.array([pushforward_log_density(log_pz, T_inv, xi) for xi in x])
return float(log_pz(T_inv(x)) + log_abs_det_jacobian(T_inv, x))
```

</details>
