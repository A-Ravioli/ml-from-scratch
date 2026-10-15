# smoothness & lipschitz continuity · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

evaluate `f` once per point and stack the results into an array `F` of shape `(k, m)` (wrap each value in `np.atleast_1d` so scalar outputs become length 1). then you need two `(k, k)` matrices: input distances and output distances.

</details>

<details><summary>hint 2: the approach</summary>

`P[:, None, :] - P[None, :, :]` has shape `(k, k, n)`: all pairwise differences. take `np.linalg.norm(..., axis=-1)` for the distance matrix `dx`, and the same for `F` to get `df`. build a mask of the pairs to use: upper triangle only (`np.triu(..., k=1)`) and nonzero distance. return the max of `df / dx` over the mask.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
P = np.asarray(points, dtype=float)
F = np.array([np.atleast_1d(f(p)) for p in P])
dx = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
df = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1)
mask = np.triu(dx > 0, k=1)
return float(np.max(df[mask] / dx[mask]))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the smoothness estimate is the Lipschitz estimate of the *gradient*: one line, reusing stage 1. the strong-convexity estimate has the same pairwise structure, but with a dot product in the numerator and a squared distance in the denominator, and a minimum instead of a maximum.

</details>

<details><summary>hint 2: the approach</summary>

stack the gradients into `G` (shape `(k, n)`). with `D = P[:, None, :] - P[None, :, :]`, the numerator is `np.sum((G[:, None, :] - G[None, :, :]) * D, axis=-1)` and the denominator `np.sum(D * D, axis=-1)`. mask the upper triangle with nonzero denominators and take the minimum ratio.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
num = np.sum((G[:, None, :] - G[None, :, :]) * D, axis=-1)
den = np.sum(D * D, axis=-1)
mask = np.triu(den > 0, k=1)
return float(np.min(num[mask] / den[mask]))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

the gap is "bound minus function": write the right-hand side of the descent lemma, subtract $f(y)$. for local smoothness, you need every eigenvalue of the Hessian, and then the largest one in absolute value.

</details>

<details><summary>hint 2: the approach</summary>

with `d = y - x`: `f(x) + grad_f(x) @ d + 0.5 * L * (d @ d) - f(y)`. for `hessian_smoothness`: `np.linalg.eigvalsh(numerical_hessian(f, x))` gives the eigenvalues of a symmetric matrix; take `np.max(np.abs(...))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# descent_lemma_gap
d = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
return float(f(x) + np.asarray(grad_f(x)) @ d + 0.5 * L * (d @ d) - f(y))

# hessian_smoothness
return float(np.max(np.abs(np.linalg.eigvalsh(numerical_hessian(f, np.asarray(x, dtype=float))))))
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`gradient_descent` keeps a list of iterates, each a *new* array (never `x -= ...` on the caller's array). `largest_stable_step` runs it once per candidate step and judges each run by the final gradient norm.

</details>

<details><summary>hint 2: the approach</summary>

start with `xs = [np.array(x0, dtype=float)]` (a copy) and append `xs[-1] - step * grad_f(xs[-1])` `n_iters` times. for the threshold: compute `g0 = norm(grad_f(x0))`; for each `s` in `steps`, take the last iterate; it's stable if it's finite (`np.all(np.isfinite(last))`) and `norm(grad_f(last)) <= tol * g0`. keep the largest stable `s`. wrap the run in `np.errstate(over="ignore", invalid="ignore")` to silence overflow warnings from diverging runs.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
g0 = np.linalg.norm(grad_f(np.asarray(x0, dtype=float)))
best = 0.0
for s in steps:
    with np.errstate(over="ignore", invalid="ignore"):
        last = gradient_descent(grad_f, x0, s, n_iters)[-1]
        ok = np.all(np.isfinite(last)) and np.linalg.norm(grad_f(last)) <= tol * g0
    if ok:
        best = max(best, float(s))
return best
```

</details>
