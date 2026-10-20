# eigenvalues & eigenvectors · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`rng = np.random.default_rng(rng)` accepts `None`, an int, or an existing Generator, so start with that line. draw `v = rng.normal(size=n)` and normalize it.

</details>

<details><summary>hint 2: the approach</summary>

each iteration: `w = A @ v`, `lam = v @ w` (the Rayleigh quotient, since `v` is a unit vector), check the residual `norm(w - lam * v) <= tol * norm(w)` and break if it passes, otherwise `v = w / norm(w)`. computing the residual costs nothing extra because you need `w` anyway.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for _ in range(max_iters):
    w = A @ v
    lam = float(v @ w)
    if np.linalg.norm(w - lam * v) <= tol * max(np.linalg.norm(w), np.finfo(float).tiny):
        break
    v = w / np.linalg.norm(w)
return lam, v
```

the `max(..., tiny)` guards the case `A @ v = 0`, where the residual and the threshold are both zero.

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

it's stage 1 with "multiply by $A$" replaced by "solve with $A - \mu I$". factor *before* the loop.

</details>

<details><summary>hint 2: the approach</summary>

`factors = lu(A - mu * np.eye(n))`. if any diagonal entry of `factors[2]` (that's `U`) is exactly zero, add a tiny amount like `1e-10 * (1 + abs(mu))` to the shift and refactor. in the loop: `w = lu_solve(factors, v)`, normalize to get the new `v`, then `lam = v @ A @ v` and test the residual `norm(A v - lam v)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for _ in range(max_iters):
    w = lu_solve(factors, v)
    v = w / np.linalg.norm(w)
    Av = A @ v
    lam = float(v @ Av)
    if np.linalg.norm(Av - lam * v) <= tol * max(np.linalg.norm(Av), np.finfo(float).tiny):
        break
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

three lines inside a loop: a stopping test, `Q, R = qr(Ak)`, and `Ak = R @ Q`.

</details>

<details><summary>hint 2: the approach</summary>

the docstring's stopping test looks at the strictly lower part, `np.tril(Ak, -1)`, and compares its norm with `tol * np.linalg.norm(A)`. (that's one reasonable choice, the one the checks expect to converge in time. watching only the subdiagonal, or the change in the diagonal, also works.) after the loop, return `np.sort(np.diag(Ak))[::-1]`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
Ak = np.array(A, dtype=float)
scale = np.linalg.norm(Ak)
for _ in range(max_iters):
    if np.linalg.norm(np.tril(Ak, -1)) <= tol * scale:
        break
    Q, R = qr(Ak)
    Ak = R @ Q
return np.sort(np.diag(Ak))[::-1]
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

structure: `V = np.eye(n)`; an outer loop over sweeps (up to `max_iters`) with the stopping test at the top; inside, a double loop over pairs `p < q`; for each pair whose `A[p, q]` isn't negligible (skip it if `abs(A[p, q]) <= np.finfo(float).eps**2 * norm(A)`, which also keeps `tau` from overflowing), compute `c, s` and rotate.

</details>

<details><summary>hint 2: the approach</summary>

with `tau = (A[q, q] - A[p, p]) / (2 * A[p, q])`, `t = sign(tau) / (abs(tau) + np.hypot(1, tau))` (with sign(0) = +1), `c = 1 / np.hypot(1, t)`, `s = t * c`. (`np.hypot(1, tau)` is $\sqrt{1 + \tau^2}$ without overflowing when $\tau$ is huge, which happens when $a_{pq}$ is already tiny.) the rotation updates columns `p, q` of `A` (`A[:, p], A[:, q] = c*ap - s*aq, s*ap + c*aq`), then rows `p, q` the same way, then columns `p, q` of `V`. set `A[p, q] = A[q, p] = 0` exactly afterwards. the off-diagonal norm is `np.linalg.norm(A - np.diag(np.diag(A)))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
ap, aq = A[:, p].copy(), A[:, q].copy()
A[:, p], A[:, q] = c * ap - s * aq, s * ap + c * aq
ap, aq = A[p, :].copy(), A[q, :].copy()
A[p, :], A[q, :] = c * ap - s * aq, s * ap + c * aq
A[p, q] = A[q, p] = 0.0
vp, vq = V[:, p].copy(), V[:, q].copy()
V[:, p], V[:, q] = c * vp - s * vq, s * vp + c * vq
```

at the end: `w = np.diag(A)`, `order = np.argsort(w)`, return `w[order], V[:, order]`.

</details>
