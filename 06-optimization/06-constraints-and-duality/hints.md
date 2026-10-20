# constraints & duality · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

the box is `np.clip`; the ball needs one norm and one comparison. for the simplex, the answer is `max(v - theta, 0)`: everything is about finding `theta`, and the lesson's formula finds it from the sorted values and their running sums.

</details>

<details><summary>hint 2: the approach</summary>

`u = np.sort(v)[::-1]` (descending), `c = np.cumsum(u) - 1`, and the candidate thresholds `c / np.arange(1, n + 1)`. the positions where `u - c / np.arange(1, n + 1) > 0` form a prefix; the last one (`np.flatnonzero(...)[-1]`) is `j`, and `theta = c[j] / (j + 1)` (0-based `j`).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
v = np.asarray(v, dtype=float)
u = np.sort(v)[::-1]
c = np.cumsum(u) - 1.0
j = np.flatnonzero(u - c / np.arange(1, len(v) + 1) > 0)[-1]
theta = c[j] / (j + 1)
return np.maximum(v - theta, 0.0)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

this is `gradient_descent` with one extra function call per step, and a different stopping test: how far the projected step moved, compared with `tol * lr`. there's no gradient test before the first step.

</details>

<details><summary>hint 2: the approach</summary>

`x_new = project(x - lr * grad(x))`, `moved = norm(x_new - x)`, then *take* the step (`x = x_new`), update the history, `k` and the callback, and only then `break` with `converged = True` if `moved <= tol * lr`. loop `while k < max_iters`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
while k < max_iters:
    x_new = np.asarray(project(x - lr * np.asarray(grad(x), dtype=float)), dtype=float)
    moved = float(np.linalg.norm(x_new - x))
    x = x_new
    fx = float(f(x)); history.append(fx); k += 1
    if callback is not None:
        callback(x, k)
    if not np.isfinite(fx):
        break
    if moved <= tol * lr:
        converged = True
        break
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

stack the matrix from four blocks and the right-hand side from two vectors, solve once, and split the solution into its first $n$ and last $m$ entries.

</details>

<details><summary>hint 2: the approach</summary>

`K = np.block([[Q, A.T], [A, np.zeros((m, m))]])`, `rhs = np.concatenate([-c, b])`, `z = solve(K, rhs)`, return `z[:n], z[n:]`. use `np.atleast_2d(A)` and `np.atleast_1d(b)` so a single constraint works too.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
A = np.atleast_2d(np.asarray(A, dtype=float))
n, m = Q.shape[0], A.shape[0]
K = np.block([[Q, A.T], [A, np.zeros((m, m))]])
z = np.asarray(solve(K, np.concatenate([-c, np.atleast_1d(b)])), dtype=float)
return z[:n], z[n:]
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

both functions start from the same vector $r = c + G^\top\lambda$. $x(\lambda)$ solves $Qx = -r$, and the dual value is the Lagrangian at that $x$, which simplifies to $\tfrac12r^\top x - h^\top\lambda$. the residuals are four one-liners built from the slack $Gx - h$.

</details>

<details><summary>hint 2: the approach</summary>

`r = c + G.T @ lam`, `x = -solve(Q, r)`, `g = 0.5 * r @ x - h @ lam` (check: $\tfrac12r^\top x = -\tfrac12r^\top Q^{-1}r$). for the residuals, `slack = G @ x - h`; stationarity is a norm, primal `max(0, max(slack))`, dual `max(0, max(-lam))`, complementarity `max(abs(lam * slack))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
slack = G @ x - h
return {
    "stationarity": float(np.linalg.norm(Q @ x + c + G.T @ lam)),
    "primal": float(max(0.0, np.max(slack))),
    "dual": float(max(0.0, np.max(-lam))),
    "complementarity": float(np.max(np.abs(lam * slack))),
}
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

you already have everything: `qp_dual` gives $g$ and $x(\lambda)$, the dual gradient is $Gx(\lambda) - h$, and `projected_gradient_descent` minimizes. so minimize $-g$, with gradient $-(Gx(\lambda) - h)$, over the box $\lambda \ge 0$.

</details>

<details><summary>hint 2: the approach</summary>

default `lr = 1 / eigvalsh(G @ solve(Q, G.T))[-1]`. define `neg_dual(lam) = -qp_dual(...)[0]` and `neg_dual_grad(lam) = -(G @ qp_dual(...)[1] - h)`, project with `lambda lam: project_box(lam, 0.0, np.inf)`, start from `np.zeros(m)`, and pass `max_iters` and `tol` through. finally recompute `g, x = qp_dual(..., lam)` and the gap `f(x) - g`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
res = projected_gradient_descent(neg_dual, neg_dual_grad, lambda lam: project_box(lam, 0.0, np.inf),
                                 np.zeros(G.shape[0]), lr=lr, max_iters=max_iters, tol=tol)
lam = res.x
g_val, x = qp_dual(Q, c, G, h, lam)
return x, lam, float(0.5 * x @ Q @ x + c @ x) - g_val
```

</details>
