# entropic OT & sinkhorn · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`K = np.exp(-C / eps)`, `v = np.ones(m)`; inside a `for _ in range(max_iters)` loop: `u = a / (K @ v)`, `v = b / (K.T @ u)`, `P = u[:, None] * K * v[None, :]`, and `break` when `np.abs(P.sum(1) - a).sum() <= tol`. return `P` after the loop.

</details>

## stage 2

<details><summary>hint</summary>

precompute `la, lb = np.log(a), np.log(b)` and start `g = np.zeros(m)`. the two updates are `f = -eps * logsumexp(lb[None, :] + (g[None, :] - C) / eps, axis=1)` and `g = -eps * logsumexp(la[:, None] + (f[:, None] - C) / eps, axis=0)`; build `P = np.exp(la[:, None] + lb[None, :] + (f[:, None] + g[None, :] - C) / eps)` (every exponent is at most about 0 near convergence) and stop on the row error as in stage 1.

</details>

## stage 3

<details><summary>hint</summary>

for the primal, mask the positive entries (`pos = P > 0`) so `P log P` never sees a zero: `kl = np.sum(P[pos] * np.log(P[pos] / ab[pos])) - P.sum() + ab.sum()` with `ab = np.outer(a, b)`. the dual is one line: `f @ a + g @ b - eps * np.sum(ab * (np.exp((f[:, None] + g[None, :] - C) / eps) - 1))`.

</details>

## stage 4

<details><summary>hint</summary>

write a small inner function `ot(U, V, wu, wv)` that builds the cost (`pairwise_distances(U, V, squared=True)` for p = 2, else `pairwise_distances(U, V) ** p`), runs `sinkhorn_log(wu, wv, C, eps, max_iters=max_iters, tol=tol)` and returns `entropic_dual(f, g, wu, wv, C, eps)`. then `ot(X, Y, a, b) - ot(X, X, a, a) / 2 - ot(Y, Y, b, b) / 2`. reshape 1-D inputs to `(n, 1)` first.

</details>
