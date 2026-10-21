# OT in machine learning · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

flatten and detach both batches to float64 (`x.detach().reshape(B, -1).double()`), build `C[k, i] = ((b[:, None] - a[None]) ** 2).sum(-1)` with the **data** in the rows, and call `_, cols = linear_sum_assignment(C.numpy())`. `cols[k]` is then the noise index that belongs in slot `k`, so return `x0[torch.as_tensor(cols)]`. `pairing_cost` is `((x0 - x1).reshape(B, -1) ** 2).sum(1).mean()` as a float.

</details>

## stage 2

<details><summary>hint</summary>

this time put the **noise** in the rows: `C[i, k] = |x0[i] − x1[k]|²` as a float64 NumPy array, `w = np.full(B, 1 / B)`, `P, _, _ = ot16.sinkhorn_log(w, w, C, eps, max_iters=max_iters, tol=tol)`. then `Q = torch.as_tensor((P / P.sum(0, keepdims=True)).T)`, `idx = torch.multinomial(Q, 1, replacement=True, generator=generator)[:, 0]`, and return `x0[idx]`.

</details>

## stage 3

<details><summary>hint</summary>

write an inner `ot(u, v)`: `C = ((u[:, None] - v[None]) ** 2).sum(-1)` (keeps the graph), `_, f, g = ot16.sinkhorn_log(a, b, C.detach().double().numpy(), eps, max_iters=max_iters, tol=tol)`, turn `f, g, a, b` into tensors of `C`'s dtype (no gradients), and return `(f * a).sum() + (g * b).sum() - eps * (a[:, None] * b[None] * (torch.exp((f[:, None] + g[None] - C) / eps) - 1)).sum()`. then `ot(x, y)`, minus `ot(x, x) / 2 + ot(y, y) / 2` when debiased.

</details>

## stage 4

<details><summary>hint</summary>

uniform weights `a = np.full(n_s, 1 / n_s)`, `b = np.full(n_t, 1 / n_t)`, squared-distance cost; `P, _ = ot15.kantorovich_lp(a, b, C)` when `eps is None`, else `P, _, _ = ot16.sinkhorn_log(a, b, C, eps, max_iters=max_iters, tol=1e-9)`; return `ot15.barycentric_map(P, Xt)`.

</details>
