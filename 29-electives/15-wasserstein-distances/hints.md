# wasserstein distances · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

sort each sample with `np.argsort` and apply the same order to its weights; normalize the weights; take `cu = np.cumsum(wu)` and `cv = np.cumsum(wv)`. the breakpoints are `t = np.unique(np.concatenate([[0], cu, cv, [1]]))`. on each piece $(t_{k-1}, t_k]$ both quantile functions are constant: evaluate them at the midpoint with `u[np.searchsorted(cu, mid)]` (clip the index to `len(u) - 1` against rounding at $t = 1$), then sum `width * |qu - qv| ** p` and take the $p$-th root.

</details>

## stage 2

<details><summary>hint</summary>

`mlfs.utils.pairwise_distances(X, Y, squared=True)` gives the squared cost; reshape 1-D inputs to `(n, 1)` first. the constraint matrix is `np.vstack([np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))])`; check it on a random `P` before using it. then `linprog(C.ravel(), A_eq=A, b_eq=np.concatenate([a, b]), bounds=(0, None), method="highs")` and `res.x.reshape(n, m)`.

</details>

## stage 3

<details><summary>hint</summary>

keep the basis as a list of cells and rebuild an adjacency list each iteration (nodes `0..n-1` are rows, `n..n+m-1` columns; a cell $(i, j)$ is an edge between `i` and `n + j`). potentials: a breadth-first walk from row 0, setting the other end of each edge to $C_{ij}$ minus the known end. entering cell: `np.flatnonzero((C - f[:, None] - g[None, :]).ravel() < -tol)[0]`, then `divmod(k, m)`. the cycle: find the tree path from column node `n + j_e` to row node `i_e` (BFS with parent pointers); its edges, in order, are the cycle's $-, +, -, \dots, -$ cells. $\theta$ is the smallest mass on the $-$ cells; leave the one with the smallest flat index among those that reach 0, set it to exactly 0 and swap it for the entering cell. loop `max_iters + 1` times (the last time only to check optimality).

</details>

## stage 4

<details><summary>hint</summary>

`barycentric_map` is `(P @ Y) / P.sum(1)[:, None]` (no `[:, None]` for 1-D `Y`). for a symmetric PSD square root, `w, V = np.linalg.eigh(S)` and `(V * np.sqrt(np.clip(w, 0, None))) @ V.T`. compute `r = sqrt(S1)` once, then `sqrt(r @ S2 @ r)`; the map's matrix is `inv(r) @ sqrt(r @ S2 @ r) @ inv(r)` (symmetrize it against rounding) and `t = m2 - m1 @ A` (rows: `x @ A + t`; since `A` is symmetric that's also $m_2 + A(x - m_1)$ on columns).

</details>
