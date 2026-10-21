import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from collections import deque

    import marimo as mo
    import numpy as np
    from scipy.optimize import linprog

    from mlfs import checks, lesson
    from mlfs.utils import pairwise_distances  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # wasserstein distances · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `wasserstein_1d`: $W_p$ between 1-D distributions from their quantile functions |
    | 2 | `cost_matrix`, `transport_constraints`, `kantorovich_lp`: the Kantorovich problem as a linear program |
    | 3 | `northwest_corner`, `transport_simplex`: an exact transport solver from scratch, with dual potentials |
    | 4 | `barycentric_map`, `gaussian_w2`, `gaussian_monge_map`: Monge maps, and the Gaussian closed form |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def wasserstein_1d(u, v, p=1, *, u_weights=None, v_weights=None):
    """stage 1: W_p (not W_p^p) between the 1-D distributions sum_i w_i delta_{u_i} and sum_j w'_j delta_{v_j}."""
    u, v = np.asarray(u, dtype=float).ravel(), np.asarray(v, dtype=float).ravel()
    wu = np.ones(len(u)) if u_weights is None else np.asarray(u_weights, dtype=float).ravel()
    wv = np.ones(len(v)) if v_weights is None else np.asarray(v_weights, dtype=float).ravel()
    iu, iv = np.argsort(u, kind="stable"), np.argsort(v, kind="stable")
    u, wu, v, wv = u[iu], wu[iu] / wu.sum(), v[iv], wv[iv] / wv.sum()
    cu, cv = np.cumsum(wu), np.cumsum(wv)
    t = np.unique(np.concatenate([[0.0], cu, cv, [1.0]]))
    t = t[(t >= 0) & (t <= 1)]
    mids, widths = (t[1:] + t[:-1]) / 2, np.diff(t)
    qu = u[np.minimum(np.searchsorted(cu, mids, side="left"), len(u) - 1)]
    qv = v[np.minimum(np.searchsorted(cv, mids, side="left"), len(v) - 1)]
    return float(np.sum(widths * np.abs(qu - qv) ** p) ** (1 / p))


@app.cell(hide_code=True)
def _():
    checks.run(1, wasserstein_1d=wasserstein_1d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def cost_matrix(X, Y, p=2):
    """stage 2: (n, m) matrix of ||x_i - y_j||^p (Euclidean); 1-D inputs (n,) are points on the line."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    Y = Y[:, None] if Y.ndim == 1 else Y
    if p == 2:
        return pairwise_distances(X, Y, squared=True)
    return pairwise_distances(X, Y) ** p


@app.function
def transport_constraints(n, m):
    """stage 2: (n + m, n * m) matrix A with A @ P.ravel() = concat(P.sum(1), P.sum(0)) for any (n, m) P (C order)."""
    rows = np.kron(np.eye(n), np.ones((1, m)))
    cols = np.kron(np.ones((1, n)), np.eye(m))
    return np.vstack([rows, cols])


@app.function
def kantorovich_lp(a, b, C):
    """stage 2: (P (n, m), cost <C, P>) solving min <C, P> over P >= 0 with marginals a, b, by scipy's linprog."""
    a, b, C = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(C, dtype=float)
    if not np.isclose(a.sum(), b.sum(), rtol=1e-9, atol=1e-12):
        raise ValueError(f"a and b must have the same total mass, got {a.sum()} and {b.sum()}")
    n, m = C.shape
    res = linprog(C.ravel(), A_eq=transport_constraints(n, m), b_eq=np.concatenate([a, b]),
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"linprog failed: {res.message}")
    P = res.x.reshape(n, m)
    return P, float(np.sum(P * C))


@app.cell(hide_code=True)
def _():
    checks.run(2, cost_matrix=cost_matrix, transport_constraints=transport_constraints,
               kantorovich_lp=kantorovich_lp)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def northwest_corner(a, b):
    """stage 3: (P (n, m), basis): the north-west corner starting plan and its n + m - 1 basic cells (i, j), in order."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    ra, rb = a.copy(), b.copy()
    P = np.zeros((n, m))
    basis = []
    i = j = 0
    while True:
        x = max(min(ra[i], rb[j]), 0.0)
        P[i, j] = x
        ra[i] -= x
        rb[j] -= x
        basis.append((i, j))
        if i == n - 1 and j == m - 1:
            break
        if i == n - 1:
            j += 1
        elif j == m - 1:
            i += 1
        elif ra[i] <= rb[j]:
            i += 1
        else:
            j += 1
    return P, basis


@app.function
def transport_simplex(a, b, C, *, max_iters=10_000):
    """stage 3: (P, f, g): an optimal vertex plan and potentials with f[0] = 0; at most max_iters simplex pivots."""
    a, b, C = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(C, dtype=float)
    if not np.isclose(a.sum(), b.sum(), rtol=1e-9, atol=1e-12):
        raise ValueError(f"a and b must have the same total mass, got {a.sum()} and {b.sum()}")
    n, m = C.shape

    def tree(basis):  # adjacency of the basis tree: nodes 0..n-1 are rows, n..n+m-1 columns
        adj = [[] for _ in range(n + m)]
        for i, j in basis:
            adj[i].append(n + j)
            adj[n + j].append(i)
        return adj

    def potentials(adj):  # f[0] = 0 and f_i + g_j = C_ij on every basic cell
        pot = np.full(n + m, np.nan)
        pot[0] = 0.0
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if np.isnan(pot[w]):
                    i, j = (u, w - n) if u < n else (w, u - n)
                    pot[w] = C[i, j] - pot[u]
                    queue.append(w)
        return pot[:n], pot[n:]

    def path(adj, start, goal):  # the nodes on the tree path from start to goal
        parent = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if w not in parent:
                    parent[w] = u
                    queue.append(w)
        out, u = [], goal
        while u is not None:
            out.append(u)
            u = parent[u]
        return out[::-1]

    P, basis = northwest_corner(a, b)
    tol = 1e-9 * (1.0 + np.abs(C).max())
    for it in range(max_iters + 1):
        adj = tree(basis)
        f, g = potentials(adj)
        reduced = C - f[:, None] - g[None, :]
        neg = np.flatnonzero(reduced.ravel() < -tol)
        if len(neg) == 0:
            return P, f, g
        if it == max_iters:
            break
        ie, je = divmod(int(neg[0]), m)  # Bland: the entering cell with the smallest flat index
        nodes = path(adj, n + je, ie)  # column je ... row ie; with (ie, je) it closes a cycle
        cells = [(u, w - n) if u < n else (w, u - n) for u, w in zip(nodes[:-1], nodes[1:])]
        minus = cells[0::2]  # the cycle alternates -, +, -, ... starting next to the entering cell
        plus = cells[1::2]
        theta = min(P[c] for c in minus)
        leave = min((c for c in minus if P[c] <= theta), key=lambda c: c[0] * m + c[1])
        for c in minus:
            P[c] -= theta
        for c in plus:
            P[c] += theta
        P[ie, je] = theta
        P[leave] = 0.0
        basis.remove(leave)
        basis.append((ie, je))
    raise RuntimeError(f"transport_simplex: plan still not optimal after max_iters={max_iters} pivots")


@app.cell(hide_code=True)
def _():
    checks.run(3, northwest_corner=northwest_corner, transport_simplex=transport_simplex)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def barycentric_map(P, Y):
    """stage 4: (n, d) images T(x_i) = sum_j P_ij y_j / sum_j P_ij of a plan P (n, m) onto targets Y (m, d) or (m,)."""
    P, Y = np.asarray(P, dtype=float), np.asarray(Y, dtype=float)
    return (P @ Y) / P.sum(1).reshape((-1,) + (1,) * (Y.ndim - 1))


@app.function
def gaussian_w2(m1, S1, m2, S2):
    """stage 4: W_2 (not squared) between N(m1, S1) and N(m2, S2): the Bures closed form."""
    m1, m2 = np.atleast_1d(np.asarray(m1, dtype=float)), np.atleast_1d(np.asarray(m2, dtype=float))
    S1, S2 = np.atleast_2d(np.asarray(S1, dtype=float)), np.atleast_2d(np.asarray(S2, dtype=float))

    def sqrtm(S):  # symmetric PSD square root by eigendecomposition
        w, V = np.linalg.eigh((S + S.T) / 2)
        return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

    r1 = sqrtm(S1)
    cross = sqrtm(r1 @ S2 @ r1)
    w2 = np.sum((m1 - m2) ** 2) + np.trace(S1) + np.trace(S2) - 2 * np.trace(cross)
    return float(np.sqrt(max(w2, 0.0)))


@app.function
def gaussian_monge_map(m1, S1, m2, S2):
    """stage 4: (A, t) with T(x) = x @ A + t the optimal map from N(m1, S1) to N(m2, S2); A symmetric (S1 invertible)."""
    m1, m2 = np.atleast_1d(np.asarray(m1, dtype=float)), np.atleast_1d(np.asarray(m2, dtype=float))
    S1, S2 = np.atleast_2d(np.asarray(S1, dtype=float)), np.atleast_2d(np.asarray(S2, dtype=float))

    def sqrtm(S):
        w, V = np.linalg.eigh((S + S.T) / 2)
        return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

    r1 = sqrtm(S1)
    r1_inv = np.linalg.inv(r1)
    A = r1_inv @ sqrtm(r1 @ S2 @ r1) @ r1_inv
    A = (A + A.T) / 2
    return A, m2 - m1 @ A


@app.cell(hide_code=True)
def _():
    checks.run(4, barycentric_map=barycentric_map, gaussian_w2=gaussian_w2, gaussian_monge_map=gaussian_monge_map)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it pits the Wasserstein
    distance against KL and JS on distributions that don't overlap, races your exact solver
    against a production LP solver, and measures how slowly empirical Wasserstein distances
    converge in high dimension. topic 16 makes OT fast by smoothing it.
    """)
    return


if __name__ == "__main__":
    app.run()
