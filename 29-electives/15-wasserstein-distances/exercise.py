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
    # wasserstein distances · build

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
    raise NotImplementedError("stage1: implement wasserstein_1d")


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
    raise NotImplementedError("stage2: implement cost_matrix")


@app.function
def transport_constraints(n, m):
    """stage 2: (n + m, n * m) matrix A with A @ P.ravel() = concat(P.sum(1), P.sum(0)) for any (n, m) P (C order)."""
    raise NotImplementedError("stage2: implement transport_constraints")


@app.function
def kantorovich_lp(a, b, C):
    """stage 2: (P (n, m), cost <C, P>) solving min <C, P> over P >= 0 with marginals a, b, by scipy's linprog."""
    raise NotImplementedError("stage2: implement kantorovich_lp")


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
    raise NotImplementedError("stage3: implement northwest_corner")


@app.function
def transport_simplex(a, b, C, *, max_iters=10_000):
    """stage 3: (P, f, g): an optimal vertex plan and potentials with f[0] = 0; at most max_iters simplex pivots."""
    raise NotImplementedError("stage3: implement transport_simplex")


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
    raise NotImplementedError("stage4: implement barycentric_map")


@app.function
def gaussian_w2(m1, S1, m2, S2):
    """stage 4: W_2 (not squared) between N(m1, S1) and N(m2, S2): the Bures closed form."""
    raise NotImplementedError("stage4: implement gaussian_w2")


@app.function
def gaussian_monge_map(m1, S1, m2, S2):
    """stage 4: (A, t) with T(x) = x @ A + t the optimal map from N(m1, S1) to N(m2, S2); A symmetric (S1 invertible)."""
    raise NotImplementedError("stage4: implement gaussian_monge_map")


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
