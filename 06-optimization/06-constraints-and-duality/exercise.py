import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linalg import solve  # yours, from ch 02 · topic 03
    from mlfs.optim import OptimResult  # yours, from topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # constraints & duality · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end you can minimize over
    boxes, balls and the probability simplex, solve equality-constrained quadratic programs
    through their KKT system, and solve inequality-constrained ones through their dual, the
    route chapter 09's support vector machine takes.

    | stage | you build |
    |---|---|
    | 1 | `project_box`, `project_l2_ball`, `project_simplex`: the closest feasible point → `mlfs.optim` |
    | 2 | `projected_gradient_descent`: step, then project → `mlfs.optim` |
    | 3 | `solve_eq_qp`: an equality-constrained QP in one linear solve (the KKT system) → `mlfs.optim` |
    | 4 | `qp_dual`, `kkt_residuals`: the dual function of an inequality QP, and a certificate of optimality |
    | 5 | `solve_qp_dual`: solve the dual by projected gradient ascent, recover the primal, measure the gap |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def project_box(v, lower, upper):
    """stage 1: the Euclidean projection onto the box {lower <= x <= upper}: np.clip.

    lower and upper may be scalars or arrays broadcastable to v's shape (-np.inf / np.inf
    for no bound). returns a new float array.
    """
    raise NotImplementedError("stage1: implement project_box")


@app.function
def project_l2_ball(v, radius=1.0):
    """stage 1: the Euclidean projection onto the ball {||x||_2 <= radius}.

    v (n,): returned unchanged (as a new float array) if it's inside, otherwise scaled
    radially onto the sphere: v * radius / ||v||.
    """
    raise NotImplementedError("stage1: implement project_l2_ball")


@app.function
def project_simplex(v):
    """stage 1: the Euclidean projection of v (n,) onto the probability simplex
    {x : x >= 0, sum(x) = 1} (Duchi et al., 2008).

    the answer has the form x = max(v - theta, 0) for a threshold theta. sort u = v in
    DESCENDING order, let c = cumsum(u) - 1, find the largest index j (0-based) with
    u_j - c_j / (j + 1) > 0, set theta = c_j / (j + 1). O(n log n), no loop over candidates
    needed (vectorize). returns a new float array.
    """
    raise NotImplementedError("stage1: implement project_simplex")


@app.cell(hide_code=True)
def _():
    checks.run(1, project_box=project_box, project_l2_ball=project_l2_ball, project_simplex=project_simplex)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def projected_gradient_descent(f, grad, project, x0, *, lr=0.01, max_iters=1000, tol=1e-06, callback=None):
    """stage 2: minimize f over a convex set, given its projection.

    x0 is copied (it may be infeasible; every later iterate is feasible). each step:
        x_new = project(x - lr * grad(x))
    stop (converged) when ||x_new - x||_2 <= tol * lr, i.e. when the "gradient mapping"
    (x - x_new) / lr has norm <= tol: the constrained version of a small gradient. the step
    that meets the test is still taken (x <- x_new). also stop after max_iters steps or when
    f(x) is not finite. history = [f(x0), f(x1), ...]; callback(x, k) after step k (return value
    ignored); one grad
    call per step. returns OptimResult with converged = whether the test passed.
    """
    raise NotImplementedError("stage2: implement projected_gradient_descent")


@app.cell(hide_code=True)
def _():
    checks.run(2, projected_gradient_descent=projected_gradient_descent, project_box=project_box,
               project_l2_ball=project_l2_ball, project_simplex=project_simplex)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def solve_eq_qp(Q, c, A, b):
    """stage 3: min 1/2 x^T Q x + c^T x  subject to  A x = b, through the KKT system.

    Q (n, n) symmetric, c (n,), A (m, n), b (m,). with the Lagrangian
    L = 1/2 x^T Q x + c^T x + lam^T (A x - b), stationarity and feasibility are the linear
    system
        [[Q, A^T], [A, 0]] [x; lam] = [-c; b]
    solve it with mlfs.linalg.solve (it's symmetric but INDEFINITE: no Cholesky). returns
    (x (n,), lam (m,)). raises np.linalg.LinAlgError if the KKT matrix is singular.
    """
    raise NotImplementedError("stage3: implement solve_eq_qp")


@app.cell(hide_code=True)
def _():
    checks.run(3, solve_eq_qp=solve_eq_qp)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def qp_dual(Q, c, G, h, lam):
    """stage 4: the dual function of  min 1/2 x^T Q x + c^T x  s.t.  G x <= h  (Q positive definite).

    for multipliers lam (m,) >= 0, the Lagrangian 1/2 x^T Q x + c^T x + lam^T (G x - h) is
    minimized over x at x(lam) = -Q^{-1} (c + G^T lam) (use mlfs.linalg.solve, not an
    inverse), and the dual value is its minimum,
        g(lam) = -1/2 (c + G^T lam)^T Q^{-1} (c + G^T lam) - h^T lam.
    returns (g(lam) as a float, x(lam) (n,)). by weak duality g(lam) <= f(x) for every
    feasible x.
    """
    raise NotImplementedError("stage4: implement qp_dual")


@app.function
def kkt_residuals(Q, c, G, h, x, lam):
    """stage 4: how far (x, lam) is from satisfying the KKT conditions of the QP above.

    returns a dict of four nonnegative floats:
      "stationarity":    ||Q x + c + G^T lam||_2
      "primal":          max(0, max_i (G x - h)_i)       (worst constraint violation)
      "dual":            max(0, max_i (-lam_i))           (worst negative multiplier)
      "complementarity": max_i |lam_i (G x - h)_i|
    all four are 0 exactly at a primal-dual optimal pair (for a convex QP).
    """
    raise NotImplementedError("stage4: implement kkt_residuals")


@app.cell(hide_code=True)
def _():
    checks.run(4, qp_dual=qp_dual, kkt_residuals=kkt_residuals)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def solve_qp_dual(Q, c, G, h, *, lr=None, max_iters=20000, tol=1e-10):
    """stage 5: solve  min 1/2 x^T Q x + c^T x  s.t.  G x <= h  through its dual.

    maximize g(lam) over lam >= 0 by running your projected_gradient_descent on -g, with
    gradient -(G x(lam) - h) and projection project_box(lam, 0, inf), from lam = 0. the
    dual is a smooth concave quadratic with curvature G Q^{-1} G^T, so lr defaults to 1 / L
    with L its largest eigenvalue (np.linalg.eigvalsh is fine). pass max_iters and tol
    through. returns (x, lam, gap): the primal point x = x(lam), the multipliers, and the
    duality gap f(x) - g(lam) (about 0 at the optimum; it can be slightly negative while x
    is still slightly infeasible).
    """
    raise NotImplementedError("stage5: implement solve_qp_dual")


@app.cell(hide_code=True)
def _():
    checks.run(5, solve_qp_dual=solve_qp_dual, qp_dual=qp_dual, kkt_residuals=kkt_residuals)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your projections, `projected_gradient_descent` and
    `solve_eq_qp` are part of `mlfs.optim`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
