import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.autodiff import numerical_hessian  # yours, from ch 03 · topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # convex sets & functions · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end you have a numerical
    convexity checker: four independent ways of hunting for a proof that a set or a function
    is *not* convex, and an honest verdict when the hunt comes up empty.

    | stage | you build |
    |---|---|
    | 1 | `in_polyhedron`, `set_convexity_violation`: membership in $\{Ax \le b\}$, and a search for chords that leave a set |
    | 2 | `jensen_gap`, `min_jensen_gap`: the definition of a convex function, tested on every pair of points |
    | 3 | `first_order_gap`, `min_first_order_gap`: does the function stay above its tangent planes? |
    | 4 | `min_hessian_eigenvalue`, `check_convexity`: the second-order test, and the checker's verdict |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def in_polyhedron(A: np.ndarray, b: np.ndarray, X: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """stage 1: which rows of X lie in the polyhedron {x : A x <= b}?

    A (m, n), b (m,), X (k, n). a point is inside when every one of the m inequalities holds
    up to tol: (A x)_i <= b_i + tol. returns a bool array (k,). vectorized: no Python loop.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.all(X @ np.asarray(A, dtype=float).T <= np.asarray(b, dtype=float) + tol, axis=1)


@app.function
def set_convexity_violation(contains, points: np.ndarray, ts=(0.25, 0.5, 0.75)):
    """stage 1: search for a chord that leaves the set: a witness that it is NOT convex.

    contains(x) -> bool tests membership of one point x (n,); points (k, n) are candidate
    points (rows with contains(row) False are not in the set and are skipped). for every pair
    i < j of member rows (i outer loop, then j, then t in the order given) test the point
    z = t x_i + (1 - t) x_j. returns the first (i, j, t) with contains(z) False, with i, j
    indices into `points` and t a float, or None if every tested point is in the set.
    """
    P = np.asarray(points, dtype=float)
    members = [i for i in range(len(P)) if contains(P[i])]
    for a, i in enumerate(members):
        for j in members[a + 1:]:
            for t in ts:
                if not contains(t * P[i] + (1 - t) * P[j]):
                    return (i, j, float(t))
    return None


@app.cell(hide_code=True)
def _():
    checks.run(1, in_polyhedron=in_polyhedron, set_convexity_violation=set_convexity_violation)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def jensen_gap(f, x: np.ndarray, y: np.ndarray, t: float) -> float:
    """stage 2: t f(x) + (1 - t) f(y) - f(t x + (1 - t) y).

    the chord minus the function, at the point a fraction t of the way from y to x.
    f is convex iff this is >= 0 for every x, y and every t in [0, 1]. returns a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(t * f(x) + (1 - t) * f(y) - f(t * x + (1 - t) * y))


@app.function
def min_jensen_gap(f, points: np.ndarray, ts=(0.25, 0.5, 0.75)):
    """stage 2: the most negative Jensen gap over all pairs i < j of rows and all t in ts.

    points (k, n), k >= 2. evaluate f ONCE per row (reuse those values for the chord ends),
    and once per (pair, t) for the interior point: at most k + (k(k - 1)/2) len(ts) calls.
    returns (gap, (i, j, t)) at the minimum; on ties keep the first in the loop order i, j, t.
    a negative gap proves f is not convex.
    """
    P = np.asarray(points, dtype=float)
    F = [float(f(p)) for p in P]
    best, where = np.inf, None
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            for t in ts:
                gap = t * F[i] + (1 - t) * F[j] - float(f(t * P[i] + (1 - t) * P[j]))
                if gap < best:
                    best, where = gap, (i, j, float(t))
    return float(best), where


@app.cell(hide_code=True)
def _():
    checks.run(2, jensen_gap=jensen_gap, min_jensen_gap=min_jensen_gap)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def first_order_gap(f, grad, x: np.ndarray, y: np.ndarray) -> float:
    """stage 3: f(y) - f(x) - grad(x) . (y - x): how far f at y sits above the tangent plane at x.

    for differentiable f this is >= 0 for all x, y iff f is convex. returns a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(f(y) - f(x) - np.asarray(grad(x), dtype=float) @ (y - x))


@app.function
def min_first_order_gap(f, grad, points: np.ndarray):
    """stage 3: the smallest first-order gap over all ORDERED pairs (i, j), i != j.

    gap[i, j] = f(x_j) - f(x_i) - grad(x_i) . (x_j - x_i): the tangent plane at x_i, tested
    at x_j. points (k, n). evaluate f and grad once per row, then build the (k, k) matrix of
    gaps with broadcasting (no loop over pairs). returns (gap, (i, j)) at the minimum (the
    first in row-major order on ties).
    """
    P = np.asarray(points, dtype=float)
    F = np.array([float(f(p)) for p in P])
    G = np.array([np.asarray(grad(p), dtype=float) for p in P])
    # gaps[i, j] = F[j] - F[i] - G[i] . (P[j] - P[i])
    gaps = F[None, :] - F[:, None] - (G @ P.T - np.sum(G * P, axis=1)[:, None])
    np.fill_diagonal(gaps, np.inf)
    i, j = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
    return float(gaps[i, j]), (int(i), int(j))


@app.cell(hide_code=True)
def _():
    checks.run(3, first_order_gap=first_order_gap, min_first_order_gap=min_first_order_gap)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def min_hessian_eigenvalue(f, points: np.ndarray, *, h: float = 1e-4):
    """stage 4: the smallest Hessian eigenvalue of f over the given points.

    at each row x of points (k, n), compute numerical_hessian(f, x, h=h) (from mlfs.autodiff)
    and its smallest eigenvalue (np.linalg.eigvalsh). returns (lam_min, index of that row).
    lam_min < 0 somewhere proves f is not convex (up to finite-difference error).
    """
    P = np.asarray(points, dtype=float)
    lams = [float(np.linalg.eigvalsh(numerical_hessian(f, p, h=h))[0]) for p in P]
    k = int(np.argmin(lams))
    return lams[k], k


@app.function
def check_convexity(f, points: np.ndarray, *, ts=(0.25, 0.5, 0.75), tol: float = 1e-9):
    """stage 4: a numerical convexity checker. it can REFUTE convexity, never prove it.

    run min_jensen_gap over the points. with scale = 1 + max |f(row)|, the verdict is
    (False, (x, y, t)) if the most negative gap is below -tol * scale, where x, y are COPIES
    of the two rows (float arrays) and t the float from min_jensen_gap; otherwise
    (True, None), meaning "no violation found", which is not a proof.
    (the relative tolerance keeps rounding error in f from counting as a violation.)
    """
    P = np.asarray(points, dtype=float)
    gap, where = min_jensen_gap(f, P, ts)
    scale = 1.0 + max(abs(float(f(p))) for p in P)
    if gap < -tol * scale:
        i, j, t = where
        return False, (P[i].copy(), P[j].copy(), float(t))
    return True, None


@app.cell(hide_code=True)
def _():
    checks.run(4, min_hessian_eigenvalue=min_hessian_eigenvalue, check_convexity=check_convexity)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
