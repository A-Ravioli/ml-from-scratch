import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # sequences, limits & convergence · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end you'll have an
    empirical convergence-rate estimator and three root finders to point it at.

    | stage | you build |
    |---|---|
    | 1 | `tail_index`: the $N$ in "for every $\varepsilon$ there is an $N$" |
    | 2 | `cauchy_diameters`: how spread out each tail of a sequence is |
    | 3 | `order_estimates`, `rate_estimates`: read the order and rate of convergence off a list of errors |
    | 4 | `bisection`, `newton_1d`, `secant`: three root finders that record their iterates |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def tail_index(a: np.ndarray, limit: float, eps: float) -> int:
    """stage 1: the smallest N such that |a[n] - limit| < eps for EVERY n >= N.

    a is a 1-D float array (K,) holding the first K terms of a sequence (index 0 first).
    returns an int in 0..K. the inequality is strict. if even the last term is not within eps,
    return K. note: this is not the first index that lands within eps, because a sequence
    can wander back out.
    """
    a = np.asarray(a, dtype=float)
    outside = np.flatnonzero(~(np.abs(a - limit) < eps))
    return 0 if outside.size == 0 else int(outside[-1] + 1)


@app.cell(hide_code=True)
def _():
    checks.run(1, tail_index=tail_index)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def cauchy_diameters(a: np.ndarray) -> np.ndarray:
    """stage 2: the diameter of every tail of the sequence a (K,).

    returns d (K,) with d[N] = max over m, n >= N of |a[m] - a[n]| = max(a[N:]) - min(a[N:]).
    d is non-increasing and d[K-1] = 0. vectorize it: running max/min from the right
    (np.maximum.accumulate on the reversed array), no Python loop over N.
    """
    a = np.asarray(a, dtype=float)
    rev = a[::-1]
    return (np.maximum.accumulate(rev) - np.minimum.accumulate(rev))[::-1]


@app.cell(hide_code=True)
def _():
    checks.run(2, cauchy_diameters=cauchy_diameters)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def order_estimates(errors: np.ndarray) -> np.ndarray:
    """stage 3: estimates of the order of convergence q from consecutive errors.

    errors (n,) are positive, e_k = |x_k - x*|. returns (n - 2,) with
    q_k = log(e_{k+2} / e_{k+1}) / log(e_{k+1} / e_k).
    raises ValueError if any error is <= 0 (drop exact hits before calling).
    """
    e = np.asarray(errors, dtype=float)
    if np.any(e <= 0):
        raise ValueError("errors must be positive; drop exact hits (zero errors) first")
    r = np.log(e[1:] / e[:-1])
    return r[1:] / r[:-1]


@app.function
def rate_estimates(errors: np.ndarray, q: float) -> np.ndarray:
    """stage 3: estimates of the rate constant C in e_{k+1} ≈ C e_k^q.

    errors (n,) positive; returns (n - 1,) with entry k equal to e_{k+1} / e_k^q.
    for linear convergence (q = 1) this is the contraction ratio, which must be < 1.
    """
    e = np.asarray(errors, dtype=float)
    return e[1:] / e[:-1] ** q


@app.cell(hide_code=True)
def _():
    checks.run(3, order_estimates=order_estimates, rate_estimates=rate_estimates)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def bisection(f, a: float, b: float, *, tol: float = 1e-12, max_iters: int = 200) -> np.ndarray:
    """stage 4: bisection for a root of f in [a, b]; returns the midpoints it visits, shape (k,).

    requires f(a) and f(b) to have opposite signs (raise ValueError otherwise). if f(a) == 0
    or f(b) == 0, that endpoint is a root: return an array holding just it. each step
    evaluates f at the midpoint m (ONE new evaluation per step: keep f(a) around) and keeps
    the half-interval whose endpoints still have opposite signs. stop after recording a
    midpoint whose half-width (b - a)/2 is <= tol, or where f(m) == 0, or after max_iters
    midpoints.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    if fa == 0 or fb == 0:
        return np.array([float(a) if fa == 0 else float(b)])
    mids = []
    for _ in range(max_iters):
        m = 0.5 * (a + b)
        mids.append(m)
        fm = f(m)
        if fm == 0 or 0.5 * (b - a) <= tol:
            break
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return np.array(mids)


@app.function
def newton_1d(f, df, x0: float, *, tol: float = 1e-12, max_iters: int = 50) -> np.ndarray:
    """stage 4: Newton's method x_{k+1} = x_k - f(x_k) / df(x_k); returns the iterates (k,).

    the array starts with x0. stop (without taking another step) at the first iterate with
    |f(x_k)| <= tol, or once max_iters steps have been taken (so at most max_iters + 1
    iterates).
    """
    xs = [float(x0)]
    for _ in range(max_iters):
        fx = f(xs[-1])
        if abs(fx) <= tol:
            break
        xs.append(xs[-1] - fx / df(xs[-1]))
    return np.array(xs)


@app.function
def secant(f, x0: float, x1: float, *, tol: float = 1e-12, max_iters: int = 50) -> np.ndarray:
    """stage 4: the secant method, Newton with the derivative replaced by the slope through
    the last two iterates: x_{k+1} = x_k - f(x_k) (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1})).

    returns the iterates (k,), starting with x0, x1. stop at the first iterate with
    |f(x_k)| <= tol, when f(x_k) == f(x_{k-1}) (the slope is undefined), or after max_iters
    new iterates. ONE new evaluation of f per step: remember f at the previous iterate.
    """
    xs = [float(x0), float(x1)]
    f_prev, f_cur = f(x0), f(x1)
    for _ in range(max_iters):
        if abs(f_cur) <= tol or f_cur == f_prev:
            break
        x_new = xs[-1] - f_cur * (xs[-1] - xs[-2]) / (f_cur - f_prev)
        xs.append(x_new)
        f_prev, f_cur = f_cur, f(x_new)
    return np.array(xs)


@app.cell(hide_code=True)
def _():
    checks.run(4, bisection=bisection, newton_1d=newton_1d, secant=secant)
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
