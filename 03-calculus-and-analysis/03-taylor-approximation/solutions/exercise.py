import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.autodiff import numerical_hessian  # yours, from topic 02
    from mlfs.numgrad import numerical_gradient  # yours, from ch 02 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # taylor approximation · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end you'll have a
    multivariate Taylor approximator built from your own numerical gradient and Hessian, and
    a tool that measures how far from its center a quadratic model can be trusted.

    | stage | you build |
    |---|---|
    | 1 | `taylor_poly_1d`: evaluate a Taylor polynomial from its derivatives |
    | 2 | `lagrange_bound`, `loglog_slope`: bound the error, and measure how fast it shrinks |
    | 3 | `quadratic_model`: the second-order model $f_0 + g^\top d + \tfrac12d^\top Hd$ as a function |
    | 4 | `taylor_model`: orders 0, 1, 2 of any $f$, from numerical derivatives |
    | 5 | `agreement_ratio`, `trust_radius`: where the model stops being trustworthy |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def taylor_poly_1d(derivs, x0: float, x):
    """stage 1: the Taylor polynomial sum_k derivs[k] / k! * (x - x0)^k.

    derivs is a sequence (K + 1,) with derivs[k] = f^(k)(x0), so the polynomial has degree K.
    x is a float or an array of any shape; the result has x's shape (a float for a float x).
    """
    x = np.asarray(x, dtype=float)
    d = x - x0
    total = np.zeros_like(d)
    term = np.ones_like(d)  # d^k / k!
    for k, c in enumerate(derivs):
        if k > 0:
            term = term * d / k
        total = total + c * term
    return float(total) if total.ndim == 0 else total


@app.cell(hide_code=True)
def _():
    checks.run(1, taylor_poly_1d=taylor_poly_1d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def lagrange_bound(M: float, x0: float, x, order: int):
    """stage 2: the Lagrange remainder bound M |x - x0|^(order+1) / (order+1)!.

    bounds |f(x) - T(x)| for the Taylor polynomial T of degree `order`, if |f^(order+1)| <= M
    between x0 and x. x is a float or an array; the result has x's shape.
    """
    x = np.asarray(x, dtype=float)
    out = M * np.abs(x - x0) ** (order + 1) / math.factorial(order + 1)
    return float(out) if out.ndim == 0 else out


@app.function
def loglog_slope(hs, errors) -> float:
    """stage 2: the least-squares slope of log(errors) against log(hs).

    hs, errors are positive arrays (k,), k >= 2. if errors ≈ C h^p, the slope is p: the
    order to which the error shrinks.
    """
    lh = np.log(np.asarray(hs, dtype=float))
    le = np.log(np.asarray(errors, dtype=float))
    lh_c = lh - lh.mean()
    return float(lh_c @ (le - le.mean()) / (lh_c @ lh_c))


@app.cell(hide_code=True)
def _():
    checks.run(2, lagrange_bound=lagrange_bound, loglog_slope=loglog_slope)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def quadratic_model(x0: np.ndarray, f0: float, g: np.ndarray, H: np.ndarray):
    """stage 3: the function m(x) = f0 + g . (x - x0) + 1/2 (x - x0)^T H (x - x0).

    x0, g are (n,), H is (n, n). returns a callable m: m(x) takes x of shape (n,) and returns
    a float, or a batch of shape (k, n) and returns (k,). pass H = 0 for a linear model.
    """
    x0 = np.asarray(x0, dtype=float)
    g = np.asarray(g, dtype=float)
    H = np.asarray(H, dtype=float)

    def m(x):
        d = np.asarray(x, dtype=float) - x0
        out = f0 + d @ g + 0.5 * np.einsum("...i,ij,...j->...", d, H, d)
        return float(out) if np.ndim(out) == 0 else out

    return m


@app.cell(hide_code=True)
def _():
    checks.run(3, quadratic_model=quadratic_model)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def taylor_model(f, x0: np.ndarray, *, order: int = 2):
    """stage 4: the Taylor model of f: (n,) -> float at x0, of order 0, 1 or 2, as a callable.

    order 0: m(x) = f(x0). order 1: adds grad f(x0) . (x - x0), with the gradient from
    numerical_gradient (mlfs.numgrad, default step). order 2: adds the Hessian term, with the
    Hessian from numerical_hessian (mlfs.autodiff, default step). returns a quadratic_model,
    so it accepts (n,) or (k, n) inputs. raises ValueError for any other order.
    """
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1 or 2, got {order}")
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    g = numerical_gradient(f, x0) if order >= 1 else np.zeros(n)
    H = numerical_hessian(f, x0) if order == 2 else np.zeros((n, n))
    return quadratic_model(x0, float(f(x0)), g, H)


@app.cell(hide_code=True)
def _():
    checks.run(4, taylor_model=taylor_model)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def agreement_ratio(f, model, x0: np.ndarray, x: np.ndarray) -> float:
    """stage 5: actual reduction / predicted reduction, (f(x0) - f(x)) / (model(x0) - model(x)).

    1 means the model predicted the change in f perfectly. x0, x are (n,).
    """
    return float((f(x0) - f(x)) / (model(x0) - model(x)))


@app.function
def trust_radius(f, model, x0: np.ndarray, direction: np.ndarray, radii, *, tol: float = 0.25) -> float:
    """stage 5: how far along `direction` the model can be trusted.

    direction (n,) is normalized to a unit vector u. radii is an increasing array of positive
    distances. walk through them in order and return the last r for which
    |agreement_ratio(f, model, x0, x0 + r u) - 1| <= tol, stopping at the FIRST r that fails
    (later radii don't count). returns 0.0 if the first radius already fails.
    """
    x0 = np.asarray(x0, dtype=float)
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    best = 0.0
    for r in radii:
        if abs(agreement_ratio(f, model, x0, x0 + r * u) - 1.0) > tol:
            break
        best = float(r)
    return best


@app.cell(hide_code=True)
def _():
    checks.run(5, agreement_ratio=agreement_ratio, trust_radius=trust_radius)
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
