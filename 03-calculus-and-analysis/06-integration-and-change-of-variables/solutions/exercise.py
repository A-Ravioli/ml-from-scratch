import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.autodiff import numerical_jacobian  # yours, from topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # integration & change of variables · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end you can integrate in
    any number of dimensions (on a grid, or by Monte Carlo with an error bar) and transform a
    probability density through an invertible map, the core computation of normalizing flows
    (chapter 21).

    | stage | you build |
    |---|---|
    | 1 | `midpoint_rule`: a tensor-product grid integrator in $d$ dimensions |
    | 2 | `monte_carlo_integrate`: an integral and its standard error from random points |
    | 3 | `log_abs_det_jacobian`: how much a map stretches volume, on a log scale |
    | 4 | `pushforward_log_density`: the density of $X = T(Z)$ |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def midpoint_rule(f, low, high, n: int) -> float:
    """stage 1: the integral of f over the box [low_1, high_1] x ... x [low_d, high_d] by the
    tensor-product midpoint rule with n cells per axis (n^d evaluation points in total).

    low, high are arrays (d,) (or floats for d = 1). f is VECTORIZED: it takes an array of
    points of shape (m, d) and returns their values, shape (m,). call it on all n^d cell
    midpoints at once and return sum(values) * (volume of one cell) as a float.
    """
    low = np.atleast_1d(np.asarray(low, dtype=float))
    high = np.atleast_1d(np.asarray(high, dtype=float))
    h = (high - low) / n
    axes = [lo + (np.arange(n) + 0.5) * hi for lo, hi in zip(low, h)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, low.size)
    return float(np.sum(f(grid)) * np.prod(h))


@app.cell(hide_code=True)
def _():
    checks.run(1, midpoint_rule=midpoint_rule)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def monte_carlo_integrate(f, low, high, n: int, *, rng=None) -> tuple[float, float]:
    """stage 2: the integral of f over a box by plain Monte Carlo, with its standard error.

    draw n points uniformly in the box (one (n, d) array), evaluate the vectorized f once,
    and return (V * mean(values), V * std(values, ddof=1) / sqrt(n)) with V the box volume.
    rng is None, an int seed, or a np.random.Generator (normalize it with
    np.random.default_rng(rng)).
    """
    rng = np.random.default_rng(rng)
    low = np.atleast_1d(np.asarray(low, dtype=float))
    high = np.atleast_1d(np.asarray(high, dtype=float))
    vol = float(np.prod(high - low))
    pts = low + (high - low) * rng.random((n, low.size))
    vals = np.asarray(f(pts), dtype=float)
    return vol * float(vals.mean()), vol * float(vals.std(ddof=1)) / np.sqrt(n)


@app.cell(hide_code=True)
def _():
    checks.run(2, monte_carlo_integrate=monte_carlo_integrate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def log_abs_det_jacobian(T, z: np.ndarray) -> float:
    """stage 3: log |det J_T(z)| for a map T: (d,) -> (d,) at the point z (d,).

    use numerical_jacobian (from mlfs.autodiff) and np.linalg.slogdet, which returns
    (sign, log|det|) without overflow. a negative determinant (a reflection) is fine: take
    the absolute value.
    """
    J = numerical_jacobian(T, np.atleast_1d(np.asarray(z, dtype=float)))
    return float(np.linalg.slogdet(J)[1])


@app.cell(hide_code=True)
def _():
    checks.run(3, log_abs_det_jacobian=log_abs_det_jacobian)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def pushforward_log_density(log_pz, T_inv, x):
    """stage 4: log p_X(x) for X = T(Z), where Z has log-density log_pz and T is invertible.

        log p_X(x) = log_pz(T_inv(x)) + log |det J_{T_inv}(x)|.

    log_pz takes one point (d,) and returns a float; T_inv maps (d,) -> (d,). x is one point
    (d,) (returns a float) or a batch (m, d) (returns (m,)).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return np.array([pushforward_log_density(log_pz, T_inv, xi) for xi in x])
    return float(log_pz(T_inv(x)) + log_abs_det_jacobian(T_inv, x))


@app.cell(hide_code=True)
def _():
    checks.run(4, pushforward_log_density=pushforward_log_density)
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
