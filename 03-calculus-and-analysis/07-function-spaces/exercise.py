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
    # function spaces · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. functions become vectors: you'll
    measure angles and lengths between them, build an orthogonal basis of polynomials, and
    project functions onto Legendre and Fourier bases.

    | stage | you build |
    |---|---|
    | 1 | `l2_inner`, `l2_norm`: the inner product and length of functions on an interval |
    | 2 | `legendre`: Legendre polynomials by their three-term recurrence |
    | 3 | `legendre_coefficients`, `legendre_series`: project onto polynomials and rebuild |
    | 4 | `fourier_coefficients`, `fourier_series`: project onto sines and cosines and rebuild |

    every function `f` here is vectorized: it takes a NumPy array of points and returns an
    array of values. you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def l2_inner(f, g, a: float, b: float, *, n: int = 4096) -> float:
    """stage 1: <f, g> = integral of f(x) g(x) over [a, b], by the midpoint rule with n cells.

    f, g are vectorized (array of points -> array of values). returns a float.
    """
    raise NotImplementedError("stage1: implement l2_inner")


@app.function
def l2_norm(f, a: float, b: float, *, n: int = 4096) -> float:
    """stage 1: ||f|| = sqrt(<f, f>) on [a, b]."""
    raise NotImplementedError("stage1: implement l2_norm")


@app.cell(hide_code=True)
def _():
    checks.run(1, l2_inner=l2_inner, l2_norm=l2_norm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def legendre(k: int, x) -> np.ndarray:
    """stage 2: the Legendre polynomial P_k evaluated at x (any shape), by the recurrence

        P_0 = 1,  P_1 = x,  (j + 1) P_{j+1} = (2j + 1) x P_j - j P_{j-1}.

    returns a float array shaped like x. (don't use np.polynomial: build it.)
    """
    raise NotImplementedError("stage2: implement legendre")


@app.cell(hide_code=True)
def _():
    checks.run(2, legendre=legendre)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def legendre_coefficients(f, K: int, *, n: int = 4096) -> np.ndarray:
    """stage 3: the coefficients c_0..c_K of the projection of f onto P_0..P_K on [-1, 1].

    c_k = <f, P_k> / <P_k, P_k> = (2k + 1)/2 * integral of f P_k over [-1, 1], using
    l2_inner with n cells. returns (K + 1,).
    """
    raise NotImplementedError("stage3: implement legendre_coefficients")


@app.function
def legendre_series(c, x) -> np.ndarray:
    """stage 3: sum_k c[k] P_k(x), shaped like x."""
    raise NotImplementedError("stage3: implement legendre_series")


@app.cell(hide_code=True)
def _():
    checks.run(3, legendre_coefficients=legendre_coefficients, legendre_series=legendre_series)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fourier_coefficients(f, K: int, *, n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """stage 4: Fourier coefficients of f on [-pi, pi], up to frequency K.

    a_k = (1/pi) * integral of f(x) cos(kx), b_k = (1/pi) * integral of f(x) sin(kx), for
    k = 0..K, by the midpoint rule with n cells (so b_0 = 0, and a_0 is TWICE the mean of f).
    returns (a (K + 1,), b (K + 1,)), matching fourier_series below.
    """
    raise NotImplementedError("stage4: implement fourier_coefficients")


@app.function
def fourier_series(a, b, x) -> np.ndarray:
    """stage 4: a[0]/2 + sum_{k>=1} (a[k] cos(kx) + b[k] sin(kx)), shaped like x."""
    raise NotImplementedError("stage4: implement fourier_series")


@app.cell(hide_code=True)
def _():
    checks.run(4, fourier_coefficients=fourier_coefficients, fourier_series=fourier_series)
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
