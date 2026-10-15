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
    # derivatives · build (reference solution)

    a derivative is a slope: how fast a function's output changes when you nudge its input.
    in this notebook you compute derivatives the way a computer can, from two nearby function
    values, and you build the **derivative checker** that every later chapter uses to test
    its hand-derived derivatives. `derivative` and `check_derivative` become part of
    `mlfs.numerics`.

    every function has step-by-step comments inside it: replace its last line (the one that
    raises "not implemented") with code that follows the steps. the checks under each stage
    turn green when your code is right.

    | stage | you build |
    |---|---|
    | 1 | `secant_slope`: the slope of the line through two points on a curve |
    | 2 | `derivative`: forward, backward and central finite differences |
    | 3 | `product_rule`, `chain_rule`: derivatives of combined functions |
    | 4 | `check_derivative`: is a hand-derived derivative right? |
    | 5 | `second_derivative`: the slope of the slope |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def secant_slope(f, a, b):
    """stage 1: the slope of the straight line through (a, f(a)) and (b, f(b)).

    slope = rise / run = (f(b) - f(a)) / (b - a). f is a function that accepts floats or
    NumPy arrays; a and b are floats or arrays of the same shape (then every entry is done
    separately), with a != b. returns a float or an array.
    example: secant_slope(lambda x: x**2, 1.0, 3.0) = (9 - 1) / (3 - 1) = 4.0
    """
    # step 1: compute the rise, f(b) - f(a).
    # step 2: compute the run, b - a.
    # step 3: return rise / run.
    return (f(b) - f(a)) / (b - a)


@app.cell(hide_code=True)
def _():
    checks.run(1, secant_slope=secant_slope)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def derivative(f, x, *, h=1e-5, method="central"):
    """stage 2: estimate the derivative f'(x) with a finite difference of step h.

    method "forward":  slope through x and x + h
    method "backward": slope through x - h and x
    method "central":  slope through x - h and x + h   (the most accurate; the default)
    any other method raises ValueError. f accepts floats or arrays; x is a float, or an
    array (then you get the derivative at every entry). returns a float or an array.
    example: derivative(lambda x: x**2, 3.0, h=0.1, method="forward") = (3.1^2 - 9) / 0.1 = 6.1
    """
    # step 1: x = np.asarray(x, dtype=float), so x + h works for floats and arrays alike.
    # step 2: depending on method, return secant_slope(f, ..., ...) with the right two points.
    # step 3: if method is none of the three, raise ValueError (say which methods exist).
    x = np.asarray(x, dtype=float)
    if method == "forward":
        return secant_slope(f, x, x + h)
    if method == "backward":
        return secant_slope(f, x - h, x)
    if method == "central":
        return secant_slope(f, x - h, x + h)
    raise ValueError(f"method must be 'forward', 'backward' or 'central', got {method!r}")


@app.cell(hide_code=True)
def _():
    checks.run(2, derivative=derivative, secant_slope=secant_slope)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def product_rule(f, df, g, dg):
    """stage 3: the derivative of the product p(x) = f(x) * g(x), as a function.

    f, g are functions and df, dg are their derivatives (all accept floats or arrays).
    returns a NEW function of x that computes p'(x) = f'(x) g(x) + f(x) g'(x).
    example: product_rule(np.sin, np.cos, np.exp, np.exp)(0.0) = cos(0) e^0 + sin(0) e^0 = 1.0
    """
    # step 1: define an inner function dp(x) that returns df(x) * g(x) + f(x) * dg(x).
    # step 2: return dp (the function itself).
    def dp(x):
        return df(x) * g(x) + f(x) * dg(x)

    return dp


@app.function
def chain_rule(f, df, g, dg):
    """stage 3: the derivative of the composition c(x) = f(g(x)), as a function.

    f, g are functions and df, dg are their derivatives (all accept floats or arrays).
    returns a NEW function of x that computes c'(x) = f'(g(x)) * g'(x):
    the outer slope, evaluated at the INNER value g(x), times the inner slope.
    example: chain_rule(np.sin, np.cos, lambda x: x**2, lambda x: 2*x)(x) = cos(x^2) * 2x
    """
    # step 1: define an inner function dc(x) that returns df(g(x)) * dg(x).
    # step 2: return dc.
    def dc(x):
        return df(g(x)) * dg(x)

    return dc


@app.cell(hide_code=True)
def _():
    checks.run(3, product_rule=product_rule, chain_rule=chain_rule)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def check_derivative(f, df, xs, *, h=1e-5):
    """stage 4: how wrong is a hand-derived derivative df of f? returns a float.

    at every point x in xs, compare df(x) with the central difference fd(x) =
    derivative(f, x, h=h) and compute the error  |df(x) - fd(x)| / max(1, |fd(x)|).
    return the LARGEST of these errors, as a float. rule of thumb: below 1e-6 means df is
    right; above 1e-3 means it has a bug. xs is a list or array of points.
    """
    # step 1: xs = np.asarray(xs, dtype=float).
    # step 2: fd = derivative(f, xs, h=h, method="central")  (every point at once)
    #         and analytic = np.asarray(df(xs), dtype=float).
    # step 3: err = np.abs(analytic - fd) / np.maximum(1.0, np.abs(fd)).
    # step 4: return float(np.max(err)).
    xs = np.asarray(xs, dtype=float)
    fd = derivative(f, xs, h=h, method="central")
    analytic = np.asarray(df(xs), dtype=float)
    err = np.abs(analytic - fd) / np.maximum(1.0, np.abs(fd))
    return float(np.max(err))


@app.cell(hide_code=True)
def _():
    checks.run(4, check_derivative=check_derivative, derivative=derivative)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def second_derivative(f, x, *, h=1e-4):
    """stage 5: estimate f''(x), the slope of the slope, by the central second difference

        f''(x) ≈ (f(x + h) - 2 f(x) + f(x - h)) / h^2

    f accepts floats or arrays; x is a float or an array. returns a float or an array.
    (the default h is bigger than derivative's: dividing by h^2 amplifies rounding errors.)
    example: second_derivative(lambda x: x**4, 1.0, h=0.1) = (1.1^4 - 2 + 0.9^4) / 0.01 = 12.02
    """
    # step 1: x = np.asarray(x, dtype=float).
    # step 2: return (f(x + h) - 2 * f(x) + f(x - h)) / h**2.
    x = np.asarray(x, dtype=float)
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


@app.cell(hide_code=True)
def _():
    checks.run(5, second_derivative=second_derivative)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. you'll find the best step
    size (it isn't the smallest), see what finite differences say where a function has a
    corner, and catch a chain-rule bug with your checker.
    """)
    return


if __name__ == "__main__":
    app.run()
