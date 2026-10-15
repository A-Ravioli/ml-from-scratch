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
    # functions, exponents & logs · build

    your computer can only add, subtract, multiply and divide. so how does it compute $e^x$
    or $\log x$? in this notebook you find out by building both yourself, from nothing but
    arithmetic: $e^x$ from an infinite sum, and $\log x$ by *undoing* $e^x$.

    work through the stages in order. each stage shows its part of the lesson, then gives you
    functions to write. every function has step-by-step comments inside it: replace its
    last line (the one that raises "not implemented") with code that follows the steps. the checks under
    each stage turn green when your code is right, and their messages tell you what to fix
    when it isn't.

    | stage | you build |
    |---|---|
    | 1 | `compose` and `bisect_inverse`: combining and undoing functions |
    | 2 | `exp_series`: $e^x$ as a sum of terms |
    | 3 | `exp`: an accurate $e^x$ for any $x$, using an exponent rule |
    | 4 | `newton_log_steps`: Newton's method, the fast way to undo $e^x$ |
    | 5 | `log`: an accurate $\log y$ for any positive $y$ |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def compose(f, g):
    """stage 1: the composition "f after g": a new function that sends x to f(g(x)).

    f and g are functions of one argument (they may also accept NumPy arrays).
    returns a function. g is applied FIRST, then f.
    example: compose(np.sqrt, lambda x: x + 1)(8.0) == 3.0   (8 + 1 = 9, then sqrt(9) = 3)
    """
    # step 1: inside compose, define a small function h(x) that returns f(g(x)).
    # step 2: return h itself (the function, not h called on something).
    raise NotImplementedError("stage1: implement compose")


@app.function
def bisect_inverse(f, y, lo, hi, *, tol=1e-12):
    """stage 1: solve f(x) = y for x by bisection, where f is INCREASING on [lo, hi].

    f: a function of one float. y: a float with f(lo) <= y <= f(hi); otherwise raise
    ValueError (the answer isn't in the bracket). repeatedly halve the bracket [lo, hi],
    keeping the half that contains the answer, until hi - lo <= tol (or 200 halvings, since a
    float can't be halved forever). returns the midpoint of the final bracket, a float.
    example: bisect_inverse(lambda x: x**3, 27.0, 0.0, 10.0) is 3.0 to within 1e-12.
    """
    # step 1: if y < f(lo) or y > f(hi), raise ValueError with a short message.
    # step 2: repeat at most 200 times, stopping early once hi - lo <= tol:
    #           mid = (lo + hi) / 2
    #           if f(mid) < y: the answer is to the RIGHT of mid, so set lo = mid
    #           else:          the answer is at mid or to its LEFT, so set hi = mid
    # step 3: return the midpoint (lo + hi) / 2 as a float.
    raise NotImplementedError("stage1: implement bisect_inverse")


@app.cell(hide_code=True)
def _():
    checks.run(1, compose=compose, bisect_inverse=bisect_inverse)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def exp_series(x, n_terms):
    """stage 2: the first n_terms terms of the series for e^x, added up:

        1 + x + x^2/2! + x^3/3! + ... + x^(n_terms - 1)/(n_terms - 1)!

    x: a float or a NumPy array (then every entry is done separately). n_terms: an integer >= 1.
    returns a float, or an array shaped like x.
    example: exp_series(1.0, 4) = 1 + 1 + 1/2 + 1/6 = 2.6666...
    """
    # step 1: turn x into a float array: x = np.asarray(x, dtype=float).
    # step 2: the k = 0 term is 1. set term = np.ones_like(x) and total = term.copy().
    # step 3: for k = 1, 2, ..., n_terms - 1:
    #           get term k from term k-1 by one multiplication and one division:
    #           term = term * x / k      (because x^k/k! = (x^(k-1)/(k-1)!) * x / k)
    #           total = total + term
    # step 4: return total.
    raise NotImplementedError("stage2: implement exp_series")


@app.cell(hide_code=True)
def _():
    checks.run(2, exp_series=exp_series)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def exp(x):
    """stage 3: e^x for any float x (or every entry of an array), to about 1e-13 relative error.

    the series alone is slow and inaccurate for big |x| (try exp_series(-30.0, 100)). so use
    the exponent rule e^(k + r) = e^k * e^r ("range reduction"):
      - k = the integer nearest to x, and r = x - k, so -1/2 <= r <= 1/2;
      - e^r from your exp_series with 20 terms: plenty, because r is small;
      - e^k = E**k, where E = exp_series(1.0, 20) is the number e itself.
    returns a float, or an array shaped like x.
    """
    # step 1: x = np.asarray(x, dtype=float).
    # step 2: k = np.round(x) (the nearest whole number) and r = x - k.
    # step 3: E = exp_series(1.0, 20), which is e = 2.71828...
    # step 4: return E**k * exp_series(r, 20).
    raise NotImplementedError("stage3: implement exp")


@app.cell(hide_code=True)
def _():
    checks.run(3, exp=exp, exp_series=exp_series)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def newton_log_steps(y, x0, n_steps):
    """stage 4: run n_steps of Newton's method for log(y), starting from the guess x0.

    log(y) is the x that solves e^x = y. Newton's update for that equation is

        x_new = x - 1 + y * e^(-x)

    (use your own exp). y > 0 and x0 are floats; n_steps is an integer >= 0.
    returns a 1-D NumPy array of length n_steps + 1: [x0, x1, ..., x_n_steps].
    example: newton_log_steps(2.0, 1.0, 1) = [1.0, 2/e] = [1.0, 0.7357588823...]
    """
    # step 1: start a list xs = [x0] and set x = x0.
    # step 2: repeat n_steps times:
    #           x = x - 1 + y * exp(-x)
    #           append x to xs
    # step 3: return np.array(xs, dtype=float).
    raise NotImplementedError("stage4: implement newton_log_steps")


@app.cell(hide_code=True)
def _():
    checks.run(4, newton_log_steps=newton_log_steps, exp=exp)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def log(y):
    """stage 5: the natural logarithm of a positive float y (or of every entry of an array).

    Newton's method needs a good starting guess, so first shrink y into [0.5, 1):
      - m, p = np.frexp(y) splits y into y = m * 2**p, with 0.5 <= m < 1 and p a whole number;
      - then log(y) = log(m) + p * log(2)   (the log rules: products become sums);
      - log(m): 6 Newton steps x = x - 1 + m * exp(-x), starting from x = m - 1;
      - log(2): newton_log_steps(2.0, 0.7, 6), last entry.
    raise ValueError if any y <= 0. returns a float, or an array shaped like y.
    """
    # step 1: y = np.asarray(y, dtype=float); if np.any(y <= 0), raise ValueError.
    # step 2: m, p = np.frexp(y).
    # step 3: x = m - 1, then repeat 6 times: x = x - 1 + m * exp(-x).   (now x = log(m))
    # step 4: ln2 = newton_log_steps(2.0, 0.7, 6)[-1].
    # step 5: return x + p * ln2.
    raise NotImplementedError("stage5: implement log")


@app.cell(hide_code=True)
def _():
    checks.run(5, log=log, exp=exp, newton_log_steps=newton_log_steps)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. you'll find out how many
    series terms 10-digit accuracy really takes, race Newton against bisection, and see just
    how fast "exponential growth" is.
    """)
    return


if __name__ == "__main__":
    app.run()
