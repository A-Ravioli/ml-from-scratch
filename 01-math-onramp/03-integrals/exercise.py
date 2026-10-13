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
    # integrals · build

    an integral is an area under a curve, and you compute it the way you'd measure an odd
    shaped field: cut it into thin strips and add up their areas. in this notebook you build
    three ways to do that, each much more accurate than the last, and you measure *how much*
    more accurate. `trapezoid` and `simpson` become part of `mlfs.numerics`.

    every function has step-by-step comments inside it: replace its last line (the one that
    raises "not implemented") with code that follows the steps. the checks under each stage
    turn green when your code is right.

    | stage | you build |
    |---|---|
    | 1 | `riemann_sum`: rectangles (left, right, midpoint) |
    | 2 | `trapezoid`: trapezoids, the composite trapezoid rule |
    | 3 | `simpson`: parabolas, the composite Simpson rule |
    | 4 | `cumulative_trapezoid`: the running area, and the fundamental theorem |
    | 5 | `convergence_order`: how fast does the error shrink? |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def riemann_sum(f, a, b, n=100, *, rule="left"):
    """stage 1: approximate the area under f between a and b with n rectangles of equal width.

    the width of every rectangle is h = (b - a) / n. rectangle i (i = 0, ..., n-1) covers
    [a + i h, a + (i+1) h], and its height is f at that strip's
      rule "left":     left end,  a + i h
      rule "right":    right end, a + (i+1) h
      rule "midpoint": middle,    a + (i + 1/2) h
    any other rule raises ValueError. f takes a NumPy array of points and returns an array.
    returns h times the sum of the heights, as a float.
    example: riemann_sum(lambda x: x**2, 0.0, 1.0, 4, rule="left") = (0 + 1/16 + 4/16 + 9/16) / 4 = 0.21875
    """
    # step 1: h = (b - a) / n, and i = np.arange(n) (the strip numbers 0, 1, ..., n-1).
    # step 2: pick the sample points for the rule: a + i * h, a + (i + 1) * h, or a + (i + 0.5) * h.
    #         if the rule is none of the three, raise ValueError.
    # step 3: return float(h * np.sum(f(points))).
    raise NotImplementedError("stage1: implement riemann_sum")


@app.cell(hide_code=True)
def _():
    checks.run(1, riemann_sum=riemann_sum)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def trapezoid(f, a, b, n=100):
    """stage 2: the composite trapezoid rule with n strips of width h = (b - a) / n.

    join neighbouring points of the curve with straight lines and add up the areas of the n
    trapezoids underneath. with the n + 1 grid points x_0 = a, ..., x_n = b and y_i = f(x_i):

        h * (y_0 / 2 + y_1 + y_2 + ... + y_(n-1) + y_n / 2)

    f takes a NumPy array of points and returns an array. returns a float.
    example: trapezoid(lambda x: x**2, 0.0, 1.0, 4) = 0.34375
    """
    # step 1: xs = np.linspace(a, b, n + 1) (n + 1 points make n strips), and y = f(xs).
    # step 2: h = (b - a) / n.
    # step 3: the two end points count half: return float(h * (np.sum(y) - (y[0] + y[-1]) / 2)).
    raise NotImplementedError("stage2: implement trapezoid")


@app.cell(hide_code=True)
def _():
    checks.run(2, trapezoid=trapezoid)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def simpson(f, a, b, n=100):
    """stage 3: the composite Simpson rule with n strips (n must be EVEN; otherwise ValueError).

    fit a parabola through each group of three neighbouring points and add up the areas under
    the parabolas. with h = (b - a) / n, grid points x_0, ..., x_n and y_i = f(x_i):

        (h / 3) * (y_0 + 4 y_1 + 2 y_2 + 4 y_3 + 2 y_4 + ... + 2 y_(n-2) + 4 y_(n-1) + y_n)

    odd-numbered points get weight 4, even-numbered INNER points weight 2, the two ends 1.
    f takes a NumPy array of points and returns an array. returns a float.
    example: simpson(lambda x: x**4, 0.0, 1.0, 2) = (0.5 / 3) * (0 + 4 * 0.0625 + 1) = 0.208333...
    """
    # step 1: if n is odd (n % 2 != 0), raise ValueError.
    # step 2: xs = np.linspace(a, b, n + 1), y = f(xs), h = (b - a) / n.
    # step 3: odd = np.sum(y[1:-1:2])  (y_1, y_3, ..., y_(n-1))
    #         even = np.sum(y[2:-1:2]) (y_2, y_4, ..., y_(n-2): the inner even points)
    # step 4: return float(h / 3 * (y[0] + 4 * odd + 2 * even + y[-1])).
    raise NotImplementedError("stage3: implement simpson")


@app.cell(hide_code=True)
def _():
    checks.run(3, simpson=simpson)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def cumulative_trapezoid(f, a, b, n=100):
    """stage 4: the running area F(x) = (area under f from a to x), at every grid point.

    grid: xs = np.linspace(a, b, n + 1). returns (xs, F), two arrays of shape (n + 1,), with
    F[0] = 0 (no area yet) and F[k] = F[k-1] + (area of trapezoid k between xs[k-1] and xs[k]),
    where that trapezoid's area is h * (f(xs[k-1]) + f(xs[k])) / 2. so F[-1] equals
    trapezoid(f, a, b, n).
    example: cumulative_trapezoid(lambda x: 2 * x, 0.0, 1.0, 2) gives F = [0, 0.25, 1.0]
    """
    # step 1: xs = np.linspace(a, b, n + 1), y = f(xs), h = (b - a) / n.
    # step 2: the n trapezoid areas at once: strips = h * (y[:-1] + y[1:]) / 2.
    # step 3: F = np.concatenate([[0.0], np.cumsum(strips)])  (running totals, starting at 0).
    # step 4: return xs, F.
    raise NotImplementedError("stage4: implement cumulative_trapezoid")


@app.cell(hide_code=True)
def _():
    checks.run(4, cumulative_trapezoid=cumulative_trapezoid)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def convergence_order(ns, errors):
    """stage 5: estimate p in "error ≈ C / n^p" from measured errors. returns a float.

    ns: numbers of strips (1-D, positive); errors: the matching errors (1-D, positive).
    if error = C / n^p, then log(error) = log(C) - p log(n): a straight line in
    (log n, log error) with slope -p. fit that line by least squares through ALL the points
    and return -slope.
    example: convergence_order([10, 20, 40], [1e-2, 2.5e-3, 6.25e-4]) = 2.0
    """
    # step 1: u = np.log(ns) and v = np.log(errors), as float arrays.
    # step 2: the least-squares slope of v against u:
    #         slope = sum((u - mean(u)) * (v - mean(v))) / sum((u - mean(u))^2)
    # step 3: return float(-slope).
    raise NotImplementedError("stage5: implement convergence_order")


@app.cell(hide_code=True)
def _():
    checks.run(5, convergence_order=convergence_order)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. you'll measure each
    rule's convergence rate, find a function where the humble trapezoid rule beats Simpson,
    and one where Simpson loses its advantage completely.
    """)
    return


if __name__ == "__main__":
    app.run()
