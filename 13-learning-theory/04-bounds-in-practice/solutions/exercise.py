import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    from scipy import stats

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bounds in practice · build (reference solution)

    work through the stages in order. you'll implement the two uniform bounds with explicit
    constants (finite classes and VC classes), the one bound that is tight in practice (the
    test-set bound, an exact binomial tail inversion), a calculator for how much data the VC
    bound needs, and a report that fits a real model and puts all of them next to its measured
    errors.

    | stage | you build |
    |---|---|
    | 1 | `finite_class_bound` → `mlfs.theory`: Hoeffding + union bound, for classes of any size |
    | 2 | `vc_bound` → `mlfs.theory`: Vapnik's bound with Sauer's lemma, explicit constants |
    | 3 | `test_set_bound`: the exact binomial upper confidence bound on a held-out error rate |
    | 4 | `vc_sample_size`: the smallest $n$ that makes the VC bound's complexity term $\le\varepsilon$ |
    | 5 | `generalization_report`: fit a model, measure, bound |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def finite_class_bound(n, class_size, delta, emp_risk=0.0):
    """stage 1: emp_risk + sqrt((ln|H| + ln(1/delta)) / (2n)), Hoeffding + union bound; class_size may be a huge Python int."""
    return float(emp_risk + math.sqrt((math.log(class_size) + math.log(1 / delta)) / (2 * n)))


@app.cell(hide_code=True)
def _():
    checks.run(1, finite_class_bound=finite_class_bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def vc_bound(n, vc_dim, delta, emp_risk=0.0):
    """stage 2: emp_risk + sqrt((8/n)(ln Pi(2n) + ln(4/delta))), ln Pi(2n) <= min(d ln(2en/d) [Sauer, 2n >= d], 2n ln 2) (0 if d = 0)."""
    d = vc_dim
    if d == 0:
        log_growth = 0.0
    elif 2 * n >= d:  # both Sauer's bound and the trivial 2^(2n) hold; the min keeps the bound monotone in n
        log_growth = min(d * math.log(2 * math.e * n / d), 2 * n * math.log(2))
    else:
        log_growth = 2 * n * math.log(2)
    return float(emp_risk + math.sqrt(8 / n * (log_growth + math.log(4 / delta))))


@app.cell(hide_code=True)
def _():
    checks.run(2, vc_bound=vc_bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def test_set_bound(n_errors, n, delta):
    """stage 3: the largest p with P(Binomial(n, p) <= n_errors) >= delta (Clopper-Pearson upper limit, Langford 2005)."""
    if n_errors >= n:
        return 1.0
    return float(stats.beta.ppf(1 - delta, n_errors + 1, n - n_errors))


@app.cell(hide_code=True)
def _():
    checks.run(3, test_set_bound=test_set_bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def vc_sample_size(vc_dim, epsilon, delta):
    """stage 4: the smallest integer n >= max(vc_dim, 1) with vc_bound(n, vc_dim, delta) <= epsilon (doubling, then bisection)."""
    lo = max(vc_dim, 1)
    if vc_bound(lo, vc_dim, delta) <= epsilon:
        return lo
    hi = 2 * lo
    while vc_bound(hi, vc_dim, delta) > epsilon:  # the bound decreases in n for n >= d, so this ends
        lo, hi = hi, 2 * hi
    while hi - lo > 1:  # invariant: bound(lo) > epsilon >= bound(hi)
        mid = (lo + hi) // 2
        if vc_bound(mid, vc_dim, delta) <= epsilon:
            hi = mid
        else:
            lo = mid
    return hi


@app.cell(hide_code=True)
def _():
    checks.run(4, vc_sample_size=vc_sample_size, vc_bound=vc_bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def generalization_report(model, X_train, y_train, X_test, y_test, *, vc_dim, n_params, delta=0.05, bits=32):
    """stage 5: fit model on the training set and return dict(n_train, n_test, train_error, test_error, gap, vc_bound, finite_bound, test_bound)."""
    model.fit(X_train, y_train)
    train_error = float(np.mean(model.predict(X_train) != y_train))
    test_errors = int(np.sum(model.predict(X_test) != y_test))
    n, m = len(y_train), len(y_test)
    return {
        "n_train": n,
        "n_test": m,
        "train_error": train_error,
        "test_error": test_errors / m,
        "gap": test_errors / m - train_error,
        "vc_bound": vc_bound(n, vc_dim, delta, train_error),
        "finite_bound": finite_class_bound(n, 2 ** (bits * n_params), delta, train_error),
        "test_bound": test_set_bound(test_errors, m, delta),
    }


@app.cell(hide_code=True)
def _():
    checks.run(5, generalization_report=generalization_report, vc_bound=vc_bound, finite_class_bound=finite_class_bound,
               test_set_bound=test_set_bound)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `finite_class_bound` and `vc_bound` are part of `mlfs.theory`.
    then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
