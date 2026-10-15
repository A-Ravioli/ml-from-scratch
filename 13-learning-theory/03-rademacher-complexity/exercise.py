import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # rademacher complexity · build

    work through the stages in order. the Rademacher complexity of a class on a sample asks:
    how well can the class correlate with **pure noise**, random signs $\sigma_i = \pm1$? you'll
    estimate it by Monte Carlo for finite classes and for norm-bounded linear classes, bound
    it with Massart's lemma and with Jensen's inequality, turn it into a generalization bound,
    and finally estimate it the way Zhang et al. (2017) did for neural networks: by training a
    real learner on random labels.

    | stage | you build |
    |---|---|
    | 1 | `empirical_rademacher` → `mlfs.theory`: Monte Carlo over random signs, for a finite class |
    | 2 | `massart_bound`: the finite-class bound $\max_h\|h\|_2\sqrt{2\ln N}/n$ |
    | 3 | `norm_ball_rademacher`, `l2_rademacher_bound`: linear classes with $\|w\|_p \le B$, in closed form inside the expectation |
    | 4 | `rademacher_bound`: risk $\le$ empirical risk $+\ 2\hat{\mathfrak R} + 3\sqrt{\ln(2/\delta)/(2n)}$ |
    | 5 | `rademacher_via_erm`: fit random labels, measure the correlation |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def empirical_rademacher(predictions, *, n_draws=1000, rng=None):
    """stage 1: Monte Carlo estimate of E_sigma max_h (1/n) sum_i sigma_i h(x_i), predictions (n_hypotheses, n), sigma uniform on {-1, +1}^n."""
    raise NotImplementedError("stage1: implement empirical_rademacher")


@app.cell(hide_code=True)
def _():
    checks.run(1, empirical_rademacher=empirical_rademacher)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def massart_bound(predictions):
    """stage 2: Massart's lemma, max_h ||h||_2 sqrt(2 ln N) / n, with N the number of DISTINCT rows of predictions (n_hypotheses, n)."""
    raise NotImplementedError("stage2: implement massart_bound")


@app.cell(hide_code=True)
def _():
    checks.run(2, massart_bound=massart_bound, empirical_rademacher=empirical_rademacher)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def norm_ball_rademacher(X, B, *, norm=2, n_draws=1000, rng=None):
    """stage 3: Monte Carlo estimate of E_sigma sup_{||w||_norm <= B} (1/n) sum_i sigma_i w . x_i = (B/n) E ||sum_i sigma_i x_i||_dual, norm in {1, 2, inf}."""
    raise NotImplementedError("stage3: implement norm_ball_rademacher")


@app.function
def l2_rademacher_bound(X, B):
    """stage 3: the Jensen bound for the L2 ball, B sqrt(sum_i ||x_i||^2) / n."""
    raise NotImplementedError("stage3: implement l2_rademacher_bound")


@app.cell(hide_code=True)
def _():
    checks.run(3, norm_ball_rademacher=norm_ball_rademacher, l2_rademacher_bound=l2_rademacher_bound,
               empirical_rademacher=empirical_rademacher)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def rademacher_bound(emp_risk, rademacher, n, delta):
    """stage 4: emp_risk + 2 rademacher + 3 sqrt(ln(2/delta) / (2n)), for a loss class with values in [0, 1] (Mohri et al., Thm 3.3)."""
    raise NotImplementedError("stage4: implement rademacher_bound")


@app.cell(hide_code=True)
def _():
    checks.run(4, rademacher_bound=rademacher_bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def rademacher_via_erm(fit_predict, X, *, n_draws=50, rng=None):
    """stage 5: mean over n_draws random sign vectors sigma of (1/n) sum_i sigma_i h(x_i), where h = fit_predict(X, sigma) in {-1, +1}^n."""
    raise NotImplementedError("stage5: implement rademacher_via_erm")


@app.cell(hide_code=True)
def _():
    checks.run(5, rademacher_via_erm=rademacher_via_erm, empirical_rademacher=empirical_rademacher)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `empirical_rademacher` is part of `mlfs.theory` (try
    `from mlfs.theory import empirical_rademacher` in a scratch cell). then open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
