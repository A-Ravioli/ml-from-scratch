import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linear import LinearRegression, LogisticRegression  # yours, from ch 08 · topics 01 and 04


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # potential outcomes & weighting · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. throughout, `t` is a 0/1
    treatment array, `y` the observed outcomes, `X` the covariates, and `e` the propensity
    scores $P(T = 1 \mid X)$, all of length $n$.

    | stage | you build |
    |---|---|
    | 1 | `propensity_scores`, `effective_sample_size`: estimating $e(x)$ and measuring what weighting costs |
    | 2 | `ipw_ate`, `ipw_att`: inverse propensity weighting (Horvitz–Thompson and Hájek) |
    | 3 | `outcome_models`: the outcome-regression (g-formula) route to the same effect |
    | 4 | `aipw_ate`: the doubly robust estimator, with a standard error |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def propensity_scores(X, t, *, alpha=0.0, clip=None):
    """stage 1: P(T=1 | X) from a LogisticRegression(alpha=alpha) fit of t on X, optionally clipped to [clip, 1 - clip]."""
    raise NotImplementedError("stage1: implement propensity_scores")


@app.function
def effective_sample_size(w):
    """stage 1: Kish's effective sample size (sum w)^2 / sum w^2 of nonnegative weights w (n,)."""
    raise NotImplementedError("stage1: implement effective_sample_size")


@app.cell(hide_code=True)
def _():
    checks.run(1, propensity_scores=propensity_scores, effective_sample_size=effective_sample_size)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def ipw_ate(y, t, e, *, normalized=False):
    """stage 2: the IPW estimate of the ATE: Horvitz-Thompson, or Hajek (self-normalized) if normalized=True."""
    raise NotImplementedError("stage2: implement ipw_ate")


@app.function
def ipw_att(y, t, e):
    """stage 2: the ATT: mean treated outcome minus the controls' mean weighted by e / (1 - e)."""
    raise NotImplementedError("stage2: implement ipw_att")


@app.cell(hide_code=True)
def _():
    checks.run(2, ipw_ate=ipw_ate, ipw_att=ipw_att)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def outcome_models(X, y, t, make_model=None):
    """stage 3: (mu0, mu1), each (n,): predictions for ALL units from models fitted on the control / treated units only."""
    raise NotImplementedError("stage3: implement outcome_models")


@app.cell(hide_code=True)
def _():
    checks.run(3, outcome_models=outcome_models)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def aipw_ate(y, t, e, mu0, mu1):
    """stage 4: (estimate, standard error) of the ATE from the AIPW (doubly robust) scores."""
    raise NotImplementedError("stage4: implement aipw_ate")


@app.cell(hide_code=True)
def _():
    checks.run(4, aipw_ate=aipw_ate, ipw_ate=ipw_ate, outcome_models=outcome_models, propensity_scores=propensity_scores)
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
