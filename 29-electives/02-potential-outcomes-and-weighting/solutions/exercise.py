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
    X = np.asarray(X, dtype=float)
    t = np.asarray(t).astype(int)
    model = LogisticRegression(alpha=alpha).fit(X, t)
    e = model.predict_proba(X)[:, 1]
    if clip is not None:
        e = np.clip(e, clip, 1 - clip)
    return e


@app.function
def effective_sample_size(w):
    """stage 1: Kish's effective sample size (sum w)^2 / sum w^2 of nonnegative weights w (n,)."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


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
    y, t, e = (np.asarray(a, dtype=float) for a in (y, t, e))
    w1, w0 = t / e, (1 - t) / (1 - e)
    if normalized:
        return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))
    return float(np.mean(w1 * y) - np.mean(w0 * y))


@app.function
def ipw_att(y, t, e):
    """stage 2: the ATT: mean treated outcome minus the controls' mean weighted by e / (1 - e)."""
    y, t, e = (np.asarray(a, dtype=float) for a in (y, t, e))
    treated = t == 1
    w = e[~treated] / (1 - e[~treated])
    return float(y[treated].mean() - np.sum(w * y[~treated]) / np.sum(w))


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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)
    if make_model is None:
        make_model = LinearRegression
    m0 = make_model().fit(X[t == 0], y[t == 0])
    m1 = make_model().fit(X[t == 1], y[t == 1])
    return np.asarray(m0.predict(X), dtype=float), np.asarray(m1.predict(X), dtype=float)


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
    y, t, e, mu0, mu1 = (np.asarray(a, dtype=float) for a in (y, t, e, mu0, mu1))
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))


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
