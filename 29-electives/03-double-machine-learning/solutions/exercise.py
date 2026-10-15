import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # double machine learning · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. every nuisance model is passed
    in as a factory `make_model` (a class or a zero-argument function) whose instances have
    `fit(X, y)` and `predict(X)`, like every regressor you built in part II.

    | stage | you build |
    |---|---|
    | 1 | `cross_fit_predict`: out-of-fold predictions, so no unit is predicted by a model that saw it |
    | 2 | `partialling_out`, `naive_plugin`: the orthogonal score and the non-orthogonal one it replaces |
    | 3 | `dml_plr`: double machine learning for the partially linear model, with repeated cross-fitting |
    | 4 | `dml_irm`: cross-fitted AIPW for a binary treatment (the interactive model) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def cross_fit_predict(make_model, X, y, *, n_folds=5, response="predict", folds=None, rng=None):
    """stage 1: out-of-fold predictions (n,) of make_model() over n_folds random folds (or the given folds); response 'predict' or 'proba'."""
    if response not in ("predict", "proba"):
        raise ValueError(f"response must be 'predict' or 'proba', got {response!r}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(y)
    out = np.empty(n)
    if folds is None:
        folds = np.array_split(as_rng(rng).permutation(n), n_folds)
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        model = make_model().fit(X[train], y[train])
        out[test] = model.predict_proba(X[test])[:, 1] if response == "proba" else model.predict(X[test])
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, cross_fit_predict=cross_fit_predict)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def partialling_out(y_res, d_res):
    """stage 2: (theta, se) from residuals y - l(X) and d - m(X): theta = sum(d_res y_res) / sum(d_res^2), sandwich se."""
    y_res = np.asarray(y_res, dtype=float)
    d_res = np.asarray(d_res, dtype=float)
    theta = float(np.sum(d_res * y_res) / np.sum(d_res**2))
    psi = d_res * (y_res - theta * d_res)
    J = np.mean(d_res**2)
    se = float(np.sqrt(np.mean(psi**2) / J**2 / len(y_res)))
    return theta, se


@app.function
def naive_plugin(X, y, d, make_model, *, n_iters=10):
    """stage 2: alternate g = make_model() fit on (X, y - theta d) and theta = sum(d (y - g(X))) / sum(d^2), from theta = 0."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    theta = 0.0
    for _ in range(n_iters):
        g = make_model().fit(X, y - theta * d).predict(X)
        theta = float(np.sum(d * (y - g)) / np.sum(d**2))
    return theta


@app.cell(hide_code=True)
def _():
    checks.run(2, partialling_out=partialling_out, naive_plugin=naive_plugin)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def dml_plr(X, y, d, make_model_y, make_model_d, *, n_folds=5, n_repeats=1, rng=None):
    """stage 3: DML for y = theta d + g(X) + u; returns dict(theta, se, ci, thetas) aggregated over n_repeats by the median."""
    g = as_rng(rng)
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    thetas, ses = [], []
    for _ in range(n_repeats):
        folds = np.array_split(g.permutation(len(y)), n_folds)  # one partition shared by both nuisances
        y_res = y - cross_fit_predict(make_model_y, X, y, folds=folds)
        d_res = d - cross_fit_predict(make_model_d, X, d, folds=folds)
        th, se = partialling_out(y_res, d_res)
        thetas.append(th)
        ses.append(se)
    thetas, ses = np.array(thetas), np.array(ses)
    theta = float(np.median(thetas))
    se = float(np.sqrt(np.median(ses**2 + (thetas - theta) ** 2)))
    return {"theta": theta, "se": se, "ci": (theta - 1.96 * se, theta + 1.96 * se), "thetas": thetas}


@app.cell(hide_code=True)
def _():
    checks.run(3, dml_plr=dml_plr, cross_fit_predict=cross_fit_predict, partialling_out=partialling_out)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def dml_irm(X, y, d, make_model_y, make_classifier_d, *, n_folds=5, clip=0.01, rng=None):
    """stage 4: cross-fitted AIPW estimate of the ATE of a binary d; returns dict(theta, se, ci, scores)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.asarray(d).astype(int)
    n = len(y)
    mu0, mu1, e = np.empty(n), np.empty(n), np.empty(n)
    for test in np.array_split(as_rng(rng).permutation(n), n_folds):
        train = np.setdiff1d(np.arange(n), test)
        tr0, tr1 = train[d[train] == 0], train[d[train] == 1]
        mu0[test] = make_model_y().fit(X[tr0], y[tr0]).predict(X[test])
        mu1[test] = make_model_y().fit(X[tr1], y[tr1]).predict(X[test])
        e[test] = make_classifier_d().fit(X[train], d[train]).predict_proba(X[test])[:, 1]
    e = np.clip(e, clip, 1 - clip)
    psi = mu1 - mu0 + d * (y - mu1) / e - (1 - d) * (y - mu0) / (1 - e)
    theta = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    return {"theta": theta, "se": se, "ci": (theta - 1.96 * se, theta + 1.96 * se), "scores": psi}


@app.cell(hide_code=True)
def _():
    checks.run(4, dml_irm=dml_irm)
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
