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
    raise NotImplementedError("stage1: implement cross_fit_predict")


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
    raise NotImplementedError("stage2: implement partialling_out")


@app.function
def naive_plugin(X, y, d, make_model, *, n_iters=10):
    """stage 2: alternate g = make_model() fit on (X, y - theta d) and theta = sum(d (y - g(X))) / sum(d^2), from theta = 0."""
    raise NotImplementedError("stage2: implement naive_plugin")


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
    raise NotImplementedError("stage3: implement dml_plr")


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
    raise NotImplementedError("stage4: implement dml_irm")


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
