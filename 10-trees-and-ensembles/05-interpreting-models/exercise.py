import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # interpreting models · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. everything here is model-agnostic:
    it only calls `score`, `predict` or `predict_proba`, so it works for your forests and
    boosters and for anything else in the course.

    | stage | you build |
    |---|---|
    | 1 | `permutation_importance`: how much the score drops when one column is shuffled → `mlfs.inspection` |
    | 2 | `ice_curves`, `partial_dependence`: what the model does as one feature moves → `mlfs.inspection` |
    | 3 | `masked_predictions`, `shapley_values_exact`: Shapley values by brute force |
    | 4 | `sample_coalitions`, `kernel_shap`: Shapley values by weighted least squares → `mlfs.inspection` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def permutation_importance(estimator, X, y, *, scoring=None, n_repeats=5, rng=None):
    """stage 1: score drop per shuffled column -> dict(importances_mean (d,), importances_std (d,), importances (d, n_repeats))."""
    raise NotImplementedError("stage1: implement permutation_importance")


@app.cell(hide_code=True)
def _():
    checks.run(1, permutation_importance=permutation_importance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def model_response(estimator, X, response="auto"):
    """the model output that interpretation looks at: 'auto' = predict_proba[:, 1] for binary classifiers, else predict."""
    if response == "auto":
        classes = getattr(estimator, "classes_", None)
        response = "predict_proba" if classes is not None and len(classes) == 2 else "predict"
    if response == "predict_proba":
        return np.asarray(estimator.predict_proba(X))[:, 1]
    if response == "decision_function":
        return np.asarray(estimator.decision_function(X), dtype=float)
    if response == "predict":
        return np.asarray(estimator.predict(X), dtype=float)
    raise ValueError(f"response must be 'auto', 'predict', 'predict_proba' or 'decision_function', got {response!r}")


@app.function
def default_grid(column, grid_resolution=50):
    """stage 2: the unique values if there are at most grid_resolution of them, else unique quantiles at linspace(0.05, 0.95, grid_resolution)."""
    raise NotImplementedError("stage2: implement default_grid")


@app.function
def ice_curves(estimator, X, feature, grid, response="auto"):
    """stage 2: (n, g): row i's prediction with column `feature` set to grid[k], every other column kept; one batched call."""
    raise NotImplementedError("stage2: implement ice_curves")


@app.function
def partial_dependence(estimator, X, feature, *, grid=None, grid_resolution=50, response="auto"):
    """stage 2: (grid (g,), averaged (g,)): the mean of the ICE curves, on default_grid(X[:, feature]) unless a grid is given."""
    raise NotImplementedError("stage2: implement partial_dependence")


@app.cell(hide_code=True)
def _():
    checks.run(2, ice_curves=ice_curves, partial_dependence=partial_dependence, default_grid=default_grid)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def masked_predictions(predict, X_background, x, masks):
    """stage 3: v(S) for each mask row (m, d): the mean of predict over background rows with the masked-in features set to x."""
    raise NotImplementedError("stage3: implement masked_predictions")


@app.function
def shapley_values_exact(value, d):
    """stage 3: exact Shapley values (d,) of a set function value(masks (m, d) bool) -> (m,), by enumerating all 2^d coalitions."""
    raise NotImplementedError("stage3: implement shapley_values_exact")


@app.cell(hide_code=True)
def _():
    checks.run(3, masked_predictions=masked_predictions, shapley_values_exact=shapley_values_exact)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def sample_coalitions(d, n_samples, rng=None):
    """stage 4: bool masks (n_samples, d): size s in 1..d-1 drawn with probability proportional to (d-1)/(s(d-s)), then a uniform subset of that size."""
    raise NotImplementedError("stage4: implement sample_coalitions")


@app.function
def kernel_shap(predict, X_background, x, *, n_samples=2048, rng=None):
    """stage 4: KernelSHAP: phi (d,) from a Shapley-kernel weighted least-squares fit, with phi.sum() + E[f] == f(x) enforced."""
    raise NotImplementedError("stage4: implement kernel_shap")


@app.cell(hide_code=True)
def _():
    checks.run(4, kernel_shap=kernel_shap, sample_coalitions=sample_coalitions,
               shapley_values_exact=shapley_values_exact, masked_predictions=masked_predictions)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `permutation_importance`, `partial_dependence` and
    `kernel_shap` are part of `mlfs.inspection`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
