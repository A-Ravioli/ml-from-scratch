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
    # interpreting models · build (reference solution)

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
    X = np.asarray(X, dtype=float)
    gen = as_rng(rng)
    score = (lambda Xs: estimator.score(Xs, y)) if scoring is None else (lambda Xs: scoring(estimator, Xs, y))
    base = score(X)
    d = X.shape[1]
    imp = np.zeros((d, n_repeats))
    for j in range(d):
        Xp = X.copy()
        for r in range(n_repeats):
            Xp[:, j] = X[gen.permutation(len(X)), j]
            imp[j, r] = base - score(Xp)
    return {"importances_mean": imp.mean(axis=1), "importances_std": imp.std(axis=1), "importances": imp}


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
    vals = np.unique(column)
    if len(vals) <= grid_resolution:
        return vals.astype(float)
    return np.unique(np.quantile(column, np.linspace(0.05, 0.95, grid_resolution)))


@app.function
def ice_curves(estimator, X, feature, grid, response="auto"):
    """stage 2: (n, g): row i's prediction with column `feature` set to grid[k], every other column kept; one batched call."""
    X = np.asarray(X, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n, g = len(X), len(grid)
    Xg = np.repeat(X, g, axis=0)               # row i repeated g times
    Xg[:, feature] = np.tile(grid, n)
    return model_response(estimator, Xg, response).reshape(n, g)


@app.function
def partial_dependence(estimator, X, feature, *, grid=None, grid_resolution=50, response="auto"):
    """stage 2: (grid (g,), averaged (g,)): the mean of the ICE curves, on default_grid(X[:, feature]) unless a grid is given."""
    X = np.asarray(X, dtype=float)
    grid = default_grid(X[:, feature], grid_resolution) if grid is None else np.asarray(grid, dtype=float)
    return grid, ice_curves(estimator, X, feature, grid, response).mean(axis=0)


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
    Xb = np.asarray(X_background, dtype=float)
    x = np.asarray(x, dtype=float)
    masks = np.atleast_2d(np.asarray(masks, dtype=bool))
    m, nb = len(masks), len(Xb)
    Z = np.where(masks[:, None, :], x[None, None, :], Xb[None, :, :])   # (m, nb, d)
    return np.asarray(predict(Z.reshape(m * nb, -1)), dtype=float).reshape(m, nb).mean(axis=1)


@app.function
def shapley_values_exact(value, d):
    """stage 3: exact Shapley values (d,) of a set function value(masks (m, d) bool) -> (m,), by enumerating all 2^d coalitions."""
    masks = np.array(list(itertools.product([False, True], repeat=d)), dtype=bool)   # all 2^d subsets
    v = np.asarray(value(masks), dtype=float)
    index = {tuple(mk): k for k, mk in enumerate(masks)}
    sizes = masks.sum(axis=1)
    phi = np.zeros(d)
    for j in range(d):
        without = np.flatnonzero(~masks[:, j])
        with_j = masks[without].copy()
        with_j[:, j] = True
        gains = v[[index[tuple(mk)] for mk in with_j]] - v[without]
        s = sizes[without]
        weights = np.array([math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d) for k in s])
        phi[j] = np.sum(weights * gains)
    return phi


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
    gen = as_rng(rng)
    sizes = np.arange(1, d)
    p = (d - 1) / (sizes * (d - sizes))
    s = gen.choice(sizes, size=n_samples, p=p / p.sum())
    ranks = np.argsort(np.argsort(gen.random((n_samples, d)), axis=1), axis=1)
    return ranks < s[:, None]   # the s features with the smallest random keys: a uniform subset of size s


@app.function
def kernel_shap(predict, X_background, x, *, n_samples=2048, rng=None):
    """stage 4: KernelSHAP: phi (d,) from a Shapley-kernel weighted least-squares fit, with phi.sum() + E[f] == f(x) enforced."""
    Xb = np.asarray(X_background, dtype=float)
    x = np.asarray(x, dtype=float)
    d = len(x)
    f0 = float(np.mean(predict(Xb)))
    fx = float(np.asarray(predict(x[None, :]), dtype=float)[0])
    if d == 1:
        return np.array([fx - f0])
    if 2 ** d - 2 <= n_samples:  # few features: use every coalition with its exact kernel weight
        masks = np.array([mk for mk in itertools.product([False, True], repeat=d) if 0 < sum(mk) < d], dtype=bool)
        s = masks.sum(axis=1)
        comb = np.array([math.comb(d, k) for k in range(d + 1)], dtype=float)
        w = (d - 1) / (comb[s] * s * (d - s))
    else:  # sampled in proportion to the kernel, so every sample gets the same weight
        masks = sample_coalitions(d, n_samples, rng)
        w = np.ones(n_samples)
    v = masked_predictions(predict, Xb, x, masks)
    Z = masks.astype(float)
    # eliminate phi_d with the efficiency constraint sum(phi) = fx - f0, then weighted least squares
    target = v - f0 - Z[:, -1] * (fx - f0)
    A = Z[:, :-1] - Z[:, -1:]
    sw = np.sqrt(w)
    head = np.linalg.lstsq(A * sw[:, None], target * sw, rcond=None)[0]
    return np.append(head, (fx - f0) - head.sum())


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
