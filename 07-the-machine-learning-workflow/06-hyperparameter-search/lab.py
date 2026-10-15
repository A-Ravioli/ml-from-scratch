import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.base import Classifier, Regressor, check_X_y  # yours, from topic 01
    from mlfs.metrics import get_scorer  # yours, from topic 03
    from mlfs.model_selection import KFold, StratifiedKFold  # yours, from topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    MSE = get_scorer("neg_mean_squared_error")


@app.function
def minkowski(A, B, p):
    """(n, m) Minkowski distances (sum |a - b|^p)^(1/p) between the rows of A and B."""
    return (np.abs(A[:, None, :] - B[None, :, :]) ** p).sum(axis=2) ** (1.0 / p)


@app.class_definition
class KNNRegressor(Regressor):
    """mean target of the k nearest training rows in Minkowski-p distance (a lab helper;
    chapter 09 builds k-NN properly). two hyperparameters: k, and p."""

    def __init__(self, k=5, p=2.0):
        self.k = k
        self.p = p

    def fit(self, X, y):
        self.X_, self.y_ = check_X_y(X, y)
        return self

    def predict(self, X):
        X, _ = check_X_y(X)
        k = min(int(self.k), len(self.X_))
        nearest = np.argpartition(minkowski(X, self.X_, self.p), k - 1, axis=1)[:, :k]
        return self.y_[nearest].mean(axis=1)


@app.class_definition
class KNNClassifier(Classifier):
    """majority vote of the k nearest training rows (Euclidean), a lab helper."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_, y = check_X_y(X, y)
        self.classes_, self.y_idx_ = np.unique(y, return_inverse=True)
        return self

    def predict_proba(self, X):
        X, _ = check_X_y(X)
        k = min(int(self.k), len(self.X_))
        nearest = np.argpartition(minkowski(X, self.X_, 2.0), k - 1, axis=1)[:, :k]
        return np.stack([np.mean(self.y_idx_[nearest] == c, axis=1) for c in range(len(self.classes_))], axis=1)


@app.function
def regression_data(n, rng):
    """4 features; the target is a smooth function of all four plus noise (sd 0.5)."""
    X = rng.uniform(-1, 1, size=(n, 4))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2] - 0.5 * X[:, 3] + 0.5 * rng.normal(size=n)
    return X, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hyperparameter search · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the model in experiments 1 and 3 is a
    $k$-nearest-neighbour regressor with two hyperparameters: $k$, the number of neighbours
    averaged (it matters a lot), and $p$, the exponent of the Minkowski distance
    $(\sum_j |a_j - b_j|^p)^{1/p}$ (it matters little).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: grid versus random, 9 trials each

    300 training points. your `GridSearchCV` tries the 3 × 3 grid $k \in \{5, 50, 95\}$,
    $p \in \{1, 2, 3\}$. your `RandomizedSearchCV` tries 9 random settings, $k$ uniform on
    $1..100$ and $p$ uniform on $[1, 3]$. both use the same 5 folds. we then measure each
    search's chosen model on 5,000 fresh test points, and repeat the random search with 20
    different seeds.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with the same budget of 9 trials, which search finds the better model, and how often? "
        "(the grid is evenly spaced; random is ... random.)"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _X, _y = regression_data(300, np.random.default_rng(0))
    _Xte, _yte = regression_data(5000, np.random.default_rng(1))
    _cv = list(KFold(5, shuffle=True, rng=0).split(_X))
    _grid = ex.GridSearchCV(KNNRegressor(), {"k": [5, 50, 95], "p": [1.0, 2.0, 3.0]}, cv=_cv, scoring=MSE).fit(_X, _y)
    _grid_test = -MSE(_grid, _Xte, _yte)
    _rand_test, _rand_k = [], []
    for _seed in range(20):
        _s = ex.RandomizedSearchCV(KNNRegressor(), {"k": list(range(1, 101)), "p": lambda g: float(g.uniform(1, 3))},
                                   n_iter=9, cv=_cv, scoring=MSE, rng=_seed).fit(_X, _y)
        _rand_test.append(-MSE(_s, _Xte, _yte))
        _rand_k.append(_s.best_params_["k"])
    _ks = np.arange(1, 101)
    _curve = [-MSE(KNNRegressor(int(k), 2.0).fit(_X, _y), _Xte, _yte) for k in _ks]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _a1.plot(_ks, _curve, "k-", lw=1, label="test MSE (p = 2)")
    _a1.plot([5, 50, 95], [_curve[4], _curve[49], _curve[94]], "s", ms=8, color="C0", label="the grid's k values")
    _a1.plot(_rand_k, [_curve[k - 1] for k in _rand_k], "o", color="C1", alpha=0.6, label="random search's chosen k")
    _a1.set(xlabel="k", ylabel="test MSE", title="what k does to the error")
    _a1.legend(fontsize=8)
    _a2.hist(_rand_test, bins=10, color="C1", alpha=0.7, label="random search, 20 seeds")
    _a2.axvline(_grid_test, color="C0", lw=2, label="grid search")
    _a2.set(xlabel="test MSE of the chosen model", ylabel="count")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _wins = np.mean(np.array(_rand_test) < _grid_test)
    mo.vstack([_fig, mo.md(
        f"grid search chose {_grid.best_params_} (test MSE **{_grid_test:.3f}**). random search beat it in "
        f"**{_wins:.0%}** of 20 runs (median test MSE {np.median(_rand_test):.3f}); best possible ≈ {min(_curve):.3f} "
        f"at k = {int(_ks[np.argmin(_curve)])}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how many distinct values of $k$ did each search try? $p$ hardly matters:
    what did the grid spend two thirds of its budget on? would a finer grid fix it, and what
    would it cost with, say, 5 hyperparameters?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the search's own score is optimistic

    pure noise: 80 points, 10 random features, coin-flip labels. your `GridSearchCV` tunes a
    $k$-NN classifier over $k \in \{1, 3, 5, \dots, 39\}$ with 5-fold stratified CV, and
    reports `best_score_`. your `nested_cross_validate` wraps the whole search in an outer
    5-fold CV. 10 datasets.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("on coin-flip labels, what will best_score_ be on average? and the nested CV score?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _best, _nested = [], []
    for _seed in range(10):
        _rng = np.random.default_rng(10 + _seed)
        _X = _rng.normal(size=(80, 10))
        _y = np.repeat([0, 1], 40)
        _make = lambda: ex.GridSearchCV(KNNClassifier(), {"k": list(range(1, 40, 2))},  # noqa: E731
                                        cv=StratifiedKFold(5, shuffle=True, rng=_seed))
        _best.append(_make().fit(_X, _y).best_score_)
        _nested.append(np.mean(ex.nested_cross_validate(_make(), _X, _y, cv=StratifiedKFold(5, shuffle=True, rng=99))))
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    _ax.boxplot([_best, _nested])
    _ax.set_xticks([1, 2], ["best_score_ of the search", "nested CV"])
    _ax.axhline(0.5, color="gray", ls=":", label="true accuracy")
    _ax.set(ylabel="accuracy (10 noise datasets)", ylim=(0.2, 0.9))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"mean best_score_: **{np.mean(_best):.3f}**; mean nested CV: **{np.mean(_nested):.3f}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** every candidate is equally useless. why is the best of 20 cross-validated
    scores above 0.5? what does nested CV do differently, and what is its number an estimate
    *of*? why is the nested estimate so variable from dataset to dataset?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how many random trials do you need?

    back to the regression problem of experiment 1. your `GridSearchCV` first scores a dense
    reference grid of 500 settings ($k = 1..100$, five values of $p$) once. call a setting
    "good" if it's in the top 5% of those 500. then we draw random searches with your
    `RandomizedSearchCV.candidate_params` (no refitting: each sampled setting is looked up in
    the table) and record how often the best of $n$ samples is good.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "how many random trials do you need to land in the top 5% with 95% probability, whatever "
        "the number of hyperparameters?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X, _y = regression_data(300, np.random.default_rng(0))
    _cv = list(KFold(5, shuffle=True, rng=0).split(_X))
    _ps = [1.0, 1.5, 2.0, 2.5, 3.0]
    _ref = ex.GridSearchCV(KNNRegressor(), {"k": list(range(1, 101)), "p": _ps}, cv=_cv, scoring=MSE, refit=False).fit(_X, _y)
    _table = {(c["k"], c["p"]): s for c, s in zip(_ref.cv_results_["params"], _ref.cv_results_["mean_test_score"])}
    _cut = np.quantile(list(_table.values()), 0.95)
    _ns = np.array([1, 2, 5, 10, 20, 30, 45, 60])
    _hit = []
    for _n in _ns:
        _good = 0
        for _seed in range(300):
            _cands = ex.RandomizedSearchCV(KNNRegressor(), {"k": list(range(1, 101)), "p": _ps}, n_iter=int(_n),
                                           rng=_seed).candidate_params()
            _good += max(_table[(c["k"], c["p"])] for c in _cands) >= _cut
        _hit.append(_good / 300)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.plot(_ns, _hit, "o-", label="measured (300 random searches each)")
    _nn = np.arange(1, 61)
    _ax.plot(_nn, 1 - 0.95**_nn, "k--", lw=1, label=r"$1 - 0.95^n$")
    _ax.axhline(0.95, color="gray", ls=":")
    _ax.set(xlabel="random trials n", ylabel="P(best trial is in the top 5%)", ylim=(0, 1.02))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(" | ".join(f"n={n}: {h:.2f}" for n, h in zip(_ns, _hit)))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** derive the dashed curve: what's the probability that $n$ independent
    draws *all* miss the top 5%? why doesn't the answer depend on how many hyperparameters
    there are? what does "top 5% of the search space" *not* guarantee?
    """)
    return


if __name__ == "__main__":
    app.run()
