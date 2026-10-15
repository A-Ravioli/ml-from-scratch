import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.base import DummyRegressor, Regressor, Transformer, check_X_y, clone  # yours, from topic 01
    from mlfs.metrics import get_scorer, r2_score  # yours, from topic 03
    from mlfs.model_selection import (  # yours, from topics 02, 04 and 06
        GridSearchCV,
        KFold,
        nested_cross_validate,
        train_test_split,
    )
    from mlfs.preprocessing import PolynomialFeatures, StandardScaler, make_pipeline  # yours, from topic 05
    from mlfs.stats import bootstrap_ci, permutation_test  # yours, from ch 05
    from mlfs.utils import as_rng, pairwise_distances  # yours, from ch 00

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    N_BOOT = 1_000 if SMALL else 2_000
    KS = [1, 3, 5, 10, 20, 40] if SMALL else [1, 3, 5, 10, 20, 40, 80]
    OUTER = 3  # outer folds of the nested CV
    NEG_MSE = get_scorer("neg_mean_squared_error")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 · capstone: an end-to-end tabular project (reference solution)

    read [`brief.md`](brief.md) first. you'll predict the median house value of California
    districts from eight numbers, and, more importantly, *measure* how well you do without
    fooling yourself: a locked test set, baselines, cross-validation with confidence
    intervals, paired comparisons, honest tuning, and one final look at the test set.

    | stage | you build |
    |---|---|
    | 1 | `KNNRegressor`: $k$-nearest-neighbour regression, computed in chunks so memory stays bounded |
    | 2 | `LeastSquaresRegressor`: linear least squares with an intercept |
    | 3 | `cross_val_predict`: out-of-fold predictions, one per row, each from a model that never saw it |
    | 4 | `rmse_ci`, `paired_comparison`: an RMSE with a bootstrap CI, and a paired test between two models |
    | 5 | `tuned_knn`: a scaler + kNN pipeline whose $k$ is chosen by grid search, inside the folds |

    the data loading and the evaluation report (experiments A–E) are given: they run on your
    functions as soon as they're green. the write-up prompts are at the end.
    """)
    return


@app.class_definition
class KNNRegressor(Regressor):
    """predicts the mean target of the k training rows nearest to x (Euclidean distance)."""

    def __init__(self, k=5, *, chunk_size=512):
        self.k = k
        self.chunk_size = chunk_size

    def fit(self, X, y):
        """stage 1: memorize the training data (X_ float (n, d), y_ float (n,)) and return self.
        X, y = check_X_y(X, y); k < 1 raises ValueError."""
        if self.k < 1:
            raise ValueError(f"k must be at least 1, got {self.k}")
        self.X_, y = check_X_y(X, y)
        self.y_ = y.astype(float)
        return self

    def predict(self, X):
        """stage 1: the mean of y_ over the k nearest training rows, for every row of X.

        X = check_X_y(X)[0], with the same number of columns as in fit (else ValueError);
        k > the number of training rows raises ValueError. process X in chunks of chunk_size
        rows: for each chunk, the squared distances to every training row with
        pairwise_distances(chunk, X_, squared=True) (mlfs.utils, yours from ch 00), the k
        smallest per row with np.argpartition (no full sort needed), and the mean of their
        targets. so memory is O(chunk_size * n_train), never O(len(X) * n_train).
        """
        X, _ = check_X_y(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError(f"fitted on {self.X_.shape[1]} features, got {X.shape[1]}")
        k = int(self.k)
        if k > len(self.X_):
            raise ValueError(f"k = {k} is more than the {len(self.X_)} training rows")
        out = np.empty(len(X))
        for start in range(0, len(X), self.chunk_size):
            D = pairwise_distances(X[start:start + self.chunk_size], self.X_, squared=True)
            nearest = np.argpartition(D, k - 1, axis=1)[:, :k]
            out[start:start + self.chunk_size] = self.y_[nearest].mean(axis=1)
        return out


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", KNNRegressor=KNNRegressor)
    return


@app.class_definition
class LeastSquaresRegressor(Regressor):
    """ordinary least squares: y ~ X @ coef_ + intercept_."""

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """stage 2: minimize ||X coef + intercept - y||^2 and return self.

        X, y = check_X_y(X, y). with fit_intercept, solve least squares for [intercept, coef]
        on the design matrix [1, X] (a column of ones first); without it, on X, and
        intercept_ = 0.0. use np.linalg.lstsq(A, y, rcond=None) (your ch 02 lstsq's fast
        equivalent). coef_ is float (d,), intercept_ a Python float.
        """
        X, y = check_X_y(X, y)
        A = np.column_stack([np.ones(len(X)), X]) if self.fit_intercept else X
        sol = np.linalg.lstsq(A, y.astype(float), rcond=None)[0]
        self.intercept_ = float(sol[0]) if self.fit_intercept else 0.0
        self.coef_ = sol[1:] if self.fit_intercept else sol
        return self

    def predict(self, X):
        """stage 2: X @ coef_ + intercept_."""
        X, _ = check_X_y(X)
        return X @ self.coef_ + self.intercept_


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", LeastSquaresRegressor=LeastSquaresRegressor)
    return


@app.function
def cross_val_predict(estimator, X, y, *, cv=5):
    """stage 3: out-of-fold predictions: a float array (n,) where row i is predicted by a
    clone of `estimator` fitted on the training rows of the fold in which i is a TEST row.

    cv: an int k (-> KFold(k)), a splitter (split(X, y)), or a list of (train, test) pairs.
    every row must be a test row exactly once, else ValueError (that's what makes the
    predictions "out of fold": a model never predicts a row it was fitted on).
    """
    X, y = np.asarray(X), np.asarray(y)
    cv = KFold(int(cv)) if isinstance(cv, (int, np.integer)) else cv
    splits = cv.split(X, y) if hasattr(cv, "split") else cv
    pred = np.full(len(y), np.nan)
    seen = np.zeros(len(y), dtype=bool)
    for train, test in splits:
        if seen[test].any():
            raise ValueError("a row appears in two test folds")
        pred[test] = clone(estimator).fit(X[train], y[train]).predict(X[test])
        seen[test] = True
    if not seen.all():
        raise ValueError(f"{int((~seen).sum())} rows are in no test fold")
    return pred


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", cross_val_predict=cross_val_predict)
    return


@app.function
def rmse_ci(y_true, y_pred, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 4: (rmse, lo, hi): the root mean squared error and a (1 - alpha) percentile
    bootstrap CI for it, resampling the per-row squared errors with mlfs.stats.bootstrap_ci
    (statistic: the square root of their mean). all three Python floats."""
    se = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    rmse = float(np.sqrt(se.mean()))
    lo, hi = bootstrap_ci(se, lambda s: float(np.sqrt(np.mean(s))), n_boot=n_boot, alpha=alpha, rng=rng)
    return rmse, float(lo), float(hi)


@app.function
def paired_comparison(y_true, pred_a, pred_b, *, n_boot=2000, n_permutations=2000, alpha=0.05, rng=None):
    """stage 4: is model b better than model a on the SAME rows?

    with per-row squared errors e_a and e_b, returns a dict:
      "mse_a", "mse_b":  their means
      "diff":            mse_a - mse_b (positive: b has the lower error)
      "ci":              (lo, hi), a percentile bootstrap CI for the mean of e_a - e_b
                         (resample the per-row differences: that keeps the pairing)
      "p_value":         from mlfs.stats.permutation_test(e_a, e_b, paired=True), two-sided
    use one generator, gen = as_rng(rng): first for the bootstrap, then for the permutations.
    """
    gen = as_rng(rng)
    y = np.asarray(y_true, dtype=float)
    ea = (y - np.asarray(pred_a, dtype=float)) ** 2
    eb = (y - np.asarray(pred_b, dtype=float)) ** 2
    lo, hi = bootstrap_ci(ea - eb, np.mean, n_boot=n_boot, alpha=alpha, rng=gen)
    _, p = permutation_test(ea, eb, paired=True, n_permutations=n_permutations, rng=gen)
    return {"mse_a": float(ea.mean()), "mse_b": float(eb.mean()), "diff": float(ea.mean() - eb.mean()),
            "ci": (float(lo), float(hi)), "p_value": float(p)}


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", rmse_ci=rmse_ci, paired_comparison=paired_comparison)
    return


@app.function
def tuned_knn(ks, *, cv=5, chunk_size=512):
    """stage 5: an UNFITTED GridSearchCV over the number of neighbours of a scaled kNN.

    the estimator is make_pipeline(StandardScaler(), KNNRegressor(chunk_size=chunk_size)), so
    its steps are named "standardscaler" and "knnregressor"; the grid is
    {"knnregressor__k": list(ks)}; scoring is get_scorer("neg_mean_squared_error"); cv is
    passed on. because the scaler is inside the pipeline, every candidate in every fold is
    scaled with statistics from its own training rows only.
    """
    pipe = make_pipeline(StandardScaler(), KNNRegressor(chunk_size=chunk_size))
    return GridSearchCV(pipe, {"knnregressor__k": list(ks)}, cv=cv, scoring=get_scorer("neg_mean_squared_error"))


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", tuned_knn=tuned_knn)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data, and the locked test set (given)

    California housing (1990 census): 8 numeric features per district, target = median
    house value in units of \$100,000. small config: the 1,000-row synthetic stand-in. 20% of
    the rows are split off **now**, before anything else, and not looked at again until
    experiment E.
    """)
    return


@app.cell
def _():
    X_all, y_all, feature_names = datasets.california_housing()
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, rng=0)
    CV = KFold(5, shuffle=True, rng=1)
    mo.md(f"{len(X_all):,} rows · train {len(X_train):,} · test {len(X_test):,} (locked) · features: "
          + ", ".join(feature_names) + f"\n\ntarget: mean {y_train.mean():.2f}, sd {y_train.std():.2f} (×$100k)")
    return CV, X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A (given): baselines and models, by cross-validation

    every model is evaluated by 5-fold CV on the training set with your `cross_val_predict`,
    and every RMSE gets a 95% bootstrap CI from your `rmse_ci`. the two dummies are the
    baselines; a model that doesn't beat them hasn't learned anything.
    """)
    return


@app.cell
def _(CV, X_train, y_train):
    models = {
        "mean (baseline)": DummyRegressor(),
        "median (baseline)": DummyRegressor(strategy="median"),
        "least squares": LeastSquaresRegressor(),
        "least squares, degree-2 features": make_pipeline(StandardScaler(), PolynomialFeatures(2), LeastSquaresRegressor()),
        "kNN (k = 10), raw features": KNNRegressor(10),
        "kNN (k = 10), standardized": make_pipeline(StandardScaler(), KNNRegressor(10)),
    }
    oof = {name: cross_val_predict(m, X_train, y_train, cv=CV) for name, m in models.items()}
    _rows = []
    for _name, _pred in oof.items():
        _r, _lo, _hi = rmse_ci(y_train, _pred, n_boot=N_BOOT, rng=0)
        _rows.append(f"| {_name} | {_r:.3f} | [{_lo:.3f}, {_hi:.3f}] | {r2_score(y_train, _pred):.3f} |")
    mo.md("| model | CV RMSE | 95% CI | CV $R^2$ |\n|---|---|---|---|\n" + "\n".join(_rows))
    return models, oof


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B (given): paired comparisons

    are the differences in experiment A real? each comparison uses the same out-of-fold rows
    for both models, with your `paired_comparison` (bootstrap CI of the MSE difference and a
    paired permutation test).
    """)
    return


@app.cell
def _(oof, y_train):
    _pairs = [("mean (baseline)", "least squares"),
              ("least squares", "least squares, degree-2 features"),
              ("kNN (k = 10), raw features", "kNN (k = 10), standardized"),
              ("least squares, degree-2 features", "kNN (k = 10), standardized")]
    _rows = []
    for _a, _b in _pairs:
        _c = paired_comparison(y_train, oof[_a], oof[_b], n_boot=N_BOOT, n_permutations=N_BOOT, rng=0)
        _rows.append(f"| {_a} → {_b} | {_c['diff']:+.4f} | [{_c['ci'][0]:+.4f}, {_c['ci'][1]:+.4f}] | {_c['p_value']:.4f} |")
    mo.md("| comparison (a → b) | MSE(a) − MSE(b) | 95% CI | permutation p |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C (given): tuning $k$, honestly

    your `tuned_knn` searches $k$ with 5-fold CV. its `best_score_` is the CV score of the
    winner, which was *chosen* for being the best. nested CV (your `nested_cross_validate`,
    3 outer folds) evaluates the whole tuning procedure on rows it never saw.
    """)
    return


@app.cell
def _(CV, X_train, y_train):
    search = tuned_knn(KS, cv=CV).fit(X_train, y_train)
    _res = search.cv_results_
    _nested = nested_cross_validate(tuned_knn(KS, cv=5), X_train, y_train, cv=KFold(OUTER, shuffle=True, rng=2),
                                    scoring=NEG_MSE)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    _ax.semilogx([p["knnregressor__k"] for p in _res["params"]], np.sqrt(-np.array(_res["mean_test_score"])), "o-")
    _ax.set(xlabel="k (neighbours)", ylabel="CV RMSE", title="validation curve of the scaled kNN")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"best k = **{search.best_params_['knnregressor__k']}**, best CV RMSE (from best_score_) "
        f"**{np.sqrt(-search.best_score_):.3f}**; nested CV RMSE per outer fold "
        f"{np.round(np.sqrt(-_nested), 3).tolist()}, mean **{np.sqrt(-np.mean(_nested)):.3f}**")])
    return (search,)


@app.class_definition
class CellTargetEncoder(Transformer):
    """(given) target encoding of location: appends one column, the mean training target of
    the districts in the same longitude/latitude cell (cell x cell degrees; columns 0 and 1),
    or the overall training mean for a cell never seen in fit. it LEARNS FROM y, so where it
    is fitted matters."""

    def __init__(self, cell=0.05):
        self.cell = cell

    def cell_ids(self, X):
        X = np.asarray(X, dtype=float)
        return [(int(a), int(b)) for a, b in np.floor(X[:, :2] / self.cell)]

    def fit(self, X, y):
        sums = {}
        for c, t in zip(self.cell_ids(X), np.asarray(y, dtype=float)):
            s, n = sums.get(c, (0.0, 0))
            sums[c] = (s + t, n + 1)
        self.means_ = {c: s / n for c, (s, n) in sums.items()}
        self.global_mean_ = float(np.mean(y))
        return self

    def transform(self, X):
        enc = [self.means_.get(c, self.global_mean_) for c in self.cell_ids(X)]
        return np.column_stack([np.asarray(X, dtype=float), enc])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment D (given): three ways to fool yourself

    1. **scaling outside the folds.** standardize all training rows first, then cross-validate
       kNN on the scaled matrix, versus the pipeline that scales inside each fold. (expect
       almost no difference: with 16,000 rows, means and standard deviations barely move
       between folds, and they carry no label information.)
    2. **a label-aware step outside the folds.** add a target-encoded location feature (the
       mean house value of each district's 0.05° cell, about 5 km across: `CellTargetEncoder`
       above) to least squares, once fitted on all training rows before CV, and once inside a
       pipeline.
    3. **tuning on the test set.** compute the test RMSE for every $k$ from 1 to 80 and report
       the best, versus the test RMSE of the $k$ that cross-validation chose. (this cell peeks
       at the test set on purpose, to show what peeking does; experiment E is the real
       evaluation. with 4,000 test rows and a smooth curve, expect the optimism to be
       small: it grows with the number of choices and shrinks with the size of the test set.)
    """)
    return


@app.cell
def _(CV, X_test, X_train, search, y_test, y_train):
    _k = search.best_params_["knnregressor__k"]
    _rmse = lambda pred, y: float(np.sqrt(np.mean((pred - y) ** 2)))  # noqa: E731
    _Xs = StandardScaler().fit_transform(X_train)
    _leaky = cross_val_predict(KNNRegressor(_k), _Xs, y_train, cv=CV)
    _clean = cross_val_predict(make_pipeline(StandardScaler(), KNNRegressor(_k)), X_train, y_train, cv=CV)
    _Xenc = CellTargetEncoder().fit(X_train, y_train).transform(X_train)
    _te_leaky = cross_val_predict(LeastSquaresRegressor(), _Xenc, y_train, cv=CV)
    _te_clean = cross_val_predict(make_pipeline(CellTargetEncoder(), LeastSquaresRegressor()), X_train, y_train, cv=CV)
    _ls = cross_val_predict(LeastSquaresRegressor(), X_train, y_train, cv=CV)
    # test RMSE for every k in one pass: neighbours of each test row sorted once, then running means
    _sc = StandardScaler().fit(X_train)
    _A, _B = _sc.transform(X_train), _sc.transform(X_test)
    _kmax = 80
    _cum = np.empty((len(_B), _kmax))
    for _s in range(0, len(_B), 512):
        _D = pairwise_distances(_B[_s:_s + 512], _A, squared=True)
        _nn = np.argpartition(_D, _kmax - 1, axis=1)[:, :_kmax]
        _nn = np.take_along_axis(_nn, np.argsort(np.take_along_axis(_D, _nn, axis=1), axis=1), axis=1)
        _cum[_s:_s + 512] = np.cumsum(y_train[_nn], axis=1) / np.arange(1, _kmax + 1)
    _test_rmse = np.sqrt(np.mean((_cum - y_test[:, None]) ** 2, axis=0))
    _peek_k = int(np.argmin(_test_rmse)) + 1
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.plot(np.arange(1, _kmax + 1), _test_rmse)
    _ax.axvline(_k, color="C1", ls="--", label=f"CV's choice, k = {_k}")
    _ax.plot([_peek_k], [_test_rmse[_peek_k - 1]], "o", color="C3", label=f"best on the test set, k = {_peek_k}")
    _ax.set(xlabel="k", ylabel="test RMSE (peeking!)", xscale="log")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(
        f"1. scaling: CV RMSE outside the folds **{_rmse(_leaky, y_train):.4f}**, inside (pipeline) "
        f"**{_rmse(_clean, y_train):.4f}**.\n\n"
        f"2. target encoding: CV RMSE of least squares without it {_rmse(_ls, y_train):.3f}; with it, encoded "
        f"outside the folds **{_rmse(_te_leaky, y_train):.3f}**, inside (pipeline) **{_rmse(_te_clean, y_train):.3f}**.\n\n"
        f"3. peeking: best test RMSE over k = 1..{_kmax} is **{_test_rmse[_peek_k - 1]:.4f}** (k = {_peek_k}); the "
        f"CV-chosen k = {_k} gets **{_test_rmse[_k - 1]:.4f}**."), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment E (given): the one look at the test set

    the final model is the tuned search refitted on all training rows (that's what
    `search.best_estimator_` already is). it's compared with the mean baseline, fitted on the
    same training rows, on the locked test set, once.
    """)
    return


@app.cell
def _(X_test, X_train, search, y_test, y_train):
    _final = search.predict(X_test)
    _base = DummyRegressor().fit(X_train, y_train).predict(X_test)
    _r, _lo, _hi = rmse_ci(y_test, _final, n_boot=N_BOOT, rng=3)
    _rb, _lob, _hib = rmse_ci(y_test, _base, n_boot=N_BOOT, rng=3)
    _c = paired_comparison(y_test, _base, _final, n_boot=N_BOOT, n_permutations=N_BOOT, rng=4)
    mo.md(f"| model | test RMSE | 95% CI | test $R^2$ |\n|---|---|---|---|\n"
          f"| mean baseline | {_rb:.3f} | [{_lob:.3f}, {_hib:.3f}] | {r2_score(y_test, _base):.3f} |\n"
          f"| tuned kNN (k = {search.best_params_['knnregressor__k']}) | {_r:.3f} | [{_lo:.3f}, {_hi:.3f}] | "
          f"{r2_score(y_test, _final):.3f} |\n\n"
          f"improvement in MSE over the baseline: {_c['diff']:.3f}, 95% CI [{_c['ci'][0]:.3f}, {_c['ci'][1]:.3f}], "
          f"permutation p = {_c['p_value']:.4f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your evaluation report (grade it with [`rubric.md`](rubric.md)).
    answer at least:

    1. **the result.** what is your final model, and how well does it predict house values on
       the test set? give the RMSE in dollars with its CI, the baseline's, and $R^2$. is the
       test RMSE consistent with the nested CV estimate from experiment C?
    2. **what mattered.** from experiments A and B: which differences between models are
       real (CI excludes 0, small p), and which are within the noise? why does standardizing
       matter so much for kNN and not at all for least squares?
    3. **the cost of choosing.** compare `best_score_` with the nested CV estimate
       (experiment C), and the "best k on the test set" with the CV-chosen k (experiment D).
       which number would you report, and why?
    4. **leakage.** compare the two leaks of experiment D: why is the scaling leak invisible
       and the target-encoding leak large? why does the leaky encoding help most for cells with
       few districts? how does the pipeline prevent it?
    5. **limits.** what do your CIs *not* account for (think: the train/test split itself,
       the CV folds, the fact that districts near each other are similar)? what would you do
       with more time?
    """)
    return


if __name__ == "__main__":
    app.run()
