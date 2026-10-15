import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Regressor, check_X_y, clone  # yours, from topic 01
    from mlfs.model_selection import KFold, cross_validate  # yours, from topic 02
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bias–variance & complexity · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you a
    class or function to write, and checks it live as you edit. stages 2–4 join
    `mlfs.model_selection`. they use your `clone` (topic 01) and your `KFold` and
    `cross_validate` (topic 02); if those aren't finished, launch with
    `MLFS_REFERENCE=base,model_selection marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `PolynomialRegressor`: a 1-D regressor with a complexity knob, `degree` |
    | 2 | `bias_variance_decomposition`: measure bias², variance and noise by refitting on fresh data → `mlfs.model_selection` |
    | 3 | `learning_curve`: train and validation scores as the training set grows → `mlfs.model_selection` |
    | 4 | `validation_curve`: train and validation scores as a hyperparameter changes → `mlfs.model_selection` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class PolynomialRegressor(Regressor):
    """least-squares polynomial fit of one feature: y ~ sum_j coef_[j] x^j, j = 0..degree."""

    def __init__(self, degree=1):
        self.degree = degree

    def fit(self, X, y):
        """stage 1: fit the coefficients by least squares and return self.

        X, y = check_X_y(X, y) (from mlfs.base); X must have exactly one column (else
        ValueError). build the design matrix with columns 1, x, x^2, ..., x^degree
        (np.vander(x, degree + 1, increasing=True)) and solve the least-squares problem with
        np.linalg.lstsq(A, y, rcond=None) (you wrote your own lstsq in ch 02; NumPy's is the
        fast equivalent). coef_ is a float array (degree + 1,), lowest power first.
        """
        X, y = check_X_y(X, y)
        if X.shape[1] != 1:
            raise ValueError(f"PolynomialRegressor takes one feature, got {X.shape[1]}")
        A = np.vander(X[:, 0], self.degree + 1, increasing=True)
        self.coef_ = np.linalg.lstsq(A, y.astype(float), rcond=None)[0]
        return self

    def predict(self, X):
        """stage 1: the fitted polynomial at each row of X (one column): a float array (n,)."""
        X, _ = check_X_y(X)
        return np.vander(X[:, 0], self.degree + 1, increasing=True) @ self.coef_


@app.cell(hide_code=True)
def _():
    checks.run(1, PolynomialRegressor=PolynomialRegressor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def bias_variance_decomposition(estimator, sample_data, X_test, f_test, *, n_rounds=100, noise_var=0.0, rng=None):
    """stage 2: estimate the bias-variance decomposition of `estimator` by simulation.

    gen = as_rng(rng). for each of n_rounds rounds: draw a fresh training set
    X, y = sample_data(gen), fit clone(estimator) on it, and record its predictions at X_test
    (m test points). f_test (m,) holds the NOISELESS targets f(x) at X_test. with the
    predictions in an (n_rounds, m) array:
        mean prediction  = average over rounds              (m,)
        bias2            = mean over x of (mean prediction - f_test)^2
        variance         = mean over x of the variance over rounds (ddof=0)
        noise            = noise_var (the variance of y around f, which you know)
        expected_error   = bias2 + variance + noise
    returns those four as Python floats in a dict with exactly these keys.
    """
    gen = as_rng(rng)
    f_test = np.asarray(f_test, dtype=float)
    preds = np.empty((n_rounds, len(f_test)))
    for r in range(n_rounds):
        X, y = sample_data(gen)
        preds[r] = clone(estimator).fit(X, y).predict(X_test)
    mean = preds.mean(axis=0)
    bias2 = float(np.mean((mean - f_test) ** 2))
    variance = float(np.mean(preds.var(axis=0)))
    return {"bias2": bias2, "variance": variance, "noise": float(noise_var),
            "expected_error": bias2 + variance + float(noise_var)}


@app.cell(hide_code=True)
def _():
    checks.run(2, bias_variance_decomposition=bias_variance_decomposition)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def materialize_splits(cv, X, y=None, groups=None):
    """(given) a list of (train_idx, test_idx) pairs from an int (KFold), a splitter, or an
    iterable of pairs, so every candidate and every training size sees the SAME folds."""
    if isinstance(cv, (int, np.integer)):
        cv = KFold(int(cv))
    if hasattr(cv, "split"):
        return [(np.asarray(tr), np.asarray(te)) for tr, te in cv.split(X, y, groups)]
    return [(np.asarray(tr), np.asarray(te)) for tr, te in cv]


@app.function
def learning_curve(estimator, X, y, *, train_sizes=(0.1, 0.325, 0.55, 0.775, 1.0), cv=5, scoring=None):
    """stage 3: scores as a function of the training-set size.

    splits = materialize_splits(cv, X, y) (given above). n_max = the size of the SMALLEST
    training fold, and each fraction f in train_sizes (0 < f <= 1, else ValueError) becomes
    the size max(1, int(f * n_max)). for every split and every size: fit on the FIRST `size`
    rows of that split's training indices, train_idx[:size], and score on those same rows
    (train score) and on the split's test rows (validation score), with
    cross_validate(estimator, X, y, cv=[(train_idx[:size], test_idx)], scoring=scoring,
    return_train_score=True) from topic 02.
    returns (sizes int array (s,), train_scores (s, k), val_scores (s, k)).
    (to get random subsets, pass a shuffled cv: the subsets are prefixes of each fold.)
    """
    splits = materialize_splits(cv, X, y)
    n_max = min(len(tr) for tr, _ in splits)
    fractions = np.asarray(train_sizes, dtype=float)
    if np.any(fractions <= 0) or np.any(fractions > 1):
        raise ValueError(f"train_sizes must be fractions in (0, 1], got {train_sizes}")
    sizes = np.array([max(1, int(f * n_max)) for f in fractions])
    train_scores = np.empty((len(sizes), len(splits)))
    val_scores = np.empty((len(sizes), len(splits)))
    for j, (tr, te) in enumerate(splits):
        for i, size in enumerate(sizes):
            out = cross_validate(estimator, X, y, cv=[(tr[:size], te)], scoring=scoring, return_train_score=True)
            train_scores[i, j] = out["train_score"][0]
            val_scores[i, j] = out["test_score"][0]
    return sizes, train_scores, val_scores


@app.cell(hide_code=True)
def _():
    checks.run(3, learning_curve=learning_curve)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def validation_curve(estimator, X, y, param_name, param_range, *, cv=5, scoring=None):
    """stage 4: scores as a function of one hyperparameter.

    splits = materialize_splits(cv, X, y), computed ONCE so every value is scored on the same
    folds. for each value v in param_range: est = clone(estimator).set_params(**{param_name: v})
    (never modify `estimator` itself), then cross_validate(est, X, y, cv=splits,
    scoring=scoring, return_train_score=True).
    returns (train_scores (p, k), val_scores (p, k)), row i for param_range[i].
    """
    splits = materialize_splits(cv, X, y)
    train_scores, val_scores = [], []
    for value in param_range:
        est = clone(estimator).set_params(**{param_name: value})
        out = cross_validate(est, X, y, cv=splits, scoring=scoring, return_train_score=True)
        train_scores.append(out["train_score"])
        val_scores.append(out["test_score"])
    return np.array(train_scores), np.array(val_scores)


@app.cell(hide_code=True)
def _():
    checks.run(4, validation_curve=validation_curve)
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
