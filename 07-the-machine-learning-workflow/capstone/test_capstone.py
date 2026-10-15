"""Checks for capstone.py (an end-to-end tabular project).

All data here is synthetic and deterministic. The library pieces (pipelines, searches,
bootstrap) come from mlfs (the reference solutions under pytest); these checks test your
models, your out-of-fold predictions, your statistics and your tuned pipeline, plus minimum
metric thresholds on held-out synthetic data.
"""
import tracemalloc

import numpy as np
import pytest

from mlfs.testing import bounded


def _housing_like(n, seed):
    """three features on wildly different scales; the target depends on all three."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(0, 5000, n), rng.uniform(0, 0.01, n)])
    y = np.sin(2 * X[:, 0]) + (X[:, 1] / 2500 - 1) ** 2 + 100 * X[:, 2] + 0.2 * rng.normal(size=n)
    return X, y


# ---------------------------------------------------------------- stage 1: KNNRegressor


@pytest.mark.parametrize("k", [1, 4, 9])
def test_stage1_matches_brute_force(nb, k):
    rng = np.random.default_rng(k)
    X, y = rng.normal(size=(120, 3)), rng.normal(size=120)
    Xq = rng.normal(size=(37, 3))
    ours = np.asarray(nb.KNNRegressor(k, chunk_size=10).fit(X, y).predict(Xq))
    D = ((Xq[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)  # the slow, obvious oracle
    ref = y[np.argsort(D, axis=1)[:, :k]].mean(axis=1)
    assert ours.shape == (37,) and np.allclose(ours, ref), (
        f"k = {k}: each prediction is the mean target of the {k} nearest training rows; check the chunk boundaries "
        "too (chunk_size=10 with 37 query rows leaves a partial last chunk)")


def test_stage1_protocol(nb):
    X, y = _housing_like(50, 0)
    m = nb.KNNRegressor(1).fit(X, y)
    assert np.allclose(m.predict(X), y), "with k = 1, each training row is its own nearest neighbour"
    assert nb.KNNRegressor(3).get_params() == {"k": 3, "chunk_size": 512}, "__init__ stores k and chunk_size"
    assert -1 < nb.KNNRegressor(5).fit(X, y).score(X, y) <= 1, "it's a Regressor: score is R^2"
    with pytest.raises(ValueError):
        nb.KNNRegressor(51).fit(X, y).predict(X[:2])
        pytest.fail("k = 51 with 50 training rows should raise ValueError")
    with pytest.raises(ValueError):
        nb.KNNRegressor(3).fit(X, y).predict(X[:, :2])
        pytest.fail("predicting with a different number of features should raise ValueError")


def test_stage1_memory_is_chunked(nb):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(3000, 4)), rng.normal(size=3000)
    Xq = rng.normal(size=(3000, 4))
    m = nb.KNNRegressor(5, chunk_size=100).fit(X, y)
    tracemalloc.start()
    m.predict(Xq)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    full = 3000 * 3000 * 8
    assert peak < full / 4, (
        f"predict peaked at {peak / 1e6:.0f} MB; the full distance matrix would be {full / 1e6:.0f} MB. compute distances "
        "chunk_size query rows at a time, so memory stays O(chunk_size * n_train)")


# ---------------------------------------------------------------- stage 2: LeastSquaresRegressor


def test_stage2_recovers_coefficients(nb):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    y = 1.5 + X @ np.array([2.0, -1.0, 0.5])
    m = nb.LeastSquaresRegressor().fit(X, y)
    assert np.allclose(m.coef_, [2.0, -1.0, 0.5]) and np.isclose(m.intercept_, 1.5), (
        f"noiseless data should give coef_ = [2, -1, 0.5] and intercept_ = 1.5; got {m.coef_}, {m.intercept_}")
    assert isinstance(m.intercept_, float) and np.asarray(m.coef_).shape == (3,)
    assert np.allclose(m.predict(X[:4]), y[:4])


def test_stage2_normal_equations(nb):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4))
    y = X @ rng.normal(size=4) + 3 + rng.normal(size=60)
    for fi in (True, False):
        A = np.column_stack([np.ones(60), X]) if fi else X
        ref = np.linalg.solve(A.T @ A, A.T @ y)  # the normal equations (fine here: A is well conditioned)
        ours = nb.LeastSquaresRegressor(fit_intercept=fi).fit(X, y)
        assert np.allclose(ours.coef_, ref[1:] if fi else ref) and np.isclose(ours.intercept_, ref[0] if fi else 0.0), (
            f"fit_intercept={fi}: the coefficients should solve the least-squares problem (they satisfy the normal "
            "equations); without an intercept, intercept_ is 0.0")


# ---------------------------------------------------------------- stage 3: cross_val_predict


class _Spy:
    def __init__(self, tag=0):
        self.tag = tag

    def get_params(self):
        return {"tag": self.tag}

    def set_params(self, **p):
        self.tag = p.get("tag", self.tag)
        return self

    def fit(self, X, y):
        self.rows_ = set(np.asarray(X)[:, 0].tolist())
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        rows = np.asarray(X)[:, 0]
        assert not (set(rows.tolist()) & self.rows_), "a model was asked to predict a row it was fitted on"
        return np.full(len(rows), self.mean_)


def test_stage3_out_of_fold(nb):
    from mlfs.model_selection import KFold
    X = np.arange(20.0).reshape(-1, 1)
    y = np.arange(20.0) ** 2
    spy = _Spy()
    pred = np.asarray(nb.cross_val_predict(spy, X, y, cv=KFold(4, shuffle=True, rng=0)))
    assert pred.shape == (20,) and np.all(np.isfinite(pred)), "one prediction per row"
    assert not hasattr(spy, "rows_"), "fit a clone in each fold; the estimator you're given stays unfitted"
    for tr, te in KFold(4, shuffle=True, rng=0).split(X):
        assert np.allclose(pred[te], y[tr].mean()), "rows of each test fold are predicted by the model fitted on its training rows"
    assert np.allclose(nb.cross_val_predict(_Spy(), X, y, cv=4)[:5], y[5:].mean()), "an int cv means KFold(cv)"


def test_stage3_rejects_incomplete_splits(nb):
    X, y = np.arange(6.0).reshape(-1, 1), np.arange(6.0)
    with pytest.raises(ValueError):
        nb.cross_val_predict(_Spy(), X, y, cv=[(np.array([0, 1, 2]), np.array([3, 4, 5]))])
        pytest.fail("rows 0-2 are never in a test fold: out-of-fold predictions don't exist for them; raise ValueError")
    with pytest.raises(ValueError):
        nb.cross_val_predict(_Spy(), X, y, cv=[(np.array([0, 1]), np.array([2, 3, 4, 5])),
                                              (np.array([4, 5]), np.array([0, 1, 2, 3]))])
        pytest.fail("rows 2 and 3 are in two test folds: raise ValueError")


# ---------------------------------------------------------------- stage 4: rmse_ci, paired_comparison


def test_stage4_rmse_ci(nb):
    rng = np.random.default_rng(4)
    y = rng.normal(size=400)
    pred = y + rng.normal(scale=0.5, size=400)
    r, lo, hi = bounded(nb.rmse_ci, y, pred, n_boot=500, rng=0, budget=400_000, what="rmse_ci")
    assert np.isclose(r, np.sqrt(np.mean((y - pred) ** 2))) and all(isinstance(v, float) for v in (r, lo, hi))
    assert lo < r < hi and 0.03 < hi - lo < 0.15, (
        f"the 95% CI for an RMSE of about 0.5 on 400 rows should be roughly +/- 0.035; got [{lo:.3f}, {hi:.3f}]")
    assert nb.rmse_ci(y, pred, n_boot=500, rng=0) == (r, lo, hi), "the same rng must give the same interval"


def test_stage4_rmse_ci_coverage(nb):
    hits = 0
    for seed in range(60):
        rng = np.random.default_rng(100 + seed)
        y = rng.normal(size=150)
        pred = y + rng.normal(size=150)  # true RMSE = 1
        _, lo, hi = nb.rmse_ci(y, pred, n_boot=300, rng=seed)
        hits += lo <= 1.0 <= hi
    assert hits >= 48, f"a 95% CI should contain the true RMSE (1.0) in about 57 of 60 datasets; yours did in {hits}"


def test_stage4_paired_comparison(nb):
    rng = np.random.default_rng(5)
    y = rng.normal(size=300)
    good = y + rng.normal(scale=0.3, size=300)
    bad = y + rng.normal(scale=0.6, size=300)
    c = nb.paired_comparison(y, bad, good, n_boot=500, n_permutations=500, rng=0)
    assert set(c) >= {"mse_a", "mse_b", "diff", "ci", "p_value"}, f"missing keys in {sorted(c)}"
    assert np.isclose(c["diff"], np.mean((y - bad) ** 2) - np.mean((y - good) ** 2)) and c["diff"] > 0, (
        "diff = mse_a - mse_b: positive when model b (here the better one) has the lower error")
    assert c["ci"][0] > 0 and c["p_value"] < 0.01, (
        f"a clear improvement: the CI should exclude 0 and p should be small; got ci={c['ci']}, p={c['p_value']}")
    same = nb.paired_comparison(y, good, good.copy(), n_boot=200, n_permutations=200, rng=0)
    assert same["diff"] == 0.0 and same["p_value"] == 1.0, "identical predictions: no difference, p = 1"
    shrunk = 0.95 * good + 0.05 * y  # every prediction 5% closer to the truth: a tiny but consistent gain
    tiny = nb.paired_comparison(y, good, shrunk, n_boot=500, n_permutations=500, rng=2)
    assert tiny["ci"][0] > 0 and tiny["p_value"] < 0.01, (
        f"model b's error is smaller on EVERY row, by about 10%: pairing the rows makes that detectable (CI above 0, "
        f"small p), even though the two MSEs ({tiny['mse_a']:.4f}, {tiny['mse_b']:.4f}) look alike. got ci={tiny['ci']}, "
        f"p={tiny['p_value']}: resample the per-row DIFFERENCES, and use permutation_test(..., paired=True)")
    resid = y - good
    swapped = y - resid[::-1]  # the same errors, on different rows: the MSEs are exactly equal
    null = nb.paired_comparison(y, good, swapped, n_boot=500, n_permutations=500, rng=1)
    assert abs(null["diff"]) < 1e-12 and null["ci"][0] <= 0 <= null["ci"][1] and null["p_value"] > 0.5, (
        f"two models with exactly the same MSE: the CI must contain 0 and p should be large; got ci={null['ci']}, "
        f"p={null['p_value']}")
    from mlfs.stats import bootstrap_ci
    for alpha in (0.05, 0.2):
        c = nb.paired_comparison(y, bad, good, n_boot=300, n_permutations=50, alpha=alpha, rng=3)
        want = bootstrap_ci((y - bad) ** 2 - (y - good) ** 2, np.mean, n_boot=300, alpha=alpha, rng=np.random.default_rng(3))
        assert np.allclose(c["ci"], want), (
            f"alpha={alpha}: the CI should be bootstrap_ci(e_a - e_b, np.mean, n_boot, alpha, rng=gen) with gen = "
            f"as_rng(rng), drawn BEFORE the permutation test: expected {want}, got {c['ci']}")


# ---------------------------------------------------------------- stage 5: tuned_knn


def test_stage5_structure(nb):
    s = nb.tuned_knn([1, 5, 9], cv=3)
    assert type(s).__name__ == "GridSearchCV" and not hasattr(s, "best_estimator_"), "return an UNFITTED GridSearchCV"
    names = [n for n, _ in s.estimator.steps]
    assert names == ["standardscaler", "knnregressor"], f"the pipeline should be scaler then kNN, got steps {names}"
    assert s.param_grid == {"knnregressor__k": [1, 5, 9]} and s.cv == 3, "grid over knnregressor__k, cv passed on"
    class Zero:
        def predict(self, X):
            return np.zeros(len(X))

    yy = np.array([1.0, -2.0, 3.0])
    assert s.scoring is not None and np.isclose(s.scoring(Zero(), np.zeros((3, 1)), yy), -np.mean(yy**2)), (
        "scoring should be get_scorer('neg_mean_squared_error'): -MSE of the predictions")


def test_stage5_beats_the_baseline_on_held_out_data(nb):
    X, y = _housing_like(500, 6)
    Xt, yt = _housing_like(1000, 7)
    s = bounded(nb.tuned_knn([1, 3, 8, 15, 30], cv=4).fit, X, y, budget=3_000_000, what="tuned_knn(...).fit")
    pred = s.predict(Xt)
    r2 = 1 - np.mean((yt - pred) ** 2) / np.var(yt)
    assert s.best_params_["knnregressor__k"] in (3, 8, 15), f"a sensible k should win, got {s.best_params_}"
    assert r2 > 0.75, (
        f"the tuned, scaled kNN should explain most of the variance of the held-out data (R^2 > 0.75); got {r2:.3f}. "
        "is the scaler inside the pipeline?")
    scaler = s.best_estimator_.named_steps["standardscaler"]
    assert np.allclose(scaler.mean_, X.mean(axis=0)), "the refitted pipeline's scaler learns from the training rows"
