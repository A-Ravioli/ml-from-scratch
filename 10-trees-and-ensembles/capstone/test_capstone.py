"""Checks for capstone.py (a tabular benchmark).

Synthetic data only. Feature engineering is checked by hand; early stopping against the
reference booster's own staged predictions; the random search for its bookkeeping and for
actually improving on a bad default; the statistics by simulation (coverage) and by paired
cases where the answer is known; and the whole pipeline (search + early stopping) against
scikit-learn's HistGradientBoostingRegressor on a synthetic regression problem.
Under pytest, mlfs.trees and mlfs.stats are the reference implementations.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from mlfs.testing import bounded
from mlfs.trees import GradientBoostingRegressor


def _friedman(seed, n):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 6))
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
    return X, y + rng.normal(size=n)


# ---------------------------------------------------------------- stage 1: features


def test_stage1_ratio_features(nb):
    X = np.array([[-122.0, 37.0, 20.0, 600.0, 120.0, 900.0, 200.0, 5.0],
                  [-118.0, 34.0, 35.0, 0.0, 0.0, 300.0, 0.0, 2.0]])
    X0 = X.copy()
    out = np.asarray(nb.add_ratio_features(X))
    assert out.shape == (2, 11) and np.array_equal(out[:, :8], X), (
        "keep the 8 original columns first and append 3 ratios: (n, 8) -> (n, 11)")
    assert np.allclose(out[0, 8:], [600 / 200, 120 / 600, 900 / 200]), (
        f"rooms per household = col 3 / col 6, bedrooms per room = col 4 / col 3, people per household = "
        f"col 5 / col 6: expected [3.0, 0.2, 4.5], got {out[0, 8:].tolist()}")
    assert np.allclose(out[1, 8:], 0.0) and np.all(np.isfinite(out)), (
        "a zero denominator gives 0 (not inf or nan, which would break check_X_y)")
    assert np.array_equal(X, X0), "don't modify X in place"


# ---------------------------------------------------------------- stage 2: early stopping


def test_stage2_staged_rmse_and_best_iteration(nb):
    X, y = _friedman(0, 400)
    Xv, yv = _friedman(1, 300)
    m = GradientBoostingRegressor(n_estimators=60, learning_rate=0.3, max_depth=3).fit(X, y)
    curve = np.asarray(nb.staged_rmse(m, Xv, yv))
    want = np.array([np.sqrt(np.mean((p - yv) ** 2)) for p in m.staged_predict(Xv)])
    assert curve.shape == (60,) and np.allclose(curve, want), (
        "staged_rmse: one RMSE per tree, from staged_predict (the prediction after 1, 2, ... trees)")
    n, r = nb.best_iteration(curve)
    assert n == int(np.argmin(want)) + 1 and np.isclose(r, want.min()), (
        f"best_iteration counts trees from 1: argmin + 1 = {int(np.argmin(want)) + 1}, got {n}")
    assert nb.best_iteration(np.array([3.0, 2.0, 2.0, 5.0])) == (2, 2.0), "ties go to the smaller number of trees"


def test_stage2_predict_with_n_trees(nb):
    X, y = _friedman(2, 300)
    m = GradientBoostingRegressor(n_estimators=12, learning_rate=0.3).fit(X, y)
    staged = [np.array(p) for p in m.staged_predict(X)]
    for k in (1, 7, 12):
        assert np.allclose(nb.predict_with_n_trees(m, X, k), staged[k - 1]), f"n_trees={k}: the {k}-th staged prediction"
    for bad in (0, 13):
        try:
            nb.predict_with_n_trees(m, X, bad)
        except ValueError:
            continue
        raise AssertionError(f"n_trees={bad} is impossible for a 12-tree model: raise ValueError")


# ---------------------------------------------------------------- stage 3: random search


def test_stage3_random_search(nb):
    X, y = _friedman(3, 500)
    Xv, yv = _friedman(4, 300)
    space = {"learning_rate": [0.05, 0.3], "max_depth": [1, 3, 5], "reg_lambda": [0.0, 10.0]}
    seen = []

    def make_model(**params):
        seen.append(params)
        return GradientBoostingRegressor(n_estimators=40, **params)

    res = bounded(nb.random_search, make_model, space, X, y, Xv, yv, n_iter=5, rng=0, budget=30_000_000,
                  what="random_search", hint="exactly n_iter candidates")
    assert len(res) == 5 and len(seen) == 5, f"n_iter=5 candidates, each fitted once; got {len(res)} results"
    assert all(set(r) >= {"params", "model", "n_trees", "val_rmse", "seconds"} for r in res), (
        'each result is a dict with "params", "model", "n_trees", "val_rmse" and "seconds"')
    assert [r["val_rmse"] for r in res] == sorted(r["val_rmse"] for r in res), "results sorted by val_rmse, best first"
    for r in res:
        assert all(r["params"][k] in space[k] for k in space) and set(r["params"]) == set(space), (
            "every candidate draws one value for every key of the space")
        m = GradientBoostingRegressor(n_estimators=40, **r["params"]).fit(X, y)
        assert np.allclose(r["model"].predict(Xv), m.predict(Xv)), '"model" is the candidate fitted on the training rows'
        curve = np.array([np.sqrt(np.mean((p - yv) ** 2)) for p in m.staged_predict(Xv)])
        assert r["n_trees"] == int(np.argmin(curve)) + 1 and np.isclose(r["val_rmse"], curve.min()), (
            "n_trees and val_rmse must come from early stopping on the VALIDATION set")
    assert len({tuple(sorted(r["params"].items())) for r in res}) > 1, "draw each candidate afresh: 5 identical candidates"
    again = nb.random_search(make_model, space, X, y, Xv, yv, n_iter=5, rng=0)
    assert [r["params"] for r in again] == [r["params"] for r in res], "the same rng seed must draw the same candidates"


# ---------------------------------------------------------------- stage 4: statistics


def test_stage4_rmse_ci_coverage(nb):
    rng = np.random.default_rng(5)
    hits = 0
    for _ in range(60):
        e = rng.normal(scale=2.0, size=400)
        rmse, lo, hi = nb.rmse_ci(np.zeros(400), e, n_boot=400, rng=rng)
        assert np.isclose(rmse, np.sqrt(np.mean(e ** 2))) and lo <= rmse <= hi, "rmse_ci returns (rmse, lo, hi)"
        hits += lo <= 2.0 <= hi
    assert hits >= 50, f"a 95% CI should contain the true RMSE (2.0) in about 57 of 60 simulations, got {hits}"


def test_stage4_relative_gap(nb):
    rng = np.random.default_rng(6)
    y = rng.normal(size=1500)
    ref = y + rng.normal(scale=1.0, size=1500)
    worse = y + 1.25 * (ref - y) + 0.05 * rng.normal(size=1500)   # about 25% larger errors, same rows
    gap, lo, hi = nb.relative_gap_ci(y, worse, ref, n_boot=1000, rng=0)
    want = np.sqrt(np.mean((worse - y) ** 2) / np.mean((ref - y) ** 2)) - 1
    assert np.isclose(gap, want), f"gap = rmse(pred) / rmse(ref) - 1 = {want:.4f}, got {gap:.4f}"
    assert lo < gap < hi and lo > 0.2 and hi - lo < 0.03, (
        f"the errors are almost perfectly paired, so the PAIRED bootstrap CI must be narrow around +25%: got "
        f"[{lo:.3f}, {hi:.3f}]. resample the rows of (errors of pred, errors of ref) jointly (a tuple for "
        "bootstrap_ci), not each one separately.")
    same = nb.relative_gap_ci(y, ref, ref, n_boot=200, rng=0)
    assert np.allclose(same, 0.0), "a model compared with itself has gap 0 and a CI of [0, 0]"


def test_stage4_verdict(nb):
    assert nb.verdict((0.01, -0.01, 0.025), 0.03) == "within", "upper end below the margin: within"
    assert nb.verdict((0.06, 0.04, 0.08), 0.03) == "outside", "lower end above the margin: outside"
    assert nb.verdict((0.02, -0.01, 0.05), 0.03) == "inconclusive", "the margin inside the CI: inconclusive"
    assert nb.verdict((-0.05, -0.08, -0.02), 0.03) == "within", "better than the benchmark is within the margin"


def test_stage4_pipeline_close_to_benchmark(nb):
    X, y = _friedman(7, 1200)
    Xv, yv = _friedman(8, 600)
    Xt, yt = _friedman(9, 3000)
    space = {"learning_rate": [0.1], "max_depth": [2, 3, 4], "min_child_weight": [1.0, 5.0], "subsample": [0.8, 1.0]}

    def make_model(**params):
        return GradientBoostingRegressor(n_estimators=300, rng=0, **params)

    best = nb.random_search(make_model, space, X, y, Xv, yv, n_iter=4, rng=0)[0]
    ours = nb.predict_with_n_trees(best["model"], Xt, best["n_trees"])
    with threadpool_limits(1):
        bench = HistGradientBoostingRegressor(max_iter=300, early_stopping=False, random_state=0).fit(X, y)
        n_b, _ = nb.best_iteration(nb.staged_rmse(bench, Xv, yv))
        ref = nb.predict_with_n_trees(bench, Xt, n_b)
    gap, lo, hi = nb.relative_gap_ci(yt, ours, ref, n_boot=500, rng=0)
    assert gap < 0.05, (
        f"tuned on the validation set, your pipeline should be within 5% of the benchmark's test RMSE on Friedman #1; "
        f"the relative gap is {100 * gap:+.1f}% (CI [{100 * lo:+.1f}%, {100 * hi:+.1f}%])")
