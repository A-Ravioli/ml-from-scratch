"""Checks for exercise.py (interpreting models).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Most checks use
small hand-made models whose answers are known in closed form (a linear model's partial
dependence is a line; its Shapley values are w_j (x_j - mean_j)); scikit-learn's
permutation_importance and partial_dependence serve as oracles on a fitted model; KernelSHAP is
checked against brute-force Shapley values. Loops run under mlfs.testing.bounded.
"""
import numpy as np
from scipy.stats import chi2
from sklearn.ensemble import GradientBoostingRegressor as SkGB
from sklearn.inspection import partial_dependence as sk_partial_dependence
from sklearn.inspection import permutation_importance as sk_permutation_importance

from mlfs.testing import bounded


class _Linear:
    """predict = X @ w + b; score = R^2 (like a Regressor). counts predict calls and rows."""

    def __init__(self, w, b=0.0):
        self.w, self.b, self.calls, self.rows = np.asarray(w, dtype=float), b, 0, 0

    def predict(self, X):
        self.calls += 1
        self.rows += len(X)
        return np.asarray(X) @ self.w + self.b

    def score(self, X, y):
        r = y - self.predict(X)
        return 1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)


class _Logistic:
    """a binary classifier with classes_ and predict_proba = sigmoid(X @ w)."""

    classes_ = np.array([0, 1])

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.asarray(X) @ self.w))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def _sk_model(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 2 * X[:, 0] + np.sin(2 * X[:, 1]) + X[:, 2] * X[:, 3] + 0.2 * rng.normal(size=n)
    return SkGB(n_estimators=60, max_depth=3, random_state=0).fit(X, y), X, y


# ---------------------------------------------------------------- stage 1: permutation importance


def test_stage1_known_model(nb):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    model = _Linear([3.0, 0.0, 1.0])
    y = model.predict(X)
    X0 = X.copy()
    out = bounded(nb.permutation_importance, model, X, y, n_repeats=4, rng=0, budget=200_000,
                  what="permutation_importance")
    imp = np.asarray(out["importances"])
    assert set(out) >= {"importances_mean", "importances_std", "importances"} and imp.shape == (3, 4), (
        "return a dict with importances_mean (d,), importances_std (d,) and importances (d, n_repeats)")
    assert np.array_equal(X, X0), "permutation_importance must not shuffle the caller's X in place: work on a copy"
    assert np.allclose(imp[1], 0.0), "a feature the model ignores has importance exactly 0 on every repeat"
    assert imp[0].min() > 1.5 and 0.1 < imp[2].mean() < imp[0].mean(), (
        f"importance = baseline score - score with the column shuffled (R^2 drops by ~2 x var share). got means "
        f"{np.round(imp.mean(1), 3).tolist()}")
    assert np.allclose(out["importances_mean"], imp.mean(1)) and np.allclose(out["importances_std"], imp.std(1)), (
        "importances_mean / importances_std are the mean and std (ddof=0) over the repeats")
    assert len(np.unique(np.round(imp[0], 12))) > 1, "each repeat uses a fresh permutation of the ORIGINAL column"
    again = nb.permutation_importance(model, X, y, n_repeats=4, rng=0)
    assert np.allclose(again["importances"], imp), "the same rng seed must give the same importances"


def test_stage1_scoring_and_oracle(nb):
    model, X, y = _sk_model(1)
    neg_mse = lambda est, X_, y_: -np.mean((est.predict(X_) - y_) ** 2)  # noqa: E731
    ours = nb.permutation_importance(model, X, y, scoring=neg_mse, n_repeats=30, rng=0)
    theirs = sk_permutation_importance(model, X, y, scoring="neg_mean_squared_error", n_repeats=30, random_state=0)
    assert np.allclose(ours["importances_mean"], theirs.importances_mean, rtol=0.15, atol=0.03), (
        f"with scoring=neg_mse, your mean importances {np.round(ours['importances_mean'], 3).tolist()} should be "
        f"close to scikit-learn's {np.round(theirs.importances_mean, 3).tolist()} (different shuffles, 30 repeats). "
        "a scorer is called as scoring(estimator, X, y); importance = baseline - permuted (positive = matters).")


# ---------------------------------------------------------------- stage 2: partial dependence


def test_stage2_linear_model(nb):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    model = _Linear([2.0, -1.0, 0.5], b=1.0)
    grid = np.array([-1.0, 0.0, 2.0])
    ice = np.asarray(nb.ice_curves(model, X, 1, grid))
    assert ice.shape == (200, 3), f"ice_curves returns (n, g) = (200, 3), got {ice.shape}"
    for k, v in enumerate(grid):
        Xv = X.copy()
        Xv[:, 1] = v
        assert np.allclose(ice[:, k], Xv @ model.w + 1.0), "ICE[i, k] = prediction for row i with feature set to grid[k]"
    model.calls = 0
    g, pd = nb.partial_dependence(model, X, 1, grid=grid)
    assert model.calls == 1, (
        f"build all n * g modified rows and call predict ONCE (got {model.calls} calls): looping over the grid "
        "or the rows makes partial dependence far slower for real models")
    want = 1.0 + 2.0 * X[:, 0].mean() + 0.5 * X[:, 2].mean() - grid
    assert np.allclose(g, grid) and np.allclose(pd, want), (
        "for a linear model the partial dependence is a straight line with the feature's coefficient as slope")


def test_stage2_default_grid_and_response(nb):
    rng = np.random.default_rng(3)
    col = rng.normal(size=1000)
    grid = np.asarray(nb.default_grid(col, 20))
    assert np.allclose(grid, np.quantile(col, np.linspace(0.05, 0.95, 20))), (
        "with many distinct values the grid is the quantiles at linspace(0.05, 0.95, grid_resolution)")
    assert np.array_equal(nb.default_grid(np.array([3.0, 1.0, 1.0, 2.0]), 20), [1.0, 2.0, 3.0]), (
        "with at most grid_resolution distinct values, the grid is those values")
    X = rng.normal(size=(300, 2))
    clf = _Logistic([1.5, 0.0])
    g, pd = nb.partial_dependence(clf, X, 0, grid_resolution=10)
    assert len(g) == 10 and np.all(np.diff(pd) > 0) and np.all((pd > 0) & (pd < 1)), (
        "response='auto' on a binary classifier uses predict_proba[:, 1]: probabilities, rising with feature 0")
    g2, pd2 = nb.partial_dependence(clf, X, 0, grid_resolution=10, response="predict")
    assert set(np.unique(pd2)) <= set(np.linspace(0, 1, 301)), "response='predict' averages the hard labels"


def test_stage2_matches_sklearn(nb):
    model, X, _ = _sk_model(4)
    grid = np.linspace(-1.5, 1.5, 7)
    for j in (0, 2):
        _, ours = nb.partial_dependence(model, X, j, grid=grid)
        sk = sk_partial_dependence(model, X, [j], custom_values={j: grid}, method="brute", kind="average")
        assert np.allclose(ours, sk["average"][0]), f"feature {j}: partial dependence differs from scikit-learn's"


# ---------------------------------------------------------------- stage 3: exact Shapley values


def _worked_f(Z):
    Z = np.asarray(Z, dtype=float)
    return Z[:, 0] * Z[:, 1] + Z[:, 0]


def test_stage3_worked_example(nb):
    B = np.array([[0.0, 0.0], [2.0, 2.0]])
    x = np.array([1.0, 3.0])
    masks = np.array([[False, False], [True, False], [False, True], [True, True]])
    v = np.asarray(nb.masked_predictions(_worked_f, B, x, masks))
    assert np.allclose(v, [3.0, 2.0, 4.0, 4.0]), (
        f"the lesson's worked example: v(none) = 3, v({{1}}) = 2, v({{2}}) = 4, v({{1, 2}}) = f(x) = 4, got {v.tolist()}")
    phi = np.asarray(nb.shapley_values_exact(lambda m: nb.masked_predictions(_worked_f, B, x, m), 2))
    assert np.allclose(phi, [-0.5, 1.5]), f"Shapley values of the worked example: (-0.5, 1.5), got {phi.tolist()}"


def test_stage3_masked_predictions(nb):
    rng = np.random.default_rng(5)
    Xb = rng.normal(size=(40, 3))
    x = np.array([1.0, -2.0, 0.5])
    model = _Linear([2.0, 1.0, -3.0])
    masks = np.array([[False, False, False], [True, False, False], [True, True, True], [False, True, True]])
    v = np.asarray(nb.masked_predictions(model.predict, Xb, x, masks))
    mean = Xb.mean(0)
    want = [m @ (model.w * x) + (~m) @ (model.w * mean) for m in masks]
    assert v.shape == (4,) and np.allclose(v, want), (
        "v(S) = mean over background rows of f(z), where z takes x's values on S and the background row's elsewhere. "
        f"expected {np.round(want, 3).tolist()}, got {np.round(v, 3).tolist()}")


def test_stage3_glove_game_and_axioms(nb):
    def glove(masks):  # players 0 and 1 together are worth 1; player 2 adds nothing
        masks = np.atleast_2d(masks)
        return (masks[:, 0] & masks[:, 1]).astype(float)

    phi = np.asarray(nb.shapley_values_exact(glove, 3))
    assert np.allclose(phi, [0.5, 0.5, 0.0]), (
        f"the glove game: the pair splits its value (symmetry) and the dummy gets 0, expected [0.5, 0.5, 0], got {phi}")

    def game(masks):
        masks = np.atleast_2d(masks).astype(float)
        return 3 * masks[:, 0] + 2 * masks[:, 1] * masks[:, 2] + masks[:, 0] * masks[:, 3] + 1.0

    def trio(masks):  # worth 1 only when players 0, 1 and 2 are all present; player 3 is a dummy
        masks = np.atleast_2d(masks)
        return (masks[:, 0] & masks[:, 1] & masks[:, 2]).astype(float)

    phi3 = np.asarray(nb.shapley_values_exact(trio, 4))
    assert np.allclose(phi3, [1 / 3, 1 / 3, 1 / 3, 0.0]), (
        f"a three-player interaction is split three ways: expected [1/3, 1/3, 1/3, 0], got {np.round(phi3, 4).tolist()}. "
        "(equal weights over coalitions give 1/4: the weights must be |S|! (d - |S| - 1)! / d!)")
    phi = np.asarray(nb.shapley_values_exact(game, 4))
    full, empty = game(np.ones((1, 4), bool))[0], game(np.zeros((1, 4), bool))[0]
    assert np.isclose(phi.sum(), full - empty), "efficiency: the Shapley values add up to v(all) - v(none)"
    assert np.allclose(phi, [3.5, 1.0, 1.0, 0.5]), (
        f"each interaction's value is split equally between its two players: expected [3.5, 1, 1, 0.5], got {phi}. "
        "weights |S|! (d - |S| - 1)! / d! over the coalitions S without player j")


# ---------------------------------------------------------------- stage 4: KernelSHAP


def test_stage4_matches_exact_shapley(nb):
    rng = np.random.default_rng(6)
    Xb = rng.normal(size=(30, 5))
    x = rng.normal(size=5)

    def f(X):
        X = np.asarray(X)
        return np.sin(X[:, 0]) * X[:, 1] + X[:, 2] ** 2 - X[:, 3] * X[:, 4] + 0.5 * X[:, 4] + X[:, 0] * X[:, 2] * X[:, 3]

    exact = nb.shapley_values_exact(lambda m: nb.masked_predictions(f, Xb, x, m), 5)
    phi = np.asarray(bounded(nb.kernel_shap, f, Xb, x, rng=0, budget=500_000, what="kernel_shap"))
    assert phi.shape == (5,) and np.allclose(phi, exact, atol=1e-8), (
        f"with 5 features, all 2^5 - 2 = 30 proper coalitions fit in n_samples: KernelSHAP with the exact Shapley "
        f"kernel weights must reproduce the brute-force values {np.round(exact, 4).tolist()}, got "
        f"{np.round(phi, 4).tolist()}. kernel weight of a coalition of size s: (d - 1) / (C(d, s) s (d - s)).")
    assert np.isclose(phi.sum() + f(Xb).mean(), f(x[None])[0]), "efficiency must hold exactly"
    worked = np.asarray(nb.kernel_shap(_worked_f, np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([1.0, 3.0])))
    assert np.allclose(worked, [-0.5, 1.5]), f"the lesson's worked example gives (-0.5, 1.5), got {worked.tolist()}"
    one = np.asarray(nb.kernel_shap(lambda X: 2 * np.asarray(X)[:, 0], Xb[:, :1], x[:1]))
    assert np.allclose(one, 2 * (x[0] - Xb[:, 0].mean())), "with one feature, phi = f(x) - mean f(background)"


def test_stage4_sample_coalitions(nb):
    d, n = 10, 30000
    masks = np.asarray(nb.sample_coalitions(d, n, rng=0))
    assert masks.shape == (n, d) and masks.dtype == bool, f"masks are bool (n_samples, d), got {masks.shape} {masks.dtype}"
    sizes = masks.sum(1)
    assert sizes.min() >= 1 and sizes.max() <= d - 1, "never the empty or the full coalition (they're fitted exactly)"
    s = np.arange(1, d)
    p = (d - 1) / (s * (d - s))
    p = p / p.sum()
    counts = np.bincount(sizes, minlength=d)[1:d]
    stat = np.sum((counts - n * p) ** 2 / (n * p))
    assert chi2.sf(stat, d - 2) > 1e-4, (
        f"coalition sizes must follow (d - 1) / (s (d - s)) (normalized): expected counts {np.round(n * p).astype(int).tolist()}, "
        f"got {counts.tolist()} (chi-square p = {chi2.sf(stat, d - 2):.1e}). small and large coalitions are the most "
        "likely; drawing sizes uniformly, or every feature independently with probability 1/2, gets this wrong.")
    for k in (1, 3, 5):  # within a size, every subset is equally likely: each feature appears equally often
        sub = masks[sizes == k]
        freq = sub.sum(0)
        exp = len(sub) * k / d
        stat = np.sum((freq - exp) ** 2 / exp) * (d - 1) / (d - k) if k < d else 0.0
        assert chi2.sf(stat, d - 1) > 1e-4, (
            f"among coalitions of size {k} every feature should appear about equally often (expected {exp:.0f} each), "
            f"got {freq.tolist()}: draw a UNIFORM subset of the chosen size")
    again = np.asarray(nb.sample_coalitions(d, 50, rng=3))
    assert np.array_equal(again, np.asarray(nb.sample_coalitions(d, 50, rng=3))), "the same seed must give the same masks"


def test_stage4_sampled_coalitions(nb):
    rng = np.random.default_rng(7)
    d = 13  # 2^13 - 2 coalitions > n_samples: KernelSHAP must sample
    Xb = rng.normal(size=(20, d))
    x = rng.normal(size=d)
    w = rng.normal(size=d)
    lin = lambda X: np.asarray(X) @ w  # noqa: E731
    phi = np.asarray(bounded(nb.kernel_shap, lin, Xb, x, n_samples=600, rng=0, budget=500_000, what="kernel_shap"))
    assert np.allclose(phi, w * (x - Xb.mean(0)), atol=1e-8), (
        "for a linear model every coalition value is linear in the mask, so even a sampled weighted least-squares "
        "fit is exact: phi_j = w_j (x_j - mean_j)")

    def f(X):
        X = np.asarray(X)
        return X @ w + 2 * X[:, 0] * X[:, 1] - X[:, 2] * X[:, 5] + np.sin(X[:, 7]) + 3 * X[:, 3] * X[:, 4] * X[:, 6]

    exact = nb.shapley_values_exact(lambda m: nb.masked_predictions(f, Xb, x, m), d)
    phis = [np.asarray(nb.kernel_shap(f, Xb, x, n_samples=4000, rng=r)) for r in (1, 2, 3)]
    phi = phis[0]
    err = np.median([np.abs(p - exact).max() for p in phis])
    assert np.isclose(phi.sum() + f(Xb).mean(), f(x[None])[0]), "efficiency must hold exactly with sampling too"
    assert err < 0.15 * np.abs(exact).max(), (
        f"with 4,000 sampled coalitions the estimate should be close to the exact values (median over 3 seeds of "
        f"the max error {err:.3f}, "
        f"largest |phi| {np.abs(exact).max():.3f}). sample coalition sizes s with probability proportional to "
        "(d - 1) / (s (d - s)), then a uniform subset of that size, and give every sample the same weight.")
    # the sampled branch, rebuilt from the spec: masks from sample_coalitions(d, n_samples, rng), equal weights,
    # efficiency enforced by eliminating the last feature
    masks = np.asarray(nb.sample_coalitions(d, 500, 9))
    v = np.asarray(nb.masked_predictions(f, Xb, x, masks))
    f0, fx = f(Xb).mean(), f(x[None])[0]
    Z = masks.astype(float)
    head = np.linalg.lstsq(Z[:, :-1] - Z[:, -1:], v - f0 - Z[:, -1] * (fx - f0), rcond=None)[0]
    want = np.append(head, fx - f0 - head.sum())
    got = np.asarray(nb.kernel_shap(f, Xb, x, n_samples=500, rng=9))
    assert np.allclose(got, want), (
        "the sampling branch should use the masks of sample_coalitions(d, n_samples, rng) with EQUAL weights: they "
        "were drawn in proportion to the Shapley kernel already, so weighting them by the kernel again counts it twice")
    again = np.asarray(nb.kernel_shap(f, Xb, x, n_samples=4000, rng=1))
    assert np.allclose(again, phi), "the same rng seed must give the same estimate"
