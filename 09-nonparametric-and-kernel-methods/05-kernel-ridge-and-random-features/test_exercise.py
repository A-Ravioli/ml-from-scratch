"""Checks for exercise.py (kernel ridge & random features).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scikit-learn's
KernelRidge is used only as an oracle. Random-feature checks average over several seeds or
use many features, so they don't depend on the order in which you draw your random numbers.
"""
import numpy as np
import pytest


def _rbf(A, B, gamma):
    return np.exp(-gamma * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def _data(n, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    return X, np.sin(X[:, 0]) + 0.5 * X[:, -1] + 0.1 * rng.normal(size=n)


# ---------------------------------------------------------------- stage 1: kernel ridge


def test_stage1_worked_example(nb):
    m = nb.KernelRidge(0.5, kernel="rbf", gamma=1.0).fit(np.array([[0.0], [1.0]]), np.array([1.0, 2.0]))
    assert np.allclose(m.dual_coef_, [0.361400, 1.244702], atol=1e-6), (
        f"(K + 0.5 I) beta = y with K = [[1, e^-1], [e^-1, 1]]: beta = [0.3614, 1.2447], got {m.dual_coef_}")
    got = m.predict(np.array([[0.0], [1.0]]))
    assert np.allclose(got, [0.819303, 1.377654], atol=1e-6), (
        f"the fitted values are shrunk towards 0: f(0) = 0.8193, f(1) = 1.3777, got {got}")


@pytest.mark.parametrize("kw", [dict(kernel="rbf", gamma=0.4), dict(kernel="poly", degree=2, gamma=0.5, coef0=1.0),
                                dict(kernel="linear"), dict(kernel="laplacian", gamma=0.3)])
def test_stage1_matches_sklearn(nb, kw):
    from sklearn.kernel_ridge import KernelRidge
    X, y = _data(60, 3, 0)
    Q, _ = _data(20, 3, 1)
    ours = nb.KernelRidge(0.3, **kw)
    assert ours.fit(X, y) is ours, "fit returns self"
    ref = KernelRidge(alpha=0.3, **kw).fit(X, y)
    assert np.allclose(ours.dual_coef_, ref.dual_coef_), f"{kw}: dual_coef_ = (K + alpha I)^-1 y"
    assert np.allclose(ours.predict(Q), ref.predict(Q)), f"{kw}: predict(X) = K(X, X_fit_) @ dual_coef_"
    assert np.array_equal(ours.X_fit_, X), "X_fit_ stores the training inputs"


def test_stage1_linear_kernel_is_ridge(nb):
    X, y = _data(50, 4, 2)
    w = np.linalg.solve(X.T @ X + 2.0 * np.eye(4), X.T @ y)
    Q, _ = _data(10, 4, 3)
    got = nb.KernelRidge(2.0, kernel="linear").fit(X, y).predict(Q)
    assert np.allclose(got, Q @ w), (
        "with the linear kernel, kernel ridge is ordinary ridge regression without an intercept: "
        "X (X^T X + alpha I)^-1 X^T y = X X^T (X X^T + alpha I)^-1 y (the push-through identity)")


def test_stage1_limits_and_protocol(nb):
    X, y = _data(30, 2, 4)
    y_clean = np.sin(X[:, 0]) + 0.5 * X[:, 1]  # noise-free targets: interpolating them is well-posed
    tiny = nb.KernelRidge(1e-10, gamma=1.0).fit(X, y_clean)
    assert np.allclose(tiny.predict(X), y_clean, atol=1e-4), "alpha -> 0 gives the minimum-norm interpolant of topic 03"
    huge = nb.KernelRidge(1e8, gamma=1.0).fit(X, y)
    assert np.allclose(huge.predict(X), 0.0, atol=1e-5), "alpha -> infinity shrinks every prediction to 0 (no intercept)"
    m = nb.KernelRidge(0.5, kernel="poly", degree=2)
    assert m.get_params() == {"alpha": 0.5, "kernel": "poly", "gamma": 1.0, "degree": 2, "coef0": 1.0}
    custom = nb.KernelRidge(0.5, kernel=lambda A, B=None: A @ (A if B is None else B).T).fit(X, y)
    assert np.allclose(custom.predict(X), nb.KernelRidge(0.5, kernel="linear").fit(X, y).predict(X)), (
        "a callable kernel works like the named one (use get_kernel)")
    assert -1 < m.fit(X, y).score(X, y) <= 1, "score is R^2 (inherited from Regressor)"
    dup = np.vstack([X[:5], X[:5]])
    zero = nb.KernelRidge(0.0, gamma=1.0).fit(dup, np.concatenate([y_clean[:5], y_clean[:5]]))
    assert np.allclose(zero.predict(X[:5]), y_clean[:5], atol=1e-6), (
        "alpha = 0 is allowed: with repeated points K is singular, so fall back to least squares when solve fails")


# ---------------------------------------------------------------- stage 2: random Fourier features


def test_stage2_shapes_and_distribution(nb):
    X = np.random.default_rng(5).normal(size=(30, 3))
    rff = nb.RandomFourierFeatures(5000, gamma=0.7, rng=0)
    assert rff.fit(X) is rff, "fit returns self"
    W, b = np.asarray(rff.random_weights_), np.asarray(rff.random_offset_)
    assert W.shape == (3, 5000) and b.shape == (5000,), (
        f"random_weights_ is (d, D) = (3, 5000) and random_offset_ is (D,); got {W.shape}, {b.shape}")
    assert abs(W.var() - 1.4) < 0.08 and abs(W.mean()) < 0.03, (
        f"the frequencies are N(0, 2 gamma) = N(0, 1.4): variance {W.var():.3f}. the spectral density of "
        "exp(-gamma r^2) is a Gaussian with variance 2 gamma per coordinate")
    assert b.min() >= 0 and b.max() < 2 * np.pi and abs(b.mean() - np.pi) < 0.1, "the offsets are uniform on [0, 2 pi)"
    Z = np.asarray(rff.transform(X))
    assert Z.shape == (30, 5000), f"transform gives (n, D) = (30, 5000), got {Z.shape}"
    assert np.allclose(Z, np.sqrt(2 / 5000) * np.cos(X @ W + b)), "z(x) = sqrt(2/D) cos(x W + b)"
    again = nb.RandomFourierFeatures(5000, gamma=0.7, rng=0).fit(X).transform(X)
    assert np.allclose(again, Z), "the same rng gives the same features"
    gen_rff = nb.RandomFourierFeatures(50, gamma=0.7, rng=np.random.default_rng(3)).fit(X)
    assert np.allclose(gen_rff.transform(X), gen_rff.transform(X)), (
        "transform must reuse the frequencies drawn in fit: two calls on the same X give the same features")


def test_stage2_approximates_the_rbf_kernel(nb):
    X = np.random.default_rng(6).normal(size=(40, 2))
    K = _rbf(X, X, 0.5)
    Z = nb.RandomFourierFeatures(20_000, gamma=0.5, rng=1).fit_transform(X)
    err = np.max(np.abs(Z @ Z.T - K))
    assert err < 0.05, (
        f"with 20,000 features, z(x) . z(y) should be within a few hundredths of exp(-0.5 ||x - y||^2); worst error "
        f"{err:.3f}. check the sqrt(2/D) scale and the N(0, 2 gamma) frequencies")


def test_stage2_error_shrinks_like_one_over_sqrt_D(nb):
    X = np.random.default_rng(7).normal(size=(60, 3))
    K = _rbf(X, X, 0.3)
    ratios = []
    for seed in range(5):
        e = []
        for D in (100, 1600):
            Z = nb.RandomFourierFeatures(D, gamma=0.3, rng=seed).fit_transform(X)
            e.append(np.linalg.norm(Z @ Z.T - K) / np.linalg.norm(K))
        ratios.append(e[0] / e[1])
    r = np.median(ratios)
    assert 2.5 < r < 6.5, (
        f"16x more features should cut the error about 4x (Monte Carlo: error ~ 1/sqrt(D)); median ratio {r:.2f}")


# ---------------------------------------------------------------- stage 3: Nyström


def test_stage3_all_points_is_exact(nb):
    X, _ = _data(40, 3, 8)
    ny = nb.Nystroem(100, gamma=0.5, rng=0)
    assert ny.fit(X) is ny, "fit returns self"
    assert len(ny.component_indices_) == 40, "n_components larger than n uses every row as a landmark"
    Z = np.asarray(ny.transform(X))
    assert Z.shape == (40, 40), f"one feature per landmark: (n, m) = (40, 40), got {Z.shape}"
    assert np.allclose(Z @ Z.T, _rbf(X, X, 0.5), atol=1e-8), (
        "with every point as a landmark, Z Z^T = K K^-1 K = K exactly. normalization_ is K_LL^(-1/2): eigh, then "
        "U diag(1/sqrt(w)) U^T")


def test_stage3_landmarks_and_formula(nb):
    X, _ = _data(80, 2, 9)
    ny = nb.Nystroem(15, kernel="poly", degree=2, gamma=0.5, coef0=1.0, rng=1).fit(X)
    idx = np.asarray(ny.component_indices_)
    assert len(idx) == 15 and len(set(idx.tolist())) == 15, "15 distinct landmark rows (sample without replacement)"
    assert np.allclose(ny.components_, X[idx]), "components_ = X[component_indices_]"
    other = nb.Nystroem(15, kernel="poly", degree=2, gamma=0.5, coef0=1.0, rng=2).fit(X)
    assert set(idx.tolist()) != set(range(15)) and set(idx.tolist()) != set(other.component_indices_.tolist()), (
        "landmarks are a random sample of rows (different rng, different landmarks), not the first m rows")
    Kp = lambda A, B: (0.5 * A @ B.T + 1.0) ** 2  # noqa: E731
    Z = np.asarray(ny.transform(X))
    want = Kp(X, X[idx]) @ np.linalg.pinv(Kp(X[idx], X[idx])) @ Kp(X[idx], X)
    assert np.allclose(Z @ Z.T, want, atol=1e-6), "Z Z^T = K_nL K_LL^-1 K_Ln, the Nyström approximation"
    assert np.allclose((Z @ Z.T)[np.ix_(idx, idx)], Kp(X[idx], X[idx]), atol=1e-6), (
        "on the landmarks themselves the approximation is exact")


# ---------------------------------------------------------------- stage 4: ridge on features


def test_stage4_primal_ridge_and_gram_error(nb):
    rng = np.random.default_rng(10)
    Z, y = rng.normal(size=(30, 5)), rng.normal(size=30)
    w = np.asarray(nb.primal_ridge(Z, y, 0.7))
    assert np.allclose(w, np.linalg.solve(Z.T @ Z + 0.7 * np.eye(5), Z.T @ y)), "w = (Z^T Z + alpha I)^-1 Z^T y"
    got = nb.gram_error(np.array([[1.0], [1.0]]), np.eye(2))
    assert np.isclose(got, 1.0), f"||[[0, 1], [1, 0]]||_F / ||I||_F = sqrt2 / sqrt2 = 1, got {got}"


def test_stage4_nystroem_ridge_equals_kernel_ridge(nb):
    X, y = _data(50, 3, 11)
    Q, _ = _data(15, 3, 12)
    ny = nb.Nystroem(50, gamma=0.4, rng=0).fit(X)
    w = nb.primal_ridge(ny.transform(X), y, 0.2)
    krr = nb.KernelRidge(0.2, gamma=0.4).fit(X, y)
    assert np.allclose(ny.transform(Q) @ w, krr.predict(Q), atol=1e-6), (
        "with every training point as a landmark, ridge on Nyström features is exactly kernel ridge regression, "
        "on new points too")


def test_stage4_random_features_approach_kernel_ridge(nb):
    X, y = _data(200, 2, 13)
    Q, _ = _data(100, 2, 14)
    exact = nb.KernelRidge(0.5, gamma=0.5).fit(X, y).predict(Q)
    gaps = []
    for D in (20, 2000):
        g = []
        for seed in range(5):
            rff = nb.RandomFourierFeatures(D, gamma=0.5, rng=seed).fit(X)
            w = nb.primal_ridge(rff.transform(X), y, 0.5)
            g.append(np.sqrt(np.mean((rff.transform(Q) @ w - exact) ** 2)))
        gaps.append(np.median(g))
    assert gaps[1] < 0.05 and gaps[1] < gaps[0] / 3, (
        f"ridge on 2,000 random features should predict almost like exact kernel ridge (RMS gap {gaps[1]:.3f}), and "
        f"much closer than with 20 features ({gaps[0]:.3f})")
