"""Checks for exercise.py (gaussian mixtures & EM).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Densities are
checked against scipy, fits against scikit-learn's GaussianMixture (an oracle only), and the
lesson's worked example by hand. fit runs under mlfs.testing.bounded, so a loop without a cap
fails instead of hanging.
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as SkGMM

from mlfs.testing import bounded

TYPES = ("full", "diag", "spherical")
LINE = np.array([[0.0], [1.0], [3.0], [4.0]])


def _params(k=3, d=2, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 3, (k, d))
    A = rng.normal(0, 1, (k, d, d))
    full = A @ A.transpose(0, 2, 1) + 0.5 * np.eye(d)
    diag = rng.uniform(0.3, 2.0, (k, d))
    sph = rng.uniform(0.3, 2.0, k)
    w = rng.dirichlet(np.ones(k))
    return w, means, {"full": full, "diag": diag, "spherical": sph}


def _as_full(cov, ctype, d):
    if ctype == "full":
        return cov
    if ctype == "diag":
        return np.array([np.diag(c) for c in cov])
    return np.array([c * np.eye(d) for c in cov])


def _mixture_data(n_per=150, seed=1):
    rng = np.random.default_rng(seed)
    means = np.array([[0.0, 0.0], [6.0, 1.0], [2.0, 6.0]])
    covs = np.array([[[1.0, 0.6], [0.6, 1.0]], [[0.5, 0.0], [0.0, 2.0]], [[1.5, -0.7], [-0.7, 1.0]]])
    X = np.concatenate([rng.multivariate_normal(m, c, n_per) for m, c in zip(means, covs)])
    return X, np.repeat(np.arange(3), n_per), means


def _fit(nb, X, **kw):
    g = nb.GaussianMixture(**kw)
    return bounded(g.fit, X, budget=3_000_000, what="GaussianMixture.fit",
                   hint="the EM loop must stop after max_iters iterations at most.")


# ---------------------------------------------------------------- stage 1: densities


def test_stage1_matches_scipy(nb):
    w, means, covs = _params(seed=2)
    X = np.random.default_rng(3).normal(0, 3, (50, 2))
    for ctype in TYPES:
        got = nb.gaussian_log_density(X, means, covs[ctype], ctype)
        assert np.shape(got) == (50, 3), f"{ctype}: expected shape (n, k) = (50, 3), got {np.shape(got)}"
        full = _as_full(covs[ctype], ctype, 2)
        want = np.column_stack([multivariate_normal(means[j], full[j]).logpdf(X) for j in range(3)])
        assert np.allclose(got, want), (
            f"covariance_type={ctype!r}: log densities differ from scipy by up to {np.abs(got - want).max():.3g}. "
            "log N = -1/2 (d log 2pi + log det Sigma + (x - mu)^T Sigma^{-1} (x - mu)); "
            + {"full": "covariances[j] is a (d, d) matrix", "diag": "covariances[j] is a vector of d variances",
               "spherical": "covariances[j] is ONE variance shared by all d dimensions (log det = d log var)"}[ctype])


def test_stage1_far_points_stay_finite(nb):
    got = nb.gaussian_log_density(np.array([[1e4, -1e4]]), np.zeros((1, 2)), np.eye(2)[None], "full")
    assert np.isfinite(got).all() and np.isclose(got[0, 0], -0.5 * (2 * np.log(2 * np.pi) + 2e8)), (
        "compute the LOG density directly (a Cholesky solve for the Mahalanobis term), never log(pdf): "
        f"a point 1e4 away should give about -1e8, got {got}")


# ---------------------------------------------------------------- stage 2: E-step


def test_stage2_worked_example(nb):
    r, ll = nb.e_step(LINE, np.array([0.5, 0.5]), np.array([[0.0], [4.0]]), np.ones((2, 1, 1)), "full")
    assert np.allclose(r[:, 0], [0.99966465, 0.98201379, 0.01798621, 0.00033535]), (
        f"responsibility of the component at 0 should be 1 / (1 + exp(4x - 8)) = (0.99966, 0.98201, 0.01799, 0.00034), "
        f"got {np.round(r[:, 0], 5)}")
    assert np.isclose(ll, -1.8528430466), f"mean log-likelihood should be -1.85284, got {ll}"


def test_stage2_matches_direct(nb):
    w, means, covs = _params(seed=4)
    X = np.random.default_rng(5).normal(0, 3, (80, 2))
    for ctype in TYPES:
        r, ll = nb.e_step(X, w, means, covs[ctype], ctype)
        full = _as_full(covs[ctype], ctype, 2)
        joint = np.column_stack([w[j] * multivariate_normal(means[j], full[j]).pdf(X) for j in range(3)])
        assert np.allclose(r, joint / joint.sum(1, keepdims=True)), (
            f"{ctype}: r[i, j] = w_j N(x_i | j) / sum_l w_l N(x_i | l)")
        assert np.isclose(ll, np.mean(np.log(joint.sum(1)))), (
            f"{ctype}: the second output is the MEAN over points of log sum_j w_j N(x_i | j)")


def test_stage2_stable_far_from_everything(nb):
    X = np.array([[0.0, 0.0], [200.0, 200.0]])
    r, ll = nb.e_step(X, np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]), np.ones((2, 2)), "diag")
    assert np.isfinite(r).all() and np.allclose(r.sum(1), 1), (
        f"a point far from every component has every density underflow to 0 if you exponentiate first, giving 0/0 = NaN. "
        f"work in log space and normalize with logsumexp. got responsibilities {r}")
    assert np.isclose(r[1, 1], 1.0) and np.isfinite(ll), "the far point belongs (almost surely) to the nearer component"


# ---------------------------------------------------------------- stage 3: M-step


def test_stage3_worked_example(nb):
    r = np.array([0.99966465, 0.98201379, 0.01798621, 0.00033535])
    w, m, c = nb.m_step(LINE, np.column_stack([r, 1 - r]), "full", 0.0)
    assert np.allclose(w, [0.5, 0.5]), f"weights = N_j / n = (0.5, 0.5), got {w}"
    assert np.allclose(np.ravel(m), [0.51865691, 3.48134309]), f"means = sum r x / N_j = (0.5187, 3.4813), got {np.ravel(m)}"
    assert np.allclose(np.ravel(c), [0.30562265, 0.30562265]), f"variances = sum r (x - mu)^2 / N_j = 0.3056, got {np.ravel(c)}"


def test_stage3_hard_assignments(nb):
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (60, 3))
    labels = rng.integers(0, 3, 60)
    resp = np.eye(3)[labels]
    w, m, c = nb.m_step(X, resp, "full", 1e-3)
    for j in range(3):
        Xj = X[labels == j]
        assert np.isclose(w[j], len(Xj) / 60), "with one-hot responsibilities, weight_j = fraction of points in j"
        assert np.allclose(m[j], Xj.mean(0)), "... mean_j = the plain mean of cluster j"
        assert np.allclose(c[j], np.cov(Xj.T, bias=True) + 1e-3 * np.eye(3)), (
            "... and covariance_j = the (biased, divide by N_j) covariance of cluster j, plus reg_covar on the diagonal")


def test_stage3_covariance_types_agree(nb):
    rng = np.random.default_rng(7)
    X = rng.normal(0, 2, (100, 4))
    resp = rng.dirichlet(np.ones(3), 100)
    _, m, full = nb.m_step(X, resp, "full", 1e-4)
    _, _, diag = nb.m_step(X, resp, "diag", 1e-4)
    _, _, sph = nb.m_step(X, resp, "spherical", 1e-4)
    assert np.shape(diag) == (3, 4) and np.shape(sph) == (3,), "diag covariances are (k, d), spherical (k,)"
    assert np.allclose(diag, np.array([np.diag(f) for f in full])), "diag variances = the diagonal of the full covariance"
    assert np.allclose(sph, diag.mean(1)), "the spherical variance is the mean of the diagonal variances"
    want = np.array([(resp[:, j, None] * (X - m[j])).T @ (X - m[j]) / resp[:, j].sum() for j in range(3)])
    assert np.allclose(full, want + 1e-4 * np.eye(4)), "full_j = sum_i r_ij (x_i - mu_j)(x_i - mu_j)^T / N_j + reg I"


# ---------------------------------------------------------------- stage 4: EM


def test_stage4_monotone_likelihood(nb):
    X, _, _ = _mixture_data(seed=8)
    for ctype in TYPES:
        g = _fit(nb, X, n_components=3, covariance_type=ctype, init="random", max_iters=60, tol=0.0, reg_covar=0.0, rng=0)
        h = np.asarray(g.lower_bound_history_)
        assert len(h) == 60 and g.n_iter_ == 60, (
            f"{ctype}: with tol=0 and max_iters=60, EM runs 60 iterations and records 60 values, got {len(h)}")
        assert np.all(np.diff(h) >= -1e-8), (
            f"{ctype}: EM never decreases the likelihood, but yours dropped by {-np.diff(h).min():.3g} at iteration "
            f"{int(np.argmin(np.diff(h))) + 2}. check that the M-step uses the responsibilities from THIS E-step (this test uses reg_covar=0: the variance floor is the one thing that can make a correct EM dip)")
        assert g.score(X) >= h[-1] - 1e-8, "the final parameters (after the last M-step) score at least the last recorded value"


def test_stage4_recovers_components(nb):
    X, y, means = _mixture_data(seed=9)
    g = _fit(nb, X, n_components=3, rng=0)
    sk = SkGMM(3, random_state=0, tol=1e-6).fit(X)
    assert abs(g.score(X) - sk.score(X)) < 0.01, (
        f"mean log-likelihood {g.score(X):.4f} should match scikit-learn's {sk.score(X):.4f} on three clear clusters")
    order = [int(np.argmin(((g.means_ - m) ** 2).sum(1))) for m in means]
    assert sorted(order) == [0, 1, 2] and np.allclose(g.means_[order], means, atol=0.35), (
        f"the three fitted means should be near the true ones {means.tolist()}, got {np.round(g.means_, 2).tolist()}")
    acc = np.mean(np.asarray(order)[y] == g.predict(X))
    assert acc > 0.95, f"predict should put {acc:.0%} > 95% of points in their true component"
    assert g.converged_ and g.n_iter_ < 100, "EM on clear clusters converges well before max_iters"


def test_stage4_em_moves_beyond_the_start(nb):
    # three long, thin, tilted strips: the k-means start cuts them crosswise, far from a fixed point of EM
    rng = np.random.default_rng(15)
    X = np.concatenate([np.c_[rng.normal(0, 4, 150), rng.normal(o, 0.4, 150)] for o in (0, 2.5, 5)])
    X = X @ np.array([[np.cos(0.6), np.sin(0.6)], [-np.sin(0.6), np.cos(0.6)]])
    g = _fit(nb, X, n_components=3, rng=0)
    gain = g.lower_bound_history_[-1] - g.lower_bound_history_[0]
    assert gain > 0.01, (
        f"from a k-means start, EM should raise the mean log-likelihood noticeably (it rose by {gain:.4f}). does every "
        "iteration run an E-step AND an M-step with the new responsibilities?")
    assert len(g.lower_bound_history_) == g.n_iter_, "one history entry per iteration"
    g.fit(X)
    assert len(g.lower_bound_history_) == g.n_iter_, "refitting starts a fresh lower_bound_history_"


def test_stage4_estimator_methods(nb):
    X, _, _ = _mixture_data(seed=10)
    g = _fit(nb, X, n_components=3, covariance_type="diag", rng=1)
    assert g.get_params()["covariance_type"] == "diag" and g.get_params()["init"] == "kmeans", (
        "GaussianMixture subclasses Estimator and __init__ only stores its arguments")
    P = g.predict_proba(X)
    assert P.shape == (len(X), 3) and np.allclose(P.sum(1), 1), "predict_proba rows sum to 1"
    assert np.array_equal(g.predict(X), P.argmax(1)), "predict = argmax of predict_proba"
    s = g.score_samples(X)
    full = _as_full(g.covariances_, "diag", 2)
    want = np.log(sum(g.weights_[j] * multivariate_normal(g.means_[j], full[j]).pdf(X) for j in range(3)))
    assert np.allclose(s, want), "score_samples(X) = log sum_j w_j N(x | mu_j, Sigma_j)"
    assert np.isclose(g.score(X), s.mean()), "score = mean log-likelihood"
    assert np.isclose(g.weights_.sum(), 1) and g.covariances_.shape == (3, 2), "weights sum to 1; diag covariances (k, d)"


def test_stage4_rng_and_validation(nb):
    X, _, _ = _mixture_data(seed=11)
    a = _fit(nb, X, n_components=3, init="random", rng=3).means_
    b = _fit(nb, X, n_components=3, init="random", rng=3).means_
    assert np.allclose(a, b), "the same rng gives the same fit"
    starts = [np.sort(_fit(nb, X, n_components=6, max_iters=1, rng=s).means_[:, 0]) for s in range(5)]
    assert any(not np.allclose(starts[0], m) for m in starts[1:]), (
        "with init='kmeans', different rng values should give different k-means starts: pass the estimator's "
        "generator to _initial_resp (it's given), don't seed KMeans with a constant")
    for bad in (dict(covariance_type="tied"), dict(init="bogus")):
        try:
            nb.GaussianMixture(2, **bad).fit(X)
        except ValueError:
            continue
        raise AssertionError(f"GaussianMixture(2, {bad}) should raise ValueError in fit")


# ---------------------------------------------------------------- stage 5: sampling and BIC


def _sk_reference(seed, ctype):
    X, _, _ = _mixture_data(seed=seed)
    sk = SkGMM(3, covariance_type=ctype, random_state=0).fit(X)
    return X, sk


def _ours_from(nb, sk, ctype):
    g = nb.GaussianMixture(3, covariance_type=ctype)
    g.weights_, g.means_, g.covariances_ = sk.weights_.copy(), sk.means_.copy(), sk.covariances_.copy()
    return g


def test_stage5_bic_matches_sklearn(nb):
    for ctype in TYPES:
        X, sk = _sk_reference(12, ctype)
        g = _ours_from(nb, sk, ctype)
        assert np.isclose(g.bic(X), sk.bic(X), rtol=1e-6), (
            f"{ctype}: BIC = -2 sum log p(x_i) + p log n with p = (k - 1) + k d + covariance parameters "
            f"(full k d (d + 1) / 2, diag k d, spherical k). got {g.bic(X):.3f}, sklearn {sk.bic(X):.3f}")


def test_stage5_sample(nb):
    X, sk = _sk_reference(13, "full")
    g = _ours_from(nb, sk, "full")
    S, labels = g.sample(30000, rng=0)
    assert S.shape == (30000, 2) and labels.shape == (30000,), "sample(n) returns (X (n, d), labels (n,))"
    freq = np.bincount(labels, minlength=3) / 30000
    assert np.allclose(freq, g.weights_, atol=0.015), f"component frequencies {freq} should match the weights {g.weights_}"
    for j in range(3):
        Sj = S[labels == j]
        assert np.allclose(Sj.mean(0), g.means_[j], atol=0.08), f"samples of component {j} should have mean {g.means_[j]}"
        assert np.allclose(np.cov(Sj.T), g.covariances_[j], atol=0.12), (
            f"samples of component {j} should have covariance {g.covariances_[j].tolist()}: draw mean + L z with L L^T = Sigma "
            "(the Cholesky factor), z standard normal")
    for ctype in ("diag", "spherical"):
        _, skc = _sk_reference(14, ctype)
        S, labels = _ours_from(nb, skc, ctype).sample(20000, rng=1)
        want = skc.covariances_[0] * (np.ones(2) if ctype == "spherical" else 1)
        assert np.allclose(S[labels == 0].var(0), want, rtol=0.1), f"{ctype}: per-dimension sample variance of component 0"
    a, _ = g.sample(5, rng=7)
    b, _ = g.sample(5, rng=7)
    assert np.allclose(a, b), "the same rng gives the same samples"
