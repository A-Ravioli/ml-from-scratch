"""Checks for capstone.py (a generalization report).

All data here is small and synthetic. The models and bounds you build on come from mlfs (the
reference solutions under pytest): LogisticRegression, SVC, KNeighborsClassifier, and
vc_bound / finite_class_bound / pac_bayes_bound from this chapter. These checks test your
measurement, your normalized margins, the grids and union bounds, and the random-label fit,
against direct computations.
"""
import math

import numpy as np
from scipy import stats
from scipy.special import ndtr

from mlfs.testing import bounded


def _linear_data(n=600, d=4, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = np.where(X @ w + 0.3 > 0, 1, -1)
    flip = rng.random(n) < noise
    y[flip] *= -1
    return X, y


class _Stump:
    def __init__(self):
        self.fits = 0

    def fit(self, X, y):
        self.fits += 1
        return self

    def predict(self, X):
        return np.where(np.asarray(X)[:, 0] >= 0, 1, -1)


def _margin_ref(m, delta, gammas, sigmas):
    from mlfs.theory import pac_bayes_bound
    n = len(m)
    mb = [np.clip(1 - m / g, 0, 1).mean() + 2 / (g * math.sqrt(n)) + math.sqrt(math.log(len(gammas) / delta) / (2 * n))
          for g in gammas]
    pb = [pac_bayes_bound(float(np.mean(ndtr(-m / s))), 1 / (2 * s * s), n, delta / len(sigmas)) for s in sigmas]
    return mb, pb


# ---------------------------------------------------------------- stage 1: measuring the gap


def test_stage1_measure_gap(nb):
    X, y = _linear_data(seed=1)
    Xte, yte = _linear_data(n=500, seed=2)
    model = _Stump()
    out = nb.measure_gap(model, X, y, Xte, yte, delta=0.05, n_boot=500, rng=0)
    assert model.fits == 1, "fit the model once, on the training data"
    tr = float(np.mean(model.predict(X) != y))
    k = int(np.sum(model.predict(Xte) != yte))
    assert np.isclose(out["train_error"], tr) and np.isclose(out["test_error"], k / 500), "train and test error rates"
    assert np.isclose(out["gap"], k / 500 - tr), "gap = test error - train error"
    assert np.isclose(out["test_bound"], stats.beta.ppf(0.95, k + 1, 500 - k), atol=1e-7), (
        "test_bound is the exact binomial (Clopper-Pearson) upper bound on the test error at delta, as in topic 04")
    lo, hi = out["test_ci"]
    se = math.sqrt(k / 500 * (1 - k / 500) / 500)
    assert lo < k / 500 < hi and abs((hi - lo) - 3.92 * se) < 0.4 * 3.92 * se, (
        f"test_ci is a 95% bootstrap interval for the test error ({k / 500:.3f}); its width should be close to "
        f"3.92 * sqrt(p(1-p)/m) = {3.92 * se:.3f}, got ({lo:.3f}, {hi:.3f})")


# ---------------------------------------------------------------- stage 2: margin and PAC-Bayes bounds


def test_stage2_margin_bounds(nb):
    rng = np.random.default_rng(3)
    m = np.clip(rng.normal(0.3, 0.2, size=2000), -1, 1)
    gammas, sigmas = [0.5, 0.25, 0.1], [0.05, 0.1, 0.2]
    out = nb.margin_bounds(m, delta=0.05, gammas=gammas, sigmas=sigmas)
    mb, pb = _margin_ref(m, 0.05, gammas, sigmas)
    i, j = int(np.argmin(mb)), int(np.argmin(pb))
    assert np.isclose(out["margin_bound"], mb[i]) and np.isclose(out["margin_gamma"], gammas[i]), (
        f"margin bound over the grid {gammas}: mean ramp loss + 2 / (gamma sqrt n) + sqrt(ln(k / delta) / (2n)) with "
        f"k = len(gammas) (B = R = 1 for normalized margins); expected {mb[i]:.4f} at gamma = {gammas[i]}, got "
        f"{out['margin_bound']:.4f} at {out['margin_gamma']}")
    assert np.isclose(out["pac_bayes_bound"], pb[j], atol=1e-7) and np.isclose(out["pac_bayes_sigma"], sigmas[j]), (
        f"PAC-Bayes over the grid {sigmas}: Gibbs risk mean Phi(-m / sigma), KL = 1 / (2 sigma^2) (unit-norm u, prior "
        f"N(0, sigma^2 I)), pac_bayes_bound at delta / len(sigmas); expected {pb[j]:.4f} at sigma = {sigmas[j]}, got "
        f"{out['pac_bayes_bound']:.4f} at {out['pac_bayes_sigma']}")


def test_stage2_default_grids_and_behaviour(nb):
    good = np.full(20000, 0.5)
    bad = np.full(20000, 0.01)
    g, b = nb.margin_bounds(good), nb.margin_bounds(bad)
    assert g["margin_bound"] < 0.2 and g["pac_bayes_bound"] < 0.1, (
        f"20,000 points all with normalized margin 0.5: both bounds should be small with the default grids; got {g}")
    assert b["margin_bound"] > g["margin_bound"] and b["pac_bayes_bound"] > g["pac_bayes_bound"], (
        "small margins must give looser bounds")


# ---------------------------------------------------------------- stage 3: a linear model


def test_stage3_linear_bounds(nb):
    from mlfs.linear import LogisticRegression
    from mlfs.theory import finite_class_bound, vc_bound
    X, y01 = _linear_data(n=800, d=3, seed=4)
    y = np.where(y01 > 0, "yes", "no")  # any two labels: classes_[1] = 'yes' is the positive class
    model = LogisticRegression().fit(X, y)
    out = nb.linear_bounds(model, X, y, delta=0.05)
    tr = float(np.mean(model.predict(X) != y))
    assert np.isclose(out["train_error"], tr), "train_error of the fitted model on (X, y)"
    assert np.isclose(out["vc_bound"], vc_bound(800, 4, 0.05, tr)), "VC dimension of halfspaces with a bias in R^3 is 4"
    s32 = np.where(y == model.classes_[1], 1.0, -1.0)
    w32 = np.append(model.coef_, model.intercept_).astype(np.float32).astype(float)
    tr32 = float(np.mean(s32 * (np.hstack([X, np.ones((800, 1))]) @ w32) < 0))
    assert np.isclose(out["finite_bound"], finite_class_bound(800, 2 ** (32 * 4), 0.05, tr32)), (
        "the bit-counting bound is for the model rounded to 32 bits per parameter: 4 float32 parameters, |H| = 2^128, "
        "with the training error of the float32-rounded weights")
    s = np.where(y == model.classes_[1], 1.0, -1.0)
    Xt = np.hstack([X, np.ones((800, 1))])
    w = np.append(model.coef_, model.intercept_)
    want = s * (Xt @ w) / (np.linalg.norm(w) * np.linalg.norm(Xt, axis=1))
    got = np.asarray(out["norm_margins"])
    assert got.shape == (800,) and np.allclose(got, want), (
        "normalized margins: s_i (w . x~_i) / (||w|| ||x~_i||) with x~ = (x, 1), w = (coef_, intercept_), s = +1 for "
        "classes_[1]; they lie in [-1, 1] and are negative exactly on training errors")
    assert np.isclose(np.mean(got < 0), tr), "negative normalized margins are exactly the training errors"
    ref = nb.margin_bounds(want, delta=0.05)
    assert np.isclose(out["margin_bound"], ref["margin_bound"]) and np.isclose(out["pac_bayes_bound"], ref["pac_bayes_bound"]), (
        "linear_bounds includes margin_bounds(norm_margins) (margin_bound, margin_gamma, pac_bayes_bound, pac_bayes_sigma)")
    assert out["pac_bayes_bound"] < out["vc_bound"], "on this easy problem the PAC-Bayes bound beats the VC bound"


# ---------------------------------------------------------------- stage 4: a kernel SVM


def test_stage4_svm_bounds(nb):
    from mlfs.kernels import SVC, rbf_kernel
    rng = np.random.default_rng(5)
    X = rng.normal(size=(150, 2))
    y = np.where(np.linalg.norm(X, axis=1) > 1.1, 1, -1)
    svc = SVC(C=2.0, gamma=0.7, rng=0).fit(X, y)
    out = bounded(nb.svm_bounds, svc, X, y, delta=0.05, what="svm_bounds", budget=2_000_000)
    a = np.asarray(svc.dual_coef_)
    B = math.sqrt(a @ rbf_kernel(svc.support_vectors_, gamma=0.7) @ a)
    assert np.isclose(out["rkhs_norm"], B), (
        f"the RKHS norm of f = sum_i dual_coef_i k(sv_i, .) is sqrt(dual_coef^T K_sv dual_coef) = {B:.4f}; got {out['rkhs_norm']}")
    s = np.where(y == svc.classes_[1], 1.0, -1.0)
    want = s * svc.decision_function(X) / (math.sqrt(B**2 + svc.intercept_**2) * math.sqrt(2.0))
    assert np.allclose(out["norm_margins"], want), (
        "normalized margins: y f(x) / (||(w, b)|| ||(phi(x), 1)||) with ||phi(x)||^2 = k(x, x) = 1 for the RBF kernel")
    assert np.isclose(out["margin_bound"], nb.margin_bounds(want, delta=0.05)["margin_bound"]), "margin bound of those margins"
    assert np.isclose(out["train_error"], np.mean(svc.predict(X) != y)), "train_error of the fitted SVM"
    from mlfs.kernels import polynomial_kernel
    poly = SVC(C=1.0, kernel="poly", degree=2, gamma=0.5, coef0=1.0, rng=0).fit(X, y)
    outp = nb.svm_bounds(poly, X, y, delta=0.05)
    a = np.asarray(poly.dual_coef_)
    Bp = math.sqrt(a @ polynomial_kernel(poly.support_vectors_, degree=2, gamma=0.5, coef0=1.0) @ a)
    kxx = np.array([polynomial_kernel(X[i:i + 1], degree=2, gamma=0.5, coef0=1.0)[0, 0] for i in range(len(X))])
    sp = np.where(y == poly.classes_[1], 1.0, -1.0)
    wantp = sp * poly.decision_function(X) / (math.sqrt(Bp**2 + poly.intercept_**2) * np.sqrt(kxx + 1.0))
    assert np.allclose(outp["norm_margins"], wantp), (
        "for a polynomial kernel k(x, x) varies with x: normalize each margin by sqrt(k(x, x) + 1), computed from the "
        "SVM's own kernel (get_kernel with its kernel, gamma, degree, coef0), not by a constant")


# ---------------------------------------------------------------- stage 5: random labels


def test_stage5_random_label_correlation(nb):
    from mlfs.kernels import KNeighborsClassifier
    from mlfs.linear import LogisticRegression
    X = np.random.default_rng(6).normal(size=(200, 3))
    got = bounded(nb.random_label_correlation, lambda: KNeighborsClassifier(1), X, n_draws=4, rng=0,
                  what="random_label_correlation", budget=5_000_000)
    assert np.isclose(got, 1.0), f"1-NN memorizes any labels of distinct points: correlation exactly 1, got {got}"

    class Const:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.ones(len(X), dtype=int)

    c = nb.random_label_correlation(Const, X, n_draws=400, rng=1)
    assert abs(c) < 4.5 / math.sqrt(200 * 400), f"a constant predictor has correlation ~0 with random signs, got {c:.4f}"
    lr = nb.random_label_correlation(lambda: LogisticRegression(alpha=1e-3), X, n_draws=20, rng=2)
    assert 0.0 < lr < 0.3, f"a 4-parameter linear model fits random labels of 200 points only a little: got {lr:.3f}"
    seen = []

    class Spy(Const):
        def fit(self, X, y):
            seen.append(set(np.unique(y)))
            return self

    nb.random_label_correlation(Spy, X, n_draws=3, rng=3)
    assert len(seen) == 3 and all(s_ == {-1, 1} for s_ in seen), "train each fresh model on random labels in {-1, +1}"
