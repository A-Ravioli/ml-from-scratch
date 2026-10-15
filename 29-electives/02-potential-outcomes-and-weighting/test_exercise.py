"""Checks for exercise.py (potential outcomes & weighting).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Estimators
are checked three ways: against the lesson's six-unit worked example (exact), against
identities they must satisfy (Hajek is shift-invariant, AIPW reduces to IPW and to
regression in the right limits), and on large simulated samples with a known ATE, with
tolerances of 5+ standard errors.
"""
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------- data


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _dgp(n, rng, gamma=1.0):
    """X ~ N(0, I_3); e(x) = sigmoid(gamma (x0 - 0.5 x1)); y(0) = x0 + x1 + eps, y(1) = y(0) + 2 + x2. ATE = 2."""
    X = rng.normal(size=(n, 3))
    e = _sigmoid(gamma * (X[:, 0] - 0.5 * X[:, 1]))
    t = (rng.random(n) < e).astype(float)
    y0 = X[:, 0] + X[:, 1] + rng.normal(size=n)
    y1 = y0 + 2.0 + X[:, 2]
    return X, t, np.where(t == 1, y1, y0), e, y0, y1


# the lesson's worked example
_T = np.array([1, 1, 1, 0, 0, 0.0])
_Y = np.array([4, 6, 2, 1, 3, 2.0])
_E = np.array([0.8, 0.5, 0.25, 0.8, 0.5, 0.2])


# ---------------------------------------------------------------- stage 1: propensity scores


def test_stage1_matches_logistic_mle(nb):
    rng = np.random.default_rng(0)
    X, t, *_ = _dgp(400, rng)
    e = nb.propensity_scores(X, t)
    A = np.column_stack([np.ones(len(X)), X])

    def nll(b):
        z = A @ b
        return np.sum(np.logaddexp(0, z) - t * z)

    b = minimize(nll, np.zeros(4), method="BFGS", options={"gtol": 1e-10}).x
    want = _sigmoid(A @ b)
    assert np.shape(e) == (400,), f"return one score per unit, shape (n,); got {np.shape(e)}"
    assert np.max(np.abs(e - want)) < 1e-4, (
        f"propensity scores should be the logistic-regression probabilities P(T=1 | X) (with an intercept); "
        f"max difference from a direct maximum-likelihood fit is {np.max(np.abs(e - want)):.2e}. "
        "use predict_proba(X)[:, 1], the column for class 1.")


def test_stage1_clip_and_alpha(nb):
    rng = np.random.default_rng(1)
    X, t, *_ = _dgp(300, rng, gamma=4.0)
    e = nb.propensity_scores(X, t)
    ec = nb.propensity_scores(X, t, clip=0.05)
    assert e.min() < 0.05 and e.max() > 0.95, "(sanity) strong confounding should give some extreme scores"
    assert np.isclose(ec.min(), 0.05) and np.isclose(ec.max(), 0.95), (
        f"clip=0.05 should clip the scores to [0.05, 0.95]; got range [{ec.min():.3f}, {ec.max():.3f}]")
    mid = (e > 0.05) & (e < 0.95)
    assert np.allclose(ec[mid], e[mid]), "clipping must leave scores inside [clip, 1 - clip] unchanged"
    er = nb.propensity_scores(X, t, alpha=5.0)
    assert np.std(er) < np.std(e), "a large ridge penalty alpha shrinks the logistic weights: scores move towards the mean"


def test_stage1_effective_sample_size(nb):
    ess = nb.effective_sample_size
    assert abs(ess(1 / _E[:3]) - 7.25**2 / 21.5625) < 1e-9, (
        f"the treated weights (1.25, 2, 4) of the worked example have ESS 52.5625 / 21.5625 = 2.438; got {ess(1 / _E[:3])}")
    assert abs(ess(np.full(10, 3.0)) - 10) < 1e-9, "equal weights give an ESS equal to n"
    w = np.random.default_rng(2).exponential(size=50)
    assert abs(ess(w) - ess(7 * w)) < 1e-9, "the ESS must not depend on the scale of the weights"
    assert abs(ess(np.array([1.0, 0, 0, 0])) - 1) < 1e-9, "one nonzero weight means an ESS of 1"


# ---------------------------------------------------------------- stage 2: IPW


def test_stage2_worked_example(nb):
    ht = nb.ipw_ate(_Y, _T, _E)
    hj = nb.ipw_ate(_Y, _T, _E, normalized=True)
    assert abs(ht - 23 / 12) < 1e-9, (
        f"Horvitz-Thompson on the worked example: 25/6 - 13.5/6 = 1.9167; got {ht:.4f}. "
        "controls are weighted by 1 / (1 - e), and both sums are divided by n = 6.")
    assert abs(hj - (25 / 7.25 - 13.5 / 8.25)) < 1e-9, (
        f"Hajek divides each weighted sum by the sum of its weights (7.25 and 8.25): 3.4483 - 1.6364 = 1.8119; got {hj:.4f}")
    att = nb.ipw_att(_Y, _T, _E)
    assert abs(att - (4 - 7.5 / 5.25)) < 1e-9, (
        f"ATT on the worked example: 4 - 7.5/5.25 = 2.5714; got {att:.4f}. controls get weights e / (1 - e).")


def test_stage2_hajek_shift_invariance(nb):
    rng = np.random.default_rng(3)
    X, t, y, e, *_ = _dgp(500, rng, gamma=2.0)
    a, b = nb.ipw_ate(y, t, e, normalized=True), nb.ipw_ate(y + 100, t, e, normalized=True)
    assert abs(a - b) < 1e-8, f"adding 100 to every outcome must not change the Hajek estimate; it moved by {b - a:.3g}"
    a2, b2 = nb.ipw_ate(y, t, e), nb.ipw_ate(y + 100, t, e)
    assert abs((b2 - a2) - 100 * (np.mean(t / e) - np.mean((1 - t) / (1 - e)))) < 1e-6, (
        "Horvitz-Thompson shifts by 100 * (mean(t/e) - mean((1-t)/(1-e))) when you add 100 to y: "
        "check you divide by n, not by the sum of the weights, when normalized=False")


def test_stage2_unbiased_with_true_scores(nb):
    rng = np.random.default_rng(4)
    X, t, y, e, y0, y1 = _dgp(200_000, rng, gamma=1.0)
    naive = y[t == 1].mean() - y[t == 0].mean()
    for norm in (False, True):
        est = nb.ipw_ate(y, t, e, normalized=norm)
        assert abs(est - 2.0) < 0.06, (
            f"with the true propensity scores IPW (normalized={norm}) is consistent for the ATE 2; got {est:.3f} "
            f"(the naive difference is {naive:.3f})")
    att_true = np.mean((y1 - y0)[t == 1])
    att = nb.ipw_att(y, t, e)
    assert abs(att - att_true) < 0.04, f"the ATT here is {att_true:.3f} (sample); ipw_att gave {att:.3f}"


# ---------------------------------------------------------------- stage 3: outcome regression


def test_stage3_separate_linear_fits(nb):
    rng = np.random.default_rng(5)
    X, t, y, *_ = _dgp(300, rng, gamma=1.0)
    mu0, mu1 = nb.outcome_models(X, y, t)
    A = np.column_stack([np.ones(len(X)), X])
    b0 = np.linalg.lstsq(A[t == 0], y[t == 0], rcond=None)[0]
    b1 = np.linalg.lstsq(A[t == 1], y[t == 1], rcond=None)[0]
    assert np.shape(mu0) == (300,) and np.shape(mu1) == (300,), "predict for ALL n units, both arms"
    assert np.allclose(mu0, A @ b0, atol=1e-6) and np.allclose(mu1, A @ b1, atol=1e-6), (
        "mu0 comes from a LinearRegression fitted on the CONTROL units only, mu1 from one fitted on the TREATED "
        "units only, and both predict every unit")


def test_stage3_uses_make_model(nb):
    made = []

    class Constant:
        def fit(self, X, y):
            made.append(len(y))
            self.c = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.c)

    rng = np.random.default_rng(6)
    X, t, y, *_ = _dgp(200, rng)
    mu0, mu1 = nb.outcome_models(X, y, t, make_model=Constant)
    assert sorted(made) == sorted([int((t == 0).sum()), int((t == 1).sum())]), (
        f"make_model() should be called once per arm and fitted on that arm's rows; fit saw sizes {made}")
    assert np.allclose(mu1, y[t == 1].mean()) and np.allclose(mu0, y[t == 0].mean()), (
        "with a constant model, mu1 is the treated mean and mu0 the control mean")


def test_stage3_regression_estimate_is_consistent(nb):
    rng = np.random.default_rng(7)
    X, t, y, *_ = _dgp(50_000, rng, gamma=2.0)
    mu0, mu1 = nb.outcome_models(X, y, t)
    est = np.mean(mu1 - mu0)
    assert abs(est - 2.0) < 0.05, f"with correctly specified linear models, mean(mu1 - mu0) should be near 2; got {est:.3f}"


# ---------------------------------------------------------------- stage 4: AIPW


def test_stage4_worked_example(nb):
    mu0 = np.array([1.0, 2.0, 1.0, 2.0, 2.0, 1.0])
    mu1 = mu0 + 2.0
    est, se = nb.aipw_ate(_Y, _T, _E, mu0, mu1)
    psi = np.array([2 + 1 / 0.8, 2 + 2 / 0.5, 2 - 1 / 0.25, 2 + 1 / 0.2, 2 - 1 / 0.5, 2 - 1 / 0.8])
    assert abs(est - psi.mean()) < 1e-9, (
        f"the worked example's AIPW scores are {psi.round(3).tolist()}, mean {psi.mean():.4f}; got {est:.4f}. "
        "treated units add t (y - mu1) / e, controls subtract (1 - t)(y - mu0) / (1 - e).")
    assert abs(se - psi.std(ddof=1) / np.sqrt(6)) < 1e-9, (
        f"the standard error is std(psi, ddof=1) / sqrt(n) = {psi.std(ddof=1) / np.sqrt(6):.4f}; got {se:.4f}")


def test_stage4_limits(nb):
    rng = np.random.default_rng(8)
    X, t, y, e, *_ = _dgp(400, rng)
    z = np.zeros_like(y)
    est, _ = nb.aipw_ate(y, t, e, z, z)
    assert abs(est - nb.ipw_ate(y, t, e)) < 1e-9, "with mu0 = mu1 = 0, AIPW reduces to Horvitz-Thompson IPW"
    est2, _ = nb.aipw_ate(y, t, e, y, y)
    assert abs(est2) < 1e-9, (
        "algebraic limit: with mu0 = mu1 = y (not a realistic model, just an identity), every residual and the "
        "regression term vanish, so AIPW is exactly 0")
    mu0 = np.where(t == 0, y, 0.3 * X[:, 0])
    mu1 = np.where(t == 1, y, 1.0 + 0.3 * X[:, 0])
    est3, _ = nb.aipw_ate(y, t, e, mu0, mu1)
    assert abs(est3 - np.mean(mu1 - mu0)) < 1e-9, (
        "outcome models that fit their own arm perfectly zero out the residual correction: AIPW then equals the "
        "regression estimate mean(mu1 - mu0)")


def test_stage4_double_robustness(nb):
    rng = np.random.default_rng(9)
    X, t, y, e, *_ = _dgp(100_000, rng, gamma=1.5)
    mu0, mu1 = nb.outcome_models(X, y, t)
    good_e = nb.propensity_scores(X, t)
    bad_e = np.full_like(y, t.mean())  # ignores X entirely
    bad_mu = np.zeros_like(y)
    naive = y[t == 1].mean() - y[t == 0].mean()
    assert abs(naive - 2) > 0.3, "(sanity) the naive difference should be far from 2 here"
    for label, ee, m0, m1 in [("right e, wrong outcome models", good_e, bad_mu, bad_mu),
                              ("wrong e, right outcome models", bad_e, mu0, mu1),
                              ("both right", good_e, mu0, mu1)]:
        est, se = nb.aipw_ate(y, t, ee, m0, m1)
        assert abs(est - 2.0) < max(5 * se, 0.03), (
            f"AIPW with {label} should still be consistent for the ATE 2: got {est:.3f} (se {se:.3f})")
    est, se = nb.aipw_ate(y, t, bad_e, bad_mu, bad_mu)
    assert abs(est - 2.0) > 0.3, f"(sanity) with both wrong AIPW should be biased; got {est:.3f}"
