"""Checks for exercise.py (double machine learning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Nuisance
models are tiny classes defined here (least squares, a quadratic least squares, a
"memorizer" that only knows its training rows), so the checks are fast and test the DML
machinery rather than any particular learner. Oracles: the Frisch-Waugh-Lovell theorem
(partialling out with in-sample OLS residuals = the OLS coefficient, with HC0 standard
errors) and simulated data with a known effect.
"""
import numpy as np

# ---------------------------------------------------------------- test models


class _OLS:
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        self.b = np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.b


class _Quad(_OLS):
    """least squares on (x, x^2): flexible enough for the quadratic test worlds."""

    def fit(self, X, y):
        return super().fit(np.column_stack([X, X**2]), y)

    def predict(self, X):
        return super().predict(np.column_stack([X, X**2]))


class _Mean:
    def fit(self, X, y):
        self.c = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.c)


class _Memorizer:
    """predicts y exactly for rows it was trained on and -999 for any row it has never seen."""

    def fit(self, X, y):
        self.table = {tuple(np.round(r, 12)): v for r, v in zip(X, y)}
        return self

    def predict(self, X):
        return np.array([self.table.get(tuple(np.round(r, 12)), -999.0) for r in X])


class _LogitQuad:
    """logistic regression on (x, x^2) by a few Newton steps, with predict_proba."""

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X, X**2])
        b = np.zeros(A.shape[1])
        for _ in range(25):
            p = 1 / (1 + np.exp(-A @ b))
            H = A.T @ (A * (p * (1 - p))[:, None]) + 1e-8 * np.eye(len(b))
            b += np.linalg.solve(H, A.T @ (y - p))
        self.b = b
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.column_stack([np.ones(len(X)), X, X**2]) @ self.b))
        return np.column_stack([1 - p, p])


def _plr(n, rng, theta=0.5):
    """partially linear world: d = m(X) + v, y = theta d + g(X) + u, with quadratic m and g."""
    X = rng.normal(size=(n, 3))
    m = X[:, 0] + 0.5 * X[:, 1] ** 2
    g = X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2]
    d = m + rng.normal(size=n)
    y = theta * d + g + rng.normal(size=n)
    return X, y, d


def _irm(n, rng):
    """binary-treatment world, ATE = 1: logit e = 0.8 x0 - 0.4 x1^2 + 0.3, mu0 = 2 x0 + x1^2, effect 1 + 0.5 x0."""
    X = rng.normal(size=(n, 2))
    e = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.4 * X[:, 1] ** 2 + 0.3)))
    d = (rng.random(n) < e).astype(float)
    y = 2 * X[:, 0] + X[:, 1] ** 2 + d * (1 + 0.5 * X[:, 0]) + rng.normal(size=n)
    return X, y, d


# ---------------------------------------------------------------- stage 1: cross-fitting


def test_stage1_never_predicts_a_training_row(nb):
    rng = np.random.default_rng(0)
    X, y, d = _plr(60, rng)
    out = nb.cross_fit_predict(_Memorizer, X, y, n_folds=4, rng=1)
    assert np.shape(out) == (60,), f"one prediction per row, shape (n,); got {np.shape(out)}"
    leaked = int(np.sum(out != -999.0))
    assert leaked == 0, (
        f"{leaked} rows were predicted by a model that was trained on them: each row's prediction must come "
        "from the model fitted on the OTHER folds")


def test_stage1_folds_follow_the_spec(nb):
    rng = np.random.default_rng(2)
    X, y, _ = _plr(23, rng)
    sizes = []

    class Spy(_Mean):
        def fit(self, X, y):
            sizes.append(len(y))
            return super().fit(X, y)

    out = nb.cross_fit_predict(Spy, X, y, n_folds=5, rng=7)
    assert sorted(sizes) == [18, 18, 18, 19, 19], (
        f"5 folds of 23 rows have sizes 5, 5, 5, 4, 4, so the 5 models train on 18, 18, 18, 19, 19 rows; "
        f"fit saw {sorted(sizes)}")
    want = np.empty(23)
    for test in np.array_split(np.random.default_rng(7).permutation(23), 5):
        want[test] = np.delete(y, test).mean()
    assert np.allclose(out, want), (
        "folds should be np.array_split(as_rng(rng).permutation(n), n_folds), in that order: "
        "with a mean model, each row gets the mean of y over the other folds")
    g = np.random.default_rng(7)
    assert np.allclose(nb.cross_fit_predict(_Mean, X, y, n_folds=5, rng=g), want), "a Generator works as rng too"
    custom = [np.arange(0, 10), np.arange(10, 23)]
    got = nb.cross_fit_predict(_Mean, X, y, folds=custom, rng=123)
    assert np.allclose(got[:10], y[10:].mean()) and np.allclose(got[10:], y[:10].mean()), (
        "when folds= is given (a list of test-index arrays), use exactly those folds and ignore rng and n_folds")


def test_stage1_proba_response(nb):
    rng = np.random.default_rng(3)
    X, _, d = _irm(300, rng)
    p = nb.cross_fit_predict(_LogitQuad, X, d, n_folds=3, response="proba", rng=0)
    assert np.all((p > 0) & (p < 1)), "response='proba' returns predict_proba(X)[:, 1], probabilities in (0, 1)"
    assert abs(p.mean() - d.mean()) < 0.05, f"cross-fitted propensities should average near the treated share {d.mean():.2f}"
    try:
        nb.cross_fit_predict(_Mean, X, d, response="banana")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown response should raise ValueError")


# ---------------------------------------------------------------- stage 2: scores


def test_stage2_partialling_out_hand(nb):
    th, se = nb.partialling_out(np.array([1.0, 2, -1, 0]), np.array([1.0, 1, -1, -1]))
    assert abs(th - 1.0) < 1e-12, f"theta = sum(d_res y_res) / sum(d_res^2) = 4 / 4 = 1; got {th}"
    assert abs(se - np.sqrt(0.125)) < 1e-12, (
        f"the lesson's worked example: psi = d_res (y_res - theta d_res) = (0, 1, 0, -1), "
        f"se = sqrt(mean(psi^2) / mean(d_res^2)^2 / n) = sqrt(0.5 / 1 / 4) = 0.3536; got {se}")


def test_stage2_uncentred_residuals(nb):
    y_res, d_res = np.array([2.0, 3, 1, 4]), np.array([1.0, 2, 1, 2])
    th, se = nb.partialling_out(y_res, d_res)
    assert abs(th - 17 / 10) < 1e-12, (
        f"theta = sum(d_res y_res) / sum(d_res^2) = 17 / 10, with NO centring (the residuals are used as given); got {th}")
    psi = d_res * (y_res - 1.7 * d_res)
    assert abs(se - np.sqrt(np.mean(psi**2) / np.mean(d_res**2) ** 2 / 4)) < 1e-12, "se uses the same uncentred residuals"


def test_stage2_frisch_waugh_lovell(nb):
    rng = np.random.default_rng(4)
    X, y, d = _plr(500, rng)
    ry = y - _OLS().fit(X, y).predict(X)
    rd = d - _OLS().fit(X, d).predict(X)
    th, se = nb.partialling_out(ry, rd)
    A = np.column_stack([np.ones(500), d, X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    u = y - A @ beta
    bread = np.linalg.inv(A.T @ A)
    hc0 = bread @ (A.T * u**2) @ A @ bread
    assert abs(th - beta[1]) < 1e-8, (
        f"with in-sample OLS residuals, partialling out must reproduce the OLS coefficient on d "
        f"(Frisch-Waugh-Lovell): expected {beta[1]:.6f}, got {th:.6f}")
    assert abs(se - np.sqrt(hc0[1, 1])) < 1e-8, (
        f"...and its standard error is the heteroskedasticity-robust (HC0) one: expected {np.sqrt(hc0[1, 1]):.6f}, "
        f"got {se:.6f}. divide by n inside the means, not n - 1.")


def test_stage2_naive_plugin_first_step(nb):
    hand = nb.naive_plugin(np.zeros((3, 1)), np.array([1.0, 3, 2]), np.array([1.0, 2, 3]), _Mean, n_iters=1)
    assert abs(hand - 1 / 14) < 1e-12, (
        f"the lesson's worked example: d = (1, 2, 3), y = (1, 3, 2), a constant model: theta_1 = 1/14 = 0.0714; got {hand}. "
        "the naive score divides by sum(d^2), not by sum((d - mean d)^2).")
    rng = np.random.default_rng(5)
    X, y, d = _plr(200, rng)
    th1 = nb.naive_plugin(X, y, d, _Mean, n_iters=1)
    want = np.sum(d * (y - y.mean())) / np.sum(d**2)
    assert abs(th1 - want) < 1e-10, (
        f"starting from theta = 0, one iteration fits g on y - 0 * d (a mean model gives g = mean(y)), then sets "
        f"theta = sum(d (y - g)) / sum(d^2) = {want:.6f}; got {th1:.6f}")


def test_stage2_naive_plugin_iterations(nb):
    calls = []

    class Count(_OLS):
        def fit(self, X, y):
            calls.append(1)
            return super().fit(X, y)

    rng = np.random.default_rng(6)
    X, y, d = _plr(300, rng)
    th = nb.naive_plugin(X, y, d, Count, n_iters=200)
    assert len(calls) == 200, f"n_iters=200 means 200 fits of g; got {len(calls)}"
    beta = np.linalg.lstsq(np.column_stack([np.ones(300), d, X]), y, rcond=None)[0][1]
    assert abs(th - beta) < 1e-4, (
        f"with a linear g, alternating between g and theta converges to the joint OLS coefficient {beta:.5f}; "
        f"got {th:.5f} after 200 iterations")


# ---------------------------------------------------------------- stage 3: DML for the partially linear model


def test_stage3_composition_one_repeat(nb):
    rng = np.random.default_rng(7)
    X, y, d = _plr(300, rng)
    out = nb.dml_plr(X, y, d, _Quad, _OLS, n_folds=3, rng=11)
    folds = np.array_split(np.random.default_rng(11).permutation(300), 3)
    ry = y - nb.cross_fit_predict(_Quad, X, y, folds=folds)
    rd = d - nb.cross_fit_predict(_OLS, X, d, folds=folds)
    th, se = nb.partialling_out(ry, rd)
    assert {"theta", "se", "ci", "thetas"} <= set(out), (
        f"return a dict with theta, se, ci and thetas; got keys {sorted(out)}")
    assert np.allclose(out["thetas"], [th]), "thetas holds the per-repeat estimates (one here)"
    assert abs(out["theta"] - th) < 1e-10 and abs(out["se"] - se) < 1e-10, (
        "with one repeat, dml_plr = partialling_out on cross-fitted residuals where BOTH nuisances use the SAME "
        "fold partition: np.array_split(as_rng(rng).permutation(n), n_folds), passed to cross_fit_predict as folds=")
    lo, hi = out["ci"]
    assert abs(lo - (th - 1.96 * se)) < 1e-10 and abs(hi - (th + 1.96 * se)) < 1e-10, "ci = theta -+ 1.96 se"


def test_stage3_median_aggregation(nb):
    rng = np.random.default_rng(8)
    X, y, d = _plr(200, rng)
    out = nb.dml_plr(X, y, d, _Quad, _Quad, n_folds=2, n_repeats=5, rng=3)
    g = np.random.default_rng(3)
    ths, ses = [], []
    for _ in range(5):
        folds = np.array_split(g.permutation(200), 2)
        ry = y - nb.cross_fit_predict(_Quad, X, y, folds=folds)
        rd = d - nb.cross_fit_predict(_Quad, X, d, folds=folds)
        t, s = nb.partialling_out(ry, rd)
        ths.append(t)
        ses.append(s)
    ths, ses = np.array(ths), np.array(ses)
    med = np.median(ths)
    assert np.allclose(out["thetas"], ths), "thetas holds the 5 per-repeat estimates, in order"
    assert abs(out["theta"] - med) < 1e-10, f"theta is the median over repeats {np.round(ths, 4).tolist()}"
    want_se = np.sqrt(np.median(ses**2 + (ths - med) ** 2))
    assert abs(out["se"] - want_se) < 1e-10, (
        f"se = sqrt(median(se_s^2 + (theta_s - theta)^2)) = {want_se:.5f}; got {out['se']:.5f}. "
        "the spread between repeats is part of the uncertainty.")


def test_stage3_recovers_effect(nb):
    rng = np.random.default_rng(9)
    X, y, d = _plr(4000, rng, theta=0.5)
    out = nb.dml_plr(X, y, d, _Quad, _Quad, n_folds=5, rng=0)
    assert abs(out["theta"] - 0.5) < 4 * out["se"] + 0.01, (
        f"with flexible enough nuisance models DML should recover theta = 0.5; got {out['theta']:.3f} (se {out['se']:.3f})")
    assert 0.01 < out["se"] < 0.05, f"the standard error for n = 4000 should be about 0.016; got {out['se']:.4f}"
    lin = nb.dml_plr(X, y, d, _OLS, _OLS, n_folds=5, rng=0)
    assert abs(lin["theta"] - 0.5) > 0.1, (
        f"(sanity) with LINEAR nuisance models the quadratic confounding isn't removed; got {lin['theta']:.3f}")


# ---------------------------------------------------------------- stage 4: the interactive model


def test_stage4_matches_spec(nb):
    rng = np.random.default_rng(10)
    X, y, d = _irm(240, rng)
    out = nb.dml_irm(X, y, d, _Quad, _LogitQuad, n_folds=4, clip=0.05, rng=5)
    n = 240
    mu0, mu1, e = np.empty(n), np.empty(n), np.empty(n)
    for test in np.array_split(np.random.default_rng(5).permutation(n), 4):
        train = np.setdiff1d(np.arange(n), test)
        t0, t1 = train[d[train] == 0], train[d[train] == 1]
        mu0[test] = _Quad().fit(X[t0], y[t0]).predict(X[test])
        mu1[test] = _Quad().fit(X[t1], y[t1]).predict(X[test])
        e[test] = _LogitQuad().fit(X[train], d[train]).predict_proba(X[test])[:, 1]
    e = np.clip(e, 0.05, 0.95)
    psi = mu1 - mu0 + d * (y - mu1) / e - (1 - d) * (y - mu0) / (1 - e)
    assert abs(out["theta"] - psi.mean()) < 1e-8, (
        f"expected theta {psi.mean():.5f}, got {out['theta']:.5f}. one permutation from as_rng(rng) gives the folds; "
        "in each, fit mu0 on the training CONTROLS, mu1 on the training TREATED, the classifier on all training "
        "rows, predict the held-out fold, clip e to [clip, 1 - clip], then average the AIPW scores.")
    assert abs(out["se"] - psi.std(ddof=1) / np.sqrt(n)) < 1e-8, "se = std(scores, ddof=1) / sqrt(n)"
    assert {"ci", "scores"} <= set(out), f"return ci and scores too; got keys {sorted(out)}"
    assert np.allclose(out["scores"], psi), "scores are the n AIPW scores"
    th, se = out["theta"], out["se"]
    assert np.allclose(out["ci"], (th - 1.96 * se, th + 1.96 * se)), "ci = theta -+ 1.96 se"


def test_stage4_recovers_ate(nb):
    rng = np.random.default_rng(11)
    X, y, d = _irm(4000, rng)
    out = nb.dml_irm(X, y, d, _Quad, _LogitQuad, n_folds=5, rng=1)
    naive = y[d == 1].mean() - y[d == 0].mean()
    assert abs(naive - 1) > 0.4, "(sanity) the naive difference should be off here"
    assert abs(out["theta"] - 1.0) < 4 * out["se"] + 0.01, (
        f"the ATE is 1; cross-fitted AIPW gave {out['theta']:.3f} (se {out['se']:.3f})")


def test_stage4_clips_extreme_propensities(nb):
    class Certain:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            p = (X[:, 0] > 0).astype(float)
            return np.column_stack([1 - p, p])

    rng = np.random.default_rng(12)
    X, y, d = _irm(200, rng)
    out = nb.dml_irm(X, y, d, _Quad, Certain, n_folds=2, clip=0.02, rng=0)
    assert np.isfinite(out["theta"]) and np.isfinite(out["se"]), (
        "a classifier that outputs exactly 0 or 1 must not produce inf/nan: clip e to [clip, 1 - clip]")
