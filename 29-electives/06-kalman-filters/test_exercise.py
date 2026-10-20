"""Checks for exercise.py (kalman filters).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The filter and
smoother are checked against brute-force Gaussian conditioning: on a small problem we build
the joint Gaussian of all states and observations explicitly and condition on the observed
values with one linear solve, which is what the Kalman recursions compute one step at a time.
The steady-state gain is checked against the closed form for a random walk in noise and
against SciPy's discrete algebraic Riccati solver.
"""
import numpy as np
from scipy import linalg, stats

from mlfs.testing import bounded

# a 2-state constant-velocity model observed in position, and a 2-D observation variant
DT = 1.0
F2 = np.array([[1.0, DT], [0.0, 1.0]])
H2 = np.array([[1.0, 0.0]])
Q2 = np.array([[0.25, 0.5], [0.5, 1.0]]) * 0.3 + 0.05 * np.eye(2)
R2 = np.array([[0.8]])
M0 = np.array([0.5, -0.2])
P0 = np.array([[2.0, 0.3], [0.3, 1.0]])
H22 = np.array([[1.0, 0.0], [1.0, 1.0]])
R22 = np.array([[0.5, 0.1], [0.1, 0.7]])


def _joint(F, H, Q, R, m0, P0, T):
    """mean and covariance of (x_0..x_{T-1}) and (y_0..y_{T-1}), stacked time-major."""
    n, m = F.shape[0], H.shape[0]
    A = np.zeros((n * T, n * T))  # x = A xi, xi = (x0, w_1, ..., w_{T-1})
    for t in range(T):
        for s in range(t + 1):
            A[t * n:(t + 1) * n, s * n:(s + 1) * n] = np.linalg.matrix_power(F, t - s)
    cov_xi = linalg.block_diag(P0, *([Q] * (T - 1)))
    mu_x = A[:, :n] @ m0
    Sx = A @ cov_xi @ A.T
    Hb = np.kron(np.eye(T), H)
    Sy = Hb @ Sx @ Hb.T + np.kron(np.eye(T), R)
    return mu_x, Sx, Hb @ mu_x, Sy, Sx @ Hb.T


def _condition(y, F, H, Q, R, m0, P0, upto):
    """E[x_t | y_s observed, s < upto] and Cov for every t, by brute force."""
    T, n, m = len(y), F.shape[0], H.shape[0]
    mu_x, Sx, mu_y, Sy, Sxy = _joint(F, H, Q, R, m0, P0, T)
    obs = np.array([(s < upto) and not np.isnan(y[s]).any() for s in range(T) for _ in range(m)])
    yflat = y.reshape(-1)
    if not obs.any():
        mean, cov = mu_x, Sx
    else:
        G = np.linalg.solve(Sy[np.ix_(obs, obs)], Sxy[:, obs].T).T
        mean = mu_x + G @ (yflat[obs] - mu_y[obs])
        cov = Sx - G @ Sxy[:, obs].T
    return mean.reshape(T, n), np.array([cov[t * n:(t + 1) * n, t * n:(t + 1) * n] for t in range(T)])


def _loglik(y, F, H, Q, R, m0, P0):
    T, m = len(y), H.shape[0]
    _, _, mu_y, Sy, _ = _joint(F, H, Q, R, m0, P0, T)
    obs = ~np.isnan(y).any(axis=1).repeat(m)
    return stats.multivariate_normal(mu_y[obs], Sy[np.ix_(obs, obs)]).logpdf(y.reshape(-1)[obs])


def _sim(T, seed, H=H2, R=R2):
    g = np.random.default_rng(seed)
    x = g.multivariate_normal(M0, P0)
    xs, ys = [], []
    for t in range(T):
        if t:
            x = F2 @ x + g.multivariate_normal(np.zeros(2), Q2)
        xs.append(x)
        ys.append(H @ x + g.multivariate_normal(np.zeros(H.shape[0]), R))
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------- stage 1: simulation


def test_stage1_simulate_matches_the_spec(nb):
    x, y = nb.simulate_lgssm(F2, H22, Q2, R22, M0, P0, 7, seed=3)
    assert x.shape == (7, 2) and y.shape == (7, 2), f"x is (T, n) and y is (T, m); got {x.shape} and {y.shape}"
    g = np.random.default_rng(3)
    xs = [g.multivariate_normal(M0, P0)]
    for _ in range(6):
        xs.append(F2 @ xs[-1] + g.multivariate_normal(np.zeros(2), Q2))
    xs = np.array(xs)
    ys = xs @ H22.T + g.multivariate_normal(np.zeros(2), R22, size=7)
    assert np.allclose(x, xs), (
        "states: x[0] = rng.multivariate_normal(m0, P0), then x[t] = F x[t-1] + rng.multivariate_normal(0, Q) for "
        "t = 1..T-1, with rng = np.random.default_rng(seed) and the draws in exactly this order")
    assert np.allclose(y, ys), (
        "observations: after all the states, one call rng.multivariate_normal(zeros(m), R, size=T) for the noise, "
        "and y = x @ H.T + noise. a transposed H or noise drawn interleaved with the states fails this")


def test_stage1_simulate_statistics(nb):
    F = np.array([[0.9]])
    x, y = nb.simulate_lgssm(F, [[2.0]], [[0.19]], [[0.25]], [0.0], [[1.0]], 20_000, seed=0)
    assert abs(x[:, 0].var() - 1.0) < 0.12, (
        f"an AR(1) state with F = 0.9, Q = 0.19 and P0 = 1 is stationary with variance 0.19 / (1 - 0.81) = 1; "
        f"got {x[:, 0].var():.3f}")
    resid = y[:, 0] - 2.0 * x[:, 0]
    assert abs(resid.var() - 0.25) < 0.015, f"y - H x should have variance R = 0.25; got {resid.var():.3f}"
    lag1 = np.corrcoef(x[1:, 0], x[:-1, 0])[0, 1]
    assert abs(lag1 - 0.9) < 0.025, f"state lag-1 autocorrelation should be F = 0.9; got {lag1:.3f}"


# ---------------------------------------------------------------- stage 2: the filter


def test_stage2_hand_example(nb):
    out = nb.kalman_filter(np.array([[1.0], [2.0]]), [[1.0]], [[1.0]], [[1.0]], [[1.0]], [0.0], [[1.0]])
    assert set(out) >= {"means", "covs", "pred_means", "pred_covs", "loglik"}, (
        f"return a dict with keys means, covs, pred_means, pred_covs, loglik; got {sorted(out)}")
    assert np.allclose(out["pred_means"][:, 0], [0.0, 0.5]) and np.allclose(out["pred_covs"][:, 0, 0], [1.0, 1.5]), (
        "the prediction at t = 0 is the prior (m0, P0) itself (no F step before the first observation); at t = 1 "
        f"it is (F m, F P F' + Q) = (0.5, 1.5). got means {out['pred_means'][:, 0]}, covs {out['pred_covs'][:, 0, 0]}")
    assert np.allclose(out["means"][:, 0], [0.5, 1.4]) and np.allclose(out["covs"][:, 0, 0], [0.5, 0.6]), (
        "t=0: S = 2, K = 0.5, m = 0.5, P = 0.5. t=1: S = 2.5, K = 0.6, m = 0.5 + 0.6 * 1.5 = 1.4, P = 0.6. "
        f"got means {out['means'][:, 0]}, covs {out['covs'][:, 0, 0]}")
    want = stats.norm(0, np.sqrt(2)).logpdf(1.0) + stats.norm(0.5, np.sqrt(2.5)).logpdf(2.0)
    assert np.isclose(out["loglik"], want), (
        f"loglik = sum_t log N(y_t; H m_pred, S_t) = {want:.5f}; got {out['loglik']:.5f}")


def test_stage2_matches_brute_force_conditioning(nb):
    _, y = _sim(6, seed=1)
    out = nb.kalman_filter(y, F2, H2, Q2, R2, M0, P0)
    for t in range(6):
        mean, cov = _condition(y, F2, H2, Q2, R2, M0, P0, upto=t + 1)
        assert np.allclose(out["means"][t], mean[t]) and np.allclose(out["covs"][t], cov[t]), (
            f"filtered moments at t={t} should equal E[x_t | y_0..y_t] and Cov[x_t | y_0..y_t] from conditioning the "
            f"joint Gaussian: mean {np.round(mean[t], 4)}, got {np.round(out['means'][t], 4)}")
        pm, pc = _condition(y, F2, H2, Q2, R2, M0, P0, upto=t)
        assert np.allclose(out["pred_means"][t], pm[t]) and np.allclose(out["pred_covs"][t], pc[t]), (
            f"predicted moments at t={t} are E[x_t | y_0..y_{{t-1}}] (the prior at t = 0)")
    assert np.isclose(out["loglik"], _loglik(y, F2, H2, Q2, R2, M0, P0)), (
        "loglik must equal the log density of all observations under the model's joint Gaussian (prediction-error "
        "decomposition); check the log-determinant and the 2 pi term")


def test_stage2_two_dimensional_observations(nb):
    _, y = _sim(5, seed=2, H=H22, R=R22)
    out = nb.kalman_filter(y, F2, H22, Q2, R22, M0, P0)
    mean, cov = _condition(y, F2, H22, Q2, R22, M0, P0, upto=5)
    assert np.allclose(out["means"][-1], mean[-1]) and np.allclose(out["covs"][-1], cov[-1]), (
        "with 2-D observations and a correlated R the update is the same matrix formula; check K = P H' S^{-1}")
    assert np.isclose(out["loglik"], _loglik(y, F2, H22, Q2, R22, M0, P0)), "loglik with m = 2"


def test_stage2_missing_observations(nb):
    _, y = _sim(7, seed=4, H=H22, R=R22)
    y[2] = np.nan
    y[3, 1] = np.nan  # one NaN makes the whole row missing
    y[6] = np.nan
    m0, p0 = M0.copy(), P0.copy()
    out = nb.kalman_filter(y, F2, H22, Q2, R22, m0, p0)
    assert np.allclose(m0, M0) and np.allclose(p0, P0), "kalman_filter must not modify m0 or P0 in place"
    for t in range(7):
        mean, cov = _condition(y, F2, H22, Q2, R22, M0, P0, upto=t + 1)
        assert np.allclose(out["means"][t], mean[t]) and np.allclose(out["covs"][t], cov[t]), (
            f"t={t}: a row of y containing any NaN is missing: skip the update (filtered = predicted) and add nothing "
            "to the log-likelihood")
    assert np.allclose(out["means"][2], out["pred_means"][2]), "with y_2 missing the filtered mean is the prediction"
    assert np.isclose(out["loglik"], _loglik(y, F2, H22, Q2, R22, M0, P0)), "loglik sums over observed rows only"
    assert np.all(np.isfinite(out["means"])), "missing values must not leak NaN into the estimates"


def test_stage2_one_dimensional_y_and_scalars(nb):
    out = nb.kalman_filter(np.array([1.0, 2.0]), [[1.0]], [[1.0]], [[1.0]], [[1.0]], [0.0], [[1.0]])
    assert np.allclose(out["means"][:, 0], [0.5, 1.4]), "a 1-D y of length T is T scalar observations"


# ---------------------------------------------------------------- stage 3: the smoother


def test_stage3_matches_brute_force(nb):
    _, y = _sim(7, seed=5)
    y[3] = np.nan
    out = nb.kalman_filter(y, F2, H2, Q2, R2, M0, P0)
    ms, Ps = nb.rts_smoother(out, F2)
    mean, cov = _condition(y, F2, H2, Q2, R2, M0, P0, upto=7)
    assert ms.shape == (7, 2) and Ps.shape == (7, 2, 2), f"return (means (T, n), covs (T, n, n)); got {ms.shape}, {Ps.shape}"
    for t in range(7):
        assert np.allclose(ms[t], mean[t]) and np.allclose(Ps[t], cov[t]), (
            f"smoothed moments at t={t} should equal E[x_t | all y] and Cov[x_t | all y]: mean "
            f"{np.round(mean[t], 4)}, got {np.round(ms[t], 4)}. the gain is G_t = P_t F' (P_pred_{{t+1}})^{{-1}}, "
            "using the FILTERED P_t and the PREDICTED P at t + 1")
    assert np.allclose(ms[-1], out["means"][-1]), "at the last step smoothing and filtering agree"
    assert np.all(np.trace(Ps, axis1=1, axis2=2) <= np.trace(out["covs"], axis1=1, axis2=2) + 1e-12), (
        "conditioning on more data can't increase the uncertainty: smoothed covariances are below filtered ones")


def test_stage3_does_not_modify_filter_output(nb):
    _, y = _sim(5, seed=6)
    out = nb.kalman_filter(y, F2, H2, Q2, R2, M0, P0)
    before = {k: np.copy(v) for k, v in out.items()}
    nb.rts_smoother(out, F2)
    assert all(np.allclose(before[k], out[k]) for k in before), "rts_smoother must not modify the filter's arrays"


# ---------------------------------------------------------------- stage 4: steady state and consistency


def test_stage4_local_level_closed_form(nb):
    for q, r in [(1.0, 1.0), (0.1, 2.0), (3.0, 0.5)]:
        K, P = bounded(nb.steady_state_gain, [[1.0]], [[1.0]], [[q]], [[r]], what="steady_state_gain")
        P_star = (q + np.sqrt(q * q + 4 * q * r)) / 2
        assert np.isclose(P[0, 0], P_star, rtol=1e-6) and np.isclose(K[0, 0], P_star / (P_star + r), rtol=1e-6), (
            f"random walk in noise (q={q}, r={r}): the steady predicted variance solves P^2 - qP - qr = 0, so "
            f"P = {P_star:.5f} and K = P / (P + r) = {P_star / (P_star + r):.5f}; got P = {P[0, 0]:.5f}, K = {K[0, 0]:.5f}. "
            "return the PREDICTED covariance and the gain computed from it")


def test_stage4_tolerance_is_relative(nb):
    for scale in (1e-8, 1e8):
        q, r = 1.0 * scale, 2.0 * scale
        K, P = bounded(nb.steady_state_gain, [[1.0]], [[1.0]], [[q]], [[r]], what="steady_state_gain")
        P_star = (q + np.sqrt(q * q + 4 * q * r)) / 2
        assert np.isclose(P[0, 0], P_star, rtol=1e-6) and np.isclose(K[0, 0], P_star / (P_star + r), rtol=1e-6), (
            f"with q and r scaled by {scale:g} the answer scales too (P = {P_star:.4g}, K unchanged); got P = {P[0, 0]:.4g}, "
            f"K = {K[0, 0]:.6f}. tol is RELATIVE to the size of P: an absolute tolerance stops too early for tiny "
            "covariances and never stops for huge ones")


def test_stage4_matches_scipy_dare(nb):
    K, P = bounded(nb.steady_state_gain, F2, H22, Q2, R22, what="steady_state_gain")
    X = linalg.solve_discrete_are(F2.T, H22.T, Q2, R22)
    assert P.shape == (2, 2) and K.shape == (2, 2), "K is (n, m), P is (n, n)"
    assert np.allclose(P, X, rtol=1e-6), "the steady predicted covariance solves the discrete algebraic Riccati equation"
    assert np.allclose(K, X @ H22.T @ np.linalg.inv(H22 @ X @ H22.T + R22), rtol=1e-6), "K = P H' (H P H' + R)^{-1}"
    out = nb.kalman_filter(np.zeros((200, 2)), F2, H22, Q2, R22, M0, P0)
    assert np.allclose(out["pred_covs"][-1], X, rtol=1e-6), "the filter's covariances converge to the same P"


def test_stage4_raises_when_riccati_diverges(nb):
    try:
        bounded(nb.steady_state_gain, [[2.0]], [[0.0]], [[1.0]], [[1.0]], max_iter=500, what="steady_state_gain",
                hint="iterate at most max_iter times")
    except RuntimeError:
        return
    except (OverflowError, FloatingPointError, np.linalg.LinAlgError):
        pass
    raise AssertionError("an unstable, unobserved state (F = 2, H = 0) has no steady state: raise RuntimeError "
                         "after max_iter iterations")


def test_stage4_nis(nb):
    out = nb.kalman_filter(np.array([[1.0], [2.0]]), [[1.0]], [[1.0]], [[1.0]], [[1.0]], [0.0], [[1.0]])
    v = nb.nis(np.array([[1.0], [2.0]]), out, [[1.0]], [[1.0]])
    assert np.allclose(v, [0.5, 1.5 ** 2 / 2.5]), (
        f"NIS_t = v_t' S_t^{{-1}} v_t with v_t = y_t - H m_pred_t: (1/2, 1.5^2 / 2.5) = (0.5, 0.9); got {v}")
    _, y = _sim(3000, seed=7, H=H22, R=R22)
    y[10] = np.nan
    good = nb.nis(y, nb.kalman_filter(y, F2, H22, Q2, R22, M0, P0), H22, R22)
    assert np.isnan(good[10]) and np.isfinite(np.delete(good, 10)).all(), "NaN exactly where the observation is missing"
    assert abs(np.nanmean(good) - 2.0) < 0.2, (
        f"with the true model, NIS is chi-squared with m = 2 degrees of freedom: mean 2; got {np.nanmean(good):.3f}")
    bad = nb.nis(y, nb.kalman_filter(y, F2, H22, Q2, R22 / 20, M0, P0), H22, R22 / 20)
    assert np.nanmean(bad) > 5, f"with R 20x too small the innovations look far too large; mean NIS {np.nanmean(bad):.2f}"


# ---------------------------------------------------------------- stage 5: learning the noise


def _is_local_max(nb, y, F, H, Q, R, q, r):
    ll = nb.kalman_filter(y, F, H, [[q * Q]], [[r * R]], [0.0], [[1.0]])["loglik"]
    for dq, dr in [(1.1, 1.0), (1 / 1.1, 1.0), (1.0, 1.1), (1.0, 1 / 1.1)]:
        other = nb.kalman_filter(y, F, H, [[q * dq * Q]], [[r * dr * R]], [0.0], [[1.0]])["loglik"]
        if other > ll + 1e-6:
            return False, dq, dr, other, ll
    return True, None, None, None, ll


def test_stage5_recovers_noise_scales(nb):
    F, H = np.array([[1.0]]), np.array([[1.0]])
    _, y = nb.simulate_lgssm(F, H, [[0.5]], [[2.0]], [0.0], [[1.0]], 200, seed=8)
    q, r = bounded(nb.fit_noise_scales, y, F, H, [[1.0]], [[1.0]], [0.0], [[1.0]], budget=20_000_000,
                   what="fit_noise_scales")
    assert 0.15 < q < 1.5 and 1.0 < r < 3.5, (
        f"local-level data with q = 0.5, r = 2 (200 steps): the maximum-likelihood scales should be in the right "
        f"region; got q = {q:.3f}, r = {r:.3f}. scale Q by q and R by r and maximize kalman_filter(...)['loglik']; "
        "optimize log q and log r so they stay positive")
    ok, dq, dr, other, ll = _is_local_max(nb, y, F, H, 1.0, 1.0, q, r)
    assert ok, (
        f"the returned scales should be a local maximum of the log-likelihood: scaling q by {dq} and r by {dr} gives "
        f"{other} > {ll:.4f}. tighten the optimizer's tolerance")


def test_stage5_handles_missing_rows(nb):
    F, H = np.array([[1.0]]), np.array([[1.0]])
    _, y = nb.simulate_lgssm(F, H, [[1.0]], [[1.0]], [0.0], [[1.0]], 120, seed=10)
    y = y.copy()
    y[40:70] = np.nan
    q, r = bounded(nb.fit_noise_scales, y, F, H, [[1.0]], [[1.0]], [0.0], [[1.0]], budget=20_000_000,
                   what="fit_noise_scales")
    assert np.isfinite(q) and np.isfinite(r), f"missing rows must not turn the scales into NaN; got q = {q}, r = {r}"
    ok, dq, dr, other, ll = _is_local_max(nb, y, F, H, 1.0, 1.0, q, r)
    assert ok, (
        f"with a gap of NaN rows, the scales should maximize the log-likelihood of the series WITH the gap (missing rows "
        f"skip their update, and the state keeps evolving through them): scaling q by {dq} and r by {dr} gives "
        f"{other} > {ll:.4f}. don't replace NaNs by zeros, and don't drop the rows (that removes the time they span)")


def test_stage5_scales_multiply_the_given_matrices(nb):
    F, H = np.array([[1.0]]), np.array([[1.0]])
    _, y = nb.simulate_lgssm(F, H, [[2.0]], [[0.5]], [0.0], [[1.0]], 100, seed=9)
    q1, r1 = bounded(nb.fit_noise_scales, y, F, H, [[1.0]], [[1.0]], [0.0], [[1.0]], budget=20_000_000,
                     what="fit_noise_scales")
    q4, r4 = bounded(nb.fit_noise_scales, y, F, H, [[4.0]], [[0.25]], [0.0], [[1.0]], budget=20_000_000,
                     what="fit_noise_scales")
    assert np.isclose(4 * q4, q1, rtol=0.05) and np.isclose(r4 / 4, r1, rtol=0.05), (
        f"the scales multiply the GIVEN Q and R: with Q = 4 and R = 1/4 the best scales are q1 / 4 and 4 r1 "
        f"({q1 / 4:.3f}, {4 * r1:.3f}); got ({q4:.3f}, {r4:.3f})")
