"""Checks for exercise.py (classical forecasting).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Autocorrelations
are checked against hand examples and against SciPy's Toeplitz solver (the Yule-Walker
equations); AR fits against closed-form least squares; ARMA residuals against
`scipy.signal.lfilter`; ARIMA forecasts and their standard deviations against closed forms
for random walks, AR(1), MA(1), IMA(1, 1) and seasonal random walks; backtesting against
forecasters that record what they were shown.
"""
import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz

from mlfs.testing import bounded


def _ar(n, phi, seed, c=0.0):
    g = np.random.default_rng(seed)
    e = g.normal(size=n + 300)
    return signal.lfilter([1.0], np.r_[1.0, -np.asarray(phi)], e + c)[300:]


# ---------------------------------------------------------------- stage 1: differencing and autocorrelation


def test_stage1_difference(nb):
    x = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    assert np.allclose(nb.difference(x), [3, 5, 7, 9, 11]), "first difference x_t - x_{t-1}"
    assert np.allclose(nb.difference(x, 2), [2, 2, 2, 2]), (
        f"d=2 differences twice: the second difference of squares is constant 2; got {nb.difference(x, 2)}")
    assert np.allclose(nb.difference(x, 1, lag=2), [8, 12, 16, 20]), (
        f"lag=2: x_t - x_{{t-2}}; got {nb.difference(x, 1, lag=2)}")
    assert np.allclose(nb.difference(x, 2, lag=2), [8, 8]), "d=2, lag=2 applies (1 - B^2) twice"
    out = nb.difference(x, 0)
    assert np.allclose(out, x) and len(out) == 6, "d=0 leaves the series alone"
    ints = nb.difference(np.array([1, 4, 9]), 1)
    assert np.issubdtype(np.asarray(ints).dtype, np.floating), "difference returns a float array, even for integer input"
    seasonal = np.tile([1.0, 5.0, 2.0, 7.0], 5) + 0.5 * np.arange(20)
    assert np.allclose(nb.difference(seasonal, 1, lag=4), 2.0), (
        "a period-4 pattern plus a linear trend: one seasonal difference leaves a constant (4 * 0.5)")


def test_stage1_acf_hand_example(nb):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r = nb.acf(x, 3)
    # mean 2.5; deviations (-1.5, -0.5, 0.5, 1.5); sum of squares 5
    assert len(r) == 4, f"acf(x, nlags) returns nlags + 1 values (lags 0..nlags); got {len(r)}"
    assert np.allclose(r, [1.0, 0.25, -0.3, -0.45]), (
        f"expected [1, 0.25, -0.3, -0.45], got {np.round(r, 4).tolist()}. the lag-k autocovariance sums the "
        "n - k products but divides by n (the same denominator for every lag), and the deviations are from the "
        "overall mean. if you got 0.333 at lag 1, you divided by n - k")


def test_stage1_acf_properties(nb):
    g = np.random.default_rng(0)
    x = g.normal(size=400)
    r = nb.acf(x, 20)
    assert np.isclose(r[0], 1.0), "r_0 = 1"
    assert np.abs(r[1:]).max() < 4.5 / np.sqrt(400), (
        f"white noise: every sample autocorrelation should be within about 2/sqrt(n); got max {np.abs(r[1:]).max():.3f}")
    r2 = nb.acf(3.0 * x + 100.0, 20)
    assert np.allclose(r, r2), "autocorrelation is invariant to shifting and scaling the series (subtract the mean)"
    y = _ar(20_000, [0.8], seed=1)
    ry = nb.acf(y, 4)
    assert np.allclose(ry[1:], 0.8 ** np.arange(1, 5), atol=0.08), (
        f"an AR(1) with phi = 0.8 has autocorrelation 0.8^k; got {np.round(ry, 3).tolist()}")


def test_stage1_pacf_matches_yule_walker(nb):
    y = _ar(4000, [0.5, 0.3], seed=2)
    pa = nb.pacf(y, 6)
    r = nb.acf(y, 6)
    expected = [1.0] + [solve_toeplitz(r[:k], r[1:k + 1])[-1] for k in range(1, 7)]
    assert len(pa) == 7, f"pacf(x, nlags) returns nlags + 1 values with pacf[0] = 1; got {len(pa)}"
    assert np.allclose(pa, expected, atol=1e-10), (
        f"the lag-k partial autocorrelation is the last coefficient of the Yule-Walker AR(k) fit on acf(x). "
        f"expected {np.round(expected, 4).tolist()}, got {np.round(pa, 4).tolist()}")
    assert np.abs(pa[3:]).max() < 0.1 and pa[2] > 0.2, (
        "an AR(2)'s partial autocorrelation cuts off after lag 2")


# ---------------------------------------------------------------- stage 2: simulation and AR fitting


def _oracle_arma(n, phi, theta, c, sigma, burn, seed):
    e = np.random.default_rng(seed).normal(0.0, sigma, n + burn)
    x = np.zeros(n + burn)
    for t in range(n + burn):
        x[t] = c + e[t]
        x[t] += sum(p * x[t - i] for i, p in enumerate(phi, 1) if t - i >= 0)
        x[t] += sum(q * e[t - j] for j, q in enumerate(theta, 1) if t - j >= 0)
    return x[burn:]


def test_stage2_simulate_arma_recursion(nb):
    for phi, theta, c, sigma, burn in [((0.5,), (), 0.0, 1.0, 0), ((0.6, -0.2), (0.4,), 1.5, 2.0, 7),
                                       ((), (-0.7, 0.2), -1.0, 0.5, 3)]:
        got = nb.simulate_arma(30, phi, theta, c=c, sigma=sigma, burn=burn, seed=4)
        want = _oracle_arma(30, phi, theta, c, sigma, burn, 4)
        assert got.shape == (30,), f"return exactly n values after dropping the burn-in; got shape {got.shape}"
        assert np.allclose(got, want), (
            f"phi={phi}, theta={theta}, c={c}, sigma={sigma}, burn={burn}: the recursion "
            "x_t = c + sum phi_i x_{t-i} + e_t + sum theta_j e_{t-j}, started from zeros, with all n + burn shocks "
            "drawn in one call rng.normal(0, sigma, n + burn) from rng = np.random.default_rng(seed), then the first "
            f"`burn` values dropped. first values expected {np.round(want[:3], 4).tolist()}, got {np.round(got[:3], 4).tolist()}")


def test_stage2_fit_ar_is_least_squares(nb):
    y = _ar(500, [0.9, -0.3], seed=3, c=2.0)
    c, phi, s2 = nb.fit_ar(y, 2)
    X = np.column_stack([np.ones(498), y[1:-1], y[:-2]])
    coef, *_ = np.linalg.lstsq(X, y[2:], rcond=None)
    resid = y[2:] - X @ coef
    assert np.allclose([c, *phi], coef), (
        f"regress x_t on (1, x_{{t-1}}, x_{{t-2}}) for t = p..n-1: expected (c, phi) = {np.round(coef, 4).tolist()}, "
        f"got {np.round([c, *phi], 4).tolist()}. phi[0] is the lag-1 coefficient")
    assert np.isclose(s2, resid @ resid / 498), (
        f"sigma2 is the mean squared residual over the n - p fitted points: {resid @ resid / 498:.5f}, got {s2:.5f}")
    c0, phi0, s0 = nb.fit_ar(y, 0)
    assert np.isclose(c0, y.mean()) and len(phi0) == 0 and np.isclose(s0, y.var()), (
        "p = 0: the intercept is the mean, phi is empty, sigma2 is the variance")


def test_stage2_fit_ar_recovers_simulation(nb):
    y = nb.simulate_arma(5000, [0.5, 0.3], c=1.0, sigma=2.0, seed=5)
    c, phi, s2 = nb.fit_ar(y, 2)
    assert np.allclose(phi, [0.5, 0.3], atol=0.06), f"true phi (0.5, 0.3); got {np.round(phi, 3).tolist()}"
    assert abs(c - 1.0) < 0.35, f"true c = 1; got {c:.3f}"
    assert abs(s2 - 4.0) < 0.4, f"true sigma^2 = 4 (sigma = 2); got {s2:.3f}"


# ---------------------------------------------------------------- stage 3: ARMA by conditional sum of squares


def test_stage3_residuals_match_lfilter(nb):
    g = np.random.default_rng(6)
    x = g.normal(size=40)
    c, phi, theta = 0.3, np.array([0.5, -0.2]), np.array([0.4, 0.1])
    e = nb.arma_residuals(x, c, phi, theta)
    assert e.shape == (40,), f"one residual per observation (zeros for t < p); got shape {e.shape}"
    assert np.allclose(e[:2], 0.0), (
        f"the first p residuals are 0 (we condition on x_0..x_{{p-1}}); got {e[:2]}")
    w = x[2:] - c - phi[0] * x[1:-1] - phi[1] * x[:-2]
    want = signal.lfilter([1.0], np.r_[1.0, theta], w)
    assert np.allclose(e[2:], want), (
        "for t >= p: e_t = x_t - c - sum phi_i x_{t-i} - sum theta_j e_{t-j}, with residuals before t = p taken "
        f"as 0. expected {np.round(want[:4], 4).tolist()}, got {np.round(e[2:6], 4).tolist()}")
    ar_only = nb.arma_residuals(x, c, phi)
    assert np.allclose(ar_only[2:], w), "with no theta the residuals are just the AR prediction errors"
    ma_only = nb.arma_residuals(x, 0.0, (), (0.5,))
    assert np.allclose(ma_only, signal.lfilter([1.0], [1.0, 0.5], x)), "MA(1): e_t = x_t - 0.5 e_{t-1}, e_{-1} = 0"


def test_stage3_fit_arma_q0_is_fit_ar(nb):
    y = _ar(600, [0.7], seed=7, c=0.5)
    c, phi, theta, s2 = bounded(nb.fit_arma, y, 1, 0, what="fit_arma")
    c2, phi2, s22 = nb.fit_ar(y, 1)
    assert len(theta) == 0, "q = 0: theta is empty"
    assert np.allclose([c, *phi, s2], [c2, *phi2, s22], atol=1e-4), (
        "with q = 0 the conditional sum of squares is exactly least squares: fit_arma(x, p, 0) must agree with fit_ar")


def test_stage3_fit_arma_recovers_parameters(nb):
    for phi, theta in [((0.6,), (0.4,)), ((-0.5,), (-0.4,)), ((), (-0.6,)), ((0.5, -0.3), (0.5,))]:
        y = nb.simulate_arma(4000, phi, theta, c=1.0, seed=8)
        c, ph, th, s2 = bounded(nb.fit_arma, y, len(phi), len(theta), what="fit_arma")
        assert np.allclose(ph, phi, atol=0.1) and np.allclose(th, theta, atol=0.1), (
            f"ARMA with phi={phi}, theta={theta}: fitted phi={np.round(ph, 3).tolist()}, theta={np.round(th, 3).tolist()}. "
            "minimize the mean squared conditional residual over (c, phi, theta); a sign flip on theta means "
            "your residual recursion subtracts where it should add (or the reverse)")
        assert abs(s2 - 1.0) < 0.1, f"sigma2 should be close to 1 (unit shocks); got {s2:.3f}"
        e = nb.arma_residuals(y, c, ph, th)[len(phi):]
        assert np.isclose(s2, e @ e / len(e), rtol=1e-6), (
            "sigma2 is the mean squared residual over t >= p at the returned parameters")
        e_true = nb.arma_residuals(y, 1.0, phi, theta)[len(phi):]
        assert s2 <= e_true @ e_true / len(e_true) + 1e-6, (
            "the fit should minimize the conditional sum of squares: the true parameters shouldn't do better")


# ---------------------------------------------------------------- stage 4: ARIMA


def test_stage4_random_walk_with_drift(nb):
    g = np.random.default_rng(9)
    y = np.cumsum(0.2 + g.normal(size=300))
    model = nb.ARIMA((0, 1, 0))
    assert model.fit(y) is model, "fit returns self"
    dy = np.diff(y)
    assert np.isclose(model.c_, dy.mean()) and np.isclose(model.sigma2_, dy.var()), (
        f"ARIMA(0, 1, 0): the constant is the mean of the differences ({dy.mean():.4f}), sigma2 their variance; "
        f"got c_={model.c_:.4f}, sigma2_={model.sigma2_:.4f}")
    mean, std = model.forecast(4, return_std=True)
    j = np.arange(1, 5)
    assert np.allclose(mean, y[-1] + dy.mean() * j), (
        f"a random walk with drift forecasts y_n + c j; expected {np.round(y[-1] + dy.mean() * j, 3).tolist()}, "
        f"got {np.round(mean, 3).tolist()}")
    assert np.allclose(std, np.sqrt(dy.var() * j)), (
        f"its h-step standard deviation is sigma sqrt(h) (psi weights all 1); got {np.round(std, 3).tolist()}")
    again = model.forecast(4)
    assert np.allclose(again, mean) and np.ndim(again) == 1, (
        "forecast(h) without return_std returns just the mean array, and calling forecast twice gives the same answer")


def test_stage4_ar1_closed_form(nb):
    y = _ar(800, [0.7], seed=10, c=3.0)
    model = nb.ARIMA((1, 0, 0)).fit(y)
    c, phi, s2 = nb.fit_ar(y, 1)
    assert np.isclose(model.c_, c, atol=1e-4) and np.allclose(model.phi_, phi, atol=1e-4), (
        "ARIMA((1, 0, 0)).fit should give the same AR(1) as fit_ar")
    mean, std = model.forecast(6, return_std=True)
    ph, mu = model.phi_[0], model.c_ / (1 - model.phi_[0])
    j = np.arange(1, 7)
    assert np.allclose(mean, mu + ph ** j * (y[-1] - mu)), (
        "AR(1) forecasts decay geometrically toward the mean c / (1 - phi)")
    assert np.allclose(std, np.sqrt(model.sigma2_ * (1 - ph ** (2 * j)) / (1 - ph ** 2))), (
        "AR(1): psi_i = phi^i, so the h-step variance is sigma2 (1 - phi^(2h)) / (1 - phi^2)")


def test_stage4_ma1_uses_last_residual(nb):
    y = nb.simulate_arma(1500, (), (0.6,), c=2.0, seed=11)
    model = nb.ARIMA((0, 0, 1)).fit(y)
    e = nb.arma_residuals(y, model.c_, (), model.theta_)
    mean, std = model.forecast(3, return_std=True)
    th = model.theta_[0]
    assert np.allclose(mean, [model.c_ + th * e[-1], model.c_, model.c_]), (
        f"MA(1): the 1-step forecast is c + theta e_n (the last in-sample residual), later ones are c; "
        f"expected {[round(model.c_ + th * e[-1], 4), round(model.c_, 4)]}, got {np.round(mean, 4).tolist()}")
    assert np.allclose(std, np.sqrt(model.sigma2_ * np.array([1, 1 + th ** 2, 1 + th ** 2]))), (
        "MA(1): psi = (1, theta, 0, ...)")


def test_stage4_ima11(nb):
    level = np.cumsum(nb.simulate_arma(600, (), (-0.5,), c=0.1, seed=12))
    model = nb.ARIMA((0, 1, 1)).fit(level)
    w = np.diff(level)
    e = nb.arma_residuals(w, model.c_, (), model.theta_)
    th, c = model.theta_[0], model.c_
    mean, std = model.forecast(4, return_std=True)
    first = level[-1] + c + th * e[-1]
    assert np.allclose(mean, first + c * np.arange(4)), (
        "IMA(1, 1): forecast the differences (c + theta e_n, then c, c, ...) and add them up from the last level")
    assert np.allclose(std ** 2 / model.sigma2_, [1, 1 + (1 + th) ** 2, 1 + 2 * (1 + th) ** 2, 1 + 3 * (1 + th) ** 2]), (
        "with d = 1 the psi weights come from theta(B) / (1 - B): (1, 1 + theta, 1 + theta, ...). build them "
        "from the INTEGRATED AR polynomial phi(B) (1 - B)^d, not from phi alone")


def test_stage4_seasonal_difference(nb):
    g = np.random.default_rng(13)
    pattern = np.array([10.0, 12.0, 9.0, 15.0])
    y = np.tile(pattern, 30) + np.cumsum(g.normal(scale=0.3, size=120))
    model = nb.ARIMA((0, 0, 0), seasonal_period=4).fit(y)
    w = y[4:] - y[:-4]
    assert np.isclose(model.c_, w.mean()) and np.isclose(model.sigma2_, w.var()), (
        "seasonal_period=4 with order (0, 0, 0): fit the seasonal differences y_t - y_{t-4} (their mean and variance)")
    mean, std = model.forecast(9, return_std=True)
    j = np.arange(9)
    want = y[len(y) - 4 + j % 4] + w.mean() * (1 + j // 4)
    assert np.allclose(mean, want), (
        f"a seasonal random walk repeats the last season (plus the drift once per season passed); expected "
        f"{np.round(want, 2).tolist()}, got {np.round(mean, 2).tolist()}")
    assert np.allclose(std, np.sqrt(w.var() * (1 + j // 4))), (
        "its psi weights are 1 at lags 0, 4, 8, ... and 0 elsewhere, so the variance steps up once per season")


def test_stage4_seasonal_and_regular_difference(nb):
    g = np.random.default_rng(14)
    y = np.tile([0.0, 3.0, 1.0], 40) + np.cumsum(np.cumsum(g.normal(scale=0.1, size=120)))
    model = nb.ARIMA((0, 1, 0), seasonal_period=3).fit(y)
    w = np.diff(y[3:] - y[:-3])
    assert np.isclose(model.c_, w.mean()), "difference once at the seasonal lag and d times at lag 1, then fit"
    yy = list(y)
    for _ in range(5):
        yy.append(w.mean() + yy[-1] + yy[-3] - yy[-4])
    assert np.allclose(model.forecast(5), yy[-5:]), (
        "(1 - B)(1 - B^3) y_t = c: y_t = c + y_{t-1} + y_{t-3} - y_{t-4}; feed forecasts back in as you go")


def test_stage4_ar_and_ma_with_differencing(nb):
    for order, theta in [((1, 1, 0), ()), ((1, 1, 1), (0.4,))]:
        w_true = nb.simulate_arma(800, (0.6,), theta, c=0.2, seed=18)
        y = np.cumsum(w_true) + 5.0
        model = nb.ARIMA(order).fit(y)
        w = np.diff(y)
        c, phi = model.c_, model.phi_[0]
        th = model.theta_[0] if order[2] else 0.0
        e = nb.arma_residuals(w, c, (phi,), (th,) if order[2] else ())
        wf = [c + phi * w[-1] + th * e[-1]]
        for _ in range(5):
            wf.append(c + phi * wf[-1])
        want = y[-1] + np.cumsum(wf)
        mean, std = model.forecast(6, return_std=True)
        assert np.allclose(mean, want), (
            f"ARIMA{order}: forecast the differences (c + phi w_n + theta e_n, then c + phi * previous) and add them up "
            f"from the last level; expected {np.round(want, 3).tolist()}, got {np.round(mean, 3).tolist()}")
        psi_w = np.r_[1.0, (phi + th) * phi ** np.arange(5)]  # psi weights of the ARMA(1, 1) differences
        psi = np.cumsum(psi_w)  # dividing by (1 - B) accumulates them
        assert np.allclose(std, np.sqrt(model.sigma2_ * np.cumsum(psi ** 2))), (
            f"ARIMA{order}: the psi weights of theta(B) / ((1 - phi B)(1 - B)) are the running sums of the ARMA's "
            "(1, phi + theta, (phi + theta) phi, ...). build the integrated AR polynomial by multiplying phi(B) with "
            "(1 - B) (np.convolve), not by adding -1 at lag 1")


def test_stage4_seasonal_difference_with_ma_aligns_residuals(nb):
    g = np.random.default_rng(19)
    w_true = nb.simulate_arma(200, (), (0.5,), c=0.3, seed=19)
    y = np.tile([4.0, 1.0, 7.0, 2.0], 51)[:204] + np.r_[g.normal(size=4), np.zeros(200)]
    for t in range(4, 204):
        y[t] = y[t - 4] + w_true[t - 4]
    model = nb.ARIMA((0, 0, 1), seasonal_period=4).fit(y)
    w = y[4:] - y[:-4]
    e = nb.arma_residuals(w, model.c_, (), model.theta_)
    want = y[-4] + model.c_ + model.theta_[0] * e[-1]
    got = model.forecast(1)[0]
    assert np.isclose(got, want), (
        f"seasonal difference at lag 4 with an MA(1): the 1-step forecast is y_(n-3) + c + theta e_n, where e_n is the "
        f"last residual of the seasonally differenced series ({want:.4f}); got {got:.4f}. the residuals of w start at "
        "y's index 4 (seasonal_period + d), so align them with that offset")


def test_stage4_refit_uses_new_data(nb):
    a = _ar(300, [0.5], seed=15, c=1.0)
    b = _ar(300, [-0.5], seed=16, c=-4.0)
    model = nb.ARIMA((1, 0, 0))
    fa = model.fit(a).forecast(3)
    fb = model.fit(b).forecast(3)
    fresh = nb.ARIMA((1, 0, 0)).fit(b).forecast(3)
    assert np.allclose(fb, fresh) and not np.allclose(fa, fb), (
        "refitting the same ARIMA object on new data must forget the old fit (and forecast from the new series' end)")


# ---------------------------------------------------------------- stage 5: baselines and backtesting


def test_stage5_seasonal_naive(nb):
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert np.allclose(nb.seasonal_naive(y, 3), [7, 7, 7]), "m=1 (the default) is the naive forecast: repeat the last value"
    assert np.allclose(nb.seasonal_naive(y, 7, 3), [5, 6, 7, 5, 6, 7, 5]), (
        f"m=3: repeat the last full season (5, 6, 7) as far as the horizon needs; got {nb.seasonal_naive(y, 7, 3)}")
    assert np.allclose(nb.seasonal_naive(y, 2, 4), [4, 5]), "m=4, h=2: the values one season before each target"


def test_stage5_mase(nb):
    insample = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    # lag-1 naive errors: 2, 1, 2, 1, 2 -> mean 1.6; lag-2: 1, 1, 1, 1 -> 1
    assert np.isclose(nb.mase([6.0, 4.0], [5.0, 5.0], insample), 1.0 / 1.6), (
        "MASE = MAE of the forecast / in-sample MAE of the one-step naive forecast (here 1 / 1.6)")
    assert np.isclose(nb.mase([6.0, 4.0], [5.0, 5.0], insample, m=2), 1.0), (
        "with m=2 the scale is the in-sample MAE of the seasonal-naive forecast at lag 2 (here 1)")
    assert np.isclose(nb.mase([6.0, 4.0], [7.0, 2.0], insample), 1.5 / 1.6), "mean ABSOLUTE error, not signed"


def test_stage5_rolling_origin_no_peeking(nb):
    y = np.arange(20, dtype=float) ** 1.5
    seen = []

    def spy(history, h):
        seen.append(history.copy())
        history[:] = -1.0  # a forecaster that scribbles on its input must not corrupt the series
        return np.full(h, 7.0)

    origins, fc, act = bounded(nb.rolling_origin, y, spy, initial=10, horizon=3, step=2, what="rolling_origin")
    assert list(origins) == [10, 12, 14, 16], (
        f"origins start at `initial`, advance by `step`, and the last one is the latest t with t + horizon <= n "
        f"(16 here: the next grid point, 18, would need y[18:21] from a series of length 20); got {list(origins)}")
    for t, h in zip(origins, seen):
        assert len(h) == t and np.allclose(h, (np.arange(t, dtype=float)) ** 1.5), (
            f"at origin {t} the forecaster must see exactly y[:{t}] (nothing at or after the origin); "
            f"it saw {len(h)} values")
    assert fc.shape == (4, 3) and np.allclose(fc, 7.0), f"forecasts are (n_origins, horizon); got shape {fc.shape}"
    want = np.array([y[t:t + 3] for t in origins])
    assert np.allclose(act, want), "actuals[i] = y[origin_i : origin_i + horizon] (pass the forecaster a copy)"
    o2, _, _ = nb.rolling_origin(y, lambda hist, h: np.zeros(h), initial=15, horizon=5)
    assert list(o2) == [15], "with initial=15, horizon=5, n=20 the single origin 15 fits exactly (15 + 5 <= 20)"


def test_stage5_backtest_seasonal_naive_wins_on_pure_season(nb):
    g = np.random.default_rng(17)
    y = np.tile([5.0, 9.0, 2.0, 7.0, 4.0], 20) + 0.1 * g.normal(size=100)
    _, fc, act = nb.rolling_origin(y, lambda h_, k: nb.seasonal_naive(h_, k, 5), initial=50, horizon=5, step=5)
    _, fc1, _ = nb.rolling_origin(y, lambda h_, k: nb.seasonal_naive(h_, k, 1), initial=50, horizon=5, step=5)
    m5 = nb.mase(act.ravel(), fc.ravel(), y[:50], m=5)
    m1 = nb.mase(act.ravel(), fc1.ravel(), y[:50], m=5)
    assert m5 < 0.5 * m1, f"on a strongly seasonal series the seasonal-naive forecast should crush the naive one ({m5:.2f} vs {m1:.2f})"
