import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from scipy import optimize, signal

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # classical forecasting · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `difference`, `acf`, `pacf`: making a series stationary, and reading its memory |
    | 2 | `simulate_arma`, `fit_ar`: ARMA simulation and least-squares AR fitting |
    | 3 | `arma_residuals`, `fit_arma`: conditional sum of squares for ARMA(p, q) |
    | 4 | `ARIMA`: differencing, fitting, forecasting with intervals |
    | 5 | `seasonal_naive`, `mase`, `rolling_origin`: the baseline and honest backtesting |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def difference(x, d=1, *, lag=1):
    """stage 1: apply (1 - B^lag)^d to the 1-D series x; returns length len(x) - d * lag."""
    out = np.asarray(x, dtype=float)
    for _ in range(d):
        out = out[lag:] - out[:-lag]
    return out


@app.function
def acf(x, nlags):
    """stage 1: sample autocorrelations r_0..r_nlags (r_0 = 1), with the 1/n (biased) autocovariance."""
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    denom = z @ z
    return np.array([z[: len(z) - k] @ z[k:] / denom for k in range(nlags + 1)])


@app.function
def pacf(x, nlags):
    """stage 1: partial autocorrelations by Durbin-Levinson on acf(x); pacf[0] = 1."""
    r = acf(x, nlags)
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    phi = np.zeros(0)
    for k in range(1, nlags + 1):
        a = (r[k] - phi @ r[1:k][::-1]) / (1.0 - phi @ r[1:k])
        phi = np.concatenate([phi - a * phi[::-1], [a]])
        out[k] = a
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, difference=difference, acf=acf, pacf=pacf)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def simulate_arma(n, phi=(), theta=(), *, c=0.0, sigma=1.0, burn=200, seed=None):
    """stage 2: n values of x_t = c + sum phi_i x_{t-i} + e_t + sum theta_j e_{t-j}, e ~ N(0, sigma^2), after `burn`."""
    phi, theta = np.atleast_1d(np.asarray(phi, dtype=float)), np.atleast_1d(np.asarray(theta, dtype=float))
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n + burn)
    # the recursion started from x = e = 0 before the first draw: first u = c + theta(B) e, then phi(B) x = u
    u = c + signal.lfilter(np.r_[1.0, theta], [1.0], e)
    x = signal.lfilter([1.0], np.r_[1.0, -phi], u)
    return x[burn:]


@app.function
def fit_ar(x, p):
    """stage 2: least squares of x_t on (1, x_{t-1}, ..., x_{t-p}), t >= p; returns (c, phi (p,), sigma2)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n - p)] + [x[p - i: n - i] for i in range(1, p + 1)])
    coef, *_ = np.linalg.lstsq(X, x[p:], rcond=None)
    resid = x[p:] - X @ coef
    return float(coef[0]), coef[1:], float(resid @ resid / len(resid))


@app.cell(hide_code=True)
def _():
    checks.run(2, simulate_arma=simulate_arma, fit_ar=fit_ar)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def arma_residuals(x, c, phi=(), theta=()):
    """stage 3: conditional residuals; e_t = 0 for t < p, else x_t - c - sum phi_i x_{t-i} - sum theta_j e_{t-j}."""
    x = np.asarray(x, dtype=float)
    phi, theta = np.atleast_1d(np.asarray(phi, dtype=float)), np.atleast_1d(np.asarray(theta, dtype=float))
    p, n = len(phi), len(x)
    w = np.zeros(n)
    w[p:] = x[p:] - c - sum(phi[i - 1] * x[p - i: n - i] for i in range(1, p + 1))
    e = np.zeros(n)
    e[p:] = signal.lfilter([1.0], np.r_[1.0, theta], w[p:])
    return e


@app.function
def fit_arma(x, p, q):
    """stage 3: ARMA(p, q) by conditional sum of squares (scipy.optimize); returns (c, phi, theta, sigma2)."""
    x = np.asarray(x, dtype=float)
    if q == 0:
        c, phi, s2 = fit_ar(x, p)
        return c, phi, np.zeros(0), s2
    # Hannan-Rissanen start: a long AR gives residual estimates, then regress on lags of x and e
    m = min(max(p + q + 5, 10), len(x) // 4)
    c0, ph0, _ = fit_ar(x, m)
    ehat = arma_residuals(x, c0, ph0)
    k = max(p, q) + m
    n = len(x)
    cols = [np.ones(n - k)] + [x[k - i: n - i] for i in range(1, p + 1)] + [ehat[k - j: n - j] for j in range(1, q + 1)]
    start, *_ = np.linalg.lstsq(np.column_stack(cols), x[k:], rcond=None)
    start[1 + p:] = np.clip(start[1 + p:], -0.95, 0.95)

    def css(v):
        e = arma_residuals(x, v[0], v[1:1 + p], v[1 + p:])[p:]
        return e @ e / len(e)

    res = optimize.minimize(css, start, method="L-BFGS-B",
                            bounds=[(None, None)] * (1 + p) + [(-0.999, 0.999)] * q)
    v = res.x
    return float(v[0]), v[1:1 + p], v[1 + p:], float(css(v))


@app.cell(hide_code=True)
def _():
    checks.run(3, arma_residuals=arma_residuals, fit_arma=fit_arma, simulate_arma=simulate_arma, fit_ar=fit_ar)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class ARIMA:
    """stage 4: ARIMA(p, d, q), optionally after one seasonal difference at lag seasonal_period."""

    def __init__(self, order=(1, 0, 0), *, seasonal_period=None):
        self.order = tuple(int(v) for v in order)
        self.seasonal_period = seasonal_period

    def _integrated_ar(self):
        """coefficients a_1.. of y_t = c + sum a_i y_{t-i} + ..., from phi(B) (1 - B)^d (1 - B^m)."""
        poly = np.r_[1.0, -np.asarray(self.phi_, dtype=float)]
        for _ in range(self.order[1]):
            poly = np.convolve(poly, [1.0, -1.0])
        if self.seasonal_period:
            seas = np.zeros(self.seasonal_period + 1)
            seas[0], seas[-1] = 1.0, -1.0
            poly = np.convolve(poly, seas)
        return -poly[1:]

    def fit(self, y):
        """stage 4: difference y, fit ARMA(p, q) by CSS; sets c_, phi_, theta_, sigma2_; returns self."""
        p, d, q = self.order
        y = np.asarray(y, dtype=float)
        w = difference(y, 1, lag=self.seasonal_period) if self.seasonal_period else y
        w = difference(w, d)
        self.c_, self.phi_, self.theta_, self.sigma2_ = fit_arma(w, p, q)
        self.y_ = y.copy()
        self.resid_ = arma_residuals(w, self.c_, self.phi_, self.theta_)
        return self

    def forecast(self, h, *, return_std=False):
        """stage 4: h-step mean forecast of y after the fitted data (and std from psi weights if return_std)."""
        a = self._integrated_ar()
        theta = np.asarray(self.theta_, dtype=float)
        n = len(self.y_)
        offset = n - len(self.resid_)
        e = np.r_[np.zeros(offset), self.resid_, np.zeros(h)]
        yy = np.r_[self.y_, np.zeros(h)]
        for t in range(n, n + h):
            ar = sum(a[i - 1] * yy[t - i] for i in range(1, len(a) + 1))
            ma = sum(theta[j - 1] * e[t - j] for j in range(1, len(theta) + 1))
            yy[t] = self.c_ + ar + ma
        mean = yy[n:]
        if not return_std:
            return mean
        psi = np.zeros(h)
        for j in range(h):
            v = 1.0 if j == 0 else (theta[j - 1] if j <= len(theta) else 0.0)
            v += sum(a[i - 1] * psi[j - i] for i in range(1, min(j, len(a)) + 1))
            psi[j] = v
        return mean, np.sqrt(self.sigma2_ * np.cumsum(psi ** 2))


@app.cell(hide_code=True)
def _():
    checks.run(4, ARIMA=ARIMA, simulate_arma=simulate_arma)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def seasonal_naive(y, h, m=1):
    """stage 5: forecast y_{n+j} = y_{n+j-m*k} (last observed value in the same season); m=1 is the naive forecast."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    return np.array([y[n - m + (j % m)] for j in range(h)])


@app.function
def mase(actual, forecast, insample, m=1):
    """stage 5: mean |actual - forecast| / mean |insample_t - insample_{t-m}| (Hyndman & Koehler, 2006)."""
    insample = np.asarray(insample, dtype=float)
    scale = np.mean(np.abs(insample[m:] - insample[:-m]))
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float))) / scale)


@app.function
def rolling_origin(y, forecaster, *, initial, horizon, step=1):
    """stage 5: origins t = initial, initial + step, ... with t + horizon <= n; returns (origins, forecasts, actuals)."""
    y = np.asarray(y, dtype=float)
    origins = np.arange(initial, len(y) - horizon + 1, step)
    fc = np.array([np.asarray(forecaster(y[:t].copy(), horizon), dtype=float) for t in origins]).reshape(-1, horizon)
    act = np.array([y[t:t + horizon] for t in origins]).reshape(-1, horizon)
    return origins, fc, act


@app.cell(hide_code=True)
def _():
    checks.run(5, seasonal_naive=seasonal_naive, mase=mase, rolling_origin=rolling_origin)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it backtests your ARIMA
    against the seasonal-naive forecast on synthetic demand series with trend, two
    seasonalities, autocorrelated noise and level shifts.
    """)
    return


if __name__ == "__main__":
    app.run()
