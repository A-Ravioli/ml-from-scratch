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
    # classical forecasting · build

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
    raise NotImplementedError("stage1: implement difference")


@app.function
def acf(x, nlags):
    """stage 1: sample autocorrelations r_0..r_nlags (r_0 = 1), with the 1/n (biased) autocovariance."""
    raise NotImplementedError("stage1: implement acf")


@app.function
def pacf(x, nlags):
    """stage 1: partial autocorrelations by Durbin-Levinson on acf(x); pacf[0] = 1."""
    raise NotImplementedError("stage1: implement pacf")


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
    raise NotImplementedError("stage2: implement simulate_arma")


@app.function
def fit_ar(x, p):
    """stage 2: least squares of x_t on (1, x_{t-1}, ..., x_{t-p}), t >= p; returns (c, phi (p,), sigma2)."""
    raise NotImplementedError("stage2: implement fit_ar")


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
    raise NotImplementedError("stage3: implement arma_residuals")


@app.function
def fit_arma(x, p, q):
    """stage 3: ARMA(p, q) by conditional sum of squares (scipy.optimize); returns (c, phi, theta, sigma2)."""
    raise NotImplementedError("stage3: implement fit_arma")


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

    def fit(self, y):
        """stage 4: difference y, fit ARMA(p, q) by CSS; sets c_, phi_, theta_, sigma2_; returns self."""
        raise NotImplementedError("stage4: implement ARIMA.fit")

    def forecast(self, h, *, return_std=False):
        """stage 4: h-step mean forecast of y after the fitted data (and std from psi weights if return_std)."""
        raise NotImplementedError("stage4: implement ARIMA.forecast")


@app.cell(hide_code=True)
def _():
    checks.run(4, ARIMA=ARIMA, simulate_arma=simulate_arma, arma_residuals=arma_residuals, fit_ar=fit_ar)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def seasonal_naive(y, h, m=1):
    """stage 5: forecast y_{n+j} = y_{n+j-m*k} (last observed value in the same season); m=1 is the naive forecast."""
    raise NotImplementedError("stage5: implement seasonal_naive")


@app.function
def mase(actual, forecast, insample, m=1):
    """stage 5: mean |actual - forecast| / mean |insample_t - insample_{t-m}| (Hyndman & Koehler, 2006)."""
    raise NotImplementedError("stage5: implement mase")


@app.function
def rolling_origin(y, forecaster, *, initial, horizon, step=1):
    """stage 5: origins t = initial, initial + step, ... with t + horizon <= n; returns (origins, forecasts, actuals)."""
    raise NotImplementedError("stage5: implement rolling_origin")


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
