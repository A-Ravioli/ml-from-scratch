import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    def demand_series(seed, weeks=10):
        """a SYNTHETIC hourly demand series: trend + daily and weekly seasonality + AR(1) noise + level shifts.

        nothing here is real data. the shape imitates hourly electricity or traffic load: a daily
        cycle with two harmonics, lower weekends, a seasonal amplitude that drifts slowly, strongly
        autocorrelated noise (phi = 0.85 per hour), and a Poisson(2) number of permanent level shifts.
        """
        g = np.random.default_rng(seed)
        n = 168 * weeks
        t = np.arange(n)
        hour, day = t % 24, (t // 24) % 7
        daily = 12 * np.sin(2 * np.pi * (hour - 8) / 24) + 5 * np.sin(4 * np.pi * (hour - 3) / 24)
        weekly = np.where(day >= 5, -10.0, 0.0) + 2 * np.sin(2 * np.pi * day / 7)
        amp = 1 + 0.15 * np.sin(2 * np.pi * t / (168 * weeks) * 1.3 + g.uniform(0, 6))
        trend = 100 + 0.004 * t
        noise = signal.lfilter([1.0], [1.0, -0.85], g.normal(0, 2.0, n))
        shifts = np.zeros(n)
        for s in g.choice(np.arange(168, n), size=g.poisson(2), replace=False):
            shifts[s:] += g.normal(0, 8)
        return trend + amp * daily + weekly + noise + shifts

    N_SERIES = 10  # independent synthetic series (seeds 0..9) for every backtest
    INITIAL, HORIZON, STEP = 168 * 6, 24, 24  # six weeks of history, day-ahead forecasts made at midnight


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # classical forecasting · lab

    every experiment here runs **your** code. for each one: write down a prediction, lock it
    in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the whole lab takes well under a minute.

    **the data are synthetic**, and that is deliberate: with a simulator you know the truth
    (what the seasonality is, where the level shifts are, what the noise is), so you can tell
    *why* a method wins. the series imitate hourly demand (electricity load, web traffic):
    ten weeks per series, a daily cycle, lower weekends, AR(1) noise with $\phi = 0.85$ per
    hour, and a few permanent level shifts at random times. no real time series is cached
    for this course, and none is downloaded.
    """)
    return


@app.cell
def _():
    seed_slider = mo.ui.slider(0, N_SERIES - 1, value=0, label="series seed")
    seed_slider
    return (seed_slider,)


@app.cell
def _(seed_slider):
    _y = demand_series(seed_slider.value)
    _fig, _axes = plt.subplots(2, 1, figsize=(11, 5))
    _axes[0].plot(_y, lw=0.6)
    _axes[0].set(title=f"synthetic series {seed_slider.value}: ten weeks, hourly", xlabel="hour", ylabel="demand")
    _axes[1].plot(np.arange(168 * 2), _y[168 * 4:168 * 6], lw=1)
    for _d in range(1, 14):
        _axes[1].axvline(24 * _d, color="0.85", lw=0.5, zorder=0)
    _axes[1].set(title="two weeks (weeks 5-6): daily cycle, low weekends", xlabel="hour within the window")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: reading the memory of a series

    your `acf` and `pacf` on four series: a simulated AR(2) ($\phi = (0.6, 0.25)$), a
    simulated MA(2) ($\theta = (0.7, 0.4)$), and the synthetic demand series before and after
    a seasonal difference at lag 168 (one week). the dashed lines are $\pm 2/\sqrt n$, roughly
    where white noise would stay.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for the AR(2) and the MA(2): which plot (ACF or PACF) cuts off sharply, and after which lag? "
        "for the raw demand series, what will the ACF look like out to lag 200?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _series = {
        "AR(2), phi = (0.6, 0.25)": ex.simulate_arma(2000, (0.6, 0.25), seed=1),
        "MA(2), theta = (0.7, 0.4)": ex.simulate_arma(2000, (), (0.7, 0.4), seed=2),
        "demand, raw": demand_series(0),
        "demand, (1 - B^168) y": ex.difference(demand_series(0), 1, lag=168),
    }
    _fig, _axes = plt.subplots(4, 2, figsize=(11, 9))
    _rows = []
    for _i, (_name, _x) in enumerate(_series.items()):
        _nl = 200 if "demand" in _name else 12
        _r, _pa = ex.acf(_x, _nl), ex.pacf(_x, min(_nl, 50))
        for _ax, _v, _lab in [(_axes[_i, 0], _r, "ACF"), (_axes[_i, 1], _pa, "PACF")]:
            _ax.vlines(np.arange(1, len(_v)), 0, _v[1:], lw=1 if len(_v) > 20 else 2)
            _ax.axhline(0, color="k", lw=0.5)
            for _s in (1, -1):
                _ax.axhline(_s * 2 / np.sqrt(len(_x)), color="C3", ls="--", lw=0.8)
            _ax.set(title=f"{_name}: {_lab}", ylim=(-1, 1))
        _rows.append(f"| {_name} | " + ", ".join(f"{v:.2f}" for v in _r[1:5]) + " | "
                     + ", ".join(f"{v:.2f}" for v in _pa[1:5]) + " |")
    _axes[-1, 0].set_xlabel("lag")
    _axes[-1, 1].set_xlabel("lag")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| series | ACF lags 1–4 | PACF lags 1–4 |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the PACF of an AR(p) cut off after lag $p$ while its ACF decays
    slowly, and why is it the other way round for an MA(q)? what in the raw demand series
    makes its ACF stay large out to lag 200 with peaks every 24 lags? after the weekly
    difference, which of the two patterns (AR-like or MA-like) do you see, and what order
    would you try?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the backtest

    day-ahead forecasting: at midnight, forecast the next 24 hours. rolling origin over the
    last four weeks of each series (28 origins, six weeks of history at the first one), on
    all ten series. the score is MASE scaled by the in-sample *daily* seasonal-naive error
    (`m = 24`), so 1.0 means "as good as repeating yesterday". the contenders:

    - **naive**: repeat the last hour. **seasonal naive, m = 24**: repeat yesterday. **seasonal naive, m = 168**: repeat the same hour last week.
    - **ARIMA(2, 1, 1)**: a non-seasonal model on first differences.
    - **ARIMA(2, 0, 1) after a daily difference**, and **ARIMA(2, 0, 1) after a weekly difference** (your `seasonal_period`).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "rank the six methods by mean MASE. how much better (in %) is the best ARIMA than the best seasonal-naive "
        "forecast?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _arima(order, m=None):
        return lambda hist, h: ex.ARIMA(order, seasonal_period=m).fit(hist).forecast(h)

    methods = {
        "naive": lambda hist, h: ex.seasonal_naive(hist, h, 1),
        "seasonal naive, m = 24": lambda hist, h: ex.seasonal_naive(hist, h, 24),
        "seasonal naive, m = 168": lambda hist, h: ex.seasonal_naive(hist, h, 168),
        "ARIMA(2, 1, 1)": _arima((2, 1, 1)),
        "ARIMA(2, 0, 1), daily difference": _arima((2, 0, 1), 24),
        "ARIMA(2, 0, 1), weekly difference": _arima((2, 0, 1), 168),
    }
    scores = {k: [] for k in methods}  # per series: mean MASE over origins
    by_h = {k: [] for k in methods}  # per series: MAE by horizon
    for _s in range(N_SERIES):
        _y = demand_series(_s)
        for _name, _f in methods.items():
            _o, _fc, _act = ex.rolling_origin(_y, _f, initial=INITIAL, horizon=HORIZON, step=STEP)
            scores[_name].append(np.mean([ex.mase(a, b, _y[:t], m=24) for t, a, b in zip(_o, _act, _fc)]))
            by_h[_name].append(np.abs(_act - _fc).mean(0))
    scores = {k: np.array(v) for k, v in scores.items()}
    _fig, _ax = plt.subplots(figsize=(9, 4))
    for _name in methods:
        if _name not in ("naive", "ARIMA(2, 1, 1)"):
            _ax.plot(np.arange(1, HORIZON + 1), np.mean(by_h[_name], 0), "o-", ms=3, label=_name)
    _ax.set(xlabel="hours ahead", ylabel="mean absolute error", title="error by horizon (mean over 10 series)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {k} | {v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f} | {v.min():.2f}–{v.max():.2f} |"
             for k, v in scores.items()]
    _d = scores["ARIMA(2, 0, 1), weekly difference"] - scores["seasonal naive, m = 168"]
    _note = (f"\n\npaired over the 10 series, ARIMA (weekly difference) minus seasonal naive (m = 168): "
             f"{_d.mean():+.3f} MASE (standard error {_d.std(ddof=1) / np.sqrt(len(_d)):.3f}; "
             f"ARIMA better on {int((_d < 0).sum())} of {len(_d)} series), a relative gain of "
             f"{-_d.mean() / scores['seasonal naive, m = 168'].mean():.0%}.")
    mo.vstack([_fig, mo.md("| method | MASE, mean ± s.e. over 10 series | range |\n|---|---|---|\n"
                           + "\n".join(_rows) + _note)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does ARIMA(2, 1, 1) lose even to the naive forecast at a 24-hour
    horizon? why does the ARIMA with a *daily* difference lose to plain "repeat yesterday",
    when it contains "repeat yesterday" as a special case ($\phi = \theta = 0$)? the best
    ARIMA's advantage over "repeat last week" is concentrated at short horizons: why, and what
    happens to it by hour 24 (think about what an AR model's forecast converges to)? if you
    had to ship one of these, and had a day to do it, which would it be?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: are the intervals honest?

    your `forecast(h, return_std=True)` gives a standard deviation for every horizon. a 95%
    interval is mean ± 1.96 std. we count how often the truth lands inside, across all
    origins and series, in three settings:

    1. data simulated from the model you fit: ARIMA(1, 1, 1) series, ARIMA(1, 1, 1) fits;
    2. the demand series with the weekly-differenced AR(2) model;
    3. the demand series with the daily-differenced ARIMA(2, 0, 1) model.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "in each of the three settings, what fraction of 1-hour-ahead and 24-hour-ahead truths land inside the "
        "nominal 95% interval?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    def _coverage(series, order, m):
        hits, days = [], []
        for _y in series:
            for _t in range(INITIAL, len(_y) - HORIZON + 1, STEP):
                _mu, _sd = ex.ARIMA(order, seasonal_period=m).fit(_y[:_t]).forecast(HORIZON, return_std=True)
                hits.append(np.abs(_y[_t:_t + HORIZON] - _mu) < 1.96 * _sd)
                days.append((_t // 24) % 7)
        return np.array(hits), np.array(days)

    _sim = [np.cumsum(ex.simulate_arma(168 * 10, (0.6,), (0.3,), seed=100 + s)) for s in range(N_SERIES)]
    _dem = [demand_series(s) for s in range(N_SERIES)]
    _cases = {
        "simulated ARIMA(1,1,1), ARIMA(1,1,1) fit": _coverage(_sim, (1, 1, 1), None),
        "demand, AR(2) after weekly difference": _coverage(_dem, (2, 0, 0), 168),
        "demand, ARIMA(2,0,1) after daily difference": _coverage(_dem, (2, 0, 1), 24),
    }
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _rows = []
    for _name, (_hits, _days) in _cases.items():
        _ax.plot(np.arange(1, HORIZON + 1), _hits.mean(0), "o-", ms=3, label=_name)
        _rows.append(f"| {_name} | {_hits[:, 0].mean():.2f} | {_hits[:, 23].mean():.2f} | {_hits.mean():.2f} |")
    _ax.axhline(0.95, color="k", ls=":", lw=1)
    _ax.set(xlabel="hours ahead", ylabel="fraction inside the 95% interval", ylim=(0, 1.02))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _hits, _days = _cases["demand, ARIMA(2,0,1) after daily difference"]
    _names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _wd = " | ".join(f"{_names[d]} {_hits[_days == d, 0].mean():.2f}" for d in range(7))
    mo.vstack([_fig, mo.md("| setting | 1 h ahead | 24 h ahead | all horizons |\n|---|---|---|---|\n" + "\n".join(_rows)
                           + f"\n\ndaily-difference model, 1-hour coverage by the weekday being forecast: {_wd}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in setting 1 the intervals are right even though the parameters are
    estimated: why is parameter uncertainty a small effect here (how much data does each fit
    use)? in setting 3 the 1-hour intervals are badly overconfident, and only on certain
    weekdays: what does the daily-differenced series look like at the start of a Saturday,
    Sunday and Monday, and what does an AR model with $\phi_1 + \phi_2$ near 1 do with "the
    last few hours were 10 below yesterday"? why can't the in-sample $\hat\sigma^2$ warn you?
    """)
    return


if __name__ == "__main__":
    app.run()
