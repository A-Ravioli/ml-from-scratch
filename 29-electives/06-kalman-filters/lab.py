import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    # your ARIMA from topic 05, for experiment 4 (electives own no mlfs module, so load it directly)
    def _arima_smoke(m):  # raises NotImplementedError if a stage you need here isn't done yet
        m.ARIMA((0, 1, 1)).fit(np.cumsum(np.random.default_rng(0).normal(size=50))).forecast(2)

    prev = notebooks.load_topic("05-classical-forecasting", key="forecasting", smoke=_arima_smoke)
    fc = prev.module

    def cv_model(dt=1.0, q=0.05, r=4.0):
        """2-D constant velocity: state (px, py, vx, vy), white-noise acceleration with intensity q, position measured."""
        F1 = np.array([[1.0, dt], [0.0, 1.0]])
        Q1 = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        F = np.kron(F1, np.eye(2))  # ordering (px, py, vx, vy)
        Q = np.kron(Q1, np.eye(2))
        H = np.hstack([np.eye(2), np.zeros((2, 2))])
        R = r * np.eye(2)
        return F, H, Q, R

    F, H, Q, R = cv_model()
    M0 = np.zeros(4)
    P0 = np.diag([25.0, 25.0, 1.0, 1.0])
    T = 200
    N_SEEDS = 20


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # kalman filters · lab

    every experiment runs **your** filter, smoother and noise fit. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about half a minute.

    the main setting is tracking an object in the plane. its state is position and velocity
    $(p_x, p_y, v_x, v_y)$, it moves with constant velocity plus random accelerations
    (white-noise acceleration with intensity $q = 0.05$ per step), and a sensor reports its
    position every step with noise standard deviation 2 ($R = 4I$). the velocity is never
    measured. all trajectories are simulated with your `simulate_lgssm`, 200 steps each.
    """)
    return


@app.cell
def _():
    prev.callout()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how much does filtering buy?

    on 20 simulated trajectories: the raw measurements, your filter and your smoother, all
    against the true path. the velocity has no measurement at all; the naive estimate is the
    difference of consecutive measurements.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the sensor's position error has RMS about 2.8 (2 per axis). what RMS position error will the filter and the "
        "smoother reach? and the velocity error, compared with differencing the measurements?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)

    def _rms(a):
        return float(np.sqrt(np.mean(np.sum(a ** 2, axis=-1))))

    _res = {k: [] for k in ["measurements", "filter", "smoother", "velocity: differenced measurements",
                            "velocity: filter", "velocity: smoother"]}
    for _s in range(N_SEEDS):
        _x, _y = ex.simulate_lgssm(F, H, Q, R, M0, P0, T, seed=_s)
        _f = ex.kalman_filter(_y, F, H, Q, R, M0, P0)
        _ms, _ = ex.rts_smoother(_f, F)
        _res["measurements"].append(_rms(_y - _x[:, :2]))
        _res["filter"].append(_rms(_f["means"][:, :2] - _x[:, :2]))
        _res["smoother"].append(_rms(_ms[:, :2] - _x[:, :2]))
        _res["velocity: differenced measurements"].append(_rms(np.diff(_y, axis=0) - _x[1:, 2:]))
        _res["velocity: filter"].append(_rms(_f["means"][:, 2:] - _x[:, 2:]))
        _res["velocity: smoother"].append(_rms(_ms[:, 2:] - _x[:, 2:]))
        if _s == 0:
            _ex = (_x, _y, _f, _ms)
    _x, _y, _f, _ms = _ex
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    _tt = np.linspace(0, 2 * np.pi, 50)
    for _ax, _lo, _hi in [(_axes[0], 0, T), (_axes[1], 100, 130)]:
        _sl = slice(_lo, _hi)
        _ax.plot(*_x[_sl, :2].T, "k-", lw=1.5, label="truth")
        _ax.plot(*_y[_sl].T, ".", color="0.6", ms=4, label="measurements")
        _ax.plot(*_f["means"][_sl, :2].T, "C0-", lw=1, label="filter")
        _ax.plot(*_ms[_sl, :2].T, "C3-", lw=1, label="smoother")
        _ax.set(aspect="equal", xlabel="x", ylabel="y")
    for _t in range(100, 130, 5):  # the filter's 95% regions, zoomed panel only
        _w, _V = np.linalg.eigh(_f["covs"][_t][:2, :2])
        _e = _V @ (np.sqrt(5.991 * _w)[:, None] * np.array([np.cos(_tt), np.sin(_tt)]))
        _axes[1].plot(_f["means"][_t, 0] + _e[0], _f["means"][_t, 1] + _e[1], "C0", lw=0.7)
    _axes[0].set_title("trajectory 0, all 200 steps")
    _axes[1].set_title("steps 100-129 (ellipses: the filter's 95% regions)")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {k} | {np.mean(v):.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f} |" for k, v in _res.items()]
    mo.vstack([_fig, mo.md("| estimate | RMS error, mean ± s.e. over 20 trajectories |\n|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the filter averages measurements, but not with equal weights: what sets
    how far back it effectively looks (think of the steady-state gain)? why does the smoother
    beat the filter by so much on velocity? differencing the measurements gives a velocity
    estimate that is *worse than not measuring at all* would suggest: what is its error
    variance, in terms of $R$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what bad noise models do

    the data stay the same (the true $q$ and $R$), but the filter is given $s_Q Q$ and
    $s_R R$ for multipliers between 0.01 and 100. left: RMS position error. right: the
    fraction of steps where the true position lies inside the filter's own 95% ellipse
    (0.95 for an honest filter). 10 trajectories per cell. the sliders show one trajectory
    with your choice of multipliers.
    """)
    return


@app.cell
def _():
    sq = mo.ui.slider(-2, 2, value=0, step=0.5, label="log10 s_Q")
    sr = mo.ui.slider(-2, 2, value=0, step=0.5, label="log10 s_R")
    p2 = ui.predict(
        "which hurts the RMS error more: R 100x too small or 100x too large? and what happens to the error and to the "
        "coverage if BOTH Q and R are 100x too large?"
    )
    mo.vstack([p2, mo.hstack([sq, sr])])
    return p2, sq, sr


@app.cell
def _(p2):
    ui.gate(p2)
    scales = 10.0 ** np.arange(-2, 2.1, 1.0)
    rmse_grid = np.zeros((len(scales), len(scales)))
    cover_grid = np.zeros_like(rmse_grid)
    _data = [ex.simulate_lgssm(F, H, Q, R, M0, P0, T, seed=100 + s) for s in range(10)]
    for _i, _a in enumerate(scales):  # rows: s_Q
        for _j, _b in enumerate(scales):  # columns: s_R
            _errs, _hits = [], []
            for _x, _y in _data:
                _f = ex.kalman_filter(_y, F, H, _a * Q, _b * R, M0, P0)
                _d = _f["means"][:, :2] - _x[:, :2]
                _errs.append(np.mean(np.sum(_d ** 2, 1)))
                _m2 = np.einsum("ti,tij,tj->t", _d, np.linalg.inv(_f["covs"][:, :2, :2]), _d)
                _hits.append(np.mean(_m2 < 5.991))  # chi-squared(2) 95% quantile
            rmse_grid[_i, _j] = np.sqrt(np.mean(_errs))
            cover_grid[_i, _j] = np.mean(_hits)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for _ax, _g, _title, _fmt in [(_axes[0], rmse_grid, "RMS position error", "{:.2f}"),
                                  (_axes[1], cover_grid, "coverage of the 95% ellipse", "{:.2f}")]:
        _norm = LogNorm() if _g is rmse_grid else None
        _im = _ax.imshow(_g, origin="lower", cmap="viridis", norm=_norm)
        for (_i, _j), _v in np.ndenumerate(_g):
            _ax.text(_j, _i, _fmt.format(_v), ha="center", va="center", color="w", fontsize=8)
        _labels = [f"{s:g}" for s in scales]
        _ax.set_xticks(range(len(scales)), _labels)
        _ax.set_yticks(range(len(scales)), _labels)
        _ax.set(xlabel="s_R (model R / true R)", ylabel="s_Q (model Q / true Q)", title=_title)
        _fig.colorbar(_im, ax=_ax, shrink=0.8)
    _fig.tight_layout()
    _fig
    return cover_grid, rmse_grid, scales


@app.cell
def _(p2, sq, sr):
    ui.gate(p2)
    _x, _y = ex.simulate_lgssm(F, H, Q, R, M0, P0, T, seed=100)
    _f = ex.kalman_filter(_y, F, H, 10 ** sq.value * Q, 10 ** sr.value * R, M0, P0)
    _nis = ex.nis(_y, _f, H, 10 ** sr.value * R)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _sd = np.sqrt(_f["covs"][:, 0, 0])
    _axes[0].plot(_f["means"][:, 0] - _x[:, 0], "C0", lw=1, label="filter error in p_x")
    _axes[0].fill_between(np.arange(T), -2 * _sd, 2 * _sd, color="C0", alpha=0.2, label="filter's own ±2 std")
    _axes[0].set(xlabel="step", ylabel="error", title=f"s_Q = {10 ** sq.value:g}, s_R = {10 ** sr.value:g}")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(_nis, lw=0.6)
    _axes[1].axhline(2.0, color="k", ls=":", label="expected mean (m = 2)")
    _axes[1].set(yscale="log", xlabel="step", ylabel="NIS", title=f"mean NIS {np.mean(_nis):.2f}")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** look along the diagonal of the left heatmap, where $Q$ and $R$ are scaled
    by the same factor. why is the error (almost) the same in every diagonal cell, and what
    does that tell you about which quantity the *estimates* depend on? (hint: write the gain
    for $sQ$, $sR$ and $sP_0$.) then look at the same cells in the right heatmap: what goes
    wrong there instead? which direction of misspecification is cheaper for the estimates,
    and which for the uncertainty? how would the NIS tell you, with no ground truth, that
    $R$ is too small?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: gaps, and learning the noise from data

    the sensor drops out for 40 steps (steps 80–119 are NaN). left: the filter's and the
    smoother's position standard deviation through the gap. right: your `fit_noise_scales`,
    given $Q$ and $R$ from a model whose guesses are 10× off in opposite directions, learns
    multipliers from one trajectory with the gap; we then rerun the experiment 2 statistics
    with the learned model.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "where in the 40-step gap is the smoother least certain: at the start, the middle, or the end? and how close to "
        "the truth (10 and 0.1, to undo the wrong guesses) will the learned multipliers be?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _x, _y = ex.simulate_lgssm(F, H, Q, R, M0, P0, T, seed=7)
    _y = _y.copy()
    _y[80:120] = np.nan
    _f = ex.kalman_filter(_y, F, H, Q, R, M0, P0)
    _ms, _Ps = ex.rts_smoother(_f, F)
    _q, _r = ex.fit_noise_scales(_y, F, H, Q / 10, R * 10, M0, P0)
    _fl = ex.kalman_filter(_y, F, H, _q * Q / 10, _r * R * 10, M0, P0)
    _d = _fl["means"][:, :2] - _x[:, :2]
    _ok = ~np.isnan(_y[:, 0])
    _m2 = np.einsum("ti,tij,tj->t", _d, np.linalg.inv(_fl["covs"][:, :2, :2]), _d)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _t = np.arange(T)
    _axes[0].plot(_t, np.sqrt(_f["covs"][:, 0, 0]), label="filter std of p_x")
    _axes[0].plot(_t, np.sqrt(_Ps[:, 0, 0]), label="smoother std of p_x")
    _axes[0].axvspan(80, 119, color="0.9", label="no measurements")
    _axes[0].set(xlabel="step", ylabel="standard deviation", title="uncertainty through the gap")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(_t, _f["means"][:, 0] - _x[:, 0], "C0", lw=1, label="filter error")
    _axes[1].plot(_t, _ms[:, 0] - _x[:, 0], "C3", lw=1, label="smoother error")
    _axes[1].fill_between(_t, -2 * np.sqrt(_Ps[:, 0, 0]), 2 * np.sqrt(_Ps[:, 0, 0]), color="C3", alpha=0.2,
                          label="smoother ±2 std")
    _axes[1].set(xlabel="step", ylabel="estimate − true p_x", ylim=(-25, 25))
    _axes[1].legend(fontsize=8)
    _axes[1].axvspan(80, 119, color="0.9")
    _fig.tight_layout()
    _peak = 80 + int(np.argmax(np.sqrt(_Ps[80:120, 0, 0])))
    mo.vstack([_fig, mo.md(
        f"smoother's uncertainty peaks at step {_peak} (the gap is 80–119). "
        f"learned multipliers: q = {_q:.2f}, r = {_r:.3f} (the truth is 10 and 0.1, so the learned model has "
        f"{_q / 10:.2f}× the true Q and {_r * 10:.2f}× the true R). with the learned model, RMS position error "
        f"{np.sqrt(np.mean(np.sum(_d[_ok] ** 2, 1))):.2f} on observed steps and coverage of the 95% ellipse "
        f"{np.mean(_m2 < 5.991):.2f} over all steps.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the filter's uncertainty grow steadily through the gap and then
    collapse, while the smoother's is a bulge? (what does the smoother know at step 100 that
    the filter doesn't?) the noise multipliers were learned without ever seeing the true
    states: what in the data identifies $q$ separately from $r$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 4: the Kalman filter you already know

    a scalar random walk observed in noise (the **local level model**): $x_t = x_{t-1} + w_t$,
    $y_t = x_t + v_t$. for three signal-to-noise ratios $q/r$ we simulate 2,000 steps, learn
    $(q, r)$ with your `fit_noise_scales`, compute your `steady_state_gain` $K$, and fit your
    topic 05 `ARIMA((0, 1, 1))` to the same observations.
    """)
    return


@app.cell
def _():
    p4 = ui.predict(
        "the steady-state filter is m_t = m_{t-1} + K (y_t - m_{t-1}). what is the MA coefficient theta of the "
        "ARIMA(0, 1, 1) fitted to the same data, in terms of K?"
    )
    p4
    return (p4,)


@app.cell
def _(p4):
    ui.gate(p4)
    _rows = []
    for _q_true in [0.05, 0.5, 5.0]:
        _x, _y = ex.simulate_lgssm([[1.0]], [[1.0]], [[_q_true]], [[1.0]], [0.0], [[10.0]], 2000, seed=3)
        _q, _r = ex.fit_noise_scales(_y, [[1.0]], [[1.0]], [[1.0]], [[1.0]], [0.0], [[10.0]])
        _K, _ = ex.steady_state_gain([[1.0]], [[1.0]], [[_q]], [[_r]])
        _Ktrue, _ = ex.steady_state_gain([[1.0]], [[1.0]], [[_q_true]], [[1.0]])
        _theta = fc.ARIMA((0, 1, 1)).fit(_y[:, 0]).theta_[0]
        _rows.append(f"| {_q_true:g} | {_q:.3f}, {_r:.3f} | {_Ktrue[0, 0]:.3f} | {_K[0, 0]:.3f} | {_theta:+.3f} | "
                     f"{-(1 - _K[0, 0]):+.3f} |")
    mo.md("| true q / r | learned (q, r) | K (true model) | K (learned) | ARIMA(0,1,1) θ̂ | −(1 − K learned) |\n"
          "|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** substitute $y_t = m_{t-1} + \epsilon_t$ (the innovation) into the
    steady-state recursion and difference it: why does an ARIMA(0, 1, 1) with
    $\theta = -(1 - K)$ come out? the steady-state filter is simple exponential smoothing with
    $\alpha = K$: what does the signal-to-noise ratio say about how much to smooth? the two
    numbers in the last two columns come from completely different fitting procedures (a
    likelihood over a state-space model, a conditional sum of squares over differences):
    why do they agree only approximately?
    """)
    return


if __name__ == "__main__":
    app.run()
