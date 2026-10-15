import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from scipy import optimize

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # kalman filters · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `simulate_lgssm`: draw states and observations from a linear-Gaussian state-space model |
    | 2 | `kalman_filter`: predict, update, and the log-likelihood, with missing observations |
    | 3 | `rts_smoother`: the Rauch–Tung–Striebel backward pass |
    | 4 | `steady_state_gain`, `nis`: the limiting filter and a consistency check |
    | 5 | `fit_noise_scales`: learning Q and R by maximum likelihood |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def simulate_lgssm(F, H, Q, R, m0, P0, T, *, seed=None):
    """stage 1: x_0 ~ N(m0, P0), x_t = F x_{t-1} + w_t, y_t = H x_t + v_t; returns (x (T, n), y (T, m))."""
    F, H, Q, R = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (F, H, Q, R))
    rng = np.random.default_rng(seed)
    n, m = F.shape[0], H.shape[0]
    x = np.zeros((T, n))
    x[0] = rng.multivariate_normal(np.asarray(m0, dtype=float), np.atleast_2d(P0))
    for t in range(1, T):
        x[t] = F @ x[t - 1] + rng.multivariate_normal(np.zeros(n), Q)
    y = x @ H.T + rng.multivariate_normal(np.zeros(m), R, size=T)
    return x, y


@app.cell(hide_code=True)
def _():
    checks.run(1, simulate_lgssm=simulate_lgssm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def kalman_filter(y, F, H, Q, R, m0, P0):
    """stage 2: filtered and one-step-predicted moments plus the log-likelihood; rows of y with a NaN are skipped."""
    F, H, Q, R = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (F, H, Q, R))
    y = np.asarray(y, dtype=float).reshape(len(y), -1)
    T, n = len(y), F.shape[0]
    means, covs = np.zeros((T, n)), np.zeros((T, n, n))
    pmeans, pcovs = np.zeros((T, n)), np.zeros((T, n, n))
    m, P = np.asarray(m0, dtype=float).copy(), np.atleast_2d(np.asarray(P0, dtype=float)).copy()
    loglik = 0.0
    for t in range(T):
        if t > 0:
            m, P = F @ m, F @ P @ F.T + Q
        pmeans[t], pcovs[t] = m, P
        if not np.isnan(y[t]).any():
            v = y[t] - H @ m
            S = H @ P @ H.T + R
            K = np.linalg.solve(S, H @ P).T  # P H^T S^{-1} (S symmetric)
            m = m + K @ v
            P = P - K @ S @ K.T
            P = 0.5 * (P + P.T)
            _, logdet = np.linalg.slogdet(S)
            loglik += -0.5 * (len(v) * np.log(2 * np.pi) + logdet + v @ np.linalg.solve(S, v))
        means[t], covs[t] = m, P
    return {"means": means, "covs": covs, "pred_means": pmeans, "pred_covs": pcovs, "loglik": float(loglik)}


@app.cell(hide_code=True)
def _():
    checks.run(2, kalman_filter=kalman_filter, simulate_lgssm=simulate_lgssm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def rts_smoother(filt, F):
    """stage 3: Rauch-Tung-Striebel smoothing from kalman_filter's output; returns (means (T, n), covs (T, n, n))."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    mf, Pf, mp, Pp = filt["means"], filt["covs"], filt["pred_means"], filt["pred_covs"]
    T = len(mf)
    ms, Ps = mf.copy(), Pf.copy()
    for t in range(T - 2, -1, -1):
        G = np.linalg.solve(Pp[t + 1], F @ Pf[t]).T  # Pf F^T Pp^{-1}
        ms[t] = mf[t] + G @ (ms[t + 1] - mp[t + 1])
        Ps[t] = Pf[t] + G @ (Ps[t + 1] - Pp[t + 1]) @ G.T
        Ps[t] = 0.5 * (Ps[t] + Ps[t].T)
    return ms, Ps


@app.cell(hide_code=True)
def _():
    checks.run(3, rts_smoother=rts_smoother, kalman_filter=kalman_filter, simulate_lgssm=simulate_lgssm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def steady_state_gain(F, H, Q, R, *, tol=1e-10, max_iter=10_000):
    """stage 4: iterate the Riccati recursion to its fixed point; returns (K (n, m), P_pred (n, n)); RuntimeError if no convergence."""
    F, H, Q, R = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (F, H, Q, R))
    P = Q.copy() + np.eye(F.shape[0])
    for _ in range(max_iter):
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T
        P_new = F @ (P - K @ S @ K.T) @ F.T + Q
        P_new = 0.5 * (P_new + P_new.T)
        if np.max(np.abs(P_new - P)) <= tol * max(np.max(np.abs(P_new)), np.finfo(float).tiny):
            P = P_new
            S = H @ P @ H.T + R
            return np.linalg.solve(S, H @ P).T, P
        P = P_new
    raise RuntimeError(f"Riccati recursion did not converge in {max_iter} iterations")


@app.function
def nis(y, filt, H, R):
    """stage 4: normalized innovation squared v_t^T S_t^{-1} v_t per step, (T,); NaN where y_t is missing."""
    H, R = np.atleast_2d(np.asarray(H, dtype=float)), np.atleast_2d(np.asarray(R, dtype=float))
    y = np.asarray(y, dtype=float).reshape(len(y), -1)
    out = np.full(len(y), np.nan)
    for t in range(len(y)):
        if np.isnan(y[t]).any():
            continue
        v = y[t] - H @ filt["pred_means"][t]
        S = H @ filt["pred_covs"][t] @ H.T + R
        out[t] = v @ np.linalg.solve(S, v)
    return out


@app.cell(hide_code=True)
def _():
    checks.run(4, steady_state_gain=steady_state_gain, nis=nis, kalman_filter=kalman_filter,
               simulate_lgssm=simulate_lgssm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def fit_noise_scales(y, F, H, Q, R, m0, P0):
    """stage 5: (q, r) > 0 maximizing kalman_filter(y, F, H, q * Q, r * R, m0, P0)["loglik"]."""
    Q, R = np.atleast_2d(np.asarray(Q, dtype=float)), np.atleast_2d(np.asarray(R, dtype=float))

    def nll(v):
        return -kalman_filter(y, F, H, np.exp(v[0]) * Q, np.exp(v[1]) * R, m0, P0)["loglik"]

    res = optimize.minimize(nll, np.zeros(2), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6})
    return float(np.exp(res.x[0])), float(np.exp(res.x[1]))


@app.cell(hide_code=True)
def _():
    checks.run(5, fit_noise_scales=fit_noise_scales, kalman_filter=kalman_filter, simulate_lgssm=simulate_lgssm)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it tracks a noisy object
    in the plane with your filter and smoother, breaks the noise model on purpose, and shows
    that your Kalman filter for a random walk in noise is exponential smoothing in disguise.
    """)
    return


if __name__ == "__main__":
    app.run()
