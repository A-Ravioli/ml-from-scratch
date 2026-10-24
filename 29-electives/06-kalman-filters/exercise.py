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
    # kalman filters · build

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
    raise NotImplementedError("stage1: implement simulate_lgssm")


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
    raise NotImplementedError("stage2: implement kalman_filter")


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
    raise NotImplementedError("stage3: implement rts_smoother")


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
    raise NotImplementedError("stage4: implement steady_state_gain")


@app.function
def nis(y, filt, H, R):
    """stage 4: normalized innovation squared v_t^T S_t^{-1} v_t per step, (T,); NaN where y_t is missing."""
    raise NotImplementedError("stage4: implement nis")


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
    raise NotImplementedError("stage5: implement fit_noise_scales")


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
