import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch

    from mlfs import checks, lesson
    from mlfs.ode import odeint  # yours, from ch 20 · topic 06


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the SDE view & fast sampling · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. topic 03's DDPM needs one
    network call per training step to draw a sample. here you'll reinterpret the same
    trained noise predictor as the score of a continuous-time diffusion, and sample it
    four ways: a deterministic ODE (with your chapter 20 solver), DDIM's large deterministic
    jumps, DDIM with some noise back in, and the reverse-time SDE.

    | stage | you build |
    |---|---|
    | 1 | `alpha_bar_at`, `beta_at`, `eps_to_score`, `pf_ode_drift`: the continuous-time view of a discrete schedule |
    | 2 | `ddim_timesteps`, `ddim_sample`: DDPM's network, 20 steps instead of 1,000 → `mlfs.diffusion` |
    | 3 | `probability_flow_sample`: the probability-flow ODE with `mlfs.ode.odeint` → `mlfs.diffusion` |
    | 4 | `reverse_sde_sample`: Euler–Maruyama on the reverse-time SDE |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def alpha_bar_at(schedule, t):
    """stage 1: continuous-time alpha_bar(t), t a tensor of any shape: log alpha_bars interpolated linearly between the
    grid points t_k = k / n_steps (k = 0..n_steps-1), t clamped to [0, (n_steps - 1) / n_steps]; same shape as t."""
    raise NotImplementedError("stage1: implement alpha_bar_at")


@app.function
def beta_at(schedule, t):
    """stage 1: the continuous-time noise rate beta(t) = -d log alpha_bar / dt of alpha_bar_at: on the segment
    [k / n, (k + 1) / n) it is n_steps * (log ab_k - log ab_{k+1}); t clamped like alpha_bar_at, and the last segment
    is used at t = (n_steps - 1) / n_steps; same shape as t."""
    raise NotImplementedError("stage1: implement beta_at")


@app.function
def eps_to_score(eps, alpha_bar):
    """stage 1: the score of the noisy marginal from a noise prediction: -eps / sqrt(1 - alpha_bar); alpha_bar is a
    float or a (B,) tensor broadcast over the non-batch dims of eps."""
    raise NotImplementedError("stage1: implement eps_to_score")


@app.function
def pf_ode_drift(model, schedule, x, t, y=None):
    """stage 1: dx/dt of the probability-flow ODE of the VP-SDE, -beta(t)/2 * (x + score(x, t)), with t (B,) and
    score = eps_to_score(model(x, t, y), alpha_bar_at(schedule, t))."""
    raise NotImplementedError("stage1: implement pf_ode_drift")


@app.cell(hide_code=True)
def _():
    checks.run(1, alpha_bar_at=alpha_bar_at, beta_at=beta_at, eps_to_score=eps_to_score, pf_ode_drift=pf_ode_drift)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def ddim_timesteps(n_train, n_steps):
    """stage 2: n_steps evenly spaced training steps, round(linspace(0, n_train - 1, n_steps)) as int64, increasing;
    a single step is [n_train - 1] (start from pure noise, jump straight to x0)."""
    raise NotImplementedError("stage2: implement ddim_timesteps")


@app.function
def ddim_sample(model, schedule, shape, *, n_steps=50, eta=0.0, y=None, generator=None, device="cpu"):
    """stage 2: Song, Meng & Ermon (2021). x = torch.randn(shape, generator); over taus = ddim_timesteps from the last
    to the first: eps = model(x, full(tau / n), y), x0 = (x - sqrt(1 - ab) eps) / sqrt(ab), ab_prev = alpha_bars of the
    previous tau (1 after the first), sigma = eta sqrt((1 - ab_prev) / (1 - ab)) sqrt(1 - ab / ab_prev),
    x = sqrt(ab_prev) x0 + sqrt(1 - ab_prev - sigma^2) eps + sigma z, with z = torch.randn(shape, generator) drawn only
    when sigma > 0. no gradients; an nn.Module is evaluated in eval mode (restored afterwards)."""
    raise NotImplementedError("stage2: implement ddim_sample")


@app.cell(hide_code=True)
def _():
    checks.run(2, ddim_timesteps=ddim_timesteps, ddim_sample=ddim_sample)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def probability_flow_sample(model, schedule, shape, *, n_steps=50, method="rk4", y=None, generator=None, device="cpu"):
    """stage 3: x = torch.randn(shape, generator); integrate pf_ode_drift from t = (n - 1) / n to t = 0 with
    mlfs.ode.odeint (method, n_steps equal steps); then one denoising step at t = 0:
    x0 = (x - sqrt(1 - ab_0) model(x, 0, y)) / sqrt(ab_0). no gradients; eval mode for an nn.Module (restored)."""
    raise NotImplementedError("stage3: implement probability_flow_sample")


@app.cell(hide_code=True)
def _():
    checks.run(3, probability_flow_sample=probability_flow_sample, pf_ode_drift=pf_ode_drift)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def reverse_sde_sample(model, schedule, shape, *, n_steps=100, y=None, generator=None, device="cpu"):
    """stage 4: Euler–Maruyama on the reverse-time VP-SDE from t = (n - 1)/n to 0 in n_steps steps of size
    h = t_max / n_steps: x <- x + h beta(t) (x/2 + score) + sqrt(beta(t) h) z (z = torch.randn(shape, generator), drawn
    after x_T), with beta and score as in pf_ode_drift, then the same final denoising step as the ODE sampler."""
    raise NotImplementedError("stage4: implement reverse_sde_sample")


@app.cell(hide_code=True)
def _():
    checks.run(4, reverse_sde_sample=reverse_sde_sample)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.diffusion import ddim_sample, probability_flow_sample` gives you fast
    samplers for any noise-prediction network. topic 05 builds a model whose ODE is
    straight by design.
    """)
    return


if __name__ == "__main__":
    app.run()
