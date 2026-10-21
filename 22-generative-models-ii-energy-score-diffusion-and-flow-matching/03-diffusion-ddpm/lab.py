import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.generative import mode_coverage  # yours, from ch 21 · topic 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    STD = 0.05
    ANG = 2 * np.pi * np.arange(8) / 8
    CENTERS = 2.0 * np.c_[np.cos(ANG), np.sin(ANG)]


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # diffusion (DDPM) · lab

    every experiment uses **your** `DiffusionSchedule`, `TimeMLP`, `fit_ddpm`, `ddpm_loss`
    and `ddpm_sample`. experiment 1 uses one MNIST digit ({'the offline synthetic stand-in'
    if OFFLINE else 'from the cached dataset'}); experiments 2 and 3 use synthetic 2-D data.
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what the schedule does to an image

    the same digit and the same noise $\epsilon$, pushed through `q_sample` at several
    fractions of $T = 1000$ steps, for the linear and the cosine schedule; below, the signal
    fraction $\bar\alpha_t$ and the log signal-to-noise ratio of both.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with the default linear schedule, at which step (out of 1,000) does alpha_bar fall below 0.01, i.e. the "
        "digit's amplitude below 10% of the noise's? and with the cosine schedule?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _d = datasets.mnist()
    _x0 = torch.tensor(_d["X_train"][0], dtype=torch.float32)[None, None] / 127.5 - 1
    _eps = torch.randn(_x0.shape, generator=torch.Generator().manual_seed(0))
    _scheds = {"linear": ex.DiffusionSchedule(1000), "cosine": ex.DiffusionSchedule(1000, schedule="cosine")}
    _fracs = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.999]
    _fig, _axes = plt.subplots(2, len(_fracs) + 1, figsize=(13, 3.6),
                               gridspec_kw={"width_ratios": [1] * len(_fracs) + [2.2]})
    _rows = []
    for _r, (_name, _s) in enumerate(_scheds.items()):
        for _c, _f in enumerate(_fracs):
            _t = torch.tensor([int(_f * 1000)])
            _axes[_r, _c].imshow(_s.q_sample(_x0, _t, _eps)[0, 0], cmap="gray", vmin=-2, vmax=2)
            _axes[_r, _c].set(xticks=[], yticks=[], title=f"t={int(_f * 1000)}" if _r == 0 else "")
        _axes[_r, 0].set_ylabel(_name)
        _ab = _s.alpha_bars.double()
        _axes[_r, -1].plot(_ab.numpy(), label="alpha_bar")
        _axes[_r, -1].plot(np.log10((_ab / (1 - _ab)).numpy()) / 10, label="log10 SNR / 10")
        _axes[_r, -1].axhline(0, c="k", lw=0.5)
        _axes[_r, -1].legend(fontsize=7)
        _half = int((_ab < 0.5).nonzero()[0])
        _one = int((_ab < 0.01).nonzero()[0])
        _rows.append(f"| {_name} | {_half} | {_one} | {_ab[-1].item():.2e} |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| schedule | first step with alpha_bar < 0.5 (SNR < 1) | first step with alpha_bar < 0.01 | "
                           "alpha_bar at the last step |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** by which step can you no longer recognize the digit under each schedule?
    with the linear schedule, what fraction of the 1,000 steps (and so of sampling time) is
    spent on images that are almost pure noise? why might a model care about having more
    steps in the middle range of SNR?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how hard is it to guess the noise?

    20,000 points on the unit circle, with a radial thickness of 0.002, so the data are
    essentially one-dimensional curves in the plane. a `TimeMLP` (3 layers of 256) is trained
    with `fit_ddpm` on the default schedule for 4,000 steps. then we measure its loss at
    individual steps $t$, on 4,000 data points with fresh noise, and compare with the noise
    standard deviation $\sqrt{1 - \bar\alpha_t}$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "after training, the loss (MSE of the noise prediction, averaged over 2 coordinates) at a small t, where the "
        "noise is tiny: close to 0, close to 0.5, or close to 1? and at a large t?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _th = _rng.uniform(0, 2 * np.pi, 20000)
    _X = np.c_[np.cos(_th), np.sin(_th)] * (1 + 0.002 * _rng.normal(size=20000))[:, None]
    _s = ex.DiffusionSchedule(1000)
    torch.manual_seed(0)
    _m = ex.TimeMLP(2, hidden_dim=256, n_layers=3, time_dim=64)
    _t0 = time.perf_counter()
    _losses = ex.fit_ddpm(_m, _s, _X, n_iters=4000, batch_size=512, lr=1e-3, seed=0)
    _secs = time.perf_counter() - _t0
    _m.eval()
    _x0 = torch.tensor(_X[:4000], dtype=torch.float32)
    _steps = [0, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 800, 999]
    _mse = []
    for _k in _steps:
        _t = torch.full((4000,), _k)
        _e = torch.randn(_x0.shape, generator=torch.Generator().manual_seed(100 + _k))
        with torch.no_grad():
            _mse.append(((_m(_s.q_sample(_x0, _t, _e), _t.float() / 1000) - _e) ** 2).mean().item())
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _xs = [k + 1 for k in _steps]
    _ax.plot(_xs, _mse, "o-", label="your model's loss at step t")
    _ax.plot(_xs, [math.sqrt(1 - _s.alpha_bars[k].item()) for k in _steps], "--", label="noise std sqrt(1 - ab)")
    _ax.axhline(0.5, c="gray", lw=0.6)
    _ax.set(xscale="log", xlabel="step t + 1", ylabel="MSE per coordinate", title="data on a circle (thickness 0.002)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"training {_secs:.0f} s, final training loss {np.mean(_losses[-200:]):.3f}\n\n"
                           "| step | " + " | ".join(str(k) for k in _steps) + " |\n|" + "---|" * (len(_steps) + 1) + "\n"
                           "| loss | " + " | ".join(f"{v:.3f}" for v in _mse) + " |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** decompose the noise at a point near the circle into a component along
    the circle and one across it. once the noise is bigger than the data's thickness, which
    component can the network infer from $x_t$, and which can it not (where on the circle did
    $x_0$ come from)? what does the plateau's height tell you about the data, and what would
    it be for points on a 2-D surface in 3-D? why does the loss go to 0 at large $t$, and does
    that mean those steps are easy for sampling?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how many steps?

    the 8 gaussians (std 0.05), 10,000 points; one `TimeMLP` per number of steps
    $T \in \{5, 20, 100, 1000\}$, each with the cosine schedule (so every $T$ ends at pure
    noise) and the same training budget (3,000 steps of batch 512). we draw 5,000 samples
    from each with your `ddpm_sample` and count samples within 3 std of a mode.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "T = 5 makes sampling 200 times cheaper than T = 1000. what fraction of its samples will land within 3 std of a "
        "mode, compared with T = 1000 (the true mixture: 0.99)?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X = datasets.eight_gaussians(10000, std=STD, rng=0)
    _rows, _samples = [], {}
    for _T in (5, 20, 100, 1000):
        _s = ex.DiffusionSchedule(_T, schedule="cosine")
        torch.manual_seed(0)
        _m = ex.TimeMLP(2, hidden_dim=256, n_layers=3, time_dim=64)
        _t0 = time.perf_counter()
        ex.fit_ddpm(_m, _s, _X, n_iters=3000, batch_size=512, lr=1e-3, seed=0)
        _tt = time.perf_counter() - _t0
        _t0 = time.perf_counter()
        _xs = ex.ddpm_sample(_m, _s, (5000, 2), generator=torch.Generator().manual_seed(1)).numpy()
        _ts = time.perf_counter() - _t0
        _n, _q, _counts = mode_coverage(_xs, CENTERS, STD)
        _samples[_T] = _xs
        _rows.append(f"| {_T} | {_T} | {_n} | {_q:.3f} | {_counts.min()} – {_counts.max()} | {_tt:.0f} s | {_ts:.2f} s |")
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for _ax, (_T, _xs) in zip(_axes, _samples.items()):
        _ax.scatter(*_xs.T, s=1, alpha=0.3)
        _ax.set(xlim=(-2.8, 2.8), ylim=(-2.8, 2.8), aspect="equal", title=f"T = {_T}")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| T | network calls per sample | modes covered | within 3 std | samples per mode | training | "
                           "sampling 5,000 |\n|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each reverse step assumes $q(x_{t-1} \mid x_t)$ is close to a Gaussian.
    why is that a good approximation when $\beta_t$ is small and a poor one when a single step
    has to take noise halfway to the data? where do the $T = 5$ samples go wrong in the plot?
    going from 100 to 1,000 steps buys less than going from 5 to 20, at ten times the cost:
    what would you rather do with a fixed budget of 20 network calls (topic 04)?
    """)
    return


if __name__ == "__main__":
    app.run()
