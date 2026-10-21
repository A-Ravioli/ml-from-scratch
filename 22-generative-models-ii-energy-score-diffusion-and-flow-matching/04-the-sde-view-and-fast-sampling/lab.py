import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.diffusion import DiffusionSchedule, TimeUNet, ddpm_sample, fit_ddpm  # yours, from topic 03
    from mlfs.generative import cached_feature_net, fid_from_features, mode_coverage  # yours, from ch 21
    from mlfs.torchnn import get_device  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    STD = 0.05
    ANG = 2 * np.pi * np.arange(8) / 8
    CENTERS = torch.tensor(2.0 * np.c_[np.cos(ANG), np.sin(ANG)], dtype=torch.float32)
    SCHED = DiffusionSchedule(1000)
    RUN_ALL = os.environ.get("MLFS_LAB_RUN_ALL", "") == "1"  # press every run button (used to record the notes)


@app.class_definition
class MixtureOracle(torch.nn.Module):
    """the exact noise predictor for the 8 gaussians at any continuous t (optionally scaled, to simulate an error)."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x, t, y=None):
        a = ex.alpha_bar_at(SCHED, t)[:, None, None]
        v = a * STD ** 2 + 1 - a
        d = x[:, None, :] - a.sqrt() * CENTERS[None]
        r = torch.softmax(-0.5 * (d ** 2).sum(-1) / v[:, :, 0], dim=1)
        return self.scale * ((1 - a).sqrt() * d / v * r[:, :, None]).sum(1)


@app.function
def quality(x):
    """fraction of samples within 3 std of a mode (the true mixture: 0.99)."""
    return mode_coverage(x.numpy(), CENTERS.numpy(), STD)[1]


@app.function
def run_samplers(model, shape, nfe, seed=0):
    """each sampler at (about) the same number of network calls; returns {name: samples}."""
    g = lambda: torch.Generator().manual_seed(seed)
    out = {"DDIM": ex.ddim_sample(model, SCHED, shape, n_steps=nfe, generator=g()),
           "PF-ODE Euler": ex.probability_flow_sample(model, SCHED, shape, n_steps=nfe - 1, method="euler", generator=g()),
           "reverse SDE": ex.reverse_sde_sample(model, SCHED, shape, n_steps=nfe - 1, generator=g())}
    if nfe >= 9:
        out["PF-ODE RK4"] = ex.probability_flow_sample(model, SCHED, shape, n_steps=(nfe - 1) // 4, method="rk4",
                                                       generator=g())
    return out


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the SDE view & fast sampling · lab

    every experiment uses **your** `ddim_sample`, `probability_flow_sample` and
    `reverse_sde_sample` (and, in experiment 2, your topic 03 `TimeUNet`, `fit_ddpm` and
    `ddpm_sample`). experiments 1 and 3 use the 8 gaussians (std 0.05) with the *exact* noise
    predictor, computed in closed form at every continuous time, so the only errors are the
    samplers' own. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: quality per network call, with a perfect network

    4,000 samples from each sampler at budgets of 5 to 200 network function evaluations
    (NFE): DDIM ($\eta = 0$) with NFE steps; the PF-ODE with Euler (NFE − 1 steps, plus the
    denoising call) and RK4 ((NFE − 1)/4 steps); the reverse SDE with NFE − 1
    Euler–Maruyama steps; and ancestral DDPM, which always takes 1,000.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with a perfect network and a budget of 20 network calls, rank DDIM, PF-ODE Euler, PF-ODE RK4 and the reverse "
        "SDE by sample quality. how many calls does the reverse SDE need to catch up?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _oracle = MixtureOracle()
    _budgets = [5, 10, 20, 50, 200]
    _res = {}
    for _nfe in _budgets:
        for _name, _x in run_samplers(_oracle, (4000, 2), _nfe).items():
            _res.setdefault(_name, {})[_nfe] = quality(_x)
    _t0 = time.perf_counter()
    _ddpm = ddpm_sample(_oracle, SCHED, (4000, 2), generator=torch.Generator().manual_seed(0))
    _q_ddpm, _t_ddpm = quality(_ddpm), time.perf_counter() - _t0
    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    for _name, _d in _res.items():
        _ax.plot(list(_d), list(_d.values()), "o-", label=_name)
    _ax.plot([1000], [_q_ddpm], "k*", ms=10, label="DDPM (1,000 calls)")
    _ax.set(xscale="log", xlabel="network calls per sample", ylabel="within 3 std of a mode", ylim=(0, 1.02))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _hdr = "| sampler | " + " | ".join(f"{b} NFE" for b in _budgets) + " |\n|" + "---|" * (len(_budgets) + 1) + "\n"
    _rows = [f"| {_n} | " + " | ".join(f"{_d.get(b, float('nan')):.3f}" for b in _budgets) + " |" for _n, _d in _res.items()]
    mo.vstack([_fig, mo.md(_hdr + "\n".join(_rows) + f"\n\nancestral DDPM, 1,000 calls: **{_q_ddpm:.3f}** ({_t_ddpm:.1f} s)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the network is exact, so every gap here is discretization error. why
    does the deterministic DDIM converge so much faster than the reverse SDE (what does
    each step of the SDE add, and how many steps does it take to average out)? why does RK4,
    a fourth-order method, lose at a fixed number of calls (how many steps does it get, and
    how smooth is the drift near $t = 0$ for modes of width 0.05)? which sampler would you
    ship at 20 calls?
    """)
    return


@app.function
def steps_vs_fid(iters, n, plan, batch_size=128):
    """train a small TimeUNet with your fit_ddpm on MNIST (or the offline stand-in), sample it with each (sampler,
    network calls) in `plan`, and return (figure, markdown table)."""
    d = datasets.mnist()
    dev = torch.device("cpu") if OFFLINE else get_device()
    X = torch.tensor(d["X_train"][:, None] / 255.0, dtype=torch.float32)
    featnet = cached_feature_net(X, d["y_train"], name="mnist", epochs=2, seed=0, device=dev).to("cpu")
    real = featnet.embed(torch.tensor(d["X_test"][:2000, None] / 255.0, dtype=torch.float32))
    torch.manual_seed(0)
    unet = TimeUNet(1, base_channels=16, channel_mults=(1, 2, 2)).to(dev)
    t0 = time.perf_counter()
    fit_ddpm(unet, SCHED, (X * 2 - 1).to(dev), n_iters=iters, batch_size=batch_size, lr=2e-3, seed=0)
    train_s = time.perf_counter() - t0
    rows, grids = [], {}
    for name, nfe in plan:
        g = torch.Generator().manual_seed(1)
        t0 = time.perf_counter()
        shape = (n, 1, 28, 28)
        if name == "DDIM":
            x = ex.ddim_sample(unet, SCHED, shape, n_steps=nfe, generator=g, device=dev)
        elif name == "PF-ODE Euler":
            x = ex.probability_flow_sample(unet, SCHED, shape, n_steps=nfe - 1, method="euler", generator=g, device=dev)
        elif name == "reverse SDE":
            x = ex.reverse_sde_sample(unet, SCHED, shape, n_steps=nfe - 1, generator=g, device=dev)
        else:
            x = ddpm_sample(unet, SCHED, shape, generator=g, device=dev)
        secs = time.perf_counter() - t0
        img = ((x.cpu() + 1) / 2).clamp(0, 1)
        grids[(name, nfe)] = img[:8]
        rows.append(f"| {name} | {nfe} | {fid_from_features(real, featnet.embed(img)):.1f} | {secs:.1f} s |")
    ref = fid_from_features(real, featnet.embed(X[:n]))
    fig, axes = plt.subplots(len(plan), 1, figsize=(7, 0.95 * len(plan)))
    for ax, ((name, nfe), img) in zip(axes, grids.items()):
        ax.imshow(img[:, 0].permute(1, 0, 2).reshape(28, -1), cmap="gray")
        ax.set_axis_off()
        ax.set_title(f"{name}, {nfe} calls", fontsize=8)
    fig.tight_layout()
    table = (f"device {dev} · training {iters} steps in {train_s:.0f} s · {n} samples per row · FID of {n} real training "
             f"digits: **{ref:.1f}** (the floor at this sample size)\n\n| sampler | network calls | FID | sampling time |\n"
             "|---|---|---|---|\n" + "\n".join(rows))
    return fig, table


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 2: steps against FID on digits

    a small `TimeUNet` (base width 16, two downsamplings) trained by your `fit_ddpm` on
    {'the 2,000-image offline stand-in, briefly (a smoke test, not a result)' if OFFLINE else
    'the 60,000 MNIST training digits'} with the default 1,000-step schedule, then sampled
    with each method at several budgets. FID against test digits in the feature space of
    chapter 21's `FeatureNet` (course-internal: not comparable with published FIDs, which use
    an Inception network). the default run (1,000 training steps, 500 samples per row, the
    fast samplers) takes 3–4 minutes on a laptop GPU; a button below runs the longer version.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the trained model sampled with 20-step DDIM versus the full 1,000-step ancestral sampler: how much worse (or "
        "better) will the FID be?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _plan = [("DDIM", 10), ("DDIM", 20), ("DDIM", 50), ("PF-ODE Euler", 20), ("PF-ODE Euler", 50), ("reverse SDE", 50)]
    _t0 = time.perf_counter()
    _fig, _table = steps_vs_fid(40, 64, _plan, batch_size=32) if OFFLINE else steps_vs_fid(1000, 500, _plan)
    mo.vstack([_fig, mo.md(_table + f"\n\n{time.perf_counter() - _t0:.0f} s in all")])
    return


@app.cell
def _():
    long_button = mo.ui.run_button(label="train for 4,000 steps and add the 200-step SDE and the 1,000-step DDPM "
                                         "(15-20 minutes on a laptop GPU)")
    long_button
    return (long_button,)


@app.cell
def _(long_button):
    if (long_button.value or RUN_ALL) and not OFFLINE:
        _plan = [("DDIM", 10), ("DDIM", 20), ("DDIM", 50), ("PF-ODE Euler", 20), ("PF-ODE Euler", 50),
                 ("reverse SDE", 50), ("reverse SDE", 200), ("DDPM", 1000)]
        _fig, _table = steps_vs_fid(4000, 1000, _plan)
        _out = mo.vstack([_fig, mo.md(_table)])
    else:
        _out = mo.md(
            "reference run (`mps`, 4,000 training steps, 1,000 samples per row; real-digit floor 8.8):\n\n"
            "| sampler | network calls | FID | sampling time |\n|---|---|---|---|\n"
            "| DDIM | 10 | 97.6 | 1.2 s |\n| DDIM | 20 | 80.8 | 3.1 s |\n| DDIM | 50 | 74.4 | 9.4 s |\n"
            "| PF-ODE Euler | 20 | 101.9 | 4.1 s |\n| PF-ODE Euler | 50 | 80.8 | 10.6 s |\n"
            "| reverse SDE | 50 | 91.1 | 11.8 s |\n| reverse SDE | 200 | 95.2 | 46.1 s |\n"
            "| DDPM | 1000 | 83.2 | 221 s |\n\npress the button to run it yourself.")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the same network gives very different FIDs depending only on how you
    integrate it. in the default run, which sampler is best at 50 calls, and does the ranking
    of experiment 1 survive when the network is imperfect? in the longer run's table, which
    samplers reach the 1,000-call FID with 50 calls or fewer, and did four times the training
    change the ranking? why does FID at $n$ samples have a floor even for real digits
    (chapter 21 · topic 01)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: when the network is wrong

    the exact predictor again, but scaled by 0.9: a systematic 10% underestimate of the noise
    (and so of the score), the kind of smooth error a network makes. 4,000 samples per
    sampler at 50, 200 and 1,000 calls, plus DDIM with $\eta = 1$ (fresh noise at every step).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with a 10% too-small noise prediction, will more sampling steps fix the deterministic samplers? will they fix "
        "the stochastic ones?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _bad = MixtureOracle(scale=0.9)
    _rows, _show = [], {}
    for _nfe in (50, 200, 1000):
        _g = lambda: torch.Generator().manual_seed(2)
        _out = {"DDIM (eta = 0)": ex.ddim_sample(_bad, SCHED, (4000, 2), n_steps=_nfe, generator=_g()),
                "DDIM (eta = 1)": ex.ddim_sample(_bad, SCHED, (4000, 2), n_steps=_nfe, eta=1.0, generator=_g()),
                "PF-ODE Euler": ex.probability_flow_sample(_bad, SCHED, (4000, 2), n_steps=_nfe - 1, method="euler",
                                                           generator=_g()),
                "reverse SDE": ex.reverse_sde_sample(_bad, SCHED, (4000, 2), n_steps=_nfe - 1, generator=_g())}
        _rows.append(f"| {_nfe} | " + " | ".join(f"{quality(_x):.3f}" for _x in _out.values()) + " |")
        if _nfe == 1000:
            _show = _out
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for _ax, (_name, _x) in zip(_axes, _show.items()):
        _ax.scatter(*_x.numpy().T, s=1, alpha=0.3)
        _ax.set(xlim=(-2.8, 2.8), ylim=(-2.8, 2.8), aspect="equal", title=f"{_name}, 1,000 calls")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("within 3 std of a mode, noise predictor scaled by 0.9\n\n"
                           "| network calls | " + " | ".join(_show) + " |\n|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a too-small $\epsilon$ is a too-small score. what does the deterministic
    ODE do with a score that pulls too weakly, and why can't more steps help (what is it
    converging *to*)? the stochastic samplers use the same wrong score: what in their update
    corrects the error (compare the drift coefficient on the score, and think of each
    stochastic step as a little Langevin correction at the current noise level)? what does
    this suggest about choosing between deterministic and stochastic samplers in practice?
    """)
    return


if __name__ == "__main__":
    app.run()
