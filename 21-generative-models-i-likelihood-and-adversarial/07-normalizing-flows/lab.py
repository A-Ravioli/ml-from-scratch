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
    from mlfs.generative import mode_coverage  # yours, from topic 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    torch.set_num_threads(4)
    STD = 0.05
    ANG = 2 * np.pi * np.arange(8) / 8
    CENTERS = 2.0 * np.c_[np.cos(ANG), np.sin(ANG)]
    STEPS = 600 if OFFLINE else 2000
    X_GAUSS = datasets.eight_gaussians(5000, std=STD, rng=0)
    X_GAUSS_TEST = datasets.eight_gaussians(5000, std=STD, rng=1)
    X_MOONS = datasets.make_moons(5000, noise=0.05, rng=0)[0]
    X_MOONS_TEST = datasets.make_moons(5000, noise=0.05, rng=1)[0]


@app.function
def true_log_prob_8g(x):
    """the exact log-density of the 8-gaussian mixture (equal weights, isotropic std STD)."""
    d2 = ((x[:, None, :] - CENTERS[None, :, :]) ** 2).sum(-1)
    comp = -d2 / (2 * STD ** 2) - np.log(2 * np.pi * STD ** 2)
    m = comp.max(1, keepdims=True)
    return (m + np.log(np.exp(comp - m).mean(1, keepdims=True)))[:, 0]


@app.function
def train(X, n_layers=8, seed=0, steps=STEPS):
    """your RealNVP with n_layers couplings, trained with your fit_flow; returns (flow, seconds)."""
    torch.manual_seed(seed)
    flow = ex.RealNVP(2, n_layers=n_layers, hidden_dim=128)
    t0 = time.perf_counter()
    ex.fit_flow(flow, X, n_steps=steps, batch_size=256, lr=1e-3, seed=seed)
    return flow.eval(), time.perf_counter() - t0


@app.function
def density_grid(flow, lim=3.0, n=200):
    """exp(log_prob) on an n x n grid over [-lim, lim]^2, and the numerical integral of the density."""
    g = torch.linspace(-lim, lim, n)
    xx = torch.cartesian_prod(g, g)
    with torch.no_grad():
        p = torch.exp(flow.log_prob(xx)).view(n, n).T.numpy()  # rows = y
    h = (g[1] - g[0]).item()
    return p, float(p.sum() * h * h)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # normalizing flows · lab

    every experiment trains **your** `RealNVP` with your `fit_flow` ({STEPS} steps of batch
    256, Adam $10^{{-3}}$) on 2-D toy densities: the 8 gaussians (std {STD}) and two moons,
    5,000 training and 5,000 test points each (synthetic, so offline runs are the same
    experiment, only shorter). for each experiment: write down a prediction, lock it in,
    run it, then explain what happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: exact likelihoods

    a flow computes $\log p(x)$ exactly. we check it two ways: the density integrates to 1
    (a sum over a fine grid), and, for the 8 gaussians, whose true density is known, we
    compare the flow's test NLL with the true one, $\mathbb{E}[-\log p_\text{data}(x)]$:
    the gap is exactly $\operatorname{KL}(p_\text{data} \,\|\, p_\text{flow})$ (in
    expectation).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the true test NLL of the 8 gaussians is about -1.08 nats. how close will an 8-layer RealNVP get after "
        "training (0.01 nats? 0.1? 1?)"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    flow8, _secs = train(X_GAUSS)
    _flowm, _secs_m = train(X_MOONS)
    _rows = []
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for _ax, (_name, _f, _Xte) in zip(_axes, (("8 gaussians", flow8, X_GAUSS_TEST), ("two moons", _flowm, X_MOONS_TEST))):
        _p, _mass = density_grid(_f)
        _ax.imshow(np.log(_p + 1e-6), extent=(-3, 3, -3, 3), origin="lower", cmap="magma", vmin=-8)
        _ax.set(title=f"{_name}: log density", aspect="equal")
        with torch.no_grad():
            _nll = -_f.log_prob(torch.tensor(_Xte, dtype=torch.float32)).mean().item()
        _true = -true_log_prob_8g(_Xte).mean() if _name == "8 gaussians" else float("nan")
        _tr = "—" if np.isnan(_true) else f"{_true:.3f}"
        _gap = "—" if np.isnan(_true) else f"{_nll - _true:.3f}"
        _rows.append(f"| {_name} | {_mass:.4f} | {_nll:.3f} | {_tr} | {_gap} |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"training: {_secs:.0f} s and {_secs_m:.0f} s\n\n| data | ∫ p(x) dx (grid) | flow test NLL | "
                           "true test NLL | gap = KL (nats) |\n|---|---|---|---|---|\n" + "\n".join(_rows))])
    return (flow8,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the grid integral come out at 1 without the training ever
    enforcing it (what in the change-of-variables formula guarantees it)? the gap to the
    true NLL is a KL divergence: where in the plot does the flow put density that the
    data doesn't have? why can't a likelihood-trained flow be "too sharp" the way a
    collapsed GAN is?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: can a flow separate the modes?

    a flow is a continuous, invertible map from a single Gaussian blob. the 8 gaussians
    are 8 separate blobs. we draw 5,000 samples from the 8-layer flow and score them with
    topic 05's `mode_coverage`: how many modes, and what fraction of the samples lies
    within 3 std of a mode (for the true mixture, about 98.9%; 5,000 samples give 98.7% below).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "how many of the 8 modes will the flow cover? what fraction of its samples will fall *between* the modes "
        "(more than 3 std from any mode)? the true mixture puts 1.1% there."
    )
    p2
    return (p2,)


@app.cell
def _(flow8, p2):
    ui.gate(p2)
    _s = flow8.sample(5000, generator=torch.Generator().manual_seed(0)).numpy()
    _n, _q, _counts = mode_coverage(_s, CENTERS, STD)
    _true = datasets.eight_gaussians(5000, std=STD, rng=5)
    _nt, _qt, _ = mode_coverage(_true, CENTERS, STD)
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 4))
    _axes[0].scatter(*_s.T, s=1, alpha=0.4)
    _axes[0].scatter(*CENTERS.T, c="red", marker="x")
    _axes[0].set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title="5,000 samples from your flow")
    with torch.no_grad():
        _z = flow8(torch.tensor(X_GAUSS[:2000], dtype=torch.float32))[0].numpy()
    _lab = np.argmin(((X_GAUSS[:2000, None] - CENTERS[None]) ** 2).sum(-1), 1)
    _axes[1].scatter(*_z.T, c=_lab, cmap="tab10", s=2)
    _axes[1].set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title="data mapped to latent space, by mode")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"| | modes covered | within 3 std of a mode | between modes |\n|---|---|---|---|\n"
        f"| your flow | {_n} | {_q:.3f} | **{1 - _q:.3f}** |\n| true mixture | {_nt} | {_qt:.3f} | {1 - _qt:.3f} |\n\n"
        f"samples per mode (within 3 std): {', '.join(str(int(c)) for c in _counts)}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** maximum likelihood punishes a missing mode infinitely ($-\log 0$), so
    the flow covers everything; why does that come with mass in between? the right panel
    shows the data pushed through the flow into latent space: what must the map do to the
    Gaussian's single connected blob to produce 8 disconnected islands, and why can a
    continuous invertible map only approximate that, leaving thin bridges? compare with
    topic 05's GAN on the same kind of data: which failure would you rather have?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: depth

    the same flow with 2, 4, 8 and 12 coupling layers: test NLL, the fraction of samples
    between modes, and the training time.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "going from 8 to 12 coupling layers costs 50% more. how much does it improve the NLL and the "
        "between-modes fraction? and going from 2 to 4?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rows = []
    _true = -true_log_prob_8g(X_GAUSS_TEST).mean()
    for _L in (2, 4, 8, 12):
        _f, _secs = train(X_GAUSS, n_layers=_L)
        with torch.no_grad():
            _nll = -_f.log_prob(torch.tensor(X_GAUSS_TEST, dtype=torch.float32)).mean().item()
        _s = _f.sample(5000, generator=torch.Generator().manual_seed(0)).numpy()
        _n, _q, _ = mode_coverage(_s, CENTERS, STD)
        _rows.append(f"| {_L} | {sum(p.numel() for p in _f.parameters()):,} | {_nll:.3f} | {_nll - _true:.3f} | "
                     f"{_n} | {1 - _q:.3f} | {_secs:.0f} s |")
    mo.md(f"true test NLL {_true:.3f}\n\n| layers | parameters | test NLL | gap (nats) | modes covered | between modes | "
          "training time |\n|---|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each coupling layer transforms half the coordinates conditioned on the
    other half; why do you need several, with alternating masks, before every coordinate
    can depend on every other? where do the returns diminish, and what limits a flow's
    ability to concentrate mass on 8 islands even with many layers (think about the
    Jacobian needed to squeeze a connected region into pieces)?
    """)
    return


if __name__ == "__main__":
    app.run()
