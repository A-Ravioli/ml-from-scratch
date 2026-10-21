import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    from torch import nn

    from mlfs import notebooks, ui
    from mlfs.datasets import eight_gaussians, make_moons
    from mlfs.diffusion import TimeMLP, fit_flow_matching, sample_flow, straightness  # yours, from ch 22
    from mlfs.generative import critic_loss, gradient_penalty  # yours, from ch 21

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1); it loads your topic 15 and 16
    # code itself (the reference stands in for an unfinished topic; the callout below says when)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(1)  # small models: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # OT in machine learning · lab

    every experiment here runs **your** code: your couplings, Sinkhorn loss and domain
    adaptation from this topic (on top of your topic 15 and 16 solvers), your ch 21 WGAN-GP
    pieces and your ch 22 flow matching. for each experiment: write down a prediction, lock it
    in, run it, then explain what happened. experiment 3 is the block's **project**. the lab
    takes about 100 seconds on one CPU core. all data are synthetic (2-D), so offline runs
    are the same experiments.

    distances are measured *exactly* with the Hungarian algorithm (SciPy's
    `linear_sum_assignment`, unlocked by your topic 15 solver) between equal-size samples.
    remember topic 15's experiment 3: two samples of the *same* distribution are not at
    distance 0, so every table reports that floor.
    """)
    return


@app.cell
def _():
    # warns if your topic 15 or 16 notebook isn't finished and the reference is standing in for it
    mo.vstack([c for c in (ex.ot15_load.callout(), ex.ot16_load.callout()) if c is not None])
    return


@app.cell
def _():
    def exact_w(A, B, p):
        """exact W_p^p between two equal-size uniform samples (Hungarian algorithm)."""
        C = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1)) ** p
        r, c = linear_sum_assignment(C)
        return float(C[r, c].mean())

    return (exact_w,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: WGAN revisited

    a WGAN critic estimates $W_1$ through its dual, $\sup_{f\ \text{1-Lip}}\mathbb E_\mu f -
    \mathbb E_\nu f$, learned on minibatches; you can now compute transport *exactly*. "real"
    is 1,000 points from the 8 Gaussians; three "generators" are 1,000 points each:

    - **far**: the 8 Gaussians rotated by 22.5° (halfway between modes) and shrunk to 80%;
    - **near**: rotated by 5° and shrunk to 95%;
    - **same**: a fresh sample of the real distribution.

    per pair: the exact $W_1$ between the two 1,000-point samples; a WGAN-GP critic (an MLP
    $2\to128\to128\to1$, your ch 21 `critic_loss` and `gradient_penalty` with weight 10,
    1,500 Adam steps on batches of 256, two seeds) evaluated on all 2,000 points; and the
    average **minibatch** exact $W_1$ between batches of 256. then the same for the squared
    cost: the exact $W_2^2$ and the minibatch $W_2^2$, and below, your debiased
    `sinkhorn_loss` on the same minibatches.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for two samples of the SAME distribution: what do the critic and the minibatch exact W1 (batches of 256) "
        "report? and your debiased minibatch Sinkhorn divergence: does debiasing bring it to 0?"
    )
    p1
    return (p1,)


@app.cell
def _(exact_w, p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    real = eight_gaussians(1000, rng=1)

    def rotate(X, deg, scale):
        th = np.deg2rad(deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return scale * X @ R.T

    fakes = {"far": rotate(eight_gaussians(1000, rng=2), 22.5, 0.8),
             "near": rotate(eight_gaussians(1000, rng=2), 5.0, 0.95),
             "same": eight_gaussians(1000, rng=2)}

    def critic_estimate(A, B, seed, steps=1500):
        torch.manual_seed(seed)
        critic = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        opt = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.5, 0.9))
        g = torch.Generator().manual_seed(seed)
        At, Bt = torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)
        for _ in range(steps):
            xr, xf = At[torch.randint(len(At), (256,), generator=g)], Bt[torch.randint(len(Bt), (256,), generator=g)]
            loss = critic_loss(critic(xr).squeeze(-1), critic(xf).squeeze(-1)) \
                + 10 * gradient_penalty(critic, xr, xf, generator=g)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return float(critic(At).mean() - critic(Bt).mean())

    _g = np.random.default_rng(0)
    batches = [(_g.choice(1000, 256, replace=False), _g.choice(1000, 256, replace=False)) for _ in range(6)]
    _rows = []
    for _name, _B in fakes.items():
        _crit = [critic_estimate(real, _B, s) for s in (0, 1)]
        _mb1 = [exact_w(real[i], _B[j], 1) for i, j in batches]
        _mb2 = [exact_w(real[i], _B[j], 2) for i, j in batches]
        _rows.append(f"| {_name} | {exact_w(real, _B, 1):.3f} | {_crit[0]:.3f}, {_crit[1]:.3f} | "
                     f"{np.mean(_mb1):.3f} ± {np.std(_mb1):.3f} | {exact_w(real, _B, 2):.3f} | "
                     f"{np.mean(_mb2):.3f} ± {np.std(_mb2):.3f} |")
    mo.md("| generator | exact W1 (1,000 pts) | WGAN-GP critic (2 seeds) | minibatch W1 (256) | exact W2² (1,000) | "
          "minibatch W2² (256) |\n|---|---|---|---|---|---|\n" + "\n".join(_rows)
          + f"\n\nminibatch columns: mean ± sd over 6 batch pairs. took {time.perf_counter() - _t0:.0f} s.")
    return batches, fakes, real


@app.cell
def _():
    eps1 = mo.ui.dropdown(["0.1", "0.3", "1.0", "3.0"], value="1.0",
                          label="eps for your minibatch Sinkhorn divergence (0.1 and 0.3 take about 40 s each)")
    eps1
    return (eps1,)


@app.cell
def _(batches, eps1, fakes, real):
    _t0 = time.perf_counter()
    _eps = float(eps1.value)
    _rows = []
    for _name, _B in fakes.items():
        _sk = [float(ex.sinkhorn_loss(torch.tensor(real[i]), torch.tensor(_B[j]), _eps)) for i, j in batches]
        _rows.append(f"| {_name} | {np.mean(_sk):.3f} ± {np.std(_sk):.3f} |")
    mo.md(f"**your debiased `sinkhorn_loss` on the same 6 minibatch pairs, eps = {_eps:g}**\n\n"
          "| generator | S_eps (256 vs 256) |\n|---|---|\n" + "\n".join(_rows)
          + f"\n\ntook {time.perf_counter() - _t0:.0f} s.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the critic is a learned approximation of the dual. how close does it get to
    the exact $W_1$, and why does it fall short for the near pair? why is the minibatch exact
    $W_1$ far from 0 for the "same" pair (topic 15's experiment 3, at $n = 256$), and what
    does that bias do to a generator trained to minimize it (Fatras et al., 2020)? now the
    Sinkhorn divergence: debiasing removes the *entropic* bias $\mathrm{OT}_\varepsilon(\mu,
    \mu) > 0$; does it remove the *sampling* bias between two minibatches? switch
    $\varepsilon$: at large $\varepsilon$, $S_\varepsilon$ behaves like an MMD, whose sampling
    error shrinks like $n^{-1/2}$ in any dimension (Genevay et al., 2019); what does it lose in
    exchange? Genevay et al. (2018) train generators with this loss and no critic at all.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: domain adaptation

    a labelled **source** domain (300 points of two moons) and an unlabelled **target**
    domain: 200 fresh two-moons points rotated by an angle about the moons' centre and shifted
    by $(0.3, 0.2)$. a 5-nearest-neighbour classifier is fit on the source and scored on the
    target's (hidden) labels: as is, after mapping the source onto the target with your
    `ot_domain_adaptation` (exact, and entropic with $\varepsilon = 0.05$), and for reference
    when trained on labelled target data (the "oracle"). accuracy, mean ± sd over 5 seeds.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the target is rotated by 20, 40, 60 and 90 degrees. at which angles does mapping the source onto the target "
        "with optimal transport help the classifier, and could it ever hurt?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _t0 = time.perf_counter()

    def knn_predict(Xtr, ytr, X, k=5):
        d = ((X[:, None] - Xtr[None]) ** 2).sum(-1)
        nn_idx = np.argsort(d, axis=1)[:, :k]
        return (ytr[nn_idx].mean(1) > 0.5).astype(int)

    def domains(deg, seed):
        Xs, ys = make_moons(300, noise=0.08, rng=seed)
        Xt, yt = make_moons(200, noise=0.08, rng=seed + 100)
        th = np.deg2rad(deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        centre = np.array([0.5, 0.25])
        return Xs, ys, (Xt - centre) @ R.T + centre + np.array([0.3, 0.2]), yt

    da = {}
    for _deg in (20, 40, 60, 90):
        for _seed in range(5):
            _Xs, _ys, _Xt, _yt = domains(_deg, _seed)
            _Xo, _yo = domains(_deg, _seed + 50)[2:]  # labelled target data from other seeds, for the oracle
            _acc = {"source only": knn_predict(_Xs, _ys, _Xt),
                    "OT, exact": knn_predict(ex.ot_domain_adaptation(_Xs, _Xt), _ys, _Xt),
                    "OT, eps = 0.05": knn_predict(ex.ot_domain_adaptation(_Xs, _Xt, eps=0.05), _ys, _Xt),
                    "oracle (target labels)": knn_predict(_Xo, _yo, _Xt)}
            for _k, _pred in _acc.items():
                da.setdefault((_k, _deg), []).append(float((_pred == _yt).mean()))
    _methods = ["source only", "OT, exact", "OT, eps = 0.05", "oracle (target labels)"]
    _rows = [f"| {m} | " + " | ".join(f"{np.mean(da[m, d]):.3f} ± {np.std(da[m, d]):.3f}" for d in (20, 40, 60, 90)) + " |"
             for m in _methods]
    _Xs, _ys, _Xt, _yt = domains(40, 0)
    _Xm = ex.ot_domain_adaptation(_Xs, _Xt)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].scatter(*_Xs.T, c=_ys, cmap="coolwarm", s=10, label="source (labelled)")
    _axes[0].scatter(*_Xt.T, c="gray", s=10, alpha=0.5, label="target (unlabelled)")
    _axes[0].set(title="before: 40° rotation", aspect="equal")
    _axes[0].legend(fontsize=8)
    _axes[1].scatter(*_Xm.T, c=_ys, cmap="coolwarm", s=10, label="source, mapped by exact OT")
    _axes[1].scatter(*_Xt.T, c="gray", s=10, alpha=0.5)
    _axes[1].set(title="after", aspect="equal")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| method | 20° | 40° | 60° | 90° |\n|---|---|---|---|---|\n" + "\n".join(_rows)
                           + f"\n\ntook {time.perf_counter() - _t0:.0f} s.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** optimal transport never sees a label: it moves the source cloud onto the
    target cloud as cheaply as possible, and the labels ride along. when does "as cheaply as
    possible" put each moon onto the right moon, and when does it swap pieces of them (look at
    the large angles, and think about which matching a *rotation* needs versus which one a
    cheap transport finds)? what could fix it (Courty et al., 2017, add a class-based
    regularizer; others align with a few target labels)? and why does entropic blur help or
    hurt here?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3 (project): OT couplings straighten flow-matching paths

    your ch 22 flow matching trains a velocity field on pairs $(x_0, x_1)$: noise and data. with
    the **independent** coupling, $x_0$ and $x_1$ are paired at random; with a **minibatch OT**
    coupling (Tong et al., 2024; Pooladian et al., 2023), each training batch is re-paired by
    your `minibatch_ot_coupling`, plugged into `fit_flow_matching(..., coupling=...)`. we vary
    the size $k$ of the OT problems: the batch of 256 is split into blocks of $k$ = 8, 32 or 128
    that are each solved exactly ($k = 256$ costs 2–3× more training time for little extra
    straightness).

    data: 20,000 points from the 8 Gaussians; held-out: 1,000. model: your ch 22 `TimeMLP`
    (hidden 128, 3 layers, time embedding 32); 1,500 Adam steps at lr $2\times10^{-3}$, then
    500 at $4\times10^{-4}$, batch 256, three training seeds per coupling (the same three seeds
    for every coupling, so results pair up). we measure:

    - **pairing cost** of the coupling on a fresh batch (your `pairing_cost`; independent pairs
      cost $\mathbb E\|x_0 - x_1\|^2$);
    - **straightness** of the learned flow (your ch 22 `straightness`, 64 steps, 1,000 noise
      samples; 0 = perfectly straight);
    - **sample quality vs Euler steps**: the exact $W_2$ between 1,000 samples and the 1,000
      held-out points, for 1, 2, 4, 8 and 64 Euler steps (the same noise for every model). the
      **floor** is the $W_2$ between two independent held-out samples of 1,000.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with one Euler step, how much better (in W2) is the OT-coupled flow than the independent one? with 64 steps? "
        "and does the block size k matter more for straightness or for quality?"
    )
    p3
    return (p3,)


@app.cell
def _(exact_w, p3):
    ui.gate(p3)
    _t0 = time.perf_counter()
    Xtr = eight_gaussians(20000, rng=0).astype(np.float32)
    Xte = eight_gaussians(1000, rng=1)
    floor = np.sqrt(exact_w(Xte, eight_gaussians(1000, rng=2), 2))

    def blocked(k):
        def coupling(x0, x1):
            return torch.cat([ex.minibatch_ot_coupling(x0[i:i + k], x1[i:i + k]) for i in range(0, len(x0), k)])
        return coupling

    couplings = {"independent": None, "OT, k = 8": blocked(8), "OT, k = 32": blocked(32), "OT, k = 128": blocked(128)}
    steps_grid = [1, 2, 4, 8, 64]
    project = {}  # (coupling, seed) -> dict of measurements
    models = {}
    for _name, _cp in couplings.items():
        _gb = torch.Generator().manual_seed(123)
        _x1 = torch.tensor(Xtr[torch.randint(len(Xtr), (256,), generator=_gb).numpy()])
        _x0 = torch.randn(256, 2, generator=_gb)
        _pc = ex.pairing_cost(_x0 if _cp is None else _cp(_x0, _x1), _x1)
        for _seed in (0, 1, 2):
            torch.manual_seed(_seed)
            _m = TimeMLP(2, hidden_dim=128, n_layers=3, time_dim=32)
            _t = time.perf_counter()
            fit_flow_matching(_m, Xtr, coupling=_cp, n_iters=1500, batch_size=256, lr=2e-3, seed=_seed)
            _loss = fit_flow_matching(_m, Xtr, coupling=_cp, n_iters=500, batch_size=256, lr=4e-4, seed=_seed + 1000)
            _train_s = time.perf_counter() - _t
            _z = torch.randn(1000, 2, generator=torch.Generator().manual_seed(500 + _seed))
            _w = {k: np.sqrt(exact_w(sample_flow(_m, (1000, 2), n_steps=k, x0=_z).numpy(), Xte, 2)) for k in steps_grid}
            project[_name, _seed] = dict(pairing=_pc, straight=straightness(_m, _z, n_steps=64), w2=_w,
                                         loss=float(np.mean(_loss[-100:])), train_s=_train_s)
            models[_name, _seed] = _m
    _seeds = (0, 1, 2)
    _rows = []
    for _name in couplings:
        _r = [project[_name, s] for s in _seeds]
        _cells = [f"{_r[0]['pairing']:.2f}", ", ".join(f"{x['straight']:.3f}" for x in _r)]
        _cells += [", ".join(f"{x['w2'][k]:.2f}" for x in _r) for k in steps_grid]
        _cells.append(f"{np.mean([x['train_s'] for x in _r]):.1f}")
        _rows.append(f"| {_name} | " + " | ".join(_cells) + " |")
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for _i, _name in enumerate(couplings):
        _m = np.array([[project[_name, s]["w2"][k] for k in steps_grid] for s in _seeds])
        _axes[0].plot(steps_grid, _m.mean(0), "o-", color=f"C{_i}", label=_name)
        _axes[0].fill_between(steps_grid, _m.min(0), _m.max(0), color=f"C{_i}", alpha=0.2)
    _axes[0].axhline(floor, color="k", ls="--", lw=0.8, label="floor: two held-out samples")
    _axes[0].set(xscale="log", yscale="log", xlabel="Euler steps", ylabel="exact W2 to held-out data",
                 title="quality vs steps (mean, band = min–max over 3 seeds)")
    _axes[0].minorticks_off()
    _axes[0].set_xticks(steps_grid, [str(k) for k in steps_grid])
    _axes[0].set_yticks([0.2, 0.3, 0.5, 1.0, 2.0], ["0.2", "0.3", "0.5", "1", "2"])
    _axes[0].legend(fontsize=8)
    for _ax, _name in zip(_axes[1:], ["independent", "OT, k = 128"]):
        _m = models[_name, 0]
        _z = torch.randn(40, 2, generator=torch.Generator().manual_seed(7))
        _xs = [_z]
        with torch.no_grad():
            for _k in range(32):
                _xs.append(_xs[-1] + _m(_xs[-1], torch.full((40,), _k / 32)) / 32)
        _traj = torch.stack(_xs).numpy()
        _ax.scatter(*Xte[:400].T, s=4, c="0.8")
        _ax.plot(_traj[:, :, 0], _traj[:, :, 1], color="C0", lw=0.8)
        _ax.scatter(*_traj[0].T, s=8, c="k", zorder=3)
        _ax.set(title=f"{_name}: 32-step paths (seed 0)", aspect="equal", xlim=(-3.2, 3.2), ylim=(-3.2, 3.2))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| coupling | pairing cost | straightness (seeds 0, 1, 2) | W2, 1 step | W2, 2 steps | W2, 4 steps | "
        "W2, 8 steps | W2, 64 steps | train s |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n\nfloor (two held-out samples of 1,000): W2 = {floor:.3f}. per-seed values, seeds 0, 1, 2 in order. "
        f"took {time.perf_counter() - _t0:.0f} s.")])
    return couplings, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with independent pairs, why does one Euler step land everything near the
    origin (ch 22: what is the learned velocity at $t = 0$)? why do OT pairs make the
    *training lines* stop crossing, and why does that make the learned flow straight? the
    pairing cost falls steeply with $k$ while the 64-step quality barely moves: why can't the
    coupling change *what* distribution the flow generates, only *how* it gets there? read
    the per-seed columns: which differences between couplings are larger than the spread
    between seeds, and which aren't? how far above the floor is the best model, and what
    would you need to claim that one coupling gives better *final* samples than another?
    """)
    return


@app.cell
def _(couplings, project):
    _seeds = (0, 1, 2)
    _lines = ["| comparison (paired by seed) | 1 step | 2 steps | 64 steps | straightness |", "|---|---|---|---|---|"]
    for _name in list(couplings)[1:]:
        _d = {k: [project[_name, s]["w2"][k] - project["independent", s]["w2"][k] for s in _seeds] for k in (1, 2, 64)}
        _ds = [project[_name, s]["straight"] / project["independent", s]["straight"] for s in _seeds]
        _lines.append(f"| {_name} − independent | " + " | ".join(
            f"{np.mean(_d[k]):+.2f} [{min(_d[k]):+.2f}, {max(_d[k]):+.2f}]" for k in (1, 2, 64))
            + f" | ×{np.mean(_ds):.3f} [{min(_ds):.3f}, {max(_ds):.3f}] |")
    mo.md("**paired differences** in W2 (negative = better than independent) and straightness ratios, mean "
          "[min, max] over the 3 seeds. these per-seed numbers are what the chapter capstone's portfolio entry "
          "uses.\n\n" + "\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
