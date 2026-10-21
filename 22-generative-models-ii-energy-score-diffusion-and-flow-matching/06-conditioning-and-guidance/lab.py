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
    import torch.nn.functional as F

    from mlfs import notebooks, ui
    from mlfs.diffusion import DiffusionSchedule, TimeEmbedding, TimeMLP, ddim_sample, sample_flow  # yours, topics 03-05
    from mlfs.generative import frechet_distance  # yours, from ch 21 · topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    K = 3
    SD = 0.5
    MEANS = np.array([[np.cos(a), np.sin(a)] for a in 2 * np.pi * np.arange(K) / K])
    ITERS = 2500
    N_PER = 2000


@app.function
def make_data(n, seed):
    """n points from 3 overlapping Gaussian classes (std 0.5, centres on the unit circle) and their labels."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    return MEANS[y] + SD * rng.normal(size=(n, 2)), y


@app.function
def posterior(x):
    """the exact class posteriors p(k | x) of the clean mixture, (n, K)."""
    d2 = ((x[:, None, :] - MEANS[None]) ** 2).sum(-1)
    logp = -d2 / (2 * SD ** 2)
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(1, keepdims=True)


@app.function
def evaluate(samples_by_class):
    """per guidance setting: Bayes accuracy, mean posterior of the intended class, per-class std, distance of the class
    mean from the origin (the true one: 1.0), and the Frechet distance to the true class Gaussian (mean over classes)."""
    acc, post, std, rad, fd = [], [], [], [], []
    for k, s in samples_by_class.items():
        p = posterior(s)
        acc.append((p.argmax(1) == k).mean())
        post.append(p[:, k].mean())
        std.append(s.std(0).mean())
        rad.append(np.linalg.norm(s.mean(0)))
        fd.append(frechet_distance(MEANS[k], SD ** 2 * np.eye(2), s.mean(0), np.cov(s.T)))
    return dict(acc=np.mean(acc), post=np.mean(post), std=np.mean(std), rad=np.mean(rad), fd=np.mean(fd))


@app.function
def sample_classes(model, w, *, flow=True, schedule=None, seed=0):
    """N_PER samples per class with classifier-free guidance scale w (your cfg_model)."""
    g = ex.cfg_model(model, w, K)
    out = {}
    for k in range(K):
        y = torch.full((N_PER,), k)
        gen = torch.Generator().manual_seed(seed + k)
        if flow:
            out[k] = sample_flow(g, (N_PER, 2), n_steps=40, method="midpoint", y=y, generator=gen).numpy()
        else:
            out[k] = ddim_sample(g, schedule, (N_PER, 2), n_steps=50, y=y, generator=gen).numpy()
    return out


@app.class_definition
class NoisyClassifier(torch.nn.Module):
    """p(y | x_t, t): an MLP on x and a time embedding, for classifier guidance."""

    def __init__(self, hidden=128):
        super().__init__()
        self.temb = TimeEmbedding(32)
        self.net = torch.nn.Sequential(torch.nn.Linear(2 + 32, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.SiLU(), torch.nn.Linear(hidden, K))

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], 1))


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # conditioning & guidance · lab

    every experiment uses **your** `drop_labels` (through `fit_conditional`), `cfg_model`,
    `classifier_guided_eps` and `fit_conditional`, with topic 03's `TimeMLP`. the data are
    three **overlapping** Gaussian classes (std {SD}, centres on the unit circle), 20,000
    labelled points, synthetic, so offline runs are the same experiments. because the classes
    are known exactly we can measure fidelity (the true posterior of the intended class at
    each sample) and diversity (the samples' spread, and the Fréchet distance to the true
    class distribution). compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 1: the guidance scale

    a class-conditional flow ({ITERS} steps, $p_\text{{uncond}} = 0.2$), sampled with
    classifier-free guidance at $w \in \{{0, 1, 2, 4, 8\}}$ ({N_PER} samples per class, 40
    midpoint steps).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "as w goes from 1 to 8: will the Bayes accuracy of the samples go up or down? their spread? the distance of each "
        "class's samples from the origin (the true class means are at distance 1)? at which w is the Frechet distance to "
        "the true class distribution smallest?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    X, Y = make_data(20000, 0)
    torch.manual_seed(0)
    flow = TimeMLP(2, hidden_dim=256, n_layers=3, time_dim=64, n_classes=K)
    _t0 = time.perf_counter()
    ex.fit_conditional(flow, X, Y, K, objective="flow", p_uncond=0.2, n_iters=ITERS, batch_size=512, lr=1e-3, seed=0)
    _secs = time.perf_counter() - _t0
    _rows, _show = [], {}
    _Xr, _Yr = make_data(60000, 9)
    _true = {k: _Xr[_Yr == k][:N_PER] for k in range(K)}
    _e = evaluate(_true)
    _rows.append(f"| real data | {_e['acc']:.3f} | {_e['post']:.3f} | {_e['std']:.3f} | {_e['rad']:.3f} | {_e['fd']:.4f} |")
    for _w in (0.0, 1.0, 2.0, 4.0, 8.0):
        _s = sample_classes(flow, _w)
        _e = evaluate(_s)
        _rows.append(f"| w = {_w:g} | {_e['acc']:.3f} | {_e['post']:.3f} | {_e['std']:.3f} | {_e['rad']:.3f} | {_e['fd']:.4f} |")
        if _w in (0.0, 1.0, 4.0, 8.0):
            _show[_w] = _s
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for _ax, (_w, _s) in zip(_axes, _show.items()):
        for _k in range(K):
            _ax.scatter(*_s[_k][:600].T, s=1, alpha=0.4)
        _ax.scatter(*MEANS.T, c="k", marker="x")
        _ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title=f"w = {_w:g}")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"training {_secs:.0f} s · per class, averaged over the 3 classes\n\n| samples | Bayes accuracy | posterior of the "
        "intended class | std | distance of the class mean from 0 | Fréchet distance to the true class |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return X, Y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** guidance weights each point by $p(y \mid x)^w$. use that to explain the
    trend of every column. why is the Fréchet distance smallest near $w = 1$ while the
    accuracy keeps rising (which of the two would a human rating "does this look like class
    $k$?" track)? on paper 2 predicted an overshoot: where do you see it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how often to drop the label

    the same flow trained with $p_\text{uncond} \in \{0, 0.2, 0.5\}$, each sampled at $w = 1$
    (no guidance) and $w = 4$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with p_uncond = 0 the null label is never seen in training. what happens at w = 1? at w = 4? does dropping half "
        "the labels (0.5) hurt the conditional model?"
    )
    p2
    return (p2,)


@app.cell
def _(X, Y, p2):
    ui.gate(p2)
    _rows = []
    for _p in (0.0, 0.2, 0.5):
        torch.manual_seed(0)
        _m = TimeMLP(2, hidden_dim=256, n_layers=3, time_dim=64, n_classes=K)
        ex.fit_conditional(_m, X, Y, K, objective="flow", p_uncond=_p, n_iters=ITERS, batch_size=512, lr=1e-3, seed=0)
        for _w in (1.0, 4.0):
            _e = evaluate(sample_classes(_m, _w))
            _rows.append(f"| {_p} | {_w:g} | {_e['acc']:.3f} | {_e['std']:.3f} | {_e['rad']:.3f} | {_e['fd']:.4f} |")
    mo.md("| p_uncond | w | Bayes accuracy | std | distance of the class mean from 0 | Fréchet distance |\n"
          "|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at $w = 1$ the null label is never used, so why can $p_\text{uncond}$
    still matter there? at $w = 4$ with $p_\text{uncond} = 0$, what is the "unconditional"
    prediction, and what does extrapolating away from it do? what does the 0.5 row cost, and
    why is 0.1–0.2 the usual choice?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: classifier guidance against classifier-free guidance

    noise-prediction models this time (`objective="ddpm"`, a 200-step linear schedule), all
    sampled with 50 DDIM steps. classifier guidance: an *unconditional* model (the same
    network with every label dropped) plus a separately trained noisy classifier
    $p_\phi(y \mid x_t, t)$, combined with your `classifier_guided_eps`. classifier-free: a
    conditional model ($p_\text{uncond} = 0.2$) with your `cfg_model`. both at $w = 1$ and
    $w = 4$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "at the same w, which kind of guidance will give the higher Bayes accuracy, and which the smaller Frechet "
        "distance?"
    )
    p3
    return (p3,)


@app.cell
def _(X, Y, p3):
    ui.gate(p3)
    _s = DiffusionSchedule(200, beta_start=5e-4, beta_end=0.1)  # the default linear schedule, compressed to 200 steps
    torch.manual_seed(0)
    _cond = TimeMLP(2, hidden_dim=256, n_layers=3, time_dim=64, n_classes=K)
    ex.fit_conditional(_cond, X, Y, K, objective="ddpm", schedule=_s, p_uncond=0.2, n_iters=ITERS, batch_size=512,
                       lr=1e-3, seed=0)
    torch.manual_seed(0)
    _unc = TimeMLP(2, hidden_dim=256, n_layers=3, time_dim=64, n_classes=K)
    ex.fit_conditional(_unc, X, Y, K, objective="ddpm", schedule=_s, p_uncond=1.0, n_iters=ITERS, batch_size=512,
                       lr=1e-3, seed=0)
    torch.manual_seed(0)
    _clf = NoisyClassifier()
    _opt = torch.optim.Adam(_clf.parameters(), lr=1e-3)
    _Xt, _Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y)
    _g = torch.Generator().manual_seed(0)
    for _ in range(ITERS):
        _i = torch.randint(len(_Xt), (512,), generator=_g)
        _ts = torch.randint(200, (512,), generator=_g)
        _xt = _s.q_sample(_Xt[_i], _ts, torch.randn(512, 2, generator=_g))
        _loss = F.cross_entropy(_clf(_xt, _ts.float() / 200), _Yt[_i])
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    _rows = []
    for _w in (1.0, 4.0):
        _cg = ex.classifier_guided_eps(_unc, _clf, _s, _w)  # _unc(x, t) uses the null label
        _samp = {}
        for _k in range(K):
            _samp[_k] = ddim_sample(_cg, _s, (N_PER, 2), n_steps=50, y=torch.full((N_PER,), _k),
                                    generator=torch.Generator().manual_seed(_k)).numpy()
        _e = evaluate(_samp)
        _rows.append(f"| classifier guidance | {_w:g} | {_e['acc']:.3f} | {_e['std']:.3f} | {_e['rad']:.3f} | {_e['fd']:.4f} |")
        _e = evaluate(sample_classes(_cond, _w, flow=False, schedule=_s))
        _rows.append(f"| classifier-free | {_w:g} | {_e['acc']:.3f} | {_e['std']:.3f} | {_e['rad']:.3f} | {_e['fd']:.4f} |")
    mo.md("| guidance | w | Bayes accuracy | std | distance of the class mean from 0 | Fréchet distance |\n"
          "|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with an exact classifier and $w = 1$, classifier guidance would be
    exactly the conditional model (on paper 1); how close does the learned noisy classifier
    get? which errors does each method amplify when $w > 1$: a separately trained
    classifier's, or the difference between two outputs of one network? which would you
    rather deploy, and what would change if the classifier had been trained only on clean
    data?
    """)
    return


if __name__ == "__main__":
    app.run()
