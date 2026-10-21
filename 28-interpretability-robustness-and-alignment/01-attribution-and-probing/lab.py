import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import datasets, notebooks, ui
    from mlfs.cnn import LeNet  # yours, from ch 16 · 03
    from mlfs.torchnn import fit, make_loader  # yours, from ch 15 · 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OFFLINE = datasets.is_offline()
    _d = datasets.mnist()  # a synthetic stand-in offline
    N_TRAIN = 2_000 if OFFLINE else 12_000
    X_TRAIN = torch.tensor(_d["X_train"][:N_TRAIN], dtype=torch.float32)[:, None] / 255.0
    Y_TRAIN = torch.tensor(_d["y_train"][:N_TRAIN])
    X_TEST = torch.tensor(_d["X_test"][:2_000], dtype=torch.float32)[:, None] / 255.0
    Y_TEST = torch.tensor(_d["y_test"][:2_000])


@app.function
def grad_times_input(model, x, target):
    """gradient x input: the signed version of saliency, multiplied by the pixel values."""
    x = x.detach().clone().requires_grad_(True)
    model.eval()
    out = model(x)
    tgt = torch.as_tensor(target).long().expand(out.shape[0])
    (g,) = torch.autograd.grad(out.gather(1, tgt[:, None]).sum(), x)
    return g * x.detach()


@app.function
def layer_features(model, x, names):
    """{name: (n, features) ndarray} of module outputs (2x2 average-pooled when spatial), via forward hooks."""
    feats, hooks = {}, []
    modules = dict(model.named_modules())
    for name in names:
        def hook(_m, _inp, out, name=name):
            out = F.avg_pool2d(out, 4) if out.dim() == 4 else out
            feats.setdefault(name, []).append(out.flatten(1).detach().numpy())
        hooks.append(modules[name].register_forward_hook(hook))
    try:
        model.eval()
        with torch.no_grad():
            for i in range(0, len(x), 500):
                model(x[i:i + 500])
    finally:
        for h in hooks:
            h.remove()
    return {k: np.concatenate(v) for k, v in feats.items()}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # attribution & probing · lab

    every experiment here runs **your** `saliency`, `integrated_gradients`, `probe_selectivity`
    and sanity-check tools on a LeNet (your chapter 16 model) trained for two epochs on 12,000
    MNIST digits. for each experiment: write down a prediction, lock it in, run it, then
    explain what happened. compare your explanations with `solutions/lab_notes.md`
    afterwards. under a minute on a laptop CPU.
    """)
    return


@app.cell
def _():
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    net = LeNet()
    fit(net, F.cross_entropy, torch.optim.Adam(net.parameters(), lr=2e-3),
        make_loader(X_TRAIN.numpy(), Y_TRAIN.numpy(), batch_size=128, seed=0), epochs=2)
    net.eval()
    with torch.no_grad():
        _acc = (net(X_TEST).argmax(1) == Y_TEST).float().mean().item()
    mo.md(f"LeNet trained in {time.perf_counter() - _t0:.0f} s: test accuracy **{_acc:.3f}** on 2,000 test digits.")
    return (net,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: do the attributions add up?

    integrated gradients promises **completeness**: the attributions of one image sum to
    $f(x) - f(\text{baseline})$, the change in the class logit between a black image and the
    digit. that's exact for the integral, and only approximate for your $n$-step trapezoid
    rule. below: the relative completeness error, averaged over 32 test digits, as `n_steps`
    grows; then three attribution maps for four digits, each for the predicted class. use the
    knobs to change IG's number of steps and its baseline.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("how many steps does the trapezoid rule need before the attributions add up to within 1% of "
                    "f(x) - f(0)? and will saliency and IG highlight the same pixels?")
    p1
    return (p1,)


@app.cell
def _(net, p1):
    ui.gate(p1)
    _x = X_TEST[:32]
    with torch.no_grad():
        _pred = net(_x).argmax(1)
        _gap = (net(_x) - net(torch.zeros_like(_x))).gather(1, _pred[:, None])[:, 0]
    _steps = [1, 2, 4, 8, 16, 32, 64, 128]
    completeness = []
    for _n in _steps:
        _attr = ex.integrated_gradients(net, _x, _pred, n_steps=_n)
        completeness.append(((_attr.flatten(1).sum(1) - _gap).abs() / _gap.abs()).mean().item())
    _fig, _a0 = plt.subplots(figsize=(5, 3.2))
    _a0.loglog(_steps, completeness, "o-")
    _a0.axhline(0.01, color="gray", ls=":", label="1%")
    _a0.set(xlabel="n_steps", ylabel="mean |sum(attr) - gap| / |gap|", title="completeness error")
    _a0.legend()
    mo.vstack([_fig, mo.md("| n_steps | " + " | ".join(map(str, _steps)) + " |\n|---|" + "---|" * len(_steps)
                           + "\n| mean relative error | " + " | ".join(f"{e:.4f}" for e in completeness) + " |")])
    return


@app.cell
def _():
    ig_steps = mo.ui.slider(1, 128, value=64, step=1, label="IG n_steps for the maps")
    ig_baseline = mo.ui.dropdown(["black", "mean digit", "blurred digit"], value="black", label="IG baseline")
    mo.hstack([ig_steps, ig_baseline])
    return ig_baseline, ig_steps


@app.cell
def _(ig_baseline, ig_steps, net, p1):
    ui.gate(p1)
    _x4 = X_TEST[:4]
    with torch.no_grad():
        _p4 = net(_x4).argmax(1)
    _base = {"black": torch.zeros_like(_x4), "mean digit": X_TRAIN.mean(0, keepdim=True).expand_as(_x4),
             "blurred digit": F.avg_pool2d(_x4, 7, stride=1, padding=3, count_include_pad=False)}[ig_baseline.value]
    _ig = ex.integrated_gradients(net, _x4, _p4, baseline=_base, n_steps=ig_steps.value)
    with torch.no_grad():
        _gap = (net(_x4) - net(_base)).gather(1, _p4[:, None])[:, 0]
    _err = ((_ig.flatten(1).sum(1) - _gap).abs() / _gap.abs()).mean().item()
    _maps = [("input", _x4), ("saliency", ex.saliency(net, _x4, _p4)), ("grad x input", grad_times_input(net, _x4, _p4)),
             ("IG", _ig)]
    _fig, _axes = plt.subplots(4, 4, figsize=(6, 6))
    for _r, (_name, _m) in enumerate(_maps):
        for _c in range(4):
            _img = _m[_c, 0].detach().numpy()
            _lim = np.abs(_img).max() + 1e-12
            _axes[_r, _c].imshow(_img, cmap="gray" if _r < 2 else "RdBu_r", vmin=0 if _r < 2 else -_lim,
                                 vmax=1 if _r == 0 else _lim)
            _axes[_r, _c].set_xticks([])
            _axes[_r, _c].set_yticks([])
        _axes[_r, 0].set_ylabel(_name, fontsize=9)
    mo.vstack([_fig, mo.md(f"IG with a **{ig_baseline.value}** baseline and **{ig_steps.value}** steps: mean "
                           f"relative completeness error {_err:.4f} on these four digits.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how fast does the error fall with $n$, and why that rate for the trapezoid
    rule? (a ReLU network's gradient is piecewise constant along the path.) why do saliency
    maps light up the background around the stroke while gradient × input and IG are zero
    there? which of the three would you trust to say "the model used this pixel"? switch the
    baseline to the mean digit or a blurred copy: what question does IG answer now?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: sanity checks for saliency maps

    Adebayo et al. (2018) asked a blunt question: if you destroy the model, does the
    explanation change? your `cascading_randomization` re-initializes LeNet's layers from
    the top (fc3, then fc3 and fc2, ...) and measures the Spearman rank correlation between the
    original map and the map of the damaged model, for 16 test digits (the correlation is
    over all their pixels at once). an explanation that depends on what the model learned
    should fall towards 0 as layers are destroyed.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("after every layer of the network is re-initialized at random, which of saliency, gradient x "
                    "input and IG will still produce maps that correlate with the trained model's maps?")
    p2
    return (p2,)


@app.cell
def _(net, p2):
    ui.gate(p2)
    _x = X_TEST[100:116]
    with torch.no_grad():
        _pred = net(_x).argmax(1)
    _methods = {"saliency": ex.saliency, "grad x input": grad_times_input,
                "IG (16 steps)": lambda m, x, t: ex.integrated_gradients(m, x, t, n_steps=16)}
    cascade = {}
    for _name, _fn in _methods.items():
        cascade[_name] = ex.cascading_randomization(net, _fn, _x, _pred, generator=torch.Generator().manual_seed(0))
    _layers = ["fc3", "fc2", "fc1", "conv2", "conv1"]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})
    for (_name, _r), _style in zip(cascade.items(), ["o-", "s-", "^--"]):
        _axes[0].plot(range(1, len(_r) + 1), _r, _style, label=_name, alpha=0.8)
    _axes[0].set_xticks(range(1, 6), [f"+{n}" for n in _layers])
    _axes[0].axhline(0, color="gray", lw=0.5)
    _axes[0].set(xlabel="layers re-initialized (cumulative, from the top)", ylabel="Spearman vs trained",
                 ylim=(-0.2, 1.05), title="cascading randomization")
    _axes[0].legend(fontsize=8)
    _rand = ex.randomize_layers(net, 5, generator=torch.Generator().manual_seed(0))
    _stroke = (_x > 0).numpy()  # the pixels where the digit is: everywhere else x = 0
    on_stroke = {_name: ex.rank_correlation(_fn(net, _x, _pred).detach().abs().numpy()[_stroke],
                                            _fn(_rand, _x, _pred).detach().abs().numpy()[_stroke])
                 for _name, _fn in _methods.items()}
    _axes[1].axis("off")
    for _r, (_name, _fn) in enumerate(_methods.items()):
        for _c, (_lbl, _model) in enumerate([("trained", net), ("random", _rand)]):
            _ax = _fig.add_axes([0.6 + 0.19 * _c, 0.68 - 0.3 * _r, 0.17, 0.26])
            _ax.imshow(_fn(_model, _x[:1], _pred[:1])[0, 0].detach().abs().numpy(), cmap="gray")
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _r == 0:
                _ax.set_title(_lbl, fontsize=8)
            if _c == 0:
                _ax.set_ylabel(_name, fontsize=8)
    mo.vstack([_fig, mo.md("| method | " + " | ".join(f"+{n}" for n in _layers)
                           + " | all random, stroke pixels only |\n|---|" + "---|" * 6 + "\n"
                           + "\n".join(f"| {k} | " + " | ".join(f"{v:.2f}" for v in r) + f" | {on_stroke[k]:.2f} |"
                                       for k, r in cascade.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a map that survives the destruction of the model can't be explaining the
    model. what is it showing instead? (look at the factor of $x$ in two of the three
    methods: what is $|x \cdot g|$ when $g$ is random, and what fraction of MNIST pixels are
    exactly 0?) the last column correlates only the stroke pixels: how much of the
    "stability" was the zero background? does this make IG wrong, or does it mean the
    *picture* of an attribution is the wrong thing to look at?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what is a probe finding?

    a linear probe on each layer of the trained LeNet (spatial layers 4×4 average-pooled)
    predicts whether the digit is **even**. its control task (your `probe_selectivity`) gives
    each digit class a random binary label (the "type" here is the class), so it asks the
    same question about an arbitrary grouping of the digits. 1,500 training and 1,000 test
    digits; the control is averaged over 5 random label tables.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("parity is probed from the last hidden layer with high accuracy. how well will the probe do "
                    "on a random grouping of the digits into two sets?")
    p3
    return (p3,)


@app.cell
def _(net, p3):
    ui.gate(p3)
    _names = ["conv1", "conv2", "fc1", "fc2", "fc3"]
    _tr, _te = slice(0, 1500), slice(1500, 2500)
    _xs = torch.cat([X_TRAIN[:1500], X_TEST[:1000]])
    _ys = torch.cat([Y_TRAIN[:1500], Y_TEST[:1000]]).numpy()
    _feats = layer_features(net, _xs, _names)
    _feats["pixels"] = F.avg_pool2d(_xs, 4).flatten(1).numpy()
    _parity = (_ys % 2 == 0).astype(np.int64)
    probes = {}
    for _name in ["pixels"] + _names:
        _f = _feats[_name]
        _runs = [ex.probe_selectivity(_f[_tr], _parity[_tr], _f[_te], _parity[_te], _ys[_tr], _ys[_te], rng=s)
                 for s in range(5)]
        probes[_name] = (_runs[0]["task"], float(np.mean([r["control"] for r in _runs])),
                         float(np.min([r["control"] for r in _runs])), _f.shape[1])
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ks = list(probes)
    _ax.plot(_ks, [probes[k][0] for k in _ks], "o-", label="task: is the digit even?")
    _ax.plot(_ks, [probes[k][1] for k in _ks], "s--", label="control: random grouping (mean of 5)")
    _ax.set(ylabel="probe test accuracy", ylim=(0.45, 1.02), xlabel="layer")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| layer | features | task | control (mean) | control (worst) | selectivity |\n"
                           "|---|---|---|---|---|---|\n"
                           + "\n".join(f"| {k} | {v[3]} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {v[0] - v[1]:+.3f} |"
                                       for k, v in probes.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** if the probe does as well on a random grouping as on parity, what has it
    found in the layer: "parity", or something else? why does a linear probe on a layer that
    separates the ten classes decode *every* binary function of the class about equally
    well? why is the worst random grouping noticeably worse in the early layers? what would
    you need to show before claiming "layer fc2 represents parity"?
    """)
    return


if __name__ == "__main__":
    app.run()
