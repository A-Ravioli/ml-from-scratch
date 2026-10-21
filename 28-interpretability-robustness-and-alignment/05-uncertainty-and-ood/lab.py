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
    from scipy import stats
    from scipy.special import softmax

    from mlfs import datasets, notebooks, ui
    from mlfs.metrics import expected_calibration_error, reliability_diagram  # yours, from ch 07 · 03
    from mlfs.torchnn import MLP, fit, make_loader  # yours, from ch 15 · 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OFFLINE = datasets.is_offline()
    _c = datasets.cifar10()  # synthetic stand-ins offline
    _m = datasets.mnist()
    N_TRAIN = 2_000 if OFFLINE else 5_000
    CX = (torch.tensor(_c["X_train"][:N_TRAIN], dtype=torch.float32).reshape(N_TRAIN, -1) / 255.0 - 0.47) / 0.25
    CY = torch.tensor(_c["y_train"][:N_TRAIN])
    CX_TEST = torch.tensor(_c["X_test"][:3000], dtype=torch.float32) / 255.0  # (n, 32, 32, 3)
    CY_TEST = _c["y_test"][:3000]
    MX = torch.tensor(_m["X_train"][:12_000], dtype=torch.float32).reshape(-1, 784) / 255.0
    MY = torch.tensor(_m["y_train"][:12_000])
    MX_TEST = torch.tensor(_m["X_test"][:1000], dtype=torch.float32).reshape(-1, 784) / 255.0
    MY_TEST = _m["y_test"][:1000]
    CLASSES = datasets.CIFAR10_CLASSES


@app.function
def logits_of(model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + 500]) for i in range(0, len(x), 500)]).numpy().astype(np.float64)


@app.function
def nll(probs, y):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # uncertainty & OOD · lab

    the experiments run **your** `fit_temperature`, conformal prediction and ensemble
    uncertainty on real models: an MLP (3072 → 512 → 512 → 10) trained to memorise 5,000
    CIFAR-10 images (it ends up confident and often wrong, which is what makes it
    interesting), and five small MLPs on MNIST. for each experiment: write down a prediction, lock it in, run it, then explain.
    about a minute on a laptop CPU.
    """)
    return


@app.cell
def _():
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    cnet = MLP(3072, [512, 512], 10)
    fit(cnet, F.cross_entropy, torch.optim.Adam(cnet.parameters(), lr=1e-3),
        make_loader(CX.numpy(), CY.numpy(), batch_size=128, seed=0), epochs=30)
    test_logits = logits_of(cnet, (CX_TEST.reshape(len(CX_TEST), -1) - 0.47) / 0.25)  # standardised like CX
    _train_acc = (logits_of(cnet, CX[:2000]).argmax(1) == CY[:2000].numpy()).mean()
    mo.md(f"MLP trained for 30 epochs in {time.perf_counter() - _t0:.0f} s: training accuracy "
          f"**{_train_acc:.3f}**, test accuracy **{(test_logits.argmax(1) == CY_TEST).mean():.3f}** "
          "(first 3,000 test images).")
    return (test_logits,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: overconfidence, and one number that fixes it

    the first 1,000 of 3,000 test images are a validation set: your `fit_temperature` picks
    $T$ there. the other 2,000 measure accuracy, NLL and the expected calibration error (ECE,
    15 bins, your chapter 07 metric) before and after dividing the logits by $T$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("when this model says it's 90% sure, how often is it right? what temperature will fix it, and "
                    "how much will accuracy change?")
    p1
    return (p1,)


@app.cell
def _(p1, test_logits):
    ui.gate(p1)
    _h = len(CY_TEST) // 3  # 1,000 to fit T, 2,000 to evaluate (133 / 267 offline)
    _zv, _yv, _zt, _yt = test_logits[:_h], CY_TEST[:_h], test_logits[_h:], CY_TEST[_h:]
    temperature = ex.fit_temperature(_zv, _yv)
    _before, _after = softmax(_zt, axis=1), softmax(_zt / temperature, axis=1)
    calib = {name: (float((p.argmax(1) == _yt).mean()), nll(p, _yt), float(expected_calibration_error(_yt, p, n_bins=15)),
                    float(p.max(1).mean()))
             for name, p in [("T = 1", _before), (f"T = {temperature:.2f}", _after)]}
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for _ax, (_name, _p) in zip(_axes, [("before (T = 1)", _before), (f"after (T = {temperature:.2f})", _after)]):
        reliability_diagram((_p.argmax(1) == _yt).astype(int), _p.max(1), n_bins=15, ax=_ax)
        _ax.set_title(_name)
        _ax.set_xlabel("confidence (max probability)")
    mo.vstack([_fig, mo.md("| | accuracy | NLL | ECE | mean confidence |\n|---|---|---|---|---|\n"
                           + "\n".join(f"| {k} | {a:.3f} | {n:.3f} | {e:.3f} | {c:.3f} |"
                                       for k, (a, n, e, c) in calib.items()))])
    return (temperature,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does a network trained to minimise cross-entropy end up overconfident
    on the test set? (look at the training accuracy.) why can't temperature scaling change
    the accuracy, and why is one parameter enough to fix most of the miscalibration? what
    kind of miscalibration could it *not* fix?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does the conformal guarantee hold?

    split conformal prediction with your `conformal_prediction_sets` (at $\alpha = 0.1$ unless
    you move the slider), on the temperature-scaled probabilities of the 2,000 test images that
    were *not* used to fit $T$: 1,000 random splits into 500 calibration and 1,500 evaluation
    images. the theory says each split's coverage is random but its average is exactly
    $\lceil 501 (1 - \alpha)\rceil / 501$; the histogram compares the
    1,000 coverages with the distribution the theory predicts (a beta-binomial). then the
    coverage of each true class separately, pooled over all splits.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("over 1,000 random splits, what fraction will have coverage below 90%? and will every class be "
                    "covered 90% of the time?")
    p2
    return (p2,)


@app.cell
def _():
    alpha_pick = mo.ui.slider(0.02, 0.3, value=0.1, step=0.01, label="alpha (miscoverage)")
    alpha_pick
    return (alpha_pick,)


@app.cell
def _(alpha_pick, p2, temperature, test_logits):
    ui.gate(p2)
    _v = len(CY_TEST) // 3  # the images used to fit T are left out: the pool is exchangeable with itself
    _probs = softmax(test_logits[_v:] / temperature, axis=1)
    _y = CY_TEST[_v:]
    _rng = np.random.default_rng(0)
    _alpha = alpha_pick.value
    _n = len(_y) // 4  # 500 calibration images (75 offline)
    covs, sizes = [], []
    _hits, _counts = np.zeros(10), np.zeros(10)
    for _ in range(1000):
        _perm = _rng.permutation(len(_y))
        _c, _t = _perm[:_n], _perm[_n:]
        _sets = ex.conformal_prediction_sets(_probs[_c], _y[_c], _probs[_t], alpha=_alpha)
        _hit = _sets[np.arange(len(_t)), _y[_t]]
        covs.append(_hit.mean())
        sizes.append(_sets.sum(1).mean())
        np.add.at(_hits, _y[_t], _hit)
        np.add.at(_counts, _y[_t], 1)
    covs = np.array(covs)
    _k = math.ceil((_n + 1) * (1 - _alpha) - 1e-9)
    _m_test = len(_y) - _n
    _theory = stats.betabinom(_m_test, _k, _n + 1 - _k)  # coverage of the evaluation points over calibration sets
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    _lo, _hi = max(0.0, 1 - _alpha - 0.08), min(1.0, 1 - _alpha + 0.08)
    _a1.hist(covs, bins=np.linspace(_lo, _hi, 33), density=True, alpha=0.6, label="1,000 random splits")
    _xs = np.arange(int(_lo * _m_test), int(_hi * _m_test)) / _m_test
    _a1.plot(_xs, _theory.pmf(np.round(_xs * _m_test)) * _m_test, "k-", lw=1, label="theory (beta-binomial)")
    _a1.axvline(1 - _alpha, color="C3", ls=":", label="1 - alpha")
    _a1.set(xlabel=f"coverage on the {_m_test:,} evaluation images", ylabel="density")
    _a1.legend(fontsize=7)
    per_class = _hits / _counts
    _a2.bar(range(10), per_class)
    _a2.axhline(1 - _alpha, color="C3", ls=":")
    _a2.set_xticks(range(10), [c[:4] for c in CLASSES], fontsize=7)
    _a2.set(ylim=(0.5, 1.02), ylabel="coverage", title="coverage per true class, over all 1,000 splits")
    mo.vstack([mo.md(f"alpha = {_alpha:.2f}: mean coverage **{covs.mean():.4f}** (theory {_k / (_n + 1):.4f}); "
                     f"splits below {1 - _alpha:.2f}: **{(covs < 1 - _alpha).mean():.1%}**; middle 95% of splits "
                     f"[{np.quantile(covs, 0.025):.3f}, {np.quantile(covs, 0.975):.3f}]; mean set size "
                     f"**{np.mean(sizes):.2f}** classes."), _fig,
               mo.md("per-class coverage: " + ", ".join(f"{c} {v:.2f}" for c, v in zip(CLASSES, per_class)))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the guarantee is about an *average* over calibration sets and test points.
    why do so many individual splits fall below 90%, and what controls the width of the
    histogram? why are some classes covered far less than 90% and others almost always,
    without breaking the guarantee? what would you change to guarantee coverage per class?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: knowing when an input is unusual

    five MLPs (784 → 256 → 10, different seeds) trained on 12,000 MNIST digits. they are then
    shown 1,000 MNIST test digits (in distribution) and two kinds of out-of-distribution
    inputs: 1,000 CIFAR-10 test images turned grey and cropped to 28×28, and the 1,000 MNIST
    test digits rotated by 90°. three scores flag an input as unusual: one model's
    $1 - \max_c p_c$, the ensemble's total predictive entropy, and its epistemic part (the
    mutual information), all from your `ensemble_uncertainty`; your `ood_auroc` measures
    how well each one separates the two sets.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("which inputs are harder to flag as unusual: natural photos, or rotated digits? will the "
                    "ensemble's epistemic uncertainty beat a single model's confidence?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    members = []
    for _s in range(5):
        torch.manual_seed(_s)
        _net = MLP(784, [256], 10)
        fit(_net, F.cross_entropy, torch.optim.Adam(_net.parameters(), lr=1e-3),
            make_loader(MX.numpy(), MY.numpy(), batch_size=128, seed=_s), epochs=3)
        members.append(_net)
    _grey = CX_TEST[:len(MX_TEST)].mean(-1)[:, 2:30, 2:30].reshape(-1, 784)
    _rot = torch.rot90(MX_TEST.reshape(-1, 28, 28), 1, dims=(1, 2)).reshape(-1, 784)
    _sets = {"MNIST (in)": MX_TEST, "CIFAR, grey": _grey, "MNIST rotated 90°": _rot}
    _u = {k: ex.ensemble_uncertainty(np.stack([softmax(logits_of(m, x), axis=1) for m in members]))
          for k, x in _sets.items()}
    _single = {k: 1 - softmax(logits_of(members[0], x), axis=1).max(1) for k, x in _sets.items()}
    _acc = float((_u["MNIST (in)"]["mean"].argmax(1) == MY_TEST).mean())
    ood = {}
    for _o in ["CIFAR, grey", "MNIST rotated 90°"]:
        ood[_o] = {"1 - max p (one model)": ex.ood_auroc(_single["MNIST (in)"], _single[_o]),
                   "ensemble total entropy": ex.ood_auroc(_u["MNIST (in)"]["total"], _u[_o]["total"]),
                   "ensemble epistemic (MI)": ex.ood_auroc(_u["MNIST (in)"]["epistemic"], _u[_o]["epistemic"])}
    _fig, _axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for _ax, _key in zip(_axes, ["total", "epistemic"]):
        for _k in _sets:
            _ax.hist(_u[_k][_key], bins=40, alpha=0.5, density=True, label=_k)
        _ax.set(title=f"ensemble {_key} uncertainty (nats)", yscale="log")
    _axes[0].legend(fontsize=7)
    mo.vstack([mo.md(f"ensemble accuracy on MNIST: **{_acc:.3f}**"), _fig,
               mo.md("| score | AUROC vs CIFAR grey | AUROC vs rotated MNIST |\n|---|---|---|\n"
                     + "\n".join(f"| {s} | {ood['CIFAR, grey'][s]:.3f} | {ood['MNIST rotated 90°'][s]:.3f} |"
                                 for s in ood["CIFAR, grey"]))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are rotated digits, which a person recognises instantly as unusual,
    harder for these models to flag than photos? what does it mean that a model can be
    *confident* on an input it has never seen anything like? when does disagreement between
    ensemble members (epistemic uncertainty) carry information that a single model's
    confidence doesn't?
    """)
    return


if __name__ == "__main__":
    app.run()
