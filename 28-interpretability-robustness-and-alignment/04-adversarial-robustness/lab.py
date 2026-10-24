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
    X_TEST = torch.tensor(_d["X_test"][:1000], dtype=torch.float32)[:, None] / 255.0
    Y_TEST = torch.tensor(_d["y_test"][:1000])
    EPS_TRAIN = 0.2


@app.function
def pgd(epsilon, n_steps=10, seed=0):
    """an attack function for robust_accuracy: linf PGD with a fixed generator seed."""
    return lambda model, x, y: ex.pgd_attack(model, x, y, epsilon, n_steps=n_steps,
                                             generator=torch.Generator().manual_seed(seed))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # adversarial robustness · lab

    the experiments attack a LeNet (your chapter 16 model) trained on 12,000 MNIST digits with
    **your** `fgsm` and `pgd_attack`, harden a second LeNet with your
    `adversarial_training_step`, and test both under your `corrupt`ions. pixels are in
    $[0, 1]$, so $\epsilon = 0.1$ means every pixel may move by a tenth of the full range.
    evaluation uses the first 1,000 test digits. for each experiment: write down a
    prediction, lock it in, run it, then explain. under a minute on a laptop CPU.
    """)
    return


@app.cell
def _():
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    std_net = LeNet()
    fit(std_net, F.cross_entropy, torch.optim.Adam(std_net.parameters(), lr=2e-3),
        make_loader(X_TRAIN.numpy(), Y_TRAIN.numpy(), batch_size=128, seed=0), epochs=2)
    std_net.eval()
    mo.md(f"standard LeNet trained in {time.perf_counter() - _t0:.0f} s: clean test accuracy "
          f"**{ex.robust_accuracy(std_net, X_TEST, Y_TEST):.3f}**.")
    return (std_net,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how small can an attack be?

    accuracy of the standard model under FGSM and under 10-step PGD (random start) as
    $\epsilon$ grows, and what the attacked digits look like at $\epsilon = 0.2$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("at epsilon = 0.2 (each pixel moves by at most 20% of its range; the digit is still obvious to "
                    "you), what accuracy will the model have under PGD? under FGSM? and at epsilon = 0.1?")
    p1
    return (p1,)


@app.cell
def _(p1, std_net):
    ui.gate(p1)
    eps_grid = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    attack_curve = {"FGSM": [ex.robust_accuracy(std_net, X_TEST, Y_TEST, lambda m, x, y, e=e: ex.fgsm(m, x, y, e))
                             for e in eps_grid],
                    "PGD-10": [ex.robust_accuracy(std_net, X_TEST, Y_TEST, pgd(e)) for e in eps_grid]}
    _x, _y = X_TEST[:6], Y_TEST[:6]
    _adv = ex.pgd_attack(std_net, _x, _y, 0.2, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        _p_clean, _p_adv = std_net(_x).argmax(1), std_net(_adv).argmax(1)
    _fig = plt.figure(figsize=(11, 3.4))
    _ax = _fig.add_axes([0.06, 0.15, 0.36, 0.75])
    for _k, _v in attack_curve.items():
        _ax.plot(eps_grid, _v, "o-", label=_k)
    _ax.set(xlabel="epsilon (linf)", ylabel="accuracy", ylim=(-0.02, 1.02), title="standard LeNet")
    _ax.legend()
    for _i in range(6):
        for _r, (_img, _p) in enumerate([(_x, _p_clean), (_adv, _p_adv)]):
            _a = _fig.add_axes([0.47 + 0.088 * _i, 0.52 - 0.42 * _r, 0.08, 0.36])
            _a.imshow(_img[_i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            _a.set_title(f"{'clean' if _r == 0 else 'PGD'}: {_p[_i].item()}", fontsize=8)
            _a.axis("off")
    mo.vstack([_fig, mo.md("| epsilon | " + " | ".join(map(str, eps_grid)) + " |\n|---|" + "---|" * len(eps_grid) + "\n"
                           + "\n".join(f"| {k} | " + " | ".join(f"{a:.3f}" for a in v) + " |"
                                       for k, v in attack_curve.items()))])
    return


@app.cell
def _():
    eps_pick = mo.ui.slider(0.0, 0.3, value=0.1, step=0.01, label="epsilon for the examples below")
    eps_pick
    return (eps_pick,)


@app.cell
def _(eps_pick, p1, std_net):
    ui.gate(p1)
    _x, _y = X_TEST[6:14], Y_TEST[6:14]
    _adv = ex.pgd_attack(std_net, _x, _y, eps_pick.value, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        _pa = std_net(_adv).argmax(1)
    _fig, _axes = plt.subplots(1, 8, figsize=(10, 1.8))
    for _i, _a in enumerate(_axes):
        _a.imshow(_adv[_i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        _a.set_title(f"{_y[_i].item()} -> {_pa[_i].item()}", fontsize=8, color="C3" if _pa[_i] != _y[_i] else "k")
        _a.axis("off")
    mo.vstack([_fig, mo.md(f"PGD-10 at epsilon = {eps_pick.value:.2f}: {(_pa != _y).sum().item()} of 8 digits "
                           "misclassified (title: true label -> prediction)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can such a small change, spread over 784 pixels, flip the prediction
    of a 95%-accurate model? (think about how much a linear function $w^\top x$ can change
    when every coordinate of $x$ moves by $\epsilon$ in the direction of $\mathrm{sign}(w)$.)
    why does PGD beat FGSM, which takes one step of the full size?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 2: adversarial training

    a second LeNet, same data and optimizer, trained for 2 epochs with your
    `adversarial_training_step` (5-step PGD at $\epsilon = {EPS_TRAIN}$, a fresh attack on
    every batch). both models are then evaluated clean and under 10-step PGD at three
    budgets, and both models' input gradients are shown (signed: red raises the true class's probability).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(f"after adversarial training at epsilon = {EPS_TRAIN}: how much clean accuracy does it cost, and "
                    "what accuracy will it have under PGD at 0.1, 0.2 and 0.3? what will its input gradients look like?")
    p2
    return (p2,)


@app.cell
def _(p2, std_net):
    ui.gate(p2)
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    adv_net = LeNet()
    _opt = torch.optim.Adam(adv_net.parameters(), lr=2e-3)
    _g = torch.Generator().manual_seed(0)
    for _epoch in range(2):
        _perm = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(_epoch))
        for _i in range(0, N_TRAIN, 128):
            _b = _perm[_i:_i + 128]
            ex.adversarial_training_step(adv_net, _opt, X_TRAIN[_b], Y_TRAIN[_b], epsilon=EPS_TRAIN, n_steps=5,
                                         generator=_g)
    adv_net.eval()
    _secs = time.perf_counter() - _t0
    robust_table = {}
    for _name, _m in [("standard", std_net), ("adversarially trained", adv_net)]:
        robust_table[_name] = [ex.robust_accuracy(_m, X_TEST, Y_TEST)] + [
            ex.robust_accuracy(_m, X_TEST, Y_TEST, pgd(e)) for e in (0.1, 0.2, 0.3)]
    _x, _y = X_TEST[:5], Y_TEST[:5]

    def _grad(m, x, y):
        xg = x.clone().requires_grad_(True)
        F.cross_entropy(m(xg), y).backward()
        return -xg.grad  # the direction that makes the true class MORE likely

    _fig, _axes = plt.subplots(3, 5, figsize=(8, 5))
    for _i in range(5):
        for _r, (_lbl, _img) in enumerate([("input", _x), ("standard: -grad", _grad(std_net, _x, _y)),
                                           ("adv. trained: -grad", _grad(adv_net, _x, _y))]):
            _im = _img[_i, 0].detach().numpy()
            _lim = np.abs(_im).max() + 1e-12
            _axes[_r, _i].imshow(_im, cmap="gray" if _r == 0 else "RdBu_r", vmin=0 if _r == 0 else -_lim,
                                 vmax=1 if _r == 0 else _lim)
            _axes[_r, _i].set_xticks([])
            _axes[_r, _i].set_yticks([])
            if _i == 0:
                _axes[_r, _i].set_ylabel(_lbl, fontsize=8)
    mo.vstack([mo.md(f"adversarial training took {_secs:.0f} s."),
               mo.md("| model | clean | PGD 0.1 | PGD 0.2 | PGD 0.3 |\n|---|---|---|---|---|\n"
                     + "\n".join(f"| {k} | " + " | ".join(f"{a:.3f}" for a in v) + " |"
                                 for k, v in robust_table.items())), _fig])
    return (adv_net,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what did robustness cost in clean accuracy, and why should there be a cost
    at all (Tsipras et al., 2019: what kind of features can a robust model *not* use)? why does
    robustness collapse beyond the training budget? why do the robust model's gradients look
    like strokes of a digit while the standard model's look like noise?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: does adversarial robustness carry over to natural shifts?

    your `corruption_accuracy` for both models: Gaussian noise, lower contrast, higher
    brightness and a sideways shift, at severities 0 to 5.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("the adversarially trained model resists small worst-case perturbations. will it also be "
                    "better on random noise? on low contrast? on shifted digits?")
    p3
    return (p3,)


@app.cell
def _(adv_net, p3, std_net):
    ui.gate(p3)
    _kinds = ["gaussian_noise", "contrast", "brightness", "shift"]
    _sev = [0, 1, 2, 3, 4, 5]
    shift_acc = {_n: ex.corruption_accuracy(_m, X_TEST, Y_TEST, _kinds, _sev, generator=torch.Generator().manual_seed(0))
                 for _n, _m in [("standard", std_net), ("adversarially trained", adv_net)]}
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for _a, _k in zip(_axes, _kinds):
        for _n, _st in zip(shift_acc, ["o-", "s--"]):
            _a.plot(_sev, shift_acc[_n][_k], _st, label=_n)
        _a.set(title=_k, xlabel="severity", ylim=(0, 1.02))
    _axes[0].set_ylabel("accuracy")
    _axes[0].legend(fontsize=7)
    mo.vstack([_fig, mo.md("| corruption (severity 3 / 5) | standard | adversarially trained |\n|---|---|---|\n"
                           + "\n".join(f"| {k} | {shift_acc['standard'][k][3]:.3f} / {shift_acc['standard'][k][5]:.3f} | "
                                       f"{shift_acc['adversarially trained'][k][3]:.3f} / "
                                       f"{shift_acc['adversarially trained'][k][5]:.3f} |" for k in _kinds))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which corruptions did adversarial training help with, and which did it
    make *worse*? relate each to the threat model: what does an $\ell_\infty$ ball of radius
    0.2 contain, and what does it leave out? what does this say about calling a model
    "robust" without saying robust to what?
    """)
    return


if __name__ == "__main__":
    app.run()
