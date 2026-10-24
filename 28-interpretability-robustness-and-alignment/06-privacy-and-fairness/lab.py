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
    from mlfs.torchnn import MLP  # yours, from ch 15 · 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OFFLINE = datasets.is_offline()
    _d = datasets.mnist()  # a synthetic stand-in offline
    N_TRAIN = 2_000 if OFFLINE else 12_000
    X_TRAIN = torch.tensor(_d["X_train"][:N_TRAIN], dtype=torch.float32).reshape(-1, 784) / 255.0
    Y_TRAIN = torch.tensor(_d["y_train"][:N_TRAIN])
    X_TEST = torch.tensor(_d["X_test"][:2000], dtype=torch.float32).reshape(-1, 784) / 255.0
    Y_TEST = torch.tensor(_d["y_test"][:2000])
    _c = datasets.cifar10()
    C_MEM = (torch.tensor(_c["X_train"][:1000], dtype=torch.float32).reshape(-1, 3072) / 255.0 - 0.47) / 0.25
    CY_MEM = torch.tensor(_c["y_train"][:1000])
    C_NON = (torch.tensor(_c["X_test"][:1000], dtype=torch.float32).reshape(-1, 3072) / 255.0 - 0.47) / 0.25
    CY_NON = torch.tensor(_c["y_test"][:1000])
    DELTA = 1e-5
    BATCH = 256


@app.function
def train_dp(x, y, *, epochs, noise_multiplier, max_grad_norm=1.0, lr=0.25, seed=0, private=True, batch=BATCH):
    """an MLP (d -> 128 -> 10) trained with your dp_sgd_step (or plain SGD if private=False) on shuffled batches."""
    torch.manual_seed(seed)
    net = MLP(x.shape[1], [128], 10)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    gen = torch.Generator().manual_seed(seed)
    steps = 0
    for epoch in range(epochs):
        perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        for i in range(0, len(x) - batch + 1, batch):
            b = perm[i:i + batch]
            if private:
                ex.dp_sgd_step(net, F.cross_entropy, opt, x[b], y[b], max_grad_norm=max_grad_norm,
                               noise_multiplier=noise_multiplier, generator=gen)
            else:
                loss = F.cross_entropy(net(x[b]), y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            steps += 1
    net.eval()
    return net, steps


@app.function
def losses_of(net, x, y):
    with torch.no_grad():
        return F.cross_entropy(net(x), y, reduction="none").numpy()


@app.function
def accuracy(net, x, y):
    with torch.no_grad():
        return float((net(x).argmax(1) == y).float().mean())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # privacy & fairness · lab

    the first two experiments train small MLPs on MNIST with **your** `dp_sgd_step`, measure
    their privacy with your accountant, and attack them with your membership-inference
    attack. the third uses your fairness metrics on a **synthetic** lending dataset (no real
    demographic data is used anywhere in this course). experiment 2 uses CIFAR-10 images.
    for each experiment: write down a prediction, lock it in, run it, then explain. about a
    minute and a half on a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 1: the price of privacy

    an MLP (784 → 128 → 10) trained for 4 epochs on 12,000 digits, batches of {BATCH}, SGD with
    momentum, gradients clipped to norm 1, at four noise levels. $\epsilon$ comes from your
    `dp_sgd_epsilon` with sampling rate $q = {BATCH}/12{{,}}000$ and $\delta = 10^{{-5}}$
    (smaller than one over the number of training points). the baseline is the same
    network trained without clipping or noise, with the same learning rate (0.25).
    """)
    return


@app.cell
def _():
    p1 = ui.predict("what test accuracy will a network trained with epsilon around 1 reach, compared with a "
                    "non-private one at about 95%? what does clipping alone (no noise) cost?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    _base, _ = train_dp(X_TRAIN, Y_TRAIN, epochs=4, noise_multiplier=0.0, private=False, lr=0.25)
    privacy_rows = [("no clipping, no noise", float("inf"), accuracy(_base, X_TEST, Y_TEST))]
    for _sigma in [0.0, 0.5, 1.0, 2.0]:
        _net, _steps = train_dp(X_TRAIN, Y_TRAIN, epochs=4, noise_multiplier=_sigma)
        _eps = ex.dp_sgd_epsilon(BATCH / N_TRAIN, _sigma, _steps, DELTA) if _sigma > 0 else float("inf")
        privacy_rows.append((f"DP-SGD, sigma = {_sigma}", _eps, accuracy(_net, X_TEST, Y_TEST)))
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _fin = [r for r in privacy_rows if np.isfinite(r[1])]
    _ax.plot([r[1] for r in _fin], [r[2] for r in _fin], "o-", label="DP-SGD")
    for _r in _fin:
        _ax.annotate(_r[0].split(", ")[1], (_r[1], _r[2]), fontsize=7, xytext=(4, -10), textcoords="offset points")
    _ax.axhline(privacy_rows[0][2], color="gray", ls=":", label="non-private")
    _ax.axhline(privacy_rows[1][2], color="C2", ls="--", label="clipping only (eps = inf)")
    _ax.set(xscale="log", xlabel="epsilon (delta = 1e-5)", ylabel="test accuracy")
    _ax.legend(fontsize=7)
    mo.vstack([mo.md(f"trained five networks in {time.perf_counter() - _t0:.0f} s."), _fig,
               mo.md("| training | epsilon | test accuracy |\n|---|---|---|\n"
                     + "\n".join(f"| {n} | {e:.2f} | {a:.3f} |" for n, e, a in privacy_rows))])
    return


@app.cell
def _():
    sigma_pick = mo.ui.slider(0.3, 4.0, value=1.5, step=0.1, label="your own noise multiplier")
    clip_pick = mo.ui.dropdown(["0.1", "1.0", "10.0"], value="1.0", label="clip norm")
    mo.hstack([sigma_pick, clip_pick])
    return clip_pick, sigma_pick


@app.cell
def _(clip_pick, p1, sigma_pick):
    ui.gate(p1)
    _net, _steps = train_dp(X_TRAIN, Y_TRAIN, epochs=4, noise_multiplier=sigma_pick.value,
                            max_grad_norm=float(clip_pick.value))
    _eps = ex.dp_sgd_epsilon(BATCH / N_TRAIN, sigma_pick.value, _steps, DELTA)
    mo.md(f"sigma = {sigma_pick.value:.1f}, clip norm {clip_pick.value}: epsilon = **{_eps:.2f}** (delta = 1e-5), "
          f"test accuracy **{accuracy(_net, X_TEST, Y_TEST):.3f}**. (epsilon doesn't depend on the clip norm: why not?)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where does the accuracy go: into the clipping or into the noise? why does
    the noise hurt less than its size suggests (compare the noise's standard deviation with
    the sum of 256 clipped gradients)? the accountant assumes each batch is a Poisson sample
    of the data; this lab shuffles instead. why is that a common shortcut, and why does it
    mean the printed epsilon isn't strictly a guarantee for this code?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: can an attacker tell who was in the training set?

    MNIST models generalise well, so their training points barely stand out. this experiment
    uses a harder task where memorisation shows: three MLPs (3072 → 128 → 10) trained on
    the same **1,000 CIFAR-10 images** (the members) for 30 epochs in batches of 64: a
    non-private one (it memorises them) and two DP-SGD ones ($\sigma = 1.5$ and $\sigma = 5$,
    with smaller learning rates: noisy steps with a large learning rate let the weights drift).
    the non-members are 1,000 test images. your `loss_threshold_attack` scores members by
    low loss; `tpr_at_fpr` asks how many members it catches when it may wrongly accuse only
    1% of non-members. differential privacy bounds *any* attack: $\text{TPR} \le e^\epsilon
    \cdot \text{FPR} + \delta$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("against the memorising model, will the loss-threshold attack's AUROC be closer to 0.5 or "
                    "to 1? and what will DP-SGD with epsilon around 6 do to it?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _xm, _ym, _xn, _yn = C_MEM, CY_MEM, C_NON, CY_NON
    mia_rows = []
    for _name, _kw in [("non-private", dict(noise_multiplier=0.0, private=False, lr=0.02)),
                       ("DP-SGD, sigma = 1.5", dict(noise_multiplier=1.5, lr=0.01)),
                       ("DP-SGD, sigma = 5", dict(noise_multiplier=5.0, lr=0.003))]:
        _net, _steps = train_dp(_xm, _ym, epochs=30, batch=64, **_kw)
        _eps = ex.dp_sgd_epsilon(64 / len(_xm), _kw["noise_multiplier"], _steps, DELTA) if _kw.get("private", True) \
            else float("inf")
        _lm, _ln = losses_of(_net, _xm, _ym), losses_of(_net, _xn, _yn)
        mia_rows.append((_name, _eps, accuracy(_net, _xm, _ym), accuracy(_net, _xn, _yn),
                         ex.loss_threshold_attack(_lm, _ln), ex.tpr_at_fpr(_lm, _ln, 0.01),
                         min(1.0, np.exp(_eps) * 0.01 + DELTA) if np.isfinite(_eps) else 1.0, _lm, _ln))
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for _ax, _r in zip(_axes, mia_rows):
        _bins = np.linspace(0, max(np.quantile(_r[8], 0.98), 0.1), 40)
        _ax.hist(np.clip(_r[7], 0, _bins[-1]), bins=_bins, alpha=0.6, density=True, label="members")
        _ax.hist(np.clip(_r[8], 0, _bins[-1]), bins=_bins, alpha=0.6, density=True, label="non-members")
        _ax.set(title=_r[0], xlabel="loss", yscale="log")
    _axes[0].legend(fontsize=7)
    mo.vstack([_fig, mo.md("| model | epsilon | train acc | test acc | attack AUROC | TPR at 1% FPR | DP bound on TPR |\n"
                           "|---|---|---|---|---|---|---|\n"
                           + "\n".join(f"| {r[0]} | {r[1]:.1f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | "
                                       f"{r[6]:.3f} |" for r in mia_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what makes the non-private model's members easy to spot, and why is the
    gap between training and test accuracy a good predictor of the attack's success? compare
    the DP bound on the attack with what the attack achieves: is the bound tight, and what
    does a loose bound mean for someone choosing $\epsilon$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: which fairness?

    a **synthetic** lending problem: 20,000 applicants in groups A (70%) and B (30%), a
    repayment label with base rates 60% (A) and 40% (B), and a credit score that is equally
    informative in both groups (a noisy version of the label). three policies approve
    applicants: one threshold for everyone; group thresholds from your
    `equal_opportunity_thresholds` (equal true positive rates, 80%); and group thresholds
    that approve the same fraction of each group (demographic parity). your metrics score
    each.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("the score is equally accurate for both groups and one threshold is applied to everyone. will "
                    "the policy satisfy demographic parity? equalized odds? can any policy satisfy both?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(0)
    _n = 20_000
    _g = np.where(_rng.random(_n) < 0.7, "A", "B")
    _y = (_rng.random(_n) < np.where(_g == "A", 0.6, 0.4)).astype(int)
    _score = 1 / (1 + np.exp(-(1.6 * (2 * _y - 1) + _rng.normal(0, 1.2, _n))))
    _single = (_score >= 0.5).astype(int)
    _eo = ex.equal_opportunity_thresholds(_score, _y, _g, 0.8)
    _eo_pred = (_score >= np.where(_g == "A", _eo["A"], _eo["B"])).astype(int)
    _rate = _single.mean()
    _dp_t = {k: np.quantile(_score[_g == k], 1 - _rate) for k in ("A", "B")}
    _dp_pred = (_score >= np.where(_g == "A", _dp_t["A"], _dp_t["B"])).astype(int)

    def _row(name, pred):
        per = {k: (pred[_g == k].mean(), pred[(_g == k) & (_y == 1)].mean(), pred[(_g == k) & (_y == 0)].mean())
               for k in ("A", "B")}
        return (name, float((pred == _y).mean()), ex.demographic_parity_difference(pred, _g),
                ex.equalized_odds_difference(_y, pred, _g), per)

    fair_rows = [_row("one threshold (0.5)", _single), _row("equal opportunity (TPR 0.8)", _eo_pred),
                 _row("demographic parity", _dp_pred)]
    mo.md("| policy | accuracy | DP difference | EO difference | approval A / B | TPR A / B | FPR A / B |\n"
          "|---|---|---|---|---|---|---|\n"
          + "\n".join(f"| {n} | {a:.3f} | {d:.3f} | {e:.3f} | {p['A'][0]:.2f} / {p['B'][0]:.2f} | "
                      f"{p['A'][1]:.2f} / {p['B'][1]:.2f} | {p['A'][2]:.2f} / {p['B'][2]:.2f} |"
                      for n, a, d, e, p in fair_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does one threshold, applied to an equally accurate score, approve the
    groups at different rates? why does equalising the approval rates *break* equal error
    rates here, and vice versa? when base rates differ, which of these criteria can hold at
    the same time (Kleinberg et al., 2017; Chouldechova, 2017), and who should decide which
    one matters for a given decision?
    """)
    return


if __name__ == "__main__":
    app.run()
