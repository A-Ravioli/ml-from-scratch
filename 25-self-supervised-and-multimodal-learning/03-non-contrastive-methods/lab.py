import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.ssl import extract_features, knn_eval  # yours, from topic 01
    from mlfs.torchnn import make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(min(4, torch.get_num_threads()))  # never raise the thread count


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # non-contrastive methods · lab

    every experiment here trains with **your** `SimSiam`, `BYOL` and `train_ssl`, measures
    collapse with your `collapse_metrics`, and scores the encoder with your topic 01
    `knn_eval`. for each experiment: write down a prediction, lock it in, run it, then
    explain what happened. compare with `solutions/lab_notes.md` afterwards. the whole lab
    takes about three minutes on a laptop CPU.

    the setting for both experiments: 10,000 unlabelled MNIST digits, a small CNN encoder
    (64 features), 2 epochs of `train_ssl` with batch size 128 and Adam at lr $10^{-2}$
    (156 steps), SimSiam's projector and predictor at `proj_dim=64`, `pred_dim=16`. the
    kNN evaluation uses 5,000 labelled digits and 2,000 test digits.
    """)
    return


@app.class_definition
class SmallCNN(nn.Module):
    """three conv-BN-ReLU-pool blocks and global average pooling: 28x28 grayscale -> 64 features."""

    def __init__(self):
        super().__init__()
        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]

        self.net = nn.Sequential(*block(1, 16), *block(16, 32), *block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


@app.cell
def _():
    _d = datasets.mnist()
    X_ssl = _d["X_train"][:10_000, None].astype(np.float32) / 255
    X_lab, y_lab = X_ssl[:5000], _d["y_train"][:5000]
    X_te, y_te = _d["X_test"][:2000, None].astype(np.float32) / 255, _d["y_test"][:2000]
    return X_lab, X_ssl, X_te, y_lab, y_te


@app.function
def evaluate(encoder, X_lab, y_lab, X_te, y_te):
    """kNN accuracy and the effective rank of the encoder's features h (on the labelled set)."""
    f, l = extract_features(encoder, make_loader(X_lab, y_lab, batch_size=500, shuffle=False))
    ft, lt = extract_features(encoder, make_loader(X_te, y_te, batch_size=500, shuffle=False))
    return knn_eval(f, l, ft, lt), ex.collapse_metrics(torch.as_tensor(f))["effective_rank"]


@app.function
def run(make_model, X_ssl, X_lab, y_lab, X_te, y_te, seed):
    """build a model around a fresh SmallCNN (seeded), train it, evaluate its encoder."""
    torch.manual_seed(seed)
    enc = SmallCNN()
    model = make_model(enc)
    hist = ex.train_ssl(model, X_ssl, epochs=2, batch_size=128, lr=1e-2, strength=0.5, seed=seed)
    knn, erank = evaluate(enc, X_lab, y_lab, X_te, y_te)
    return {"loss": hist["loss"], "z_std": hist["z_std"], "knn": knn, "erank": erank}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: remove the stop-gradient

    SimSiam exactly as you built it, and the same model with `stop_gradient=False`, so
    gradients also flow into the branch that provides the target. two seeds each. we
    track the loss, the per-dimension std of the normalised projections $z$ at every step
    (collapse drives it to 0; spread-out embeddings in 64 dimensions sit near
    $1/\sqrt{64} = 0.125$), and at the end the kNN accuracy and the effective rank of the
    *encoder's* features.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "without the stop-gradient, what happens to the loss, to the std of z, and to kNN accuracy (a random, "
        "untrained encoder scores about 0.77)?"
    )
    p1
    return (p1,)


@app.cell
def _(X_lab, X_ssl, X_te, p1, y_lab, y_te):
    ui.gate(p1)
    _t0 = time.time()
    torch.manual_seed(0)
    random_knn, random_erank = evaluate(SmallCNN(), X_lab, y_lab, X_te, y_te)
    exp1 = {}
    for _name, _sg in (("SimSiam (stop-gradient)", True), ("no stop-gradient", False)):
        exp1[_name] = [run(lambda e, sg=_sg: ex.SimSiam(e, 64, proj_dim=64, pred_dim=16, stop_gradient=sg),
                           X_ssl, X_lab, y_lab, X_te, y_te, seed=s) for s in range(2)]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for (_name, _runs), _c in zip(exp1.items(), ("C0", "C3")):
        for _k, _r in enumerate(_runs):
            _axes[0].plot(_r["loss"], color=_c, alpha=0.8, label=_name if _k == 0 else None)
            _axes[1].plot(_r["z_std"], color=_c, alpha=0.8, label=_name if _k == 0 else None)
    _axes[0].set(xlabel="step", ylabel="loss (−1 is the minimum)", title="SimSiam loss, 2 seeds each")
    _axes[1].axhline(1 / 8, color="gray", ls="--", lw=1)
    _axes[1].set(xlabel="step", ylabel="std of normalised z", title="collapse metric (dashed: 1/√64)", ylim=(0, 0.14))
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| random encoder (untrained) | – | – | {random_knn:.3f} | {random_erank:.1f} |"]
    for _name, _runs in exp1.items():
        _rows.append(f"| {_name} | " + " | ".join(
            " / ".join(f"{_r[k][-1] if isinstance(_r[k], list) else _r[k]:.3f}" for _r in _runs)
            for k in ("loss", "z_std", "knn")) + " | " + " / ".join(f"{_r['erank']:.1f}" for _r in _runs) + " |")
    mo.vstack([_fig, mo.md("| model | final loss | final std of z | kNN accuracy | effective rank of h (of 64) |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows)
                           + f"\n\nseed 0 / seed 1. ({time.time() - _t0:.0f} s)")])
    return (exp1, random_knn)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** both runs without the stop-gradient reach the *best* loss of anyone,
    almost exactly $-1$, while their $z$ collapses. what is the trivial solution they
    found, and why does the loss alone not warn you? compare the two seeds: in one the
    encoder's kNN accuracy falls below a network that was never trained, in the other it
    hasn't (yet). what do the std of $z$ and the effective rank of $h$ each tell you that
    the other doesn't? the stop-gradient makes the loss non-differentiable as a whole:
    what is SimSiam optimising instead (the lesson's alternating-optimisation view)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: is the stop-gradient the only ingredient?

    pick a variant to compare with experiment 1's SimSiam (one seed, so read differences
    in kNN accuracy of a point or two as noise; collapse is not subtle):

    - **no predictor**: SimSiam with its predictor replaced by the identity;
    - **BYOL, momentum 0.99**: your `BYOL` with its slow-moving target;
    - **BYOL, momentum 0**: the target is a copy of the current online network, updated
      every step: SimSiam's setup with BYOL's loss and heads.
    """)
    return


@app.cell
def _():
    variant = mo.ui.dropdown(options=["no predictor", "BYOL, momentum 0.99", "BYOL, momentum 0"],
                             value="no predictor", label="variant")
    variant
    return (variant,)


@app.cell
def _():
    p2 = ui.predict("does the variant you picked collapse? if it does, does its encoder's kNN accuracy collapse too?")
    p2
    return (p2,)


@app.cell
def _(X_lab, X_ssl, X_te, exp1, p2, random_knn, variant, y_lab, y_te):
    ui.gate(p2)
    _t0 = time.time()

    def _make(e):
        if variant.value == "no predictor":
            m = ex.SimSiam(e, 64, proj_dim=64, pred_dim=16)
            m.predictor = nn.Identity()
            return m
        return ex.BYOL(e, 64, proj_dim=64, hidden_dim=128, momentum=0.99 if "0.99" in variant.value else 0.0)

    _r = run(_make, X_ssl, X_lab, y_lab, X_te, y_te, seed=0)
    _base = exp1["SimSiam (stop-gradient)"][0]
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.plot(_base["z_std"], color="C0", label="SimSiam (seed 0)")
    _ax.plot(_r["z_std"], color="C2", label=variant.value)
    _ax.axhline(1 / 8, color="gray", ls="--", lw=1)
    _ax.set(xlabel="step", ylabel="std of normalised z", ylim=(0, 0.14))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| model | final loss | final std of z | kNN accuracy | effective rank of h |\n|---|---|---|---|---|\n"
        f"| SimSiam (seed 0) | {_base['loss'][-1]:.3f} | {_base['z_std'][-1]:.3f} | {_base['knn']:.3f} | {_base['erank']:.1f} |\n"
        f"| {variant.value} | {_r['loss'][-1]:.3f} | {_r['z_std'][-1]:.3f} | {_r['knn']:.3f} | {_r['erank']:.1f} |\n"
        f"\nrandom untrained encoder: kNN {random_knn:.3f}. BYOL's loss is on a different scale (0 to 8). "
        f"({time.time() - _t0:.0f} s)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** without a predictor, $z$ collapses just as it did without the
    stop-gradient. why does the predictor matter, if its only job is to map $z_1$ to
    something close to $z_2$? (think about what the projector must do if the predictor is
    the identity and the target is a stop-gradient copy.) look at the kNN column: does a
    collapsed $z$ always mean a useless encoder? where in the network did the collapse
    happen? if you tried the BYOL variants: is the momentum target needed here?
    """)
    return


if __name__ == "__main__":
    app.run()
