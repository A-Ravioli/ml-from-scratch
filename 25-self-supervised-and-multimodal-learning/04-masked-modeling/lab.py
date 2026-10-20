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
    from mlfs.ssl import extract_features, knn_eval, linear_probe  # yours, from topic 01
    from mlfs.torchnn import get_device, make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(min(4, torch.get_num_threads()))  # never raise the thread count
    DEVICE = get_device()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # masked modeling · lab

    every experiment here trains **your** `MAE` with your `train_mae`, and scores its
    `encode` features with your topic 01 harness. for each experiment: write down a
    prediction, lock it in, run it, then explain what happened. compare with
    `solutions/lab_notes.md` afterwards.

    the transformers make this the slowest lab of the chapter: two to four minutes on an
    Apple-silicon GPU (MPS) or a CUDA GPU, several times that on a laptop CPU. it uses
    **{DEVICE}** here.
    """)
    return


@app.cell
def _():
    _d = datasets.mnist()
    X_ssl = _d["X_train"][:10_000, None].astype(np.float32) / 255
    X_lab, y_lab = X_ssl[:5000], _d["y_train"][:5000]
    X_te, y_te = _d["X_test"][:2000, None].astype(np.float32) / 255, _d["y_test"][:2000]
    return X_lab, X_ssl, X_te, y_lab, y_te


@app.class_definition
class Encode(nn.Module):
    """wraps an MAE so that extract_features sees encoder(x) -> (B, d)."""

    def __init__(self, mae):
        super().__init__()
        self.mae = mae

    def forward(self, x):
        return self.mae.encode(x)


@app.function
def make_mae(ratio, seed):
    """a small MAE for 28x28 digits: 7x7 = 49 patches of 4x4 pixels."""
    torch.manual_seed(seed)
    return ex.MAE(28, 4, 1, encoder_dim=64, encoder_depth=2, encoder_heads=4, decoder_dim=32, decoder_depth=1,
                  decoder_heads=4, mask_ratio=ratio)


@app.function
def scores(mae, X_lab, y_lab, X_te, y_te):
    """(kNN accuracy, linear-probe accuracy) of the MAE's encode features."""
    enc = Encode(mae).to("cpu")
    f, l = extract_features(enc, make_loader(X_lab, y_lab, batch_size=500, shuffle=False))
    ft, lt = extract_features(enc, make_loader(X_te, y_te, batch_size=500, shuffle=False))
    return knn_eval(f, l, ft, lt), linear_probe(f, l, ft, lt, max_iters=300)


@app.function
def masked_mse(mae, X, ratio, seed=0):
    """the MAE's reconstruction error on masked patches of X at `ratio`, and that of a trivial baseline that fills
    every masked patch with the mean of the image's visible pixels (same masks)."""
    mae = mae.to("cpu").eval()
    old = mae.mask_ratio
    mae.mask_ratio = ratio
    with torch.no_grad():
        torch.manual_seed(seed)
        x = torch.as_tensor(X)
        loss, pred, mask = mae(x)
        patches = ex.patchify(x, 4)
        vis = (1 - mask).unsqueeze(-1)
        fill = (patches * vis).sum((1, 2), keepdim=True) / (vis.sum((1, 2), keepdim=True) * patches.shape[-1])
        base = (((fill - patches) ** 2).mean(-1) * mask).sum() / mask.sum()
    mae.mask_ratio = old
    return loss.item(), base.item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how much should you hide?

    one MAE per mask ratio (25%, 50%, 75%, 90%), 2 seeds each, trained on 10,000
    unlabelled digits for 8 epochs (`train_mae`, batch 128, lr $2\cdot10^{-3}$). each MNIST
    digit is cut into 49 patches of 4×4 pixels; the encoder has 2 transformer blocks of
    width 64. afterwards: kNN and linear-probe accuracy of `encode` (5,000 labelled
    digits, 2,000 test digits), and the reconstruction error on the masked patches of the
    test digits, next to a trivial baseline that paints every hidden patch in the average
    grey of the visible pixels.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which mask ratio gives the best linear-probe accuracy? which gives the lowest reconstruction error? are they "
        "the same?"
    )
    p1
    return (p1,)


@app.cell
def _(X_lab, X_ssl, X_te, p1, y_lab, y_te):
    ui.gate(p1)
    _t0 = time.time()
    ratios = (0.25, 0.5, 0.75, 0.9)
    models, results = {}, {}
    _rk, _rp = scores(make_mae(0.75, 0), X_lab, y_lab, X_te, y_te)
    for _r in ratios:
        for _s in range(2):
            _m = make_mae(_r, _s)
            ex.train_mae(_m, X_ssl, epochs=8, batch_size=128, lr=2e-3, seed=_s, device=DEVICE)
            models[(_r, _s)] = _m
            results[(_r, _s)] = (*scores(_m, X_lab, y_lab, X_te, y_te), *masked_mse(_m, X_te[:1000], _r))
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for _k, (_name, _c) in enumerate((("kNN", "C0"), ("linear probe", "C1"))):
        _v = np.array([[results[(r, s)][_k] for s in range(2)] for r in ratios])
        _axes[0].plot(ratios, _v.mean(1), marker="o", color=_c, label=_name)
        _axes[0].fill_between(ratios, _v.min(1), _v.max(1), color=_c, alpha=0.2)
    _axes[0].axhline(_rp, color="C1", ls=":", lw=1, label="probe, untrained MAE")
    _axes[0].axhline(_rk, color="C0", ls=":", lw=1, label="kNN, untrained MAE")
    _axes[0].set(xlabel="mask ratio", ylabel="test accuracy", title="representation quality (band: 2 seeds)")
    _axes[0].legend(fontsize=7)
    _mse = np.array([[results[(r, s)][2] for s in range(2)] for r in ratios])
    _base = np.array([[results[(r, s)][3] for s in range(2)] for r in ratios])
    _axes[1].plot(ratios, _mse.mean(1), marker="o", label="MAE")
    _axes[1].plot(ratios, _base.mean(1), marker="s", ls="--", label="mean-of-visible baseline")
    _axes[1].set(xlabel="mask ratio", ylabel="MSE on masked patches (test)", title="reconstruction at its own ratio")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(
        f"| {r} | " + " | ".join(" / ".join(f"{results[(r, s)][k]:.3f}" for s in range(2)) for k in range(2))
        + f" | {_mse[i].mean():.4f} | {_base[i].mean():.4f} |" for i, r in enumerate(ratios))
    mo.vstack([_fig, mo.md(
        "| mask ratio | kNN (seed 0 / 1) | linear probe (seed 0 / 1) | MAE masked MSE | baseline MSE |\n"
        "|---|---|---|---|---|\n" + _rows
        + f"\n\nuntrained MAE encoder: kNN {_rk:.3f}, probe {_rp:.3f}. ({time.time() - _t0:.0f} s)")])
    return models, ratios


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the reconstruction error is flat from 25% to 50% masking and rises
    after; every MAE is far better than the trivial grey fill. so why isn't the lowest
    ratio best for the representation? with a quarter of the patches hidden, almost every
    hidden patch has visible neighbours on several sides: what can the model get away with
    then, instead of understanding the digit? and why does 90% hurt? He et al. found 75%
    best for ImageNet with 16×16 patches: what about *our* setting (49 tiny patches of a
    centred digit, a two-block encoder, 8 epochs) could move the optimum? how confident can
    you be in the ranking with 2 seeds (look at 75%)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what does the MAE paint in?

    six test digits, masked at each model's own ratio (seed 0). each row: the original, the
    visible patches, and the reconstruction with the visible patches pasted back in.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("will the reconstructions of the hidden patches be sharp strokes, or blurry? at 90% masking, will the "
                    "digits still be recognisable?")
    p2
    return (p2,)


@app.cell
def _(X_te, models, p2, ratios):
    ui.gate(p2)
    _x = torch.as_tensor(X_te[:6])
    _fig, _axes = plt.subplots(len(ratios), 18, figsize=(13, 0.85 * len(ratios) + 0.4))
    for _i, _r in enumerate(ratios):
        _m = models[(_r, 0)].to("cpu").eval()
        with torch.no_grad():
            torch.manual_seed(0)
            _, _pred, _mask = _m(_x)
        _patches = ex.patchify(_x, 4)
        _keep = (1 - _mask).unsqueeze(-1)
        _visible = ex.unpatchify(_patches * _keep + 0.5 * (1 - _keep), 4, channels=1)
        _recon = ex.unpatchify(_patches * _keep + _pred.clamp(0, 1) * (1 - _keep), 4, channels=1)
        for _j in range(6):
            for _k, _img in enumerate((_x[_j], _visible[_j], _recon[_j])):
                _ax = _axes[_i, 3 * _j + _k]
                _ax.imshow(_img[0], cmap="gray", vmin=0, vmax=1)
                _ax.set_xticks([])
                _ax.set_yticks([])
        _axes[_i, 0].set_ylabel(f"{int(_r * 100)}%", rotation=0, labelpad=18, fontsize=9)
    for _j in range(6):
        for _k, _t in enumerate(("original", "visible", "recon")):
            _axes[0, 3 * _j + _k].set_title(_t, fontsize=6)
    _fig.tight_layout(pad=0.2)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the loss is a mean squared error per pixel. when several completions are
    plausible (is that stroke the top of a 7 or of a 9?), what single prediction minimises
    the *expected* squared error, and what does it look like? why does that make MAE a
    poor image generator but still a good feature learner? at 90%, what does the model fall
    back on when only five patches are visible?
    """)
    return


if __name__ == "__main__":
    app.run()
