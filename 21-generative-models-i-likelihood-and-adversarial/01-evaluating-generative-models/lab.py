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
    from mlfs.torchnn import get_device  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    X_train = torch.tensor(_d["X_train"][:, None] / 255.0, dtype=torch.float32)
    y_train = _d["y_train"]
    X_test = torch.tensor(_d["X_test"][:, None] / 255.0, dtype=torch.float32)
    y_test = _d["y_test"]
    DEVICE = torch.device("cpu") if OFFLINE else get_device()
    _t0 = time.perf_counter()
    featnet = ex.cached_feature_net(X_train, y_train, name="mnist", epochs=2, seed=0, device=DEVICE).to("cpu")
    FEATNET_SECONDS = time.perf_counter() - _t0
    feats_test = featnet.embed(X_test)
    N_POOL = min(10_000, len(X_train))
    feats_pool = featnet.embed(X_train[:N_POOL])  # "fake" models are built from these training images
    HALF = len(feats_test) // 2


@app.cell(hide_code=True)
def _():
    _acc = float((featnet(X_test[:2000]).argmax(1).numpy() == y_test[:2000]).mean())
    mo.md(rf"""
    # evaluating generative models · lab

    every number here is computed in the feature space of **your** `FeatureNet`, trained on
    MNIST by your `cached_feature_net` (test accuracy **{_acc:.3f}**; {FEATNET_SECONDS:.0f} s to
    train or load on {DEVICE}), with your `fid_from_features`, `fid_infinity` and
    `precision_recall_knn`. the numbers are comparable with each other, not with
    Inception-based FIDs in papers. {'**offline: synthetic stand-in data, so the numbers say nothing about MNIST.**' if OFFLINE else ''}
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the FID of real data against real data

    split the test set into two disjoint random halves and compute the FID between $n$
    images from each half, five times per $n$ with fresh splits. both halves come from
    exactly the same distribution. then fit $\text{FID}(n) = a + b/n$ with your
    `fid_infinity` (one half as "real", the other as "fake").
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "two halves of the MNIST test set, n = 50 images each: roughly what FID do you expect? and with n = 5,000? "
        "(for scale: a 'model' that never draws 7, 8 or 9 scores an FID of about 100 in this feature space)"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    NS1 = [n for n in (50, 100, 250, 500, 1000, 2500, 5000) if n <= HALF]
    _rows, _means, _sds = [], [], []
    for _n in NS1:
        _v = []
        for _ in range(5):
            _p = _rng.permutation(len(feats_test))
            _v.append(ex.fid_from_features(feats_test[_p[:_n]], feats_test[_p[HALF:HALF + _n]]))
        _means.append(np.mean(_v))
        _sds.append(np.std(_v))
        _rows.append(f"| {_n} | {np.mean(_v):.2f} | {np.std(_v):.2f} |")
    _ns_inf = np.unique(np.linspace(max(HALF // 5, 20), HALF, 20).astype(int))  # n >= 1,000: the linear regime
    _fits = []
    for _r in range(5):  # five random splits: one fit is noisy
        _p = np.random.default_rng(10 + _r).permutation(len(feats_test))
        _fits.append(ex.fid_infinity(feats_test[_p[:HALF]], feats_test[_p[HALF:]], _ns_inf, rng=_r))
    _a, _b, _f = _fits[0]
    _infs = np.array([f[0] for f in _fits])
    _ref_bias = ex.fid_from_features(feats_test[_p[:HALF]], feats_pool)  # a reference half vs other real images
    _fig, _ax = plt.subplots(figsize=(6.2, 3.6))
    _ax.errorbar(NS1, _means, yerr=_sds, fmt="o-", capsize=3, label="two halves, n each")
    _ax.plot(_ns_inf, _a + _b / _ns_inf, "--", label=f"fit a + b/n to the full-half reference: a = {_a:.2f}")
    _ax.set(xscale="log", yscale="log", xlabel="n (samples per set)", ylabel="FID")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| n | FID (mean of 5) | sd |\n|---|---|---|\n" + "\n".join(_rows)
                           + f"\n\n`fid_infinity` (20 values of n from {_ns_inf[0]} to {_ns_inf[-1]}) with the full "
                             f"other half as reference, five random splits: FID∞ = **{_infs.mean():.2f} ± "
                             f"{_infs.std():.2f}** (each split: {', '.join(f'{v:.2f}' for v in _infs)}). for comparison, "
                             f"a reference half against {len(feats_pool):,} training images: FID {_ref_bias:.2f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the two sets come from the same distribution, so the *true* Fréchet
    distance is 0. where does the positive FID come from (think of the noise in a sample
    mean and a sample covariance of 128-dimensional features, and of what the covariance
    estimate looks like when $n < 128$)? why does it shrink roughly like $1/n$? what
    does this mean for a paper that reports FID on 1,000 samples next to one that used
    50,000?
    `fid_infinity` extrapolates only the *generated* side to infinitely many samples:
    why doesn't FID∞ come out at 0, and what does the reference half's own FID against
    the training images tell you?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: two flawed "models"

    two fake models made from real training images, so you know exactly what's wrong with
    each:

    - **model A (dropped modes):** perfect digits, but never a 7, 8 or 9.
    - **model B (noisy):** every digit, with Gaussian pixel noise of standard deviation
      $\sigma$ added (clipped to $[0, 1]$). the slider sets $\sigma$.

    both are compared against the test set at several $n$, and with $\text{FID}_\infty$.
    """)
    return


@app.cell
def _():
    sigma = mo.ui.slider(0.02, 0.3, step=0.02, value=0.1, label="model B's noise σ")
    sigma
    return (sigma,)


@app.cell
def _():
    p2 = ui.predict(
        "which model gets the better (lower) FID at σ = 0.1: A (30% of the digit classes missing) or B (all digits, "
        "faint noise)? and does the answer depend on n?"
    )
    p2
    return (p2,)


@app.cell
def _(p2, sigma):
    ui.gate(p2)
    _keep = y_train[:N_POOL] < 7
    feats_A = feats_pool[_keep]
    _g = torch.Generator().manual_seed(0)
    _B = (X_train[:N_POOL] + sigma.value * torch.randn(X_train[:N_POOL].shape, generator=_g)).clamp(0, 1)
    feats_B = featnet.embed(_B)
    _rng = np.random.default_rng(3)
    _ns = [n for n in (100, 300, 1000, 3000) if n <= min(len(feats_A), len(feats_B))]
    _rows = []
    for _n in _ns:
        _fa = np.mean([ex.fid_from_features(feats_test, feats_A[_rng.choice(len(feats_A), _n, replace=False)])
                       for _ in range(3)])
        _fb = np.mean([ex.fid_from_features(feats_test, feats_B[_rng.choice(len(feats_B), _n, replace=False)])
                       for _ in range(3)])
        _rows.append(f"| {_n} | {_fa:.1f} | {_fb:.1f} |")
    _top = min(len(feats_A), len(feats_B))
    _ns_inf = np.unique(np.linspace(min(1000, _top // 3), _top, 10).astype(int))
    _ia = ex.fid_infinity(feats_test, feats_A, _ns_inf, rng=4)[0]
    _ib = ex.fid_infinity(feats_test, feats_B, _ns_inf, rng=4)[0]
    _rows.append(f"| FID∞ | {_ia:.1f} | {_ib:.1f} |")
    mo.md(f"FID against all {len(feats_test):,} test images (mean of 3 draws), σ = {sigma.value:.2f}\n\n"
          "| n | model A: no 7, 8, 9 | model B: noisy |\n|---|---|---|\n" + "\n".join(_rows))
    return feats_A, feats_B


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** move the slider. at what noise level do the two models swap places?
    noise that a person barely notices at $\sigma = 0.1$ costs a lot more by $\sigma =
    0.2$: what does that say about what *your* feature network is sensitive to (it was
    trained on clean digits only)? why can't FID alone tell you *which* of the two things
    is wrong with a model?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: precision and recall

    the same two models (B at the slider's σ), plus a third, **model C (collapsed):**
    perfect images of a single digit, 1. precision asks "do the samples look real?", recall
    asks "is every part of the real data covered?". computed with your
    `precision_recall_knn` ($k = 3$) on up to 3,000 points per set.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "predict (precision, recall) for model A (no 7/8/9), model B (noisy) and model C (only 1s). which of A and B "
        "has the higher recall?"
    )
    p3
    return (p3,)


@app.cell
def _(feats_A, feats_B, p3):
    ui.gate(p3)
    _m = min(3000, len(feats_test), len(feats_A), len(feats_B))
    _rng = np.random.default_rng(5)
    _real = feats_test[_rng.choice(len(feats_test), _m, replace=False)]
    _C = feats_pool[y_train[:N_POOL] == 1]
    _models = {"A: no 7, 8, 9": feats_A, "B: noisy": feats_B, "C: only 1s": _C,
               "real (training images)": feats_pool}
    _rows, _pts = [], []
    for _name, _f in _models.items():
        _sub = _f[_rng.choice(len(_f), min(_m, len(_f)), replace=False)]
        _p, _r = ex.precision_recall_knn(_real, _sub, k=3)
        _rows.append(f"| {_name} | {_p:.3f} | {_r:.3f} | {ex.fid_from_features(_real, _sub):.1f} |")
        _pts.append((_name, _p, _r))
    _fig, _ax = plt.subplots(figsize=(4.6, 4.2))
    for _name, _p, _r in _pts:
        _ax.plot(_r, _p, "o", ms=8, label=_name)
    _ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="recall (coverage)", ylabel="precision (fidelity)",
            aspect="equal")
    _ax.legend(fontsize=7, loc="lower left")
    _fig.tight_layout()
    mo.hstack([_fig, mo.md("| model | precision | recall | FID |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** model A is missing 30% of the real data, yet its recall is only a little
    below that of real images. why (think about the $k$-NN balls around A's samples: how far do they reach, and
    where do real 7s, 8s and 9s sit in feature space)? the noisy model B contains every
    digit, so why does its *recall* fall too (what does noise do to the feature vectors:
    does it scatter them around the real ones, or move them somewhere else)? which single
    number would have warned you about model C?
    """)
    return


if __name__ == "__main__":
    app.run()
