import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, datasets
    from mlfs.torchnn import count_parameters, get_device  # yours, from ch 15 · topic 07

    plt.rcParams["figure.autolayout"] = True

    # small CPU config (offline, tests, verify) vs full config (real MNIST on a laptop GPU).
    # set MLFS_CAPSTONE=small to force the small config on the real data.
    OFFLINE = datasets.is_offline()
    FULL = not OFFLINE and os.environ.get("MLFS_CAPSTONE", "full") != "small"
    if FULL:
        CONFIG = dict(n_train=4000, n_test=5000, seeds=(0, 1, 2), steps=4000, lr=1e-3,
                      widths=[4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 100, 150, 250, 500, 1000],
                      # the clean-label control: fewer widths and seeds, enough to show there's no peak
                      control=dict(widths=[8, 32, 64, 100, 250, 1000], seeds=(0, 1)),
                      device=str(get_device()))
    else:
        CONFIG = dict(n_train=300, n_test=400, seeds=(0, 1), steps=400, lr=3e-3,
                      widths=[2, 4, 8, 16, 32, 64, 128], control=dict(widths=[4, 32, 128], seeds=(0,)), device="cpu")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 19 · capstone: reproduce a theory figure

    read [`brief.md`](brief.md) first. the required figure is **model-wise double descent with
    trained networks** (Belkin et al., 2019; Nakkiran et al., 2020): two-layer networks of
    increasing width trained to interpolate 4,000 MNIST digits, a fifth of them mislabeled,
    with the test error rising to a peak near the interpolation threshold and falling again.
    the options (NTK convergence, μTransfer) are in the brief.

    | stage | you build |
    |---|---|
    | 1 | `noisy_subset`: a random training subset with an exact fraction of wrong labels |
    | 2 | `train_interpolator`, `evaluate`: full-batch training to (near) zero training error, and the test metrics |
    | 3 | `model_wise_curve`: the whole sweep over widths and seeds |
    | 4 | `locate_threshold`: where the model starts to interpolate, where the test error (or loss) peaks |
    | 5 | `mean_ci`: means over seeds with t intervals |

    when the checks are green, scroll down to run the sweep and write up.
    """)
    return


@app.function
def noisy_subset(X, y, n, *, noise=0.0, n_classes=10, rng=None):
    """stage 1: dict X (n, d) float32 tensor in [0, 1], y_clean, y_noisy (int64 tensors): n images without replacement, exactly round(noise * n) labels moved to another class."""
    raise NotImplementedError("stage1: implement noisy_subset")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", noisy_subset=noisy_subset)
    return


@app.function
def task_loss(outputs, y, loss, n_classes):
    """the training/evaluation loss: 'mse' = squared error to one-hot summed over classes, averaged over examples; 'ce' = cross-entropy (helper, given)."""
    if loss == "mse":
        return ((outputs - F.one_hot(y, n_classes).to(outputs.dtype)) ** 2).sum(dim=1).mean()
    if loss == "ce":
        return F.cross_entropy(outputs, y)
    raise ValueError(f"loss must be 'mse' or 'ce', got {loss!r}")


@app.function
def train_interpolator(model, X, y, *, steps=4000, lr=1e-3, loss="mse", n_classes=10, device="cpu"):
    """stage 2: full-batch Adam for `steps` steps on task_loss, in train mode; returns {'loss': the steps losses before each step, 'train_error': final error on (X, y)}."""
    raise NotImplementedError("stage2: implement train_interpolator")


@app.function
def evaluate(model, X, y, *, loss="mse", n_classes=10, device="cpu"):
    """stage 2: {'error': fraction of argmax mistakes, 'loss': task_loss} with no gradients in eval mode; restores the model's mode."""
    raise NotImplementedError("stage2: implement evaluate")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", train_interpolator=train_interpolator, evaluate=evaluate)
    return


@app.function
def two_layer(in_dim, width, n_classes=10):
    """the model of the sweep: Linear(in_dim, width) -> ReLU -> Linear(width, n_classes), PyTorch's default init (helper, given)."""
    return nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(), nn.Linear(width, n_classes))


@app.function
def model_wise_curve(X_train, y_train, X_test, y_test, widths, *, seeds=(0,), steps=4000, lr=1e-3, loss="mse",
                     n_classes=10, device="cpu"):
    """stage 3: dict of (len(widths), len(seeds)) arrays train_error, test_error, test_loss, and n_params (len(widths),); model = two_layer built right after torch.manual_seed(seed)."""
    raise NotImplementedError("stage3: implement model_wise_curve")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", model_wise_curve=model_wise_curve, train_interpolator=train_interpolator,
               evaluate=evaluate)
    return


@app.function
def locate_threshold(widths, train_error, test_metric, *, tol=0.01):
    """stage 4: dict interpolation_width (first width with seed-mean train error <= tol, or None), peak_width (argmax of the seed-mean test_metric over widths >= interpolation_width; all widths if none), first_min_width (argmin of the seed-mean test_metric below the peak, or None)."""
    raise NotImplementedError("stage4: implement locate_threshold")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", locate_threshold=locate_threshold)
    return


@app.function
def mean_ci(values, *, alpha=0.05):
    """stage 5: (mean, lo, hi) arrays over the last axis: mean +- t_{1-alpha/2, n-1} * sample std / sqrt(n); lo = hi = nan when n = 1."""
    raise NotImplementedError("stage5: implement mean_ci")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", mean_ci=mean_ci)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    MNIST through `mlfs.datasets`: a random subset of the training images (4,000 in the full
    config), pixels scaled to $[0, 1]$, flattened to 784 numbers and centred by the training
    mean, and the first 5,000 test images with their true labels. the training labels come in two
    versions: clean, and with 20% moved to a wrong class.
    """)
    return


@app.cell
def _():
    _d = datasets.mnist()
    clean = noisy_subset(_d["X_train"], _d["y_train"], min(CONFIG["n_train"], len(_d["X_train"])), noise=0.0, rng=0)
    noisy = noisy_subset(_d["X_train"], _d["y_train"], min(CONFIG["n_train"], len(_d["X_train"])), noise=0.2, rng=0)
    _mu = clean["X"].mean(dim=0)
    X_train = clean["X"] - _mu
    _nt = min(CONFIG["n_test"], len(_d["X_test"]))
    X_test = torch.tensor(_d["X_test"][:_nt].reshape(_nt, -1), dtype=torch.float32) / 255.0 - _mu
    y_test = torch.tensor(_d["y_test"][:_nt])
    labels = {0.0: clean["y_noisy"], 0.2: noisy["y_noisy"]}  # 0.0: the clean-label control
    _k = X_train.shape[1]
    mo.md(f"{'**MNIST**' if not OFFLINE else '**synthetic stand-in (offline)**'}: {len(X_train)} training images "
          f"({int((noisy['y_noisy'] != noisy['y_clean']).sum())} mislabeled in the noisy version), {len(X_test)} test "
          f"images · config **{'full' if FULL else 'small'}** on `{CONFIG['device']}` · the interpolation threshold "
          f"n·K = {len(X_train) * 10:,} parameters falls at width ≈ {len(X_train) * 10 / (_k + 11):.0f}")
    return X_test, X_train, labels, y_test


@app.cell
def _(X_test, X_train, labels, y_test):
    # the sweep (yours, stage 3): every width, every seed, both label versions
    _t0 = time.perf_counter()
    curves = {}
    for _noise, _widths, _seeds in ((0.2, CONFIG["widths"], CONFIG["seeds"]),
                                    (0.0, CONFIG["control"]["widths"], CONFIG["control"]["seeds"])):
        curves[_noise] = model_wise_curve(X_train, labels[_noise], X_test, y_test, _widths,
                                          seeds=_seeds, steps=CONFIG["steps"], lr=CONFIG["lr"],
                                          device=CONFIG["device"])
    sweep_minutes = (time.perf_counter() - _t0) / 60
    mo.md(f"sweep finished in {sweep_minutes:.1f} minutes")
    return curves, sweep_minutes


@app.cell
def _(X_train, curves):
    # the figure (yours, stages 4-5)
    _wid = {0.2: np.array(CONFIG["widths"]), 0.0: np.array(CONFIG["control"]["widths"])}
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6))
    _rows = []
    for _noise, _c in curves.items():
        _lab = f"{int(_noise * 100)}% label noise"
        for _ax, _key in zip(_axes, ("train_error", "test_error", "test_loss")):
            _m, _lo, _hi = mean_ci(_c[_key])
            _ax.plot(_c["n_params"], _m, "o-", ms=3, label=_lab)
            if _c["test_error"].shape[1] > 1:
                _ax.fill_between(_c["n_params"], _lo, _hi, alpha=0.2)
        for _metric in ("test_error", "test_loss"):
            _rows.append((_noise, _metric, locate_threshold(_wid[_noise], _c["train_error"], _c[_metric])))
    for _ax, _t in zip(_axes, ("training error", "test error", "test loss (squared error)")):
        _ax.axvline(len(X_train) * 10, color="gray", ls=":", lw=0.8)
        _ax.set(xscale="log", xlabel="parameters", title=_t)
    _axes[0].legend(fontsize=8)
    _tbl = "\n".join(
        f"| {int(n * 100)}% | {metric.replace('_', ' ')} | {loc['interpolation_width']} | {loc['peak_width']} | "
        f"{loc['first_min_width']} |" for n, metric, loc in _rows)
    _detail = []
    for _noise, _c in curves.items():
        _m, _lo, _hi = mean_ci(_c["test_error"])
        _detail += [f"| {int(_noise * 100)}% | {w} | {p:,} | {_c['train_error'][i].mean():.3f} | {_m[i]:.3f} "
                    f"[{_lo[i]:.3f}, {_hi[i]:.3f}] | {_c['test_loss'][i].mean():.3f} |"
                    for i, (w, p) in enumerate(zip(_wid[_noise], _c["n_params"]))]
    mo.vstack([_fig, mo.md(
        "| label noise | metric | interpolation width | peak width | first minimum width |\n|---|---|---|---|---|\n" + _tbl
        + "\n\n| noise | width | parameters | train error | test error [95% t-interval over seeds] | test loss |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(_detail))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer each prompt in a few sentences, with numbers from your run (and their seed intervals).

    1. **the figure.** where does the training error reach zero, where does the test error peak, and how do those
       widths compare with the threshold $n K$ parameters? is your peak inside the seed interval of its neighbours,
       i.e. is it real?
    2. **noise.** compare the clean and the 20%-noise curves. how much does label noise change the height of the
       peak and the error of the widest model? relate this to topic 04's random-features results.
    3. **the right-hand descent.** every network to the right of the threshold fits the training set, mislabeled
       digits included. why does test error keep falling as width grows? which of this chapter's explanations
       (minimum-norm implicit bias, the NTK/lazy regime, flat minima) does your evidence support, and what
       experiment would distinguish them?
    4. **what's different from the theory.** the random-features curve of topic 04 has its peak exactly at $p = n$
       and an enormous spike. how do the trained networks' peak location and height differ, and why might
       training all layers (and stopping after a fixed number of steps) soften it?
    5. **fragility.** name two changes to the protocol (optimizer, loss, steps, regularization) that you expect to
       remove or move the peak, and say why. if you tried one, report it.
    6. **compute.** how long did the full sweep take on your device, and which part of the figure would you spend
       more compute on to make it more convincing?
    """)
    return


if __name__ == "__main__":
    app.run()
