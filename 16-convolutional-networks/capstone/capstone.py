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
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, datasets
    from mlfs.cnn import ResNet  # yours, from topic 04
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03
    from mlfs.torchnn import count_parameters, fit, get_device  # yours, from ch 15 · topic 07
    from mlfs.train import CosineSchedule  # yours, from ch 15 · topic 04

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    RECIPE = dict(block_counts=(2, 2, 2, 2), base_width=64, residual=True, epochs=20, batch_size=256, lr=0.2,
                  momentum=0.9, nesterov=True, weight_decay=5e-4, label_smoothing=0.1, warmup_epochs=1,
                  schedule="cosine", augment=True)
    SMALL_RECIPE = dict(RECIPE, block_counts=(1, 1, 1, 1), base_width=8, epochs=3, batch_size=64, lr=0.05)
    # the ablations use a cheaper network and budget than the full run (see brief.md)
    ABLATION_BASE = dict(RECIPE, base_width=32, epochs=10)
    ABLATIONS = {
        "full recipe": {},
        "no augmentation": dict(augment=False),
        "no label smoothing": dict(label_smoothing=0.0),
        "no weight decay": dict(weight_decay=0.0),
        "constant lr (no warmup, no cosine)": dict(schedule="constant"),
        "plain network (no residuals)": dict(residual=False),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 16 · capstone: CIFAR-10 with your ResNet

    read [`brief.md`](brief.md) first. you'll train **your** ResNet-18 (topic 04) on
    CIFAR-10 with chapter 15's recipe (augmentation, SGD with momentum and weight decay, a
    warmup + cosine schedule, label smoothing, batch norm), report its test accuracy with an
    honest interval, and measure what each ingredient of the recipe is worth in an ablation
    table.

    | stage | functions |
    |---|---|
    | 1 | `channel_stats`, `to_tensor`: per-channel normalization, uint8 NHWC → float32 NCHW |
    | 2 | `augment_batch`: random crops (with padding) and horizontal flips, on whole batches on the device |
    | 3 | `BatchLoader`, `build_model`, `make_optimizer`: batches, your ResNet, SGD with decay on weights only |
    | 4 | `predict`, `train_and_evaluate`: the recipe end to end, with your `fit` and `CosineSchedule` |
    | 5 | `ablation_table`: accuracy with a 95% interval per configuration, and the paired difference to the baseline |

    the data loading and the experiment cells are given; the write-up is at the end.
    """)
    return


@app.function
def channel_stats(X):
    """stage 1: per-channel (mean, std) of uint8 images X (N, H, W, 3), on the [0, 1] scale -> two float32 arrays of shape (3,)."""
    raise NotImplementedError("stage1: implement channel_stats")


@app.function
def to_tensor(X, mean, std):
    """stage 1: uint8 (N, H, W, C) -> float32 torch tensor (N, C, H, W), scaled to [0, 1] then standardized per channel."""
    raise NotImplementedError("stage1: implement to_tensor")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", channel_stats=channel_stats, to_tensor=to_tensor)
    return


@app.function
def augment_batch(x, *, padding=4, flip=True, generator=None):
    """stage 2: per image, a random crop of the zero-padded image (shift in [-padding, padding]) and a horizontal flip with p = 0.5; same device and shape."""
    raise NotImplementedError("stage2: implement augment_batch")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", augment_batch=augment_batch)
    return


@app.class_definition
class BatchLoader:
    """batches (x, y) of tensors already in memory, moved to `device`; shuffled and augmented each epoch when train=True."""

    def __init__(self, X, y, *, batch_size=256, train=True, augment=True, seed=0, device="cpu"):
        """stage 3: keep X (float tensor N, C, H, W) and y (int64) on the device; a seeded torch.Generator for shuffling and augmentation."""
        raise NotImplementedError("stage3: implement BatchLoader.__init__")

    def __len__(self):
        """stage 3: the number of batches, ceil(N / batch_size)."""
        raise NotImplementedError("stage3: implement BatchLoader.__len__")

    def __iter__(self):
        """stage 3: yield (x, y) batches: a fresh permutation and augment_batch when train, the original order otherwise."""
        raise NotImplementedError("stage3: implement BatchLoader.__iter__")


@app.function
def build_model(config, num_classes=10, in_channels=3):
    """stage 3: your ResNet with config's block_counts, base_width and residual, CIFAR stem (small_input=True)."""
    raise NotImplementedError("stage3: implement build_model")


@app.function
def make_optimizer(model, config):
    """stage 3: SGD(lr, momentum, nesterov); weight_decay on parameters with ndim > 1 only (no decay on biases and BN)."""
    raise NotImplementedError("stage3: implement make_optimizer")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", BatchLoader=BatchLoader, build_model=build_model,
               make_optimizer=make_optimizer)
    return


@app.function
def predict(model, X, *, batch_size=1000, device="cpu"):
    """stage 4: predicted classes (int64 ndarray) for a float tensor X, in eval mode under no_grad, restoring the mode."""
    raise NotImplementedError("stage4: implement predict")


@app.function
def train_and_evaluate(config, data, *, device="cpu", seed=0):
    """stage 4: train build_model(config) with the recipe via fit (CosineSchedule with warmup, label smoothing); dict(acc, correct, history, seconds, n_params)."""
    raise NotImplementedError("stage4: implement train_and_evaluate")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", predict=predict, train_and_evaluate=train_and_evaluate,
               build_model=build_model)
    return


@app.function
def ablation_table(results, baseline="full recipe", *, n_boot=2000, seed=0):
    """stage 5: rows dict(name, acc, lo, hi, delta, delta_lo, delta_hi): 95% bootstrap CIs; delta = acc - baseline's, paired over test images."""
    raise NotImplementedError("stage5: implement ablation_table")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", ablation_table=ablation_table)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    CIFAR-10: 50,000 training and 10,000 test images of size $32 \times 32 \times 3$ in 10
    classes. the test set is used only for the final numbers (the recipe is fixed in
    advance; nothing is tuned on it). offline, `mlfs.datasets` substitutes a small
    synthetic stand-in so everything below still runs; your write-up must use the real
    data.
    """)
    return


@app.cell
def _():
    raw = datasets.cifar10()
    _mean, _std = channel_stats(raw["X_train"])
    data = {"X_train": to_tensor(raw["X_train"], _mean, _std), "y_train": raw["y_train"],
            "X_test": to_tensor(raw["X_test"], _mean, _std), "y_test": raw["y_test"]}
    DEVICE = torch.device("cpu") if SMALL else get_device()
    mo.md(f"{'**synthetic stand-in (offline)**' if SMALL else '**real CIFAR-10**'}: "
          f"{len(data['X_train'])} training and {len(data['X_test'])} test images · per-channel mean "
          f"{np.round(_mean, 3)}, std {np.round(_std, 3)} · device **{DEVICE}**")
    return DEVICE, data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the full run (given)

    the recipe in `RECIPE` (or `SMALL_RECIPE` offline) on all the training data. on an
    Apple M2 (MPS) the full config takes about 35 minutes. while you develop, set the
    environment variable `MLFS_CAPSTONE_FULL=0` before starting marimo to skip it.
    """)
    return


@app.cell
def _(DEVICE, data):
    RUN_FULL = SMALL or os.environ.get("MLFS_CAPSTONE_FULL", "1") != "0"
    full = train_and_evaluate(SMALL_RECIPE if SMALL else RECIPE, data, device=DEVICE, seed=0) if RUN_FULL else None
    if full is not None:
        _lo, _hi = bootstrap_ci(full["correct"].astype(float), np.mean, n_boot=2000, rng=1)
        _majority = float(np.mean(data["y_test"] == np.bincount(data["y_train"]).argmax()))
        _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.4))
        _ep = np.arange(1, len(full["history"]["train_loss"]) + 1)
        _axes[0].plot(_ep, full["history"]["train_loss"], "o-")
        _axes[0].set(xlabel="epoch", ylabel="training loss (label-smoothed)")
        _axes[1].plot(_ep, full["history"]["lr"], "o-")
        _axes[1].set(xlabel="epoch", ylabel="learning rate (end of epoch)")
        _fig.tight_layout()
        _out = mo.vstack([_fig, mo.md(f"""
        | model | parameters | test accuracy | 95% bootstrap CI | time |
        |---|---|---|---|---|
        | your ResNet {'(small config)' if SMALL else '-18, full recipe'} | {full['n_params']:,} | **{full['acc']:.4f}** | [{_lo:.4f}, {_hi:.4f}] | {full['seconds'] / 60:.1f} min |
        | majority class | | {_majority:.4f} | | |
        """)])
    else:
        _out = mo.md("full run skipped (`MLFS_CAPSTONE_FULL=0`): the reference solution's run scored 0.929 "
                     "(95% CI [0.924, 0.934]) in 35 minutes on an Apple M2; see brief.md.")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the ablation table (given)

    each row removes one ingredient from the recipe, with a cheaper network (base width
    32) and budget (10 epochs): 6–10 minutes per row on an M2, under an hour for the six
    (offline: the small config). every row is evaluated on the same test images, so the
    difference to the baseline gets a **paired** interval. set `MLFS_CAPSTONE_ABLATIONS=0`
    to skip it while you develop.
    """)
    return


@app.cell
def _(DEVICE, data):
    RUN_ABLATIONS = SMALL or os.environ.get("MLFS_CAPSTONE_ABLATIONS", "1") != "0"
    _base = dict(SMALL_RECIPE, epochs=2) if SMALL else ABLATION_BASE
    results = {}
    if RUN_ABLATIONS:
        for _name, _change in ABLATIONS.items():
            results[_name] = train_and_evaluate(dict(_base, **_change), data, device=DEVICE, seed=0)
        _rows = ablation_table(results)
        _table = "\n".join(
            f"| {r['name']} | {r['acc']:.4f} | [{r['lo']:.4f}, {r['hi']:.4f}] | "
            + ("—" if r["name"] == "full recipe" else f"{r['delta']:+.4f} [{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]")
            + f" | {results[r['name']]['seconds'] / 60:.1f} min |" for r in _rows)
        _out = mo.md("| configuration | test accuracy | 95% CI | difference to full recipe (paired 95% CI) | time |\n"
                     "|---|---|---|---|---|\n" + _table)
    else:
        _out = mo.md("ablations skipped (`MLFS_CAPSTONE_ABLATIONS=0`).")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)):

    1. **the headline number.** your ResNet-18's test accuracy with its 95% interval, the
       runtime and hardware, the majority-class baseline, and (if you can afford it) the
       spread over 2–3 seeds. did you reach 90%? if not, what would you change first, and
       why?
    2. **the ablation table**, with a sentence per row: which ingredients matter at this
       budget, which differences are inside the noise (the paired interval contains 0), and
       the *mechanism* behind each real effect.
    3. **one curve that explains something**: training loss, the learning rate, or test
       accuracy per epoch for two configurations, and what it shows.
    4. **what would get you further**: a longer schedule, a wider network, stronger
       augmentation, test-time flips; estimate the cost of each.
    """)
    return


if __name__ == "__main__":
    app.run()
