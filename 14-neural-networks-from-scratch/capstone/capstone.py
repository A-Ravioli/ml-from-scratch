import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.nn import Linear, ReLU, Sequential, cross_entropy, fit, predict  # yours, from topic 04
    from mlfs.optim import Adam  # yours, from ch 06 · topic 04
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03
    from mlfs.utils import as_rng, image_grid  # yours, from ch 00

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    CONFIG = (dict(hidden=(64,), epochs=5, batch_size=64, lr=1e-3, n_val=200, n_boot=500) if SMALL else
              dict(hidden=(256, 128), epochs=12, batch_size=128, lr=1e-3, n_val=5000, n_boot=2000))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 14 · capstone: MNIST with your own framework

    read [`brief.md`](brief.md) first. you'll train a multilayer perceptron on MNIST with
    nothing but NumPy and the framework you built in this chapter: your `Tensor` and its
    backward pass, your `Linear`, `ReLU` and `cross_entropy`, your `fit` loop, and your Adam
    from chapter 06. the target is **at least 97% test accuracy**, reported with a confidence
    interval and against baselines.

    | stage | you build |
    |---|---|
    | 1 | `preprocess` and `split_validation`: pixels to features, and a validation set |
    | 2 | `build_mlp`: a ReLU network of any sizes from your layers |
    | 3 | `accuracy` and `accuracy_ci`: test accuracy with a bootstrap confidence interval |
    | 4 | `train_classifier`: Adam + your `fit`, tracking validation accuracy, keeping the best weights |

    when they're green, scroll down: the experiments run on your functions, and the write-up
    prompts are at the end.
    """)
    return


@app.function
def preprocess(images):
    """stage 1: uint8 images (n, 28, 28) or (n, 784) -> float64 (n, 784) with values in [0, 1] (divide by 255)."""
    raise NotImplementedError("stage1: implement preprocess")


@app.function
def split_validation(X, y, n_val, *, rng=None):
    """stage 1: a random split (one permutation from rng): returns (X_train, y_train, X_val, y_val) with n_val validation rows."""
    raise NotImplementedError("stage1: implement split_validation")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", preprocess=preprocess, split_validation=split_validation)
    return


@app.function
def build_mlp(sizes, *, rng=None):
    """stage 2: Sequential of Linear(sizes[i], sizes[i+1]) layers (all drawing from one rng) with ReLU between them, none after the last."""
    raise NotImplementedError("stage2: implement build_mlp")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", build_mlp=build_mlp)
    return


@app.function
def accuracy(model, X, y):
    """stage 3: the fraction of rows where the model's largest logit is the true class."""
    raise NotImplementedError("stage3: implement accuracy")


@app.function
def accuracy_ci(model, X, y, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 3: (accuracy, lo, hi): a percentile bootstrap CI (your bootstrap_ci) over the per-example correctness."""
    raise NotImplementedError("stage3: implement accuracy_ci")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", accuracy=accuracy, accuracy_ci=accuracy_ci, build_mlp=build_mlp)
    return


@app.function
def train_classifier(model, X_train, y_train, X_val, y_val, *, epochs=10, batch_size=128, lr=1e-3, rng=None):
    """stage 4: train with Adam(lr) and cross-entropy via your fit; history gets 'val_acc' per epoch; restore the best-val_acc weights."""
    raise NotImplementedError("stage4: implement train_classifier")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", train_classifier=train_classifier, build_mlp=build_mlp,
               accuracy=accuracy, preprocess=preprocess, split_validation=split_validation)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    MNIST: 60,000 training and 10,000 test images of handwritten digits, $28 \times 28$
    grayscale. offline, `mlfs.datasets` substitutes a small synthetic stand-in (2,000 + 400
    images of blurry class templates with noise) so everything below still runs; your write-up
    must use the real data.
    """)
    return


@app.cell
def _():
    data = datasets.mnist()
    X_full = preprocess(data["X_train"])
    X_test = preprocess(data["X_test"])
    y_full, y_test = data["y_train"], data["y_test"]
    X_train, y_train, X_val, y_val = split_validation(X_full, y_full, CONFIG["n_val"], rng=0)
    _ax = image_grid(data["X_test"][:16], nrow=8, titles=[str(t) for t in y_test[:16]])
    mo.vstack([mo.md(f"{'**synthetic stand-in (offline)**' if SMALL else '**real MNIST**'}: train {len(X_train)} · "
                     f"validation {len(X_val)} · test {len(X_test)} · config {CONFIG}"), _ax.figure])
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _(X_train, X_val, y_train, y_val):
    # the main model: an MLP, 784 -> hidden... -> 10
    _t0 = time.perf_counter()
    mlp = build_mlp((784, *CONFIG["hidden"], 10), rng=0)
    mlp_history = train_classifier(mlp, X_train, y_train, X_val, y_val, epochs=CONFIG["epochs"],
                                   batch_size=CONFIG["batch_size"], lr=CONFIG["lr"], rng=0)
    mlp_seconds = time.perf_counter() - _t0
    # baseline: the same pipeline with no hidden layer (softmax regression)
    linear = build_mlp((784, 10), rng=0)
    linear_history = train_classifier(linear, X_train, y_train, X_val, y_val, epochs=CONFIG["epochs"],
                                      batch_size=CONFIG["batch_size"], lr=CONFIG["lr"], rng=0)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for _name, _h in [("MLP", mlp_history), ("softmax regression", linear_history)]:
        _ep = np.arange(1, len(_h["train_loss"]) + 1)
        _axes[0].plot(_ep, _h["train_loss"], "-", label=f"{_name}: train")
        _axes[0].plot(_ep, _h["val_loss"], "--", label=f"{_name}: validation")
        _axes[1].plot(_ep, _h["val_acc"], "o-", label=_name)
    _axes[0].set(xlabel="epoch", ylabel="cross-entropy", yscale="log")
    _axes[1].set(xlabel="epoch", ylabel="validation accuracy")
    _axes[0].legend(fontsize=8)
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"MLP training time: **{mlp_seconds:.1f} s** for {CONFIG['epochs']} epochs.")])
    return linear, mlp


@app.cell
def _(X_test, X_train, linear, mlp, y_test, y_train):
    # evaluation (given): the test set is used once, here
    _acc, _lo, _hi = accuracy_ci(mlp, X_test, y_test, n_boot=CONFIG["n_boot"], rng=1)
    _lacc, _llo, _lhi = accuracy_ci(linear, X_test, y_test, n_boot=CONFIG["n_boot"], rng=1)
    _majority = float(np.mean(y_test == np.bincount(y_train).argmax()))
    _ok_mlp = (predict(mlp, X_test).argmax(1) == y_test).astype(float)
    _ok_lin = (predict(linear, X_test).argmax(1) == y_test).astype(float)
    _dlo, _dhi = bootstrap_ci((_ok_mlp, _ok_lin), lambda a, b: np.mean(a) - np.mean(b), n_boot=CONFIG["n_boot"], rng=2)
    test_errors = np.flatnonzero(_ok_mlp == 0)
    mo.md(f"""
    | model | test accuracy | 95% bootstrap CI |
    |---|---|---|
    | MLP {CONFIG['hidden']} | **{_acc:.4f}** | [{_lo:.4f}, {_hi:.4f}] |
    | softmax regression | {_lacc:.4f} | [{_llo:.4f}, {_lhi:.4f}] |
    | majority class | {_majority:.4f} | |

    paired difference MLP − softmax regression: {_acc - _lacc:+.4f}, 95% CI [{_dlo:+.4f}, {_dhi:+.4f}]
    (both models scored on the same {len(y_test)} test images, resampled jointly).
    """)
    return (test_errors,)


@app.cell
def _(X_test, mlp, test_errors, y_test):
    _show = test_errors[:16]
    if len(_show) == 0:
        _out = mo.md("no misclassified test digits.")
    else:
        _pred = predict(mlp, X_test[_show]).argmax(1)
        _ax = image_grid(X_test[_show].reshape(-1, 28, 28), nrow=8,
                         titles=[f"{t} → {p}" for t, p in zip(y_test[_show], _pred)])
        _ax.set_title("misclassified test digits (true → predicted)")
        _out = _ax.figure
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **results.** your MLP's test accuracy with its 95% confidence interval, against
       softmax regression and the majority class. is the gap to softmax regression bigger
       than the uncertainty? (use the paired interval: why is it narrower than you'd get by
       comparing the two separate intervals?)
    2. **training.** describe the learning curves. when does the validation loss stop
       improving, and does validation *accuracy* stop at the same epoch? why might they
       disagree? (your `train_classifier` keeps the weights with the best validation
       accuracy: was that early stopping needed?)
    3. **ablation.** change one thing and measure it with an interval: the width or depth,
       the learning rate, the batch size, or $\tanh$ instead of ReLU. say what you expected
       first.
    4. **failure analysis.** look at the misclassified digits. which confusions are common,
       and which mistakes would a human make too?
    5. **your framework.** how long does one epoch take? profile one training step (forward,
       backward, optimizer): where does the time go, and what would you change in your
       `Tensor` to make it faster? (chapter 15 · topic 07 compares it with PyTorch.)
    """)
    return


if __name__ == "__main__":
    app.run()
