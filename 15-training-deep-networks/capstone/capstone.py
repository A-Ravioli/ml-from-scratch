import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, datasets
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03
    from mlfs.torchnn import make_loader  # yours, from topic 07
    from mlfs.train import EarlyStopping, StepLR, get_lr  # yours, from topics 03 and 04
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    CONFIG = (dict(n_train=1800, n_val=200, hidden=(64,), epochs=4, batch_size=64, lr=0.05, lr_step_epochs=2,
                   patience=3, n_boot=500) if SMALL else
              dict(n_train=55000, n_val=5000, hidden=(256, 128), epochs=10, batch_size=128, lr=0.05, lr_step_epochs=4,
                   patience=3, n_boot=2000))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 15 · capstone: fix the training run

    read [`brief.md`](brief.md) first. the functions below are a complete MNIST training
    pipeline in PyTorch, built from the pieces of this chapter: a validation split,
    normalization, a kaiming-initialized MLP with dropout, a training loop with gradient
    clipping, your `StepLR` and `EarlyStopping` driving a torch optimizer, and an evaluation
    function. it runs without errors. it also contains **10 bugs**. your job: find each one,
    fix it, and explain it in the write-up, with a measurement of what it did.

    | stage | functions | bugs hiding there |
    |---|---|---|
    | 1 | `split_data`, `normalize` | 3 |
    | 2 | `build_model` | 2 |
    | 3 | `train_one_epoch` | 2 |
    | 4 | `evaluate`, `fit` | 3 |
    | 5 | the whole pipeline | (passes when all 10 are fixed) |

    the checks under each stage fail until that stage's bugs are fixed; their messages say
    *what* is wrong, not *where*. scroll down for the experiments and the write-up.
    """)
    return


@app.function
def split_data(X, y, n_val, *, rng=None):
    """stage 1: a random split: (X_train, y_train, X_val, y_val), validation = the first n_val indices of one permutation."""
    perm = as_rng(rng).permutation(len(X))
    X = X[perm]
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val]


@app.function
def normalize(X_train, X_other):
    """stage 1: standardize with the TRAINING set's overall pixel mean and std (two scalars); returns (X_train_n, X_other_n)."""
    both = np.concatenate([X_train, X_other])
    mean, std = both.mean(axis=0), both.std(axis=0)
    return (X_train - mean) / std, (X_other - mean) / std


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", split_data=split_data, normalize=normalize)
    return


@app.function
def build_model(*, hidden=(256, 128), dropout=0.2, seed=0):
    """stage 2: torch MLP 784 -> hidden... -> 10 of Linear -> ReLU -> Dropout blocks; kaiming-normal (relu, fan_in) weights, zero biases; returns LOGITS."""
    torch.manual_seed(seed)
    layers, n_in = [], 784
    for h in hidden:
        lin = nn.Linear(n_in, h)
        nn.init.normal_(lin.weight, std=1.0)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU(), nn.Dropout(dropout)]
        n_in = h
    layers += [nn.Linear(n_in, 10), nn.Softmax(dim=1)]
    return nn.Sequential(*layers)


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", build_model=build_model)
    return


@app.function
def train_one_epoch(model, loader, optimizer, *, grad_clip=None, scheduler=None):
    """stage 3: one pass in train mode: zero_grad -> forward -> cross-entropy -> backward -> [clip] -> step -> [per-STEP scheduler]; mean loss per example."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        loss = F.cross_entropy(model(x), y)
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", train_one_epoch=train_one_epoch, build_model=build_model)
    return


@app.function
def evaluate(model, X, y, *, batch_size=1000):
    """stage 4: (mean cross-entropy, accuracy) on ndarrays X, y, in eval mode under no_grad, restoring the model's mode."""
    total, correct = 0.0, 0
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32)
        yb = torch.as_tensor(y[i:i + batch_size], dtype=torch.int64)
        out = model(xb)
        total += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(dim=1) == yb).sum().item()
    return total / len(X), correct / len(X)


@app.function
def fit(model, X_train, y_train, X_val, y_val, *, epochs=10, batch_size=128, lr=0.05, lr_step_epochs=4,
        grad_clip=5.0, patience=3, seed=0):
    """stage 4: SGD(lr, momentum 0.9); lr halved every lr_step_epochs EPOCHS; early stopping on validation accuracy, best weights restored; history."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sched = StepLR(opt, step_size=lr_step_epochs, gamma=0.5)  # your scheduler, counting epochs here
    stopper = EarlyStopping(patience=patience)
    loader = make_loader(X_train, y_train, batch_size=batch_size, seed=seed)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_state = None
    for epoch in range(epochs):
        history["lr"].append(get_lr(opt))
        history["train_loss"].append(train_one_epoch(model, loader, opt, grad_clip=grad_clip, scheduler=sched))
        val_loss, val_acc = evaluate(model, X_val, y_val)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        stop = stopper.step(val_acc)
        if stopper.best_step == epoch:
            best_state = copy.deepcopy(model.state_dict())
        if stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", evaluate=evaluate, fit=fit, build_model=build_model,
               train_one_epoch=train_one_epoch)
    return


@app.function
def run_pipeline(X, y, X_test, *, config, seed=0):
    """the whole pipeline (given): split, normalize (validation AND test with the training statistics), build, fit; returns (model, history, X_test normalized)."""
    X_tr, y_tr, X_val, y_val = split_data(X, y, config["n_val"], rng=seed)
    _, X_val = normalize(X_tr, X_val)
    X_tr, X_test = normalize(X_tr, X_test)
    model = build_model(hidden=config["hidden"], seed=seed)
    history = fit(model, X_tr, y_tr, X_val, y_val, epochs=config["epochs"], batch_size=config["batch_size"],
                  lr=config["lr"], lr_step_epochs=config["lr_step_epochs"], patience=config["patience"], seed=seed)
    return model, history, X_test


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", run_pipeline=run_pipeline)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    MNIST: 60,000 training and 10,000 test images, flattened to 784 pixels in $[0, 1]$. the
    pipeline holds out a validation set from the training images; the test set is used once,
    at the end. offline, `mlfs.datasets` substitutes a small synthetic stand-in so everything
    below still runs; your write-up must use the real data.
    """)
    return


@app.cell
def _():
    data = datasets.mnist()
    _n = CONFIG["n_train"] + CONFIG["n_val"]
    X_full = data["X_train"][:_n].reshape(-1, 784).astype(np.float64) / 255.0
    y_full = data["y_train"][:_n]
    X_test_raw = data["X_test"].reshape(-1, 784).astype(np.float64) / 255.0
    y_test = data["y_test"]
    mo.md(f"{'**synthetic stand-in (offline)**' if SMALL else '**real MNIST**'}: {len(X_full)} training images "
          f"(of which {CONFIG['n_val']} for validation), {len(X_test_raw)} test images · config {CONFIG}")
    return X_full, X_test_raw, y_full, y_test


@app.cell
def _(X_full, X_test_raw, y_full):
    _t0 = time.perf_counter()
    model, history, X_test = run_pipeline(X_full, y_full, X_test_raw, config=CONFIG, seed=0)
    _seconds = time.perf_counter() - _t0
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6))
    _ep = np.arange(1, len(history["train_loss"]) + 1)
    _axes[0].plot(_ep, history["train_loss"], "o-", label="train")
    _axes[0].plot(_ep, history["val_loss"], "s-", label="validation")
    _axes[0].set(xlabel="epoch", ylabel="cross-entropy", yscale="log")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(_ep, history["val_acc"], "o-")
    _axes[1].set(xlabel="epoch", ylabel="validation accuracy")
    _axes[2].step(_ep, history["lr"], where="mid")
    _axes[2].set(xlabel="epoch", ylabel="learning rate (at the start of the epoch)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"trained for {len(_ep)} epochs in **{_seconds:.1f} s**; best validation accuracy "
                           f"{max(history['val_acc']):.4f}.")])
    return X_test, model


@app.cell
def _(X_test, model, y_full, y_test):
    # the test set, used once (given): accuracy with a 95% bootstrap interval, against the majority class
    _, _acc = evaluate(model, X_test, y_test)
    model.eval()
    with torch.no_grad():
        _pred = model(torch.as_tensor(X_test, dtype=torch.float32)).argmax(1).numpy()
    _correct = (_pred == y_test).astype(float)
    _lo, _hi = bootstrap_ci(_correct, np.mean, n_boot=CONFIG["n_boot"], rng=1)
    _majority = float(np.mean(y_test == np.bincount(y_full).argmax()))
    mo.md(f"""
    | model | test accuracy | 95% bootstrap CI |
    |---|---|---|
    | the fixed pipeline, MLP {CONFIG['hidden']} | **{_acc:.4f}** | [{_lo:.4f}, {_hi:.4f}] |
    | majority class | {_majority:.4f} | |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). for
    **each of the 10 bugs**, give:

    1. **the symptom**: what you observed first (a check's message, a curve, a number, a
       warning from your `DiagnosticsLogger`), before you knew the cause;
    2. **the cause**: the line, and why it's wrong;
    3. **the fix**;
    4. **the measured effect**: re-introduce that one bug into the fixed pipeline and run the
       full config: test accuracy (with its interval) or whatever it breaks (a `nan`, a
       learning rate, a stopping epoch), next to the fixed pipeline's numbers.

    then answer:

    5. **silent versus loud.** which bugs would you have noticed without the checks, and
       how? which would have shipped? rank them by how much accuracy they cost and by how hard
       they were to see; are the two rankings the same?
    6. **the honest number.** the fixed pipeline's test accuracy with its interval, the
       majority-class baseline, and the spread over at least 3 seeds. is the seed spread
       smaller or larger than the bootstrap interval?
    7. **process.** what would you add to a training script (an assertion, a logged
       statistic, a test) so that each of these bugs fails loudly next time?
    """)
    return


if __name__ == "__main__":
    app.run()
