import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.nn import Linear, ReLU, Sequential, Sigmoid, cross_entropy, evaluate, predict  # yours, ch 14 · topic 04
    from mlfs.optim import SGD, Adam, minibatch_indices  # yours, from ch 06 · topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _P = _d["X_train"].reshape(len(_d["X_train"]), -1) / 255.0
    _X = (_P - _P.mean()) / _P.std()
    X_tr, y_tr = _X[:2000], _d["y_train"][:2000]
    X_val, y_val = _X[-1000:], _d["y_train"][-1000:]
    STEPS = 200

    BUGS = {  # the six broken scripts, and what each one gets wrong
        "lr too high": "SGD with learning rate 2.0",
        "lr too low": "Adam with learning rate 1e-7",
        "deep sigmoid": "an 8-hidden-layer sigmoid network with the default init (SGD, lr 0.1)",
        "dead init": "every hidden bias initialized to -3 (SGD, lr 0.1)",
        "shuffled labels": "a data-pipeline bug: the labels were shuffled independently of the images (Adam, 1e-3)",
        "no zero_grad": "the training loop never calls optimizer.zero_grad() (SGD, lr 0.01)",
    }


@app.function
def mlp(sizes, *, act="relu", rng=0):
    """Linear layers with ReLU (or Sigmoid) between them."""
    rng = np.random.default_rng(rng)
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(Linear(a, b, rng=rng))
        if i < len(sizes) - 2:
            layers.append(ReLU() if act == "relu" else Sigmoid())
    return Sequential(*layers)


@app.function
def run_script(bug, *, steps=STEPS, seed=0, track_val=False):
    """one training script (a healthy baseline, or one of BUGS), logged with your DiagnosticsLogger every step."""
    model, labels, zero_grad = mlp((784, 128, 128, 10), rng=seed), y_tr, True
    kind, lr = "adam", 1e-3
    if bug == "lr too high":
        kind, lr = "sgd", 2.0
    elif bug == "lr too low":
        lr = 1e-7
    elif bug == "deep sigmoid":
        model, kind, lr = mlp((784,) + (128,) * 8 + (10,), act="sigmoid", rng=seed), "sgd", 0.1
    elif bug == "dead init":
        for i in (0, 2):
            model[i].bias.data[...] = -3.0
        kind, lr = "sgd", 0.1
    elif bug == "shuffled labels":
        labels = np.random.default_rng(1).permutation(y_tr)
    elif bug == "no zero_grad":
        kind, lr, zero_grad = "sgd", 0.01, False
    elif bug == "healthy sgd":
        kind, lr = "sgd", 0.01
    opt = Adam(model.parameters(), lr=lr) if kind == "adam" else SGD(model.parameters(), lr=lr)
    logger = ex.DiagnosticsLogger(model)
    rng, t, val = np.random.default_rng(seed), 0, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the exploding run overflows
        while t < steps:
            for idx in minibatch_indices(len(X_tr), 64, rng=rng):
                if zero_grad:
                    opt.zero_grad()
                loss = cross_entropy(model(X_tr[idx]), labels[idx])
                loss.backward()
                opt.step()
                logger.log(t, loss.item(), lr=lr)
                t += 1
                if track_val and t % 20 == 0:
                    val.append((t, evaluate(model, cross_entropy, X_val, y_val)))
                if t >= steps:
                    break
        acc = float(np.mean(np.nan_to_num(predict(model, X_val)).argmax(1) == y_val))
    return logger, acc, val


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # debugging training · lab

    every experiment here runs **your** `DiagnosticsLogger` from `exercise.py` on small
    training runs: a $784 \to 128 \to 128 \to 10$ ReLU network, 200 steps of batch 64 on
    2,000 MNIST digits (standardized pixels; offline, a synthetic stand-in with the same
    shapes), scored on 1,000 held-out images. for each one: write down a prediction, lock it
    in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the whole lab takes about half a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: six broken training scripts

    six training scripts each contain one bug from this list (in a hidden order, as
    scripts **A–F**), plus a healthy baseline for comparison:

    - **lr too high**: SGD with learning rate 2.0;
    - **lr too low**: Adam with learning rate $10^{-7}$;
    - **deep sigmoid**: an 8-hidden-layer sigmoid network with the default init;
    - **dead init**: every hidden bias initialized to $-3$;
    - **shuffled labels**: the labels were shuffled independently of the images;
    - **no zero_grad**: the loop never calls `optimizer.zero_grad()`.

    for each script you see your logger's `summary()`, the validation accuracy, and (pick a
    script below) its `plot()` dashboard. diagnose each one from the evidence, then open the
    answer key.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "before looking: which warnings do you expect for each bug? will every bug produce at least one warning?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _order = list(BUGS)
    np.random.default_rng(15).shuffle(_order)
    mystery = {"healthy": run_script("healthy")}
    for _letter, _bug in zip("ABCDEF", _order):
        mystery[_letter] = run_script(_bug)
    answer_key = dict(zip("ABCDEF", _order))
    _rows = []
    for _name, (_lg, _acc, _) in mystery.items():
        _warn = "<br>".join(_lg.summary()) or "*(no warnings)*"
        _rows.append(f"| **{_name}** | {_lg.history['loss'][0]:.2f} → {_lg.history['loss'][-1]:.2f} | {_acc:.3f} | {_warn} |")
    mo.md("| script | training loss, first → last step | validation accuracy | `summary()` |\n|---|---|---|---|\n"
          + "\n".join(_rows))
    return answer_key, mystery


@app.cell
def _(mystery):
    which = mo.ui.dropdown(options=list(mystery), value="healthy", label="dashboard for script")
    which
    return (which,)


@app.cell
def _(mystery, which):
    _fig = mystery[which.value][0].plot()
    _fig.axes[0].set_title(f"script {which.value}: loss")
    _fig
    return


@app.cell
def _(answer_key):
    mo.accordion({"answer key (open after you've written your diagnoses)": mo.md(
        "| script | bug |\n|---|---|\n" + "\n".join(f"| {k} | **{v}**: {BUGS[v]} |" for k, v in answer_key.items()))})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for each script, which piece of evidence identified the bug, and why does
    that bug produce that signature? the "lr too high" run reports *both* exploding gradients
    and dead ReLUs: which came first, and how did one cause the other? two of the six scripts
    produce no warnings at all. which, and why can't a logger of gradients and weights see
    those bugs? what *other* evidence (in the table) gives them away?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: is 1e-3 really the right update ratio?

    a common rule of thumb (Karpathy) says the update of a weight matrix should be about
    $10^{-3}$ of its size per step. we train the healthy network with Adam at learning rates
    from $10^{-5}$ to $10^{-1}$ and plot the validation accuracy after 200 steps against the
    median update ratio of the middle weight matrix (`update/2.weight`).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "Adam's update ratio grows roughly in proportion to the learning rate. at which update ratio will the "
        "validation accuracy after 200 steps peak: 1e-4, 1e-3, 1e-2?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rows, _pts = [], []
    for _lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        _m = mlp((784, 128, 128, 10), rng=0)
        _opt = Adam(_m.parameters(), lr=_lr)
        _lg = ex.DiagnosticsLogger(_m)
        _rng, _t = np.random.default_rng(0), 0
        while _t < STEPS:
            for _idx in minibatch_indices(len(X_tr), 64, rng=_rng):
                _opt.zero_grad()
                _loss = cross_entropy(_m(X_tr[_idx]), y_tr[_idx])
                _loss.backward()
                _opt.step()
                _lg.log(_t, _loss.item(), lr=_lr)
                _t += 1
                if _t >= STEPS:
                    break
        _ratio = float(np.median(_lg.history["update/2.weight"]))
        _acc = float(np.mean(predict(_m, X_val).argmax(1) == y_val))
        _pts.append((_ratio, _acc, _lr))
        _rows.append(f"| {_lr:g} | {_ratio:.2e} | {_acc:.3f} | {'; '.join(w.split(':')[0] for w in _lg.summary()) or '–'} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    _ax.semilogx([p[0] for p in _pts], [p[1] for p in _pts], "o")
    for _r, _a, _l in _pts:
        _ax.annotate(f"lr {_l:g}", (_r, _a), textcoords="offset points", xytext=(5, 5), fontsize=8)
    _ax.axvline(1e-3, color="gray", ls=":", label="rule of thumb: 1e-3")
    _ax.set(xlabel="median update ratio of 2.weight", ylabel="validation accuracy after 200 steps", ylim=(0, 1))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| Adam lr | median update ratio | validation accuracy | warnings |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is Adam's update ratio almost exactly proportional to its learning
    rate (what's the size of an Adam step, per weight)? where does the best accuracy sit
    relative to $10^{-3}$? is the rule of thumb a target to hit or a range to stay inside?
    what would the same plot look like for SGD, whose step is proportional to the gradient?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: catching the silent bugs

    two bugs from experiment 1 looked healthy to the logger. here is the evidence that does
    catch them. left: the training loss and the **validation** loss (every 20 steps) for the
    healthy run and the shuffled-labels run. right: the global gradient norm, step by step,
    for SGD with and without `zero_grad`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with shuffled labels, the training loss still goes down a bit. what does the validation loss do? and without "
        "zero_grad, how does the gradient norm evolve over 200 steps?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _rows = []
    for _i, _bug in enumerate(["healthy", "shuffled labels"]):
        _lg, _acc, _val = run_script(_bug, track_val=True)
        _axes[0].plot(_lg.history["step"], _lg.history["loss"], color=f"C{_i}", alpha=0.4, lw=1, label=f"{_bug}: training")
        _axes[0].plot([v[0] for v in _val], [v[1] for v in _val], "o-", color=f"C{_i}", label=f"{_bug}: validation")
        _rows.append(f"| {_bug} | {_lg.history['loss'][-1]:.2f} | {_val[-1][1]:.2f} | {_acc:.3f} |")
    _axes[0].set(xlabel="step", ylabel="cross-entropy", title="training vs validation loss")
    _axes[0].legend(fontsize=7)
    _norms = {}
    for _i, _bug in enumerate(["healthy sgd", "no zero_grad"]):
        _lg, _acc, _ = run_script(_bug)
        _norms[_bug] = _lg.history["grad_norm"]
        _axes[1].plot(_lg.history["step"], _lg.history["grad_norm"], color=f"C{_i + 2}", label=f"{_bug} (val acc {_acc:.3f})")
    _axes[1].set(xlabel="step", ylabel="global gradient norm", title="SGD, lr 0.01")
    _axes[1].legend(fontsize=7)
    _fig.tight_layout()
    _n = {k: (v[0], v[99], v[-1]) for k, v in _norms.items()}
    mo.vstack([_fig, mo.md("| run | final training loss | final validation loss | validation accuracy |\n|---|---|---|---|\n"
                           + "\n".join(_rows) + "\n\n| run | gradient norm at step 1 | step 100 | step 200 |\n|---|---|---|---|\n"
                           + "\n".join(f"| {k} | {a:.3g} | {b:.3g} | {c:.3g} |" for k, (a, b, c) in _n.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with shuffled labels, what is the network learning when its training loss
    goes down, and why must the validation loss go *up*? why is "validation loss tracks
    training loss" the single most useful check of a data pipeline? without `zero_grad`,
    the `.grad` arrays hold the *sum* of every gradient so far: why does the gradient norm
    then grow roughly linearly with the step count, and why does training still work
    tolerably at a small learning rate (think of it as a strange kind of momentum)? what
    rule would you add to `summary()` to catch it?
    """)
    return


if __name__ == "__main__":
    app.run()
