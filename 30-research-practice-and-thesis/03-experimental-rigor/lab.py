import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import datasets, notebooks, ui
    from mlfs.torchnn import MLP, evaluate, fit, make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST from the cache, or a small synthetic stand-in when offline
    X_all = _d["X_train"].reshape(len(_d["X_train"]), -1).astype(np.float32) / 255.0
    y_all = _d["y_train"]
    X_test = _d["X_test"].reshape(len(_d["X_test"]), -1).astype(np.float32) / 255.0
    y_test = _d["y_test"]
    N_VAL = min(10_000, len(X_all) // 5)
    POOL = np.arange(len(X_all) - N_VAL)  # training images are drawn from here
    VAL = np.arange(len(X_all) - N_VAL, len(X_all))  # a validation block that never overlaps the pool
    N_TRAIN = min(5_000, len(POOL) // 2)
    MU, SD = float(X_all.mean()), float(X_all.std())


@app.function
def train_mlp(*, init_seed=0, order_seed=0, subset_seed=0, lr=0.05, momentum=0.9, standardize=True, width=128,
              optimizer="sgd", epochs=3, split="test"):
    """a 784 -> width -> 10 ReLU MLP on N_TRAIN images; returns accuracy on the test set (or the validation block).

    three seeds, each controlling one source of randomness: init_seed the weights, order_seed the minibatch
    order, subset_seed which N_TRAIN images of the pool it trains on.
    """
    idx = np.random.default_rng(subset_seed).choice(POOL, N_TRAIN, replace=False)

    def prep(x):
        return (x - MU) / SD if standardize else x

    torch.manual_seed(init_seed)
    model = MLP(784, [width], 10)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    fit(model, F.cross_entropy, opt, make_loader(prep(X_all[idx]), y_all[idx], batch_size=64, seed=order_seed),
        epochs=epochs)
    Xe, ye = (X_test, y_test) if split == "test" else (X_all[VAL], y_all[VAL])
    return evaluate(model, make_loader(prep(Xe), ye, batch_size=2000, shuffle=False))["accuracy"]


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # experimental rigor · lab

    every experiment here runs **your** `mlfs.research` functions from `exercise.py` on a
    deliberately small model: a $784 \to 128 \to 10$ ReLU MLP (your ch 15 `MLP`, trained
    with your `fit`) on {N_TRAIN:,} MNIST training images for 3 epochs, scored on the
    {len(X_test):,}-image test set. one run takes well under a second on a laptop CPU,
    so the lab can afford a few hundred of them (offline it uses a small synthetic stand-in
    and the numbers mean little). for each experiment: write down a prediction, lock it in,
    run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: where does the variance come from?

    the network has three sources of randomness, each with its own seed: the initial
    weights, the minibatch order, and which 5,000 of the 50,000 pool images it trains on.
    we vary one at a time (the other two fixed at seed 0), then all three together, and
    look at the spread of test accuracy.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which single source moves test accuracy the most: init, data order, or the training subset? and how "
        "does the spread with all three varying compare with the largest single one (about equal, the sum, more)?"
    )
    p1
    return (p1,)


@app.cell
def _():
    n_seeds = mo.ui.slider(5, 30, value=20, step=1, label="seeds per source")
    n_seeds
    return (n_seeds,)


@app.cell
def _(n_seeds, p1):
    ui.gate(p1)
    _seeds = range(n_seeds.value)
    sources = {
        "init only": ex.run_seeds(lambda s: train_mlp(init_seed=s), seeds=_seeds),
        "order only": ex.run_seeds(lambda s: train_mlp(order_seed=s), seeds=_seeds),
        "subset only": ex.run_seeds(lambda s: train_mlp(subset_seed=s), seeds=_seeds),
        "all three": ex.run_seeds(lambda s: train_mlp(init_seed=s, order_seed=s, subset_seed=s), seeds=_seeds),
    }
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _jit = np.random.default_rng(0)
    _rows = []
    for _i, (_name, _v) in enumerate(sources.items()):
        _ax.scatter(_i + _jit.uniform(-0.12, 0.12, len(_v)), 100 * _v, s=16, alpha=0.7)
        _m, _lo, _hi = ex.mean_ci(_v)
        _ax.errorbar(_i + 0.3, 100 * _m, yerr=[[100 * (_m - _lo)], [100 * (_hi - _m)]], fmt="k_", capsize=4)
        _rows.append(f"| {_name} | {100 * _m:.2f} [{100 * _lo:.2f}, {100 * _hi:.2f}] | {100 * _v.std(ddof=1):.2f} | "
                     f"{100 * (_v.max() - _v.min()):.2f} |")
    _ax.set_xticks(range(len(sources)), list(sources))
    _ax.set_ylabel("test accuracy (%)")
    _ax.set_title(f"one dot per run, {n_seeds.value} seeds per source; black: mean and 95% t interval")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| source | mean accuracy, 95% CI (%) | sd of one run (points) | max − min (points) |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return (sources,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which source mattered most, and is the gap between sources bigger than the
    uncertainty in the standard deviations themselves (with 20 runs, an sd is known to about
    ±16%)? why isn't the "all three" spread the sum of the three? what does this say about a
    paper that "fixes the seed for reproducibility" and reports one run?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the single-seed ablation

    the recipe has three parts worth ablating: momentum 0.9 (removing it leaves plain SGD at
    the same learning rate), input standardization (removing it feeds raw pixels in
    $[0, 1]$), and width 128 (the ablation shrinks it to 32). we run the full recipe and the
    three ablations on the same 20 seeds (each seed varies all three sources), build the
    ablation table with your `ablation_table`, and then ask: if we had run every row once,
    with one seed, how often would that table have ranked the three ablations in the same
    order as the 20-seed table?
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "out of 20 single-seed ablation tables (one per seed), how many rank the three ablations (no momentum, "
        "no standardization, width 32) in the same order of importance as the 20-seed table?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _configs = {"full": {}, "no momentum": {"momentum": 0.0}, "no standardization": {"standardize": False},
                "width 32": {"width": 32}}
    ablation = {name: ex.run_seeds(lambda s, c=c: train_mlp(init_seed=s, order_seed=s, subset_seed=s, **c),
                                   seeds=range(20))
                for name, c in _configs.items()}
    _ablated = [n for n in ablation if n != "full"]
    _drops = {n: ablation["full"] - ablation[n] for n in _ablated}  # per-seed cost of removing each part
    _order = sorted(_ablated, key=lambda n: -_drops[n].mean())
    _single = [sorted(_ablated, key=lambda n: -_drops[n][s]) for s in range(20)]
    agree = sum(o == _order for o in _single)
    _first = sum(o[0] == _order[0] for o in _single)
    mo.vstack([
        mo.md(ex.ablation_table(ablation, baseline="full")),
        mo.md(f"20-seed order of importance: **{' > '.join(_order)}**. single-seed tables that agree on the "
              f"whole order: **{agree} of 20**; that agree on the most important part: {_first} of 20."),
        mo.md("| seed | " + " | ".join(_ablated) + " | single-seed order |\n|---|---|---|---|---|\n" + "\n".join(
            f"| {s} | " + " | ".join(f"{-100 * _drops[n][s]:+.2f}" for n in _ablated) + f" | {' > '.join(_single[s])} |"
            for s in range(6))),
        mo.md("*(first six seeds shown: the change in accuracy, in points, from removing each part.)*"),
    ])
    return (ablation,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which pair of ablations does a single seed confuse, and why that pair? look
    at the Δ column's intervals: is the paired difference *between* those two ablations
    resolved at 20 seeds? (compute it with your `compare_runs`.) what would you write in a
    paper that had run each row once?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: who wins depends on the tuning budget

    SGD with momentum 0.9 and Adam, each tuned over the learning rate by random search
    (log-uniform in $[10^{-4}, 1]$, 30 trials each, 2 epochs per trial, scored on the
    validation block, not the test set). your `expected_best` turns the 30 trials into
    the expected best validation accuracy at every budget $k = 1, \dots, 30$; the band
    is a bootstrap over the trials.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with 3 trials each, which optimizer has the higher expected best validation accuracy? with 30 each? "
        "and if SGD gets 5 trials and Adam 30, who 'wins', and by how much?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(0)
    trials = {}
    for _opt in ("sgd", "adam"):
        _lrs = 10 ** _rng.uniform(-4, 0, 30)
        trials[_opt] = np.array([train_mlp(init_seed=i, order_seed=i, lr=lr, optimizer=_opt, epochs=2, split="val")
                                 for i, lr in enumerate(_lrs)])
    return (trials,)


@app.cell
def _():
    k_sgd = mo.ui.slider(1, 30, value=5, label="SGD's budget (trials)")
    k_adam = mo.ui.slider(1, 30, value=30, label="Adam's budget (trials)")
    mo.hstack([k_sgd, k_adam])
    return k_adam, k_sgd


@app.cell
def _(k_adam, k_sgd, trials):
    _ks = np.arange(1, 31)
    _boot = np.random.default_rng(1)
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.8))
    curves, _bands = {}, {}
    for _opt in ("sgd", "adam"):
        curves[_opt] = np.asarray(ex.expected_best(trials[_opt], _ks))
        _bs = np.array([ex.expected_best(_boot.choice(trials[_opt], len(trials[_opt])), _ks) for _ in range(300)])
        _bands[_opt] = np.quantile(_bs, [0.05, 0.95], axis=0)
    for _ax, _from in zip(_axes, (1, 3)):
        _sel = _ks >= _from
        for _opt, _label in (("sgd", "SGD + momentum"), ("adam", "Adam")):
            _ax.fill_between(_ks[_sel], 100 * _bands[_opt][0, _sel], 100 * _bands[_opt][1, _sel], alpha=0.2)
            _ax.plot(_ks[_sel], 100 * curves[_opt][_sel], label=_label)
        _ax.set_xscale("log")
        _ax.minorticks_off()
        _ticks = [t for t in (1, 2, 3, 5, 10, 20, 30) if t >= _from]
        _ax.set_xticks(_ticks, [str(t) for t in _ticks])
        _ax.set(xlabel="tuning budget k (random-search trials)", ylabel="expected best val. accuracy (%)")
    _axes[0].set_title("all budgets")
    _axes[1].set_title("zoom: k ≥ 3")
    _axes[0].legend()
    _fig.suptitle("expected best of k trials; bands: 90% bootstrap over the 30 trials")
    _fig.tight_layout()
    _s, _a = curves["sgd"][k_sgd.value - 1], curves["adam"][k_adam.value - 1]
    _cross = next((int(k) for k in _ks if curves["sgd"][k - 1] > curves["adam"][k - 1]), None)
    mo.vstack([_fig, mo.md(
        f"at the budgets you chose: SGD ({k_sgd.value} trials) **{100 * _s:.2f}%**, Adam ({k_adam.value} trials) "
        f"**{100 * _a:.2f}%**, a gap of {100 * (_a - _s):+.2f} points for Adam. at equal budgets: "
        f"3 trials {100 * (curves['adam'][2] - curves['sgd'][2]):+.2f}, 30 trials "
        f"{100 * (curves['adam'][29] - curves['sgd'][29]):+.2f} points for Adam. "
        + (f"SGD's curve first passes Adam's at k = {_cross}." if _cross else "SGD's curve never passes Adam's.")
    )])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is Adam ahead at small budgets and (possibly) behind at large ones? what
    does the $k = 1$ end of each curve measure? a paper compares "our method (tuned)" with "the
    baseline (default hyperparameters)": which two points on these curves is that comparing?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done

    you've now seen, on a problem small enough to run 200 times, the three ways a single
    number lies: the seed, the single-seed table, and the tuning budget. topic 04 asks how
    many seeds you need *before* you run, and topic 05 how to show all of this in a figure.
    """)
    return


if __name__ == "__main__":
    app.run()
