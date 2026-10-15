import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.nn import Linear, ReLU, Sequential, cross_entropy, evaluate, fit, predict  # yours, ch 14 · topic 04
    from mlfs.optim import Adam  # yours, from ch 06 · topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _X = _d["X_train"].reshape(len(_d["X_train"]), -1) / 255.0
    _X = (_X - _X.mean()) / _X.std()
    X_tr, y_tr = _X[:1000], _d["y_train"][:1000]
    X_val, y_val = _X[1000:][-2000:], _d["y_train"][1000:][-2000:]  # never overlaps the training images


@app.function
def train(config, epochs, *, init_seed=0, data_seed=0):
    """a 784 -> width -> 10 ReLU network, Adam(lr), batch 32; returns (validation accuracy, validation loss)."""
    rng = np.random.default_rng(init_seed)
    net = Sequential(Linear(784, config["width"], rng=rng), ReLU(), Linear(config["width"], 10, rng=rng))
    fit(net, cross_entropy, Adam(net.parameters(), lr=config["lr"]), X_tr, y_tr, epochs=epochs, batch_size=32,
        rng=data_seed)
    return float(np.mean(predict(net, X_val).argmax(1) == y_val)), float(evaluate(net, cross_entropy, X_val, y_val))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hyperparameters & experiments · lab

    every experiment here uses **your** `ExperimentLog`, `random_search` and `hyperband`
    from `exercise.py`. the model is small on purpose, so that we can afford many runs: a
    $784 \to \text{width} \to 10$ ReLU network trained with Adam (batch 32) on 1,000 MNIST
    digits, scored on 2,000 held-out ones (offline: a synthetic stand-in, 1,000 held out).
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare your explanations with `solutions/lab_notes.md` afterwards. the whole
    lab takes about a minute and a half.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: two learning rates, ten seeds each

    a colleague tuned the learning rate and reports that $2 \times 10^{-3}$ beats
    $10^{-3}$. we train both (width 64, 3 epochs) with 10 seeds each (the seed sets the
    initialization and the minibatch order), log every run in your `ExperimentLog`, and ask
    how often a *single* pair of runs, one per learning rate, would have ranked them the same
    way as the 10-seed means.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the two learning rates' mean accuracies differ by a fraction of a percentage point. how large is the "
        "seed-to-seed spread (max - min over 10 seeds) for ONE learning rate? how often does a single-seed comparison "
        "pick the wrong winner?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    seed_log = ex.ExperimentLog()
    for _lr in [1e-3, 2e-3]:
        for _s in range(10):
            _acc, _loss = train({"lr": _lr, "width": 64}, 3, init_seed=_s, data_seed=_s)
            seed_log.log({"lr": _lr, "width": 64}, {"val_acc": _acc, "val_loss": _loss}, seed=_s)
    _groups = seed_log.group("val_acc")
    _vals = {g["config"]["lr"]: [r["metrics"]["val_acc"] for r in seed_log.runs if r["config"]["lr"] == g["config"]["lr"]]
             for g in _groups}
    _a, _b = np.array(_vals[1e-3]), np.array(_vals[2e-3])
    _winner_by_mean = np.sign(_b.mean() - _a.mean())
    _agree = np.mean([np.sign(b - a) == _winner_by_mean for a in _a for b in _b])
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    for _i, (_lr, _v) in enumerate(_vals.items()):
        _ax.plot(_v, np.full(len(_v), _i) + np.linspace(-0.1, 0.1, len(_v)), "o", alpha=0.7)
        _ax.plot([np.mean(_v)] * 2, [_i - 0.25, _i + 0.25], "k-", lw=2)
    _ax.set(yticks=[0, 1], yticklabels=[f"lr {lr:g}" for lr in _vals], xlabel="validation accuracy (10 seeds; bar = mean)")
    _fig.tight_layout()
    _rows = "\n".join(f"| {g['config']['lr']:g} | {g['n']} | {g['mean']:.4f} | {g['std']:.4f} | "
                      f"{min(_vals[g['config']['lr']]):.3f} – {max(_vals[g['config']['lr']]):.3f} |" for g in _groups)
    mo.vstack([_fig, mo.md("| learning rate | runs | mean accuracy | std over seeds | range |\n|---|---|---|---|---|\n" + _rows
                           + f"\n\ndifference of the means: {_b.mean() - _a.mean():+.4f}. of the 100 possible single-seed "
                           f"pairs, **{_agree:.0%}** rank the learning rates the same way as the means.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compare the seed-to-seed spread with the difference you're trying to
    detect. what does that say about your colleague's claim, if it rests on one run each? how
    many seeds would you need for the standard error of each mean ($\text{std}/\sqrt n$) to be
    a quarter of the difference? what *else* adds noise to the reported accuracy, even for a
    fixed trained model (how large is the binomial standard error of an accuracy near 0.9 on
    2,000 images)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: where does the variance come from?

    "the seed" controls two separate random choices: the **initial weights** and the
    **minibatch order**. we train the lr $10^{-3}$ network 10 times varying only the
    initialization, 10 times varying only the data order, and 10 times varying both.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which source of randomness causes more of the spread: the initial weights or the order of the minibatches? "
        "does varying both give roughly the sum of the two variances?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _cfg = {"lr": 1e-3, "width": 64}
    _res = {
        "initialization only": [train(_cfg, 3, init_seed=s, data_seed=0)[0] for s in range(10)],
        "data order only": [train(_cfg, 3, init_seed=0, data_seed=s)[0] for s in range(10)],
        "both": [train(_cfg, 3, init_seed=s, data_seed=100 + s)[0] for s in range(10)],
    }
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.boxplot(list(_res.values()), vert=False)
    _ax.set_yticks([1, 2, 3], list(_res))
    _ax.set(xlabel="validation accuracy (10 runs each)")
    _fig.tight_layout()
    _rows = "\n".join(f"| {k} | {np.mean(v):.4f} | {np.std(v, ddof=1):.4f} | {np.var(v, ddof=1) * 1e5:.2f} |"
                      for k, v in _res.items())
    _se = np.sqrt(0.9 * 0.1 / len(y_val))
    mo.vstack([_fig, mo.md("| what varies | mean | std | variance (×1e-5) |\n|---|---|---|---|\n" + _rows
                           + f"\n\nfor comparison, the binomial standard error of one accuracy near 0.9 measured on "
                           f"{len(y_val)} images is {_se:.4f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** both sources produce a spread of the same order. why should the
    *order of the minibatches* change the final model at all, after 3 full epochs over the
    same data? why is "both" not simply the sum of the two variances (with 10 runs each, how
    precisely can you even estimate a variance?)? when you report a result, which seeds
    would you vary, and how would you report the spread?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: hyperband versus random search, same compute

    we search over the learning rate (log-uniform from $3 \times 10^{-5}$ to $10^{-1}$) and
    the width (16, 32, 64 or 128), minimizing the validation loss, with the budget measured
    in **epochs**. your `hyperband` with `max_budget=9, eta=3` spends 78 epochs; random search
    gets as many full 9-epoch runs as fit in the same budget (8 runs, 72 epochs). we repeat the
    whole comparison with 3 different search seeds, then **re-train each search's winner with
    5 fresh seeds**, to see how much of its advantage was luck.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "hyperband tries 17 configurations for the price of random search's 8. will it find a better configuration "
        "every time? after re-training the winners with fresh seeds, will the gap between them survive?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    def _sample(rng):
        return {"lr": float(10 ** rng.uniform(-4.5, -1)), "width": int(rng.choice([16, 32, 64, 128]))}

    def _evaluate(config, budget):
        return train(config, int(round(budget)), init_seed=0, data_seed=0)[1]

    _rows, _pts = [], []
    for _seed in range(3):
        _hb_cfg, _hb_loss, _hb_hist = ex.hyperband(_sample, _evaluate, max_budget=9, eta=3, rng=_seed)
        _spent = sum(h[1] for h in _hb_hist)
        _rs_cfg, _rs_loss, _rs_hist = ex.random_search(_sample, _evaluate, n_trials=int(_spent // 9), budget=9,
                                                       rng=1000 + _seed)
        _fresh = {}
        for _name, _cfg in [("hyperband", _hb_cfg), ("random", _rs_cfg)]:
            _fresh[_name] = np.mean([train(_cfg, 9, init_seed=s, data_seed=s)[1] for s in range(1, 6)])
        _n_hb = len({str(h[0]) for h in _hb_hist})
        _rows.append(f"| {_seed} | {_n_hb} configs, {_spent:g} epochs: lr {_hb_cfg['lr']:.1e}, width {_hb_cfg['width']} → "
                     f"**{_hb_loss:.3f}** (fresh: {_fresh['hyperband']:.3f}) | {len(_rs_hist)} configs, "
                     f"{len(_rs_hist) * 9} epochs: lr {_rs_cfg['lr']:.1e}, width {_rs_cfg['width']} → **{_rs_loss:.3f}** "
                     f"(fresh: {_fresh['random']:.3f}) |")
        _pts.append((_hb_loss, _rs_loss, _fresh["hyperband"], _fresh["random"]))
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _x = np.arange(3)
    _ax.bar(_x - 0.2, [p[0] for p in _pts], 0.18, label="hyperband: search loss")
    _ax.bar(_x, [p[2] for p in _pts], 0.18, label="hyperband: fresh seeds")
    _ax.bar(_x + 0.2, [p[1] for p in _pts], 0.18, label="random: search loss")
    _ax.bar(_x + 0.4, [p[3] for p in _pts], 0.18, label="random: fresh seeds")
    _ax.set(xticks=_x + 0.1, xticklabels=[f"search seed {s}" for s in range(3)], ylabel="validation loss of the winner",
            ylim=(0.9 * min(min(p) for p in _pts), 1.1 * max(max(p) for p in _pts)))  # zoomed in: the gaps are small
    _ax.legend(fontsize=7, ncol=2)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| search seed | hyperband's winner | random search's winner |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how does hyperband try twice as many configurations for the same
    compute, and what's the risk in judging a configuration after 1 epoch? why is a winner's
    loss *during the search* an optimistic estimate of its loss with fresh seeds (think of the
    maximum of several noisy measurements)? given experiment 1, how large a difference between
    the two searches' winners would you need before calling one search better?
    """)
    return


if __name__ == "__main__":
    app.run()
