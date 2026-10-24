import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import stats as sps

    from mlfs import datasets, notebooks, ui
    from mlfs.torchnn import MLP, evaluate, fit, make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST from the cache, or a small synthetic stand-in when offline
    X_all = _d["X_train"].reshape(len(_d["X_train"]), -1).astype(np.float32) / 255.0
    y_all = _d["y_train"]
    X_test = _d["X_test"].reshape(len(_d["X_test"]), -1).astype(np.float32) / 255.0
    y_test = _d["y_test"]
    MU, SD = float(X_all.mean()), float(X_all.std())
    N_TRAIN = min(5_000, len(X_all) // 2)


@app.function
def one_epoch(optimizer, lr, seed):
    """test accuracy of a 784 -> 128 -> 10 MLP after one epoch on N_TRAIN images (seed: subset, init and order)."""
    idx = np.random.default_rng(seed).choice(len(X_all), N_TRAIN, replace=False)
    torch.manual_seed(seed)
    model = MLP(784, [128], 10)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    fit(model, F.cross_entropy, opt, make_loader((X_all[idx] - MU) / SD, y_all[idx], batch_size=64, seed=seed),
        epochs=1)
    return evaluate(model, make_loader((X_test - MU) / SD, y_test, batch_size=2000, shuffle=False))["accuracy"]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # reproducing papers · lab

    every experiment here uses **your** verdict functions and report generator from
    `exercise.py`. the first two are simulations of reproductions (fast, exact, and with a
    known truth); the third is a small real one on MNIST. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the lab takes about ten seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: "the paper's number is inside my interval"

    the most common informal criterion for a successful reproduction is that the paper's
    number falls inside your confidence interval. we simulate reproductions whose true value
    is off from the paper's by 0, 1 or 2 margins, with seed-to-seed noise equal to one margin,
    using 3 to 20 seeds, and compare how often that naive rule and your equivalence verdict
    say "reproduced". everything is in units of the margin.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the true value is 2 margins away from the paper's (a failed reproduction). with 3 seeds, how often does "
        "the naive rule ('the paper's number is in my 95% interval') say reproduced? with 20 seeds?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _ns = (3, 5, 10, 20)
    naive_rates, tost_rates = {}, {}
    for _delta in (0.0, 1.0, 2.0):
        for _n in _ns:
            _naive = _tost = 0
            for _ in range(2000):
                _runs = _rng.normal(_delta, 1.0, _n)
                _half = sps.t.ppf(0.975, _n - 1) * _runs.std(ddof=1) / np.sqrt(_n)
                _naive += abs(_runs.mean()) <= _half
                _tost += ex.equivalence_verdict(_runs, 0.0, margin=1.0)["verdict"] == "reproduced"
            naive_rates[_delta, _n] = _naive / 2000
            tost_rates[_delta, _n] = _tost / 2000
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for _ax, _rates, _title in ((_axes[0], naive_rates, "naive: paper value in my 95% CI"),
                                (_axes[1], tost_rates, "your equivalence_verdict")):
        for _delta, _ls in ((0.0, "-"), (1.0, "--"), (2.0, ":")):
            _ax.plot(_ns, [_rates[_delta, n] for n in _ns], _ls, marker="o", label=f"truth {_delta:g} margins off")
        _ax.set(xlabel="seeds", title=_title, xscale="log")
        _ax.set_xticks(_ns, [str(n) for n in _ns])
        _ax.minorticks_off()
    _axes[0].set_ylabel('P(verdict = "reproduced")')
    _axes[0].legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| truth off by | rule | 3 seeds | 5 | 10 | 20 |\n|---|---|---|---|---|---|\n" + "\n".join(
            f"| {dl:g} margins | {name} | " + " | ".join(f"{r[dl, n]:.2f}" for n in _ns) + " |"
            for dl in (0.0, 1.0, 2.0) for name, r in (("naive", naive_rates), ("equivalence", tost_rates))))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the naive rule "reproduce" a wrong result *more* often with fewer
    seeds? what does your equivalence verdict say instead with 3 seeds, and why is that the
    honest answer? what's the price the equivalence verdict pays when the reproduction is
    actually right?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: is `seeds_needed` honest?

    `seeds_needed` comes from an approximation (it treats the sample standard deviation as if
    it were the true one). we check it by simulation: for several ratios of seed noise to
    margin, run 4,000 simulated reproductions of a result that *is* right (true value = the
    paper's), with the number of seeds it recommends and with one fewer, and count how often
    your equivalence verdict says "reproduced". the target is 80%.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with seed noise equal to the margin (sd / margin = 1), how many seeds does seeds_needed recommend, and "
        "how close to 80% will the simulated power be?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _rows = []
    for _ratio in (0.25, 0.5, 1.0, 2.0):
        _n = ex.seeds_needed(_ratio, 1.0)
        _pw = []
        for _nn in (_n - 1, _n):
            if _nn < 2:
                _pw.append(float("nan"))
                continue
            _hits = sum(ex.equivalence_verdict(_rng.normal(0.0, _ratio, _nn), 0.0, margin=1.0)["verdict"]
                        == "reproduced" for _ in range(4000))
            _pw.append(_hits / 4000)
        _rows.append(f"| {_ratio:g} | {_n} | {_pw[1]:.3f} | {_n - 1} | {_pw[0]:.3f} |")
    mo.md("| sd / margin | seeds_needed | simulated power | one fewer seed | its power |\n|---|---|---|---|---|\n"
          + "\n".join(_rows) + "\n\n*(4,000 simulated reproductions per cell; Monte Carlo error about ±0.01.)*")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where does the approximation err, and in which direction? if your pilot
    runs give a seed-to-seed sd of 0.6 points and you want to confirm a paper's number within
    ±0.5 points, how many seeds should you budget, and what happens if you budget 5?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the unreported learning rate

    a reproduction on real data. a **hypothetical** paper (invented for this lab) claims:
    *"Adam (learning rate $10^{-3}$) beats SGD with momentum 0.9 by 4 points of test accuracy
    after one epoch"* on 5,000 MNIST images with a $784 \to 128 \to 10$ MLP. the paper gives
    every detail except SGD's learning rate. your claims table, written before the runs:
    reproduced if the 90% interval of the paired difference (5 seeds) lies within
    4 ± 1.5 points. the gap list has one entry: SGD's learning rate. we try five guesses.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "across SGD learning rates of 0.001, 0.003, 0.01, 0.03 and 0.1, how many different verdicts (reproduced, "
        "partly, direction only, not, inconclusive) will the same claim get? which guess 'reproduces' the paper?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    seeds = range(5)
    adam_runs = np.array([one_epoch("adam", 1e-3, s) for s in seeds])
    sgd_runs = {lr: np.array([one_epoch("sgd", lr, s) for s in seeds]) for lr in (0.001, 0.003, 0.01, 0.03, 0.1)}
    verdicts3 = {lr: ex.effect_verdict(adam_runs, r, 0.04, margin=0.015) for lr, r in sgd_runs.items()}
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _lrs = list(verdicts3)
    _ax.axhspan(100 * 0.025, 100 * 0.055, color="tab:green", alpha=0.15, label="the paper's claim: 4 ± 1.5")
    _ax.axhline(0, color="k", lw=0.8)
    for _lr in _lrs:
        _v = verdicts3[_lr]
        _ax.errorbar(_lr, 100 * _v["diff"], yerr=[[100 * (_v["diff"] - _v["ci"][0])], [100 * (_v["ci"][1] - _v["diff"])]],
                     fmt="o", color="tab:blue", capsize=4)
        _ax.annotate(_v["verdict"].replace(", ", ",\n"), (_lr, 100 * _v["ci"][1]), textcoords="offset points",
                     xytext=(0, 6),
                     ha="center", fontsize=8)
    _ax.set(xscale="log", xlabel="SGD learning rate (the unreported detail)",
            ylabel="Adam − SGD accuracy (points)", title="paired difference over 5 seeds, 90% t intervals")
    _ax.legend(loc="lower left")
    _ax.margins(y=0.2)
    _ax.set_xlim(4e-4, 2.5e-1)  # room for the labels at both ends
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Adam alone: {100 * adam_runs.mean():.2f}% (sd {100 * adam_runs.std(ddof=1):.2f}). "
                           + "; ".join(f"SGD at {lr:g}: {100 * r.mean():.2f}%" for lr, r in sgd_runs.items()))])
    return adam_runs, sgd_runs


@app.cell
def _(adam_runs, sgd_runs):
    _claims = [{"id": "C1", "kind": "effect", "statement": "Adam beats SGD with momentum by 4 points after one epoch",
                "evidence": "the hypothetical paper's table 1", "metric": "test accuracy", "paper": 0.04,
                "margin": 0.015}]
    _md, _ = ex.reproduction_report(
        "a reproduction of a hypothetical optimizer comparison", _claims, {"C1": (adam_runs, sgd_runs[0.01])},
        gaps=[{"detail": "SGD learning rate", "choice": "0.01",
               "reason": "a common default; the paper doesn't say (experiment 3 sweeps it)"}],
        checklist={"claims": "C1, fixed before the runs", "gaps": "one gap, swept in experiment 3",
                   "seeds": "5 paired seeds (0-4): each fixes the subset, init and order for both optimizers",
                   "stats": "paired 90% t interval (an equivalence test at alpha = 0.05)"})
    mo.vstack([mo.md("**the report your `reproduction_report` writes for the 0.01 guess:**"), mo.md(_md)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the same claim, the same code, five guesses and four different verdicts,
    depending on one number the paper didn't report. which verdict would you publish, and what should the report say about
    the others? why does the learning rate that "reproduces" the paper not make the paper
    right?
    """)
    return


if __name__ == "__main__":
    app.run()
