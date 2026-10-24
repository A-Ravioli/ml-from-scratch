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
    from mlfs.research import mean_ci  # yours, from topic 03
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

    DRAFT = r"""
    \begin{document}
    \begin{abstract}
    We show that wider networks generalize better on MNIST.
    \end{abstract}
    \section{Results}\label{sec:results}
    Figure~\ref{fig:width} compares the two widths over five seeds; table~\ref{tab:main} gives the numbers.
    The effect is consistent with prior work \citep{zhang2017, neyshabur2019}.
    \begin{figure}
      \label{fig:width}
      \caption{Test accuracy against width.}
    \end{figure}
    \begin{figure}
      \caption{Training curves.}\label{fig:curves}
    \end{figure}
    \begin{table}
      \caption{Main results.}\label{tab:mian}
    \end{table}
    \begin{itemize}
      \item wider is better (see \ref{sec:result}).
    \end{document}
    """
    DRAFT_BIB = ["zhang2017"]


@app.function
def mnist_run(seed, width):
    """test accuracy (%) of a 784 -> width -> 10 MLP after 3 epochs of SGD on N_TRAIN images."""
    idx = np.random.default_rng(seed).choice(len(X_all), N_TRAIN, replace=False)
    torch.manual_seed(seed)
    model = MLP(784, [width], 10)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    fit(model, F.cross_entropy, opt, make_loader((X_all[idx] - MU) / SD, y_all[idx], batch_size=64, seed=seed),
        epochs=3)
    return 100 * evaluate(model, make_loader((X_test - MU) / SD, y_test, batch_size=2000, shuffle=False))["accuracy"]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # writing & reviewing · lab

    every experiment here uses **your** tools from `exercise.py`. the first draws the same
    result three ways and lints each figure; the second simulates conference reviewing to see
    how much luck is in an accept decision; the third checks a short LaTeX draft. for each
    one: write down a prediction, lock it in, run it, then explain what happened. compare
    your explanations with `solutions/lab_notes.md` afterwards. the lab takes about ten
    seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: one result, three figures

    a real result: a $784 \to 128 \to 10$ MLP against the same MLP with 256 hidden units, 3
    epochs on 5,000 MNIST images, 5 paired seeds each. we draw it the way many papers do (a
    bar chart zoomed in on the bars' tops), the honest version of that (bars from 0 with
    intervals), and the version this topic recommends for paired seeds (the per-seed
    differences with their interval). your `lint_figure` checks each one, and we compute
    Tufte's **lie factor**: the size of the effect shown in the graphic divided by its size in
    the data.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the wider network is better by a few tenths of a point. in the zoomed bar chart, how many times bigger "
        "does the difference look than it is (the lie factor)? and how many problems will your linter find in it?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _seeds = range(5)
    _narrow = np.array([mnist_run(s, 128) for s in _seeds])
    _wide = np.array([mnist_run(s, 256) for s in _seeds])
    _d = _wide - _narrow
    _m, _lo, _hi = mean_ci(_d)
    _means = [_narrow.mean(), _wide.mean()]
    _bottom = min(_means) - 0.15

    _fig_a, _ax = plt.subplots(figsize=(3.2, 3))
    _ax.bar(["128", "256"], _means, color=["tab:gray", "tab:red"])
    _ax.set_ylim(_bottom, max(_means) + 0.05)
    _ax.tick_params(labelsize=7)
    _ax.set_title("ours is better!", fontsize=9)
    _fig_a.tight_layout()

    _fig_b, _ax = plt.subplots(figsize=(3.2, 3))
    _ci = [mean_ci(_narrow), mean_ci(_wide)]
    _ax.bar(["128", "256"], _means, yerr=[[m - lo for m, lo, _ in _ci], [hi - m for m, _, hi in _ci]],
            capsize=4, color=["tab:gray", "tab:blue"])
    _ax.set(xlabel="hidden units", ylabel="test accuracy (%)", title="means, 95% t intervals (5 seeds)")
    _fig_b.tight_layout()

    _fig_c, _ax = plt.subplots(figsize=(3.2, 3))
    _ax.axhline(0, color="k", lw=0.8)
    _ax.plot(np.zeros(5) + np.linspace(-0.08, 0.08, 5), _d, "o", color="tab:blue", label="one seed")
    _ax.errorbar([0.3], [_m], yerr=[[_m - _lo], [_hi - _m]], fmt="s", color="k", capsize=4, label="mean, 95% CI")
    _ax.set(xlim=(-0.4, 0.7), xticks=[], xlabel="paired seeds", ylabel="256 − 128 units (accuracy points)",
            title="paired differences")
    _ax.legend(fontsize=8)
    _fig_c.tight_layout()

    lint_counts = {name: ex.lint_figure(f) for name, f in (("zoomed bars", _fig_a), ("bars from 0", _fig_b),
                                                            ("paired differences", _fig_c))}
    _data_change = (_means[1] - _means[0]) / _means[0]
    _shown_change = ((_means[1] - _bottom) - (_means[0] - _bottom)) / (_means[0] - _bottom)
    lie_factor = _shown_change / _data_change
    mo.vstack([
        mo.hstack([_fig_a, _fig_b, _fig_c]),
        mo.md(f"128 units: {_means[0]:.2f}%, 256 units: {_means[1]:.2f}%; paired difference {_m:+.2f} points "
              f"(95% CI [{_lo:+.2f}, {_hi:+.2f}]). the zoomed chart's **lie factor: {lie_factor:.0f}** (the second "
              f"bar looks {100 * _shown_change:.0f}% taller; the data differ by {100 * _data_change:.2f}%)."),
        mo.md("\n".join(f"- **{name}**: {len(p)} problem{'s' if len(p) != 1 else ''}"
                        + "".join(f"\n    - `{x}`" for x in p) for name, p in lint_counts.items())),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which of your linter's rules does the zoomed chart break, and which of its
    flaws can no linter catch? the honest bar chart shows the two means as nearly identical:
    why is it still a poor figure for this result, and what does the paired-difference plot
    show that neither bar chart can?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how much luck is in an accept decision?

    in 2014, NeurIPS had about a tenth of its submissions reviewed by two independent
    committees (Cortes & Lawrence, 2021). the committees disagreed on about a quarter of those
    papers, which, at an acceptance rate around a quarter, means that roughly half of the
    papers one committee accepted, the other rejected. we simulate it: 2,000 papers with a
    true quality $q \sim N(0, 1)$; each committee scores a paper by averaging 3 reviewers, each
    of whom sees $q$ plus independent noise of standard deviation $\sigma$; each accepts its top
    25%. your `committee_agreement` measures how often they agree.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "how noisy must each reviewer be (sigma, in units of the spread of true quality) for half of one committee's "
        "accepted papers to be rejected by the other?"
    )
    p2
    return (p2,)


@app.cell
def _():
    sigma = mo.ui.slider(0.0, 3.0, value=1.0, step=0.25, label="σ: one reviewer's noise (sd, in units of quality sd)")
    sigma
    return (sigma,)


@app.cell
def _(p2, sigma):
    ui.gate(p2)
    _rng = np.random.default_rng(0)

    def _trial(s, rng):
        q = rng.normal(size=2000)
        a = q + rng.normal(0, s, (3, 2000)).mean(0)
        b = q + rng.normal(0, s, (3, 2000)).mean(0)
        return q, a, ex.committee_agreement(a, b, 0.25)

    _grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    _flip = [np.mean([_trial(s, _rng)[2]["flip_rate"] for _ in range(20)]) for s in _grid]
    _q, _a, _res = _trial(sigma.value, np.random.default_rng(1))
    _acc = np.zeros(2000, dtype=bool)
    _acc[np.argsort(-_a, kind="stable")[:500]] = True
    _pct = np.argsort(np.argsort(_q)) / 2000
    _bins = np.linspace(0, 1, 11)
    _p_acc = [_acc[(_pct >= lo) & (_pct < hi)].mean() for lo, hi in zip(_bins[:-1], _bins[1:])]
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _axes[0].plot(_grid, _flip, "o-")
    _axes[0].axhline(0.5, color="k", ls="--", lw=0.8)
    _axes[0].axvline(sigma.value, color="tab:red", lw=0.8)
    _axes[0].set(xlabel="σ (one reviewer's noise)", ylabel="flip rate", title="accepted by A, rejected by B")
    _axes[1].bar(100 * _bins[:-1] + 5, _p_acc, width=9)
    _axes[1].set(xlabel="true quality percentile", ylabel="P(accepted)", title=f"one committee at σ = {sigma.value:g}")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"at σ = {sigma.value:g}: the committees agree on {100 * _res['agree']:.0f}% of papers; {100 * _res['flip_rate']:.0f}% "
        f"of A's accepted papers are rejected by B; Cohen's κ = {_res['kappa']:.2f}. a paper in the top 10% of true "
        f"quality is accepted {100 * _p_acc[-1]:.0f}% of the time, one at the 70–80th percentile {100 * _p_acc[7]:.0f}%.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** read off the σ that gives a flip rate of one half. is that a plausible amount
    of disagreement between reviewers? why are papers near the acceptance threshold affected
    most, and what does that imply for how you should read a single rejection (or acceptance)
    of your own paper?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the draft

    a short LaTeX draft, typed quickly, the way first drafts are. it compiles (LaTeX only
    warns about most of these). before you send it to a co-author, your `check_latex` reads it
    with the bibliography it's meant to use.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("how many problems will check_latex find in the draft below? which ones would LaTeX not even warn about?")
    mo.vstack([mo.md(f"```latex\n{DRAFT}\n```"), p3])
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _problems = ex.check_latex(DRAFT, bib_keys=DRAFT_BIB)
    mo.md(f"**{len(_problems)} problems:**\n\n" + "\n".join(f"- `{p}`" for p in _problems))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which of these would a reader notice in the PDF, and which would silently
    give a wrong figure number or a "??" that nobody checks? why is a figure's `\label` before
    its `\caption` a bug even though the reference resolves?
    """)
    return


if __name__ == "__main__":
    app.run()
