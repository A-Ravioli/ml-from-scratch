import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def casino(p_fl=0.05, p_lf=0.1):
    """the dishonest casino: state 0 = fair die, state 1 = loaded die (a six half the time)."""
    h = ex.HMM(2, 6)
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[1 - p_fl, p_fl], [p_lf, 1 - p_lf]])
    h.emissionprob_ = np.vstack([np.full(6, 1 / 6), [0.1] * 5 + [0.5]])
    return h


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hidden markov models · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    the running example is Durbin et al.'s **dishonest casino**: a croupier uses a fair die,
    but switches to a loaded one (a six with probability 0.5, every other face 0.1) with
    probability 0.05 per roll, and back with probability 0.1. you only see the rolls.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: decoding the casino

    300 rolls sampled from the casino with your `sample`. we decode them three ways: your
    `viterbi` path, the posterior $P(\text{loaded} \mid \text{rolls})$ from your
    `forward_backward` thresholded at 0.5 ("posterior decoding"), and a no-memory baseline
    that calls each roll loaded if and only if it's a six.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("what fraction of the 300 rolls does Viterbi label correctly? and the 'six means loaded' rule?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _h = casino()
    _obs, _states = _h.sample(300, rng=1)
    _vit, _ = _h.viterbi(_obs)
    _, _post = _h.forward_backward(_obs)
    _pd = (_post[:, 1] > 0.5).astype(int)
    _six = (_obs == 5).astype(int)
    _acc = {k: float(np.mean(v == _states)) for k, v in [("viterbi", _vit), ("posterior", _pd), ("six rule", _six)]}
    _fig, _ax = plt.subplots(figsize=(11, 3.6))
    _t = np.arange(300)
    _ax.fill_between(_t, 0, _states, step="mid", color="C3", alpha=0.2, label="truly loaded")
    _ax.plot(_t, _post[:, 1], "C0", lw=1.2, label="P(loaded | all rolls)")
    _ax.step(_t, _vit * 1.04, "k", where="mid", lw=1, label="Viterbi (raised a little)")
    _ax.plot(_t[_obs == 5], np.full(np.sum(_obs == 5), -0.06), "|", color="C1", label="a six")
    _ax.set(xlabel="roll", ylim=(-0.12, 1.12))
    _ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"accuracy: Viterbi **{_acc['viterbi']:.3f}**, posterior decoding **{_acc['posterior']:.3f}**, "
        f"'six means loaded' **{_acc['six rule']:.3f}**. the croupier used the loaded die on "
        f"{_states.mean():.0%} of rolls; Viterbi and posterior decoding disagree on {np.sum(_vit != _pd)} rolls.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the six rule do so badly, when sixes really are five times more
    likely on the loaded die? what does the HMM know that the rule doesn't? look at where
    the posterior is near 0.5: what kind of stretch of rolls is it unsure about? Viterbi
    finds the best *path*, posterior decoding the best *state at each roll*: when, in
    general, could the second one produce a sequence of states that's impossible?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: learning the casino from scratch

    now forget the parameters. we sample $n$ rolls (slider) and run your Baum–Welch `fit`
    from 8 random starting points (8 seeds), keeping every result. for each we report the
    learned $P(6 \mid \text{loaded state})$ (the state with the larger $P(6)$), the switch
    probabilities, and the training log-likelihood, next to the log-likelihood of the
    *true* parameters on the same rolls.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("with 300 rolls, is the training log-likelihood of the learned parameters higher or lower "
                    "than that of the TRUE parameters? how close are the learned parameters to the truth?")
    n2 = mo.ui.slider(100, 5000, value=300, step=100, label="rolls")
    mo.vstack([p2, n2])
    return n2, p2


@app.cell
def _(n2, p2):
    ui.gate(p2)
    _truth = casino()
    _obs, _ = _truth.sample(n2.value, rng=2)
    _ll_true = _truth.score(_obs)
    _rows, _lls, _p6 = [], [], []
    for _seed in range(8):
        _m = ex.HMM(2, 6, max_iters=500, tol=1e-6, rng=_seed).fit(_obs)
        _l = int(np.argmax(_m.emissionprob_[:, 5]))
        _f = 1 - _l
        _ll = _m.score(_obs)
        _lls.append(_ll)
        _p6.append(_m.emissionprob_[_l, 5])
        _rows.append(f"| {_seed} | {_m.emissionprob_[_l, 5]:.3f} | {_m.emissionprob_[_f, 5]:.3f} | "
                     f"{_m.transmat_[_f, _l]:.3f} | {_m.transmat_[_l, _f]:.3f} | {_ll:.2f} | {_m.n_iter_} |")
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.scatter(_p6, _lls, c="C0")
    _ax.axhline(_ll_true, color="C3", ls="--", label="true parameters")
    _ax.axvline(0.5, color="C3", ls=":", lw=1)
    _ax.set(xlabel="learned P(6 | loaded)", ylabel="training log-likelihood")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"true parameters: log-likelihood **{_ll_true:.2f}** (P(6 | loaded) = 0.5, P(6 | fair) = 0.167, switches 0.05 / 0.1).\n\n"
        "| seed | P(6 \\| loaded) | P(6 \\| fair) | fair → loaded | loaded → fair | log-lik | iterations |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how can a model that's wrong about every parameter explain the data *better*
    than the model that generated it? what is maximum likelihood actually maximizing, and why
    does that reward fitting the noise in 300 rolls? why do different seeds end up in
    different places, and what does that say about EM's objective? move the slider to 5,000:
    what changes?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: when memory stops helping

    the HMM beats the six rule because the croupier is *sticky*: he keeps a die for about 20
    rolls (fair) or 10 rolls (loaded) at a time. we keep the long-run fraction of loaded rolls
    fixed at 1/3 and make him switch more and more often (the slider scales both switch
    probabilities), then measure the accuracy of Viterbi and posterior decoding on 5,000 rolls
    against the best you can do *without* memory. that's the six rule: with 1/3 of rolls
    loaded, $P(\text{loaded} \mid 6) = 0.6$ and $P(\text{loaded} \mid \text{not } 6) = 0.23$,
    so "six means loaded" is the optimal roll-by-roll decision, right $2/3 \cdot 5/6 + 1/3 \cdot 1/2
    = 72.2\%$ of the time (always guessing "fair" gets 66.7%).
    """)
    return


@app.cell
def _():
    p3 = ui.predict("if the croupier switches dice every couple of rolls on average, how does Viterbi compare with "
                    "the memoryless six rule (72.2%)? better, the same, or worse?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _scales = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 9.5])
    _acc_v, _acc_p, _acc_s = [], [], []
    for _s in _scales:
        _h = casino(0.05 * _s, 0.1 * _s)
        _obs, _st = _h.sample(5000, rng=3)
        _acc_v.append(np.mean(_h.viterbi(_obs)[0] == _st))
        _acc_p.append(np.mean((_h.forward_backward(_obs)[1][:, 1] > 0.5) == _st))
        _acc_s.append(np.mean((_obs == 5) == _st))
    _fig, _ax = plt.subplots(figsize=(7.5, 3.8))
    _stay = 1 / (0.1 * _scales)
    _ax.semilogx(_stay, _acc_v, "o-", label="Viterbi")
    _ax.semilogx(_stay, _acc_p, "s--", label="posterior decoding")
    _ax.semilogx(_stay, _acc_s, "^-", label="six means loaded")
    _ax.axhline(2 / 3, color="k", ls=":", label="always 'fair'")
    _ax.set(xlabel="average length of a loaded stretch (rolls)", ylabel="accuracy")
    _ax.invert_xaxis()
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {1 / (0.1 * s):.1f} | {v:.3f} | {p:.3f} | {q:.3f} |"
                      for s, v, p, q in zip(_scales, _acc_v, _acc_p, _acc_s))
    mo.vstack([_fig, mo.md("| loaded stretch (rolls) | Viterbi | posterior | six rule |\n|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a single roll carries very little information about the die (a six is
    only 5x likelier when loaded, and most rolls aren't sixes). where does the HMM's
    advantage come from, and why does it vanish as the stretches get short? when loaded
    stretches last about one roll, why can't any decoder beat the six rule? and why does
    Viterbi drop *below* it for stretches of a few rolls, while posterior decoding doesn't?
    (which of the two is trying to maximize the number of correctly labelled rolls?)
    """)
    return


if __name__ == "__main__":
    app.run()
