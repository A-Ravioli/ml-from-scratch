import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from mlfs import notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def exact_two_sided(n, p, t):
    """P(|S/n - p| >= t) for S ~ Binomial(n, p), computed exactly."""
    k = np.arange(n + 1)
    pmf = stats.binom(n, p).pmf(k)
    dev = np.abs(k / n - p)
    return np.array([pmf[dev >= tt - 1e-12].sum() for tt in np.atleast_1d(t)])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # concentration inequalities · lab

    every experiment here runs **your** bounds from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: every bound against the truth

    the quantity: the average of $n$ Bernoulli($p$) draws, and the probability that it lands
    at least $t$ away from $p$. the truth comes from 200,000 simulated averages (your
    `empirical_tail`) and from the exact binomial distribution. the bounds are yours. (Markov
    only bounds the upper tail, $P(\bar X \ge p + t) \le p/(p + t)$; the two-sided Chernoff
    bound adds the upper and lower one-sided bounds.)
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for n = 100 fair coins and t = 0.15, how many times larger than the truth is each bound "
        "(Chebyshev, Hoeffding, Bernstein, Chernoff)? which bound wins when p = 0.01?"
    )
    p1
    return (p1,)


@app.cell
def _():
    p_coin = mo.ui.dropdown(options=["0.5", "0.1", "0.01"], value="0.5", label="p")
    n_coin = mo.ui.dropdown(options=["20", "100", "500"], value="100", label="n")
    mo.hstack([p_coin, n_coin])
    return n_coin, p_coin


@app.cell
def _(n_coin, p1, p_coin):
    ui.gate(p1)
    _p, _n = float(p_coin.value), int(n_coin.value)
    _means = np.random.default_rng(0).binomial(_n, _p, size=200_000) / _n
    _ts = np.linspace(0.005, min(0.5, 1 - _p), 120)
    _emp = ex.empirical_tail(_means, _p, _ts)
    _exact = exact_two_sided(_n, _p, _ts)
    _var = _p * (1 - _p)
    _bounds = {
        "Markov (upper tail only)": ex.markov_bound(_p, _p + _ts),
        "Chebyshev": ex.chebyshev_bound(_var / _n, _ts),
        "Hoeffding": ex.hoeffding_bound(_n, _ts),
        "Bernstein": ex.bernstein_bound(_n, _ts, _var, max(_p, 1 - _p)),
        "Chernoff (two-sided)": np.minimum(1, ex.chernoff_bernoulli_bound(_n, _p, _ts)
                                           + ex.chernoff_bernoulli_bound(_n, 1 - _p, _ts)),
    }
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _label, _b in _bounds.items():
        _ax.semilogy(_ts, np.maximum(_b, 1e-300), lw=1.2, label=_label)
    _ax.semilogy(_ts, np.maximum(_exact, 1e-300), "k-", lw=2.2, label="exact")
    _ax.semilogy(_ts, np.where(_emp > 0, _emp, np.nan), "k.", ms=3, label="simulated (your empirical_tail)")
    _ax.set(xlabel="deviation t", ylabel="P(|mean - p| ≥ t)", ylim=(1e-12, 2))
    _ax.legend(fontsize=7)
    _rows = []
    for _t in (0.05, 0.1, 0.15, 0.2, 0.3):
        if _t > _ts[-1]:
            continue
        _i = int(np.argmin(np.abs(_ts - _t)))
        _truth = _exact[_i]
        _cells = " | ".join(f"{_b[_i] / _truth:.3g}×" if _truth > 0 else "—" for _b in _bounds.values())
        _rows.append(f"| {_ts[_i]:.3f} | {_truth:.3g} | {_cells} |")
    mo.vstack([_fig, mo.md("bound ÷ truth (1× would be perfect):\n\n| t | exact | " + " | ".join(_bounds)
                           + " |\n|---|---|" + "---|" * len(_bounds) + "\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does Chebyshev beat Hoeffding at small $t$ but lose by orders of
    magnitude at large $t$? at $p = 0.01$, Hoeffding is the same curve as at $p = 0.5$: why,
    and why is Bernstein so much better there? the gap between Chernoff and the truth stays
    roughly constant on a log scale: what does that say about its exponent?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how many test examples?

    you want to estimate a model's accuracy to within $\pm 0.01$, with probability at least
    95%. how many test examples does each method say you need? "exact" is the smallest $n$
    for which the true binomial probability of missing by 0.01 or more is at most 5%, when the
    true accuracy is exactly the value on the slider (something you never know in practice).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for a model with accuracy around 0.99, how many test examples does Hoeffding demand? "
        "how many are actually needed?"
    )
    p2
    return (p2,)


@app.cell
def _():
    acc = mo.ui.dropdown(options=["0.5", "0.9", "0.99"], value="0.99", label="true accuracy")
    acc
    return (acc,)


@app.function
def smallest_n(bound_at, delta, n_max=10**7):
    """smallest n with bound_at(n) <= delta, by doubling then bisection (bound_at decreasing in n)."""
    hi = 1
    while bound_at(hi) > delta:
        hi *= 2
        if hi > n_max:
            return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if bound_at(mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


@app.cell
def _(acc, p2):
    ui.gate(p2)
    _a, _t, _delta = float(acc.value), 0.01, 0.05
    _var = _a * (1 - _a)
    _rows = {
        # var / (n t^2) <= delta solved for n; rounded first so float error can't push 1980.0000000002 up to 1981
        "Chebyshev": math.ceil(round(_var / (_delta * _t**2), 6)),
        "Hoeffding": ex.hoeffding_sample_size(_t, _delta),
        "Bernstein": smallest_n(lambda n: float(ex.bernstein_bound(n, _t, _var, max(_a, 1 - _a))), _delta),
        "Chernoff (two-sided)": smallest_n(lambda n: float(ex.chernoff_bernoulli_bound(n, _a, _t)
                                                           + ex.chernoff_bernoulli_bound(n, 1 - _a, _t)), _delta),
        "CLT approximation": math.ceil(1.96**2 * _var / _t**2),
        "exact (binomial)": smallest_n(lambda n: float(exact_two_sided(n, _a, _t)[0]), _delta, n_max=200_000),
    }
    mo.md(f"accuracy {_a}, tolerance ±{_t}, confidence {1 - _delta:.0%}:\n\n| method | test examples needed |\n|---|---|\n"
          + "\n".join(f"| {_k} | {_v:,} |" for _k, _v in _rows.items()))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is Hoeffding's answer the same for every accuracy? which methods are
    *guarantees* and which are approximations? the variance-aware bounds need the variance,
    which depends on the unknown accuracy: how would you use them honestly in practice?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the winner's curse

    $k$ models, every one of them with true accuracy exactly 0.80, are each evaluated on the
    same kind of test set of $n = 1000$ examples (their mistakes are independent). we report
    the best measured accuracy, averaged over 200 repetitions. the union bound with your
    `hoeffding_bound` says: with probability 95%, every model is within $t_k$ of 0.80, where
    $k \cdot$ `hoeffding_bound(n, t_k)` $= 0.05$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with 1000 equally good models (true accuracy 0.80), what's the best measured accuracy on "
        "1000 test examples? with 10,000 models?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _n, _mu = 1000, 0.80
    _rows = []
    for _k in (1, 10, 100, 1000, 10_000):
        _best = _rng.binomial(_n, _mu, size=(200, _k)).max(axis=1) / _n
        _tk = smallest_n(lambda m: _k * float(ex.hoeffding_bound(_n, m / 10_000)), 0.05) / 10_000
        _rows.append(f"| {_k:,} | {_best.mean():.4f} | {_best.max():.4f} | {_mu + _tk:.4f} |")
    mo.md("| models k | mean best accuracy | largest best over 200 runs | union-bound limit 0.80 + t_k |\n"
          "|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** none of these models is better than 0.80. where do the "better" scores
    come from, and why do they grow so slowly with $k$ (compare the $\log k$ in $t_k$)? what
    does this say about leaderboards, hyperparameter searches evaluated on the test set, and
    papers that try many variants and report the best?
    """)
    return


if __name__ == "__main__":
    app.run()
