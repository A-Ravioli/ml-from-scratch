import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # online learning · lab

    every experiment here runs **your** Hedge, follow-the-leader and online gradient descent
    from `exercise.py`. for each one: write down a prediction, lock it in, run it, then explain
    what happened. compare your explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: regret grows like √T

    Hedge with $N = 10$ experts and its optimal learning rate $\sqrt{8\ln N/T}$, for horizons
    $T$ from 100 to 30,000, on three kinds of loss sequences:

    - **coin flips**: every expert's loss is an independent fair coin each round, so all
      experts are equally good;
    - **one better expert**: expert 0's losses are coins with mean 0.4, the others 0.5;
    - **switching**: the best expert changes every $T/10$ rounds.

    regret at the end of the run, averaged over 5 seeds, against the guarantee
    $\sqrt{(T/2)\ln N}$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "on 'coin flips', where every expert is equally good, what will Hedge's regret be after 30,000 rounds: about 0, "
        "or growing with T? on which sequence will the regret be largest?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _N, _Ts, _seeds = 10, [100, 300, 1000, 3000, 10000, 30000], 5

    def _seq(kind, T, rng):
        if kind == "coin flips":
            return (rng.random((T, _N)) < 0.5).astype(float)
        if kind == "one better expert":
            return (rng.random((T, _N)) < np.r_[0.4, np.full(_N - 1, 0.5)]).astype(float)
        best = (np.arange(T) * 10 // T) % _N  # the best expert changes every T/10 rounds
        L = (rng.random((T, _N)) < 0.5).astype(float)
        L[np.arange(T), best] = (rng.random(T) < 0.2).astype(float)
        return L

    _kinds = ["coin flips", "one better expert", "switching"]
    _res = {k: [] for k in _kinds}
    for _k in _kinds:
        for _T in _Ts:
            _regs = []
            for _s in range(_seeds):
                _L = _seq(_k, _T, np.random.default_rng(1000 * _s + _T))
                _learner = ex.run_hedge(_L, ex.hedge_optimal_lr(_T, _N))
                _regs.append(float(np.asarray(ex.regret_curve(_learner, _L))[-1]))
            _res[_k].append(float(np.mean(_regs)))
    _bound = [ex.hedge_regret_bound(T, _N, ex.hedge_optimal_lr(T, _N)) for T in _Ts]
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    for _k, _c in zip(_kinds, ["C0", "C2", "C3"]):
        _ax.loglog(_Ts, np.maximum(_res[_k], 1e-1), "o-", color=_c, label=_k)
    _ax.loglog(_Ts, _bound, "k--", label="guarantee √((T/2) ln N)")
    _ax.set(xlabel="T (rounds)", ylabel="regret at T (mean of 5 seeds)", title="Hedge, N = 10, tuned learning rate")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _slopes = {k: np.polyfit(np.log(_Ts[2:]), np.log(np.maximum(_res[k][2:], 1e-9)), 1)[0] for k in _kinds}
    _rows = "\n".join(f"| {T:,} | " + " | ".join(f"{_res[k][i]:.1f}" for k in _kinds) + f" | {_bound[i]:.1f} |"
                      for i, T in enumerate(_Ts))
    mo.vstack([_fig, mo.md(
        "| T | " + " | ".join(_kinds) + " | guarantee |\n|---|---|---|---|---|\n" + _rows
        + "\n\nlog-log slope of regret against T (T ≥ 1,000): "
        + ", ".join(f"{k}: **{v:.2f}**" for k, v in _slopes.items()) + ".")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** on coin flips no expert is better than any other, yet the regret grows
    like $\sqrt T$: whose fault is that? (what is the cumulative loss of the *best expert in
    hindsight*, compared with $T/2$?) why does that make $\sqrt{T\ln N}$ a lower bound for
    *every* algorithm, not just Hedge? the guarantee holds for every sequence: how close does
    the worst of the three come to it? why is the regret with one better expert *smaller*?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the greedy leader

    follow-the-leader (your `follow_the_leader`) against Hedge (tuned for $T = 10{,}000$) on
    two sequences of $T = 10{,}000$ rounds with 2 experts: **stochastic** (losses are coins
    with means 0.4 and 0.6) and **alternating** ($(\frac12, 0)$, then $(0, 1), (1, 0), \dots$,
    the sequence from the lesson). regret curves over time.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which algorithm has lower regret on the stochastic sequence, FTL or Hedge? on the alternating one?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _T = 10000
    _rng = np.random.default_rng(7)
    _sto = (_rng.random((_T, 2)) < np.array([0.4, 0.6])).astype(float)
    _alt = np.zeros((_T, 2))
    _alt[0] = [0.5, 0.0]
    _alt[1::2] = [0.0, 1.0]
    _alt[2::2] = [1.0, 0.0]
    _lr = ex.hedge_optimal_lr(_T, 2)
    _curves = {}
    for _name, _L in [("stochastic", _sto), ("alternating", _alt)]:
        _curves[(_name, "FTL")] = np.asarray(ex.regret_curve(ex.follow_the_leader(_L), _L))
        _curves[(_name, "Hedge")] = np.asarray(ex.regret_curve(ex.run_hedge(_L, _lr), _L))
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _ax, _name in zip(_axes, ["stochastic", "alternating"]):
        _ax.plot(_curves[(_name, "FTL")], "C3", label="follow the leader")
        _ax.plot(_curves[(_name, "Hedge")], "C0", label="Hedge")
        _ax.plot(np.sqrt(np.arange(1, _T + 1) * math.log(2) / 2), "k:", lw=1, label="√((t/2) ln 2)")
        _ax.set(title=_name, xlabel="t", ylabel="regret", yscale="symlog" if _name == "alternating" else "linear")
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {n} | {_curves[(n, 'FTL')][-1]:.1f} | {_curves[(n, 'Hedge')][-1]:.1f} |"
                      for n in ["stochastic", "alternating"])
    mo.vstack([_fig, mo.md("| sequence | FTL regret at T | Hedge regret at T |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** on the stochastic sequence FTL beats Hedge: why is jumping straight to the
    leader the right thing when losses are iid, and what is Hedge paying for? on the
    alternating sequence FTL's regret is $T/2$: what does the adversary exploit? Hedge's
    "hedging" costs a little on easy data and saves everything on hard data. what single
    knob of Hedge interpolates between it and FTL?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: online regression on a real stream

    California housing as a stream: each round, predict a district's (standardized) house
    value from its 8 standardized features plus a bias (feature vectors clipped to norm 4),
    suffer the squared loss, update. your `online_least_squares` runs OGD over the ball
    $\|w\| \le 2$, which contains the best fixed predictor in hindsight (a constrained least-
    squares fit to the whole stream). two orders of the same districts: **shuffled** and
    **sorted by latitude** (south to north); two step sizes: the theory's $\eta_t = (D/G)/\sqrt t$
    and a constant $\eta = 0.01$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "regret is measured against the best FIXED linear predictor in hindsight. can it be negative? on which stream "
        "do you expect the online learner to do best?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X, _y, _ = datasets.california_housing()
    _Z = np.hstack([(_X - _X.mean(0)) / _X.std(0), np.ones((len(_X), 1))])
    _Z *= np.minimum(1.0, 4.0 / np.linalg.norm(_Z, axis=1, keepdims=True))
    _t = (_y - _y.mean()) / _y.std()
    _R = 2.0
    _G = 4.0 * (_R * 4.0 + np.abs(_t).max())  # ||grad|| <= ||x|| (||w|| ||x|| + |y|)

    def _best_fixed(A, b):
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.linalg.norm(w) <= _R:
            return w
        lo, hi = 1e-12, 1e8
        for _ in range(200):
            lam = math.sqrt(lo * hi)
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
            lo, hi = (lam, hi) if np.linalg.norm(w) > _R else (lo, lam)
        return w

    _orders = {"shuffled": np.random.default_rng(0).permutation(len(_Z)), "sorted by latitude": np.argsort(_X[:, 1])}
    _runs, _rows = {}, []
    _wb = _best_fixed(_Z, _t)  # the same districts in any order: one best fixed predictor
    for _o, _idx in _orders.items():
        _A, _b = _Z[_idx], _t[_idx]
        _best = 0.5 * (_A @ _wb - _b) ** 2
        for _lab, _kw in [("theory: (D/G)/√t", dict(lr=2 * _R / _G, decay="sqrt")), ("constant 0.01", dict(lr=0.01, decay=None))]:
            _L, _ = ex.online_least_squares(_A, _b, radius=_R, **_kw)
            _reg = np.cumsum(_L) - np.cumsum(_best)
            _runs[(_o, _lab)] = _reg
            _rows.append(f"| {_o} | {_lab} | {np.mean(_L):.4f} | {np.mean(_best):.4f} | {_reg[-1]:+.1f} |")
    _T = len(_Z)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _ax, _o in zip(_axes, _orders):
        for _lab, _c in [("theory: (D/G)/√t", "C0"), ("constant 0.01", "C3")]:
            _ax.plot(_runs[(_o, _lab)], _c, label=_lab)
        _ax.axhline(0, color="0.5", lw=0.8)
        _ax.set(title=f"{_o} stream", xlabel="round t", ylabel="regret vs best fixed w")
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| stream | step size | mean loss (online) | mean loss (best fixed) | regret at T |\n|---|---|---|---|---|\n"
        + "\n".join(_rows)
        + f"\n\nT = {_T:,} rounds; the guarantee (3/2) G D √T for the theory step size is "
        f"**{ex.ogd_regret_bound(_T, 2 * _R, _G):,.0f}** (G = {_G:.1f}, D = {2 * _R}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the guarantee is a valid upper bound on regret, but how many times larger
    than the measured regret is it, and which ingredient makes it so pessimistic (look at $G$)?
    on the latitude-sorted stream the regret goes *negative*: the online learner beats the
    best fixed predictor in hindsight. how is that possible, and why doesn't it contradict
    anything? which step size benefits more from a drifting stream, and why? what would you
    compare against to make "beating the best fixed predictor" impossible again?
    """)
    return


if __name__ == "__main__":
    app.run()
