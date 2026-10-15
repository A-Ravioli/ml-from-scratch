import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # VC dimension · lab

    every experiment here runs **your** oracles, shattering checker and searches from
    `exercise.py`. for each one: write down a prediction, lock it in, run it, then explain what
    happened. compare your explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: three VC dimensions, by brute force

    your `vc_dimension_search` on random pools of points: 10 points on the line for intervals,
    12 points in the plane for axis-aligned boxes and for halfspaces (lines). we count how many
    times the oracle is called and plot the witness it finds, next to the four corners of a
    square, which boxes *can't* shatter.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what VC dimension will the search report for intervals, for boxes in the plane, and for halfspaces in the plane? "
        "is it the number of parameters of each class (2, 4 and 3)?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)

    class _Count:
        def __init__(self, fn):
            self.fn, self.calls = fn, 0

        def __call__(self, X, y):
            self.calls += 1
            return self.fn(X, y)

    _rng = np.random.default_rng(0)
    _pools = {"intervals": _rng.random(10), "boxes": _rng.random((12, 2)), "halfspaces": _rng.random((12, 2))}
    _oracles = {"intervals": ex.interval_realizable, "boxes": ex.rectangle_realizable,
                "halfspaces": ex.halfspace_realizable}
    _found, _rows = {}, []
    for _name in _pools:
        _c = _Count(_oracles[_name])
        _t0 = time.perf_counter()
        _d, _w = ex.vc_dimension_search(_pools[_name], _c, max_d=6)
        _found[_name] = (_d, _w)
        _rows.append(f"| {_name} | {_d} | {_c.calls:,} | {time.perf_counter() - _t0:.2f} s |")
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4))
    for _ax, _name in zip(_axes[:2], ["boxes", "halfspaces"]):
        _P = _pools[_name]
        _W = _P[list(_found[_name][1])]
        _ax.plot(_P[:, 0], _P[:, 1], "o", color="0.7", label="pool")
        _ax.plot(_W[:, 0], _W[:, 1], "o", color="C3", ms=9, label=f"shattered {len(_W)}-set")
        _ax.set(title=f"{_name}: witness", aspect="equal", xlim=(-0.05, 1.05), ylim=(-0.05, 1.3))
        _ax.legend(fontsize=7, loc="upper center", ncol=2)
    _sq = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    _dia = np.array([[0.5, 1.2], [1.2, 0.5], [0.5, -0.2], [-0.2, 0.5]])
    _axes[2].plot(_sq[:, 0], _sq[:, 1], "s", color="C0", ms=9,
                  label=f"square corners: shattered = {ex.is_shattered(_sq, ex.rectangle_realizable)}")
    _axes[2].plot(_dia[:, 0], _dia[:, 1], "D", color="C2", ms=9,
                  label=f"diamond: shattered = {ex.is_shattered(_dia, ex.rectangle_realizable)}")
    _axes[2].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ls=":", color="C0"))
    _axes[2].set(title="boxes: which 4-sets are shattered?", aspect="equal", xlim=(-0.5, 1.5), ylim=(-0.5, 1.6))
    _axes[2].legend(fontsize=7, loc="upper right")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| class | VC dimension found | oracle calls | time |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the square's set of corners not shattered by boxes, while the
    diamond is? (which labeling fails for the square?) argue that *no* 5 points in the plane
    can be shattered by boxes, so the search's answer is the true VC dimension and not just a
    property of this pool. the search is only ever a *lower* bound: when could it report less
    than the true VC dimension? why does the halfspace search take far longer than the other
    two, even though its answer is smaller than the boxes'?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the growth function against Sauer's lemma

    for halfspaces in the plane (VC dimension 3) and intervals (VC dimension 2), the growth
    function $\Pi(m)$ is the most labelings any $m$ points admit. we measure it with your
    `growth_lower_bound` (the best of 2 random $m$-point subsets of a pool) for
    $m = 1, \dots, 10$ and compare it with your `sauer_bound` and with $2^m$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for halfspaces in the plane with m = 10 points: how many of the 1,024 labelings are realizable? "
        "how does that compare with Sauer's bound?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _pool2, _pool1 = _rng.normal(size=(30, 2)), _rng.random(30)
    _ms = np.arange(1, 11)
    _half = [ex.growth_lower_bound(_pool2, ex.halfspace_realizable, int(m), n_subsets=2, rng=int(m)) for m in _ms]
    _intv = [ex.growth_lower_bound(_pool1, ex.interval_realizable, int(m), n_subsets=2, rng=int(m)) for m in _ms]
    _s3 = [ex.sauer_bound(int(m), 3) for m in _ms]
    _s2 = [ex.sauer_bound(int(m), 2) for m in _ms]
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.semilogy(_ms, 2.0**_ms, "k:", label="$2^m$ (shattered)")
    _ax.semilogy(_ms, _s3, "C3--", label="Sauer, d = 3")
    _ax.semilogy(_ms, _half, "C3o-", label="halfspaces in the plane (measured)")
    _ax.semilogy(_ms, _s2, "C0--", label="Sauer, d = 2")
    _ax.semilogy(_ms, _intv, "C0s-", label="intervals (measured)")
    _ax.set(xlabel="m (number of points)", ylabel="number of labelings")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {m} | {2**m} | {h} | {s} | {i} | {t} |" for m, h, s, i, t in zip(_ms, _half, _s3, _intv, _s2))
    mo.vstack([_fig, mo.md("| m | 2^m | halfspaces | Sauer d=3 | intervals | Sauer d=2 |\n|---|---|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** up to $m = d$ the growth function is $2^m$; after that it bends into a
    polynomial. check the measured halfspace counts against Cover's formula $m^2 - m + 2$, and
    the interval counts against $1 + m + \binom m2$. intervals meet Sauer's bound exactly: why?
    halfspaces fall short of it: which labelings does Sauer count that no line can produce?
    why does the polynomial growth, not its exact constant, matter for learning?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: one parameter, infinite VC dimension

    the class $h_\omega(x) = \mathbb 1[\sin(\omega x) > 0]$ has a single real parameter. on the
    points $x_i = 10^{-i}$, $i = 1, \dots, n$, the frequency
    $\omega = \pi\big(1 + \sum_i (1 - y_i)10^i\big)$ produces any labeling $y$ you want
    (Vapnik's classic construction). we wrap that recipe in an oracle and ask your
    `count_labelings` how many of the $2^n$ labelings it achieves, for $n = 1, \dots, 16$, in
    double precision.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "a class with ONE real parameter: can it shatter 10 points? 16? what will stop it in practice?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    def _sine_oracle(x, y):
        y = np.asarray(y, dtype=int)
        omega = np.pi * (1 + sum((1 - y[i]) * 10.0 ** (i + 1) for i in range(len(y))))
        return bool(np.array_equal((np.sin(omega * np.asarray(x)) > 0).astype(int), y))

    _rows, _ns, _frac = [], list(range(1, 17)), []
    for _n in _ns:
        _x = 10.0 ** -np.arange(1, _n + 1)
        _k = ex.count_labelings(_x, _sine_oracle)
        _frac.append(_k / 2**_n)
        _rows.append(f"| {_n} | {_k:,} | {2**_n:,} | {'yes' if _k == 2**_n else 'no'} |")
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _xs = np.geomspace(5e-6, 0.2, 40000)
    _y = np.array([1, 0, 1, 1, 0])
    _om = np.pi * (1 + sum((1 - _y[i]) * 10.0 ** (i + 1) for i in range(5)))
    _a1.step(_xs, np.sin(_om * _xs) > 0, lw=0.3, color="0.5", where="mid")
    _a1.plot(10.0 ** -np.arange(1, 6), _y, "C3o", ms=8)
    _a1.set(xscale="log", xlim=(5e-6, 0.2), xlabel="x (log scale)", ylabel="h(x)",
            title=f"labels (1, 0, 1, 1, 0) at x = 0.1 ... 1e-5: ω = {_om:.4g}")
    _a2.plot(_ns, _frac, "ko-")
    _a2.set(xlabel="n (points 10^-1 ... 10^-n)", ylabel="fraction of 2^n labelings realized", ylim=(0, 1.05))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| n | labelings realized | 2^n | shattered? |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a single parameter shatters 15 points here, and in exact arithmetic it
    shatters any number: the VC dimension is infinite. so what does the VC dimension measure, if
    not the number of parameters? what goes wrong at $n = 16$ (look at the size of $\omega$),
    and does that rescue the class in practice? what does this mean for the finite-class bound
    of topic 01 applied to "one float64 parameter"?
    """)
    return


if __name__ == "__main__":
    app.run()
