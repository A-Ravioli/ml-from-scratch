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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # graphical models · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: explaining away

    Pearl's alarm: a burglary ($B$) or an earthquake ($E$) can set off your alarm ($A$).
    burglaries are rare, $P(B) = 0.001$; earthquakes too, $P(E) = 0.002$. the alarm goes off
    with probability 0.95 if both happen, 0.94 for a burglary alone, 0.29 for an earthquake
    alone, and 0.001 otherwise. $B$ and $E$ are independent causes:
    `d_separated` says so. we compute $P(B = 1 \mid \cdot)$ under different evidence with your
    `joint_query`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "your alarm goes off: what's P(burglary)? then the radio reports an earthquake: does P(burglary) go up, "
        "stay the same, or go down, and by how much?"
    )
    p1
    return (p1,)


@app.cell
def _():
    alarm_factors = [
        ex.Factor(("B",), (2,), [0.999, 0.001]),
        ex.Factor(("E",), (2,), [0.998, 0.002]),
        ex.Factor(("B", "E", "A"), (2, 2, 2), [[[0.999, 0.001], [0.71, 0.29]], [[0.06, 0.94], [0.05, 0.95]]]),
    ]
    alarm_edges = [("B", "A"), ("E", "A")]
    return alarm_edges, alarm_factors


@app.cell
def _(alarm_edges, alarm_factors, p1):
    ui.gate(p1)
    _cases = [("prior", {}), ("alarm", {"A": 1}), ("alarm + quake", {"A": 1, "E": 1}),
              ("alarm, no quake", {"A": 1, "E": 0}), ("quake only", {"E": 1})]
    _probs = [float(ex.joint_query(alarm_factors, "B", ev).values[1]) for _, ev in _cases]
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _ax.bar([c for c, _ in _cases], _probs, color=["C7", "C3", "C0", "C3", "C7"])
    _ax.set(yscale="log", ylabel="P(burglary | evidence)", ylim=(5e-4, 2))
    for _i, _p in enumerate(_probs):
        _ax.text(_i, _p * 1.15, f"{_p:.4f}", ha="center", fontsize=9)
    _fig.tight_layout()
    _ds = [ex.d_separated(alarm_edges, "B", "E", g) for g in [(), ("A",)]]
    mo.vstack([_fig, mo.md(
        f"`d_separated(B, E)` = **{_ds[0]}**; `d_separated(B, E | A)` = **{_ds[1]}**. "
        f"P(B | alarm) = **{_probs[1]:.4f}**, P(B | alarm, quake) = **{_probs[2]:.4f}**: the quake divides "
        f"the burglary probability by **{_probs[1] / _probs[2]:.0f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** $B$ and $E$ are independent, and the earthquake says nothing about burglars
    by itself (compare "prior" and "quake only"). so why does learning about the earthquake
    change your belief about a burglary once you've heard the alarm? which of the three
    patterns (chain, fork, collider) is this, and what does observing the collider do? why is
    $P(B \mid \text{alarm})$ only about 0.37, even though the alarm is 94% reliable?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: Berkson's paradox

    a population of 20,000 people with two independent, standard-normal traits, *talent* and
    *looks*. someone becomes *famous* if talent + looks exceeds a threshold (the slider). in
    graph terms: talent → famous ← looks. we measure the correlation between talent and looks
    in the whole population and among the famous.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("among famous people only (threshold 1.5), what's the correlation between talent and looks?")
    thr2 = mo.ui.slider(-1.0, 3.0, value=1.5, step=0.25, label="fame threshold")
    mo.vstack([p2, thr2])
    return p2, thr2


@app.cell
def _(p2, thr2):
    ui.gate(p2)
    _rng = np.random.default_rng(2)
    _t, _l = _rng.normal(size=(2, 20000))
    _fam = _t + _l > thr2.value
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].scatter(_t[~_fam], _l[~_fam], s=2, c="0.75", label="not famous")
    _axes[0].scatter(_t[_fam], _l[_fam], s=2, c="C3", label="famous")
    _axes[0].set(xlabel="talent", ylabel="looks", aspect="equal")
    _axes[0].legend(fontsize=8, markerscale=4)
    _ths = np.linspace(-1, 3, 17)
    _rs = [np.corrcoef(_t[_t + _l > h], _l[_t + _l > h])[0, 1] for h in _ths]
    _axes[1].plot(_ths, _rs, "o-")
    _axes[1].axhline(0, color="k", lw=0.8)
    _axes[1].axvline(thr2.value, color="C3", ls="--", lw=1)
    _axes[1].set(xlabel="fame threshold", ylabel="corr(talent, looks) among the famous")
    _fig.tight_layout()
    _ds = ex.d_separated([("talent", "famous"), ("looks", "famous")], "talent", "looks", ["famous"])
    mo.vstack([_fig, mo.md(
        f"whole population: corr = **{np.corrcoef(_t, _l)[0, 1]:+.3f}**. famous only ({_fam.sum()} people): "
        f"corr = **{np.corrcoef(_t[_fam], _l[_fam])[0, 1]:+.3f}**. `d_separated(talent, looks | famous)` = **{_ds}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** nothing connects talent and looks in the population, yet among the famous
    they're strongly anti-correlated. where did the correlation come from? restricting a
    dataset to the famous is conditioning on which node? think of a study of hospital
    patients, or of published papers: what could a selection like this make you conclude?
    why does a *higher* threshold make the correlation more negative?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: one more observation, dependence again

    the sprinkler network: cloudy ($C$) → sprinkler ($S$), $C$ → rain ($R$), and both $S$ and
    $R$ → wet grass ($W$). we compute the exact conditional mutual information
    $I(S; R \mid Z)$ in bits from your `joint_query`, for three conditioning sets $Z$, and
    compare with `d_separated`. we also estimate it from $n$ samples drawn from the network
    (the slider), with the plug-in estimator, to see what a data analyst would measure.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("rank I(S;R), I(S;R | C) and I(S;R | C, W) from smallest to largest. which ones are exactly zero?")
    n3 = mo.ui.slider(200, 20000, value=2000, step=200, label="samples for the estimate")
    mo.vstack([p3, n3])
    return n3, p3


@app.function
def cmi_bits(p):
    """I(X; Y | Z) in bits for a table p of shape (|X|, |Y|, |Z|) (Z flattened)."""
    p = p / p.sum()
    pz = p.sum(axis=(0, 1))
    pxz = p.sum(axis=1)
    pyz = p.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * np.log2(p * pz[None, None, :] / (pxz[:, None, :] * pyz[None, :, :]))
    return float(np.nansum(term))


@app.cell
def _(n3, p3):
    ui.gate(p3)
    _fs = [
        ex.Factor(("C",), (2,), [0.5, 0.5]),
        ex.Factor(("C", "S"), (2, 2), [[0.5, 0.5], [0.9, 0.1]]),
        ex.Factor(("C", "R"), (2, 2), [[0.8, 0.2], [0.2, 0.8]]),
        ex.Factor(("S", "R", "W"), (2, 2, 2), [[[1.0, 0.0], [0.1, 0.9]], [[0.1, 0.9], [0.01, 0.99]]]),
    ]
    _edges = [("C", "S"), ("C", "R"), ("S", "W"), ("R", "W")]
    _joint = ex.joint_query(_fs, ("S", "R", "C", "W"))  # (S, R, C, W)
    _rng = np.random.default_rng(3)
    _flat = _rng.choice(16, size=n3.value, p=_joint.values.ravel())
    _counts = np.bincount(_flat, minlength=16).reshape(2, 2, 2, 2).astype(float)
    _rows, _vals = [], []
    for _name, _z in [("I(S;R)", ()), ("I(S;R ∣ C)", ("C",)), ("I(S;R ∣ C, W)", ("C", "W"))]:
        _drop = tuple(ax for ax, v in zip((2, 3), ("C", "W")) if v not in _z)
        _exact = abs(cmi_bits(_joint.values.sum(axis=_drop).reshape(2, 2, -1)))
        _est = cmi_bits(_counts.sum(axis=_drop).reshape(2, 2, -1))
        _vals.append((_exact, _est))
        _rows.append(f"| {_name} | {ex.d_separated(_edges, 'S', 'R', _z)} | {_exact:.5f} | {_est:.5f} |")
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _x = np.arange(3)
    _ax.bar(_x - 0.2, [v[0] for v in _vals], 0.4, label="exact")
    _ax.bar(_x + 0.2, [v[1] for v in _vals], 0.4, label=f"estimated from n = {n3.value}")
    _ax.set_xticks(_x, ["I(S;R)", "I(S;R | C)", "I(S;R | C, W)"])
    _ax.set(ylabel="bits")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| quantity | d-separated? | exact (bits) | estimated (bits) |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are $S$ and $R$ dependent with nothing observed, independent given $C$,
    and dependent *again* given $C$ and $W$? trace the paths between $S$ and $R$ and say which
    are blocked in each case. the estimate for the d-separated case isn't zero: how big is
    it, how does it change with $n$, and what does that mean for an algorithm that tries to
    *learn* the graph from data by testing independences?
    """)
    return


if __name__ == "__main__":
    app.run()
