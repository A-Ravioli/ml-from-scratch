import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def lazy_walk(A):
    """the lazy random walk on a graph with adjacency matrix A: stay with prob 1/2, else move to a random neighbour."""
    A = np.asarray(A, dtype=float)
    return 0.5 * np.eye(len(A)) + 0.5 * A / A.sum(axis=1, keepdims=True)


@app.function
def cycle_graph(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A


@app.function
def complete_graph(n):
    return np.ones((n, n)) - np.eye(n)


@app.function
def barbell_graph(n):
    """two complete graphs on n // 2 nodes each, joined by a single edge."""
    h = n // 2
    A = np.zeros((2 * h, 2 * h))
    A[:h, :h] = complete_graph(h)
    A[h:, h:] = complete_graph(h)
    A[h - 1, h] = A[h, h - 1] = 1
    return A


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # markov chains & stochastic processes · lab

    every experiment here runs **your** chain tools from `exercise.py`. for each one: write
    down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: mixing time tracks the spectral gap

    three families of graphs, each with $n$ nodes, and the lazy random walk on each (stay put
    with probability 1/2, otherwise move to a uniformly random neighbor): a **cycle**, a
    **complete graph** (everyone connected to everyone), and a **barbell** (two complete
    graphs on $n/2$ nodes joined by a single edge). for each, your `spectral_gap` and your
    `mixing_time` ($\varepsilon = 1/4$).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with n = 40 nodes, roughly how many steps does each walk need to mix (cycle, complete, "
        "barbell)? which grows fastest with n?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _sizes = [6, 10, 16, 24, 32, 40]
    _families = {"cycle": cycle_graph, "complete": complete_graph, "barbell": barbell_graph}
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    _rows = []
    _all_gap, _all_t = [], []
    for _name, _make in _families.items():
        _gaps, _tmix = [], []
        for _n in _sizes:
            _P = lazy_walk(_make(_n))
            _gaps.append(ex.spectral_gap(_P))
            _tmix.append(ex.mixing_time(_P, 0.25, max_steps=50_000))
        _a1.semilogy(_sizes, _tmix, "o-", label=_name)
        _a2.loglog(1 / np.array(_gaps), _tmix, "o", label=_name)
        _all_gap += _gaps
        _all_t += _tmix
        _rows.append(f"| {_name} | " + " | ".join(f"{_t} (1/γ = {1 / _g:.1f})" for _t, _g in zip(_tmix, _gaps)) + " |")
    _g = np.array([1 / min(_all_gap), 1 / max(_all_gap)])
    _a2.loglog(_g, _g, "k:", lw=0.8, label="t = 1/γ")
    _a1.set(xlabel="number of nodes n", ylabel="mixing time (ε = 1/4)")
    _a2.set(xlabel="1 / spectral gap", ylabel="mixing time")
    _a1.legend(fontsize=8)
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| graph | " + " | ".join(f"n = {_n}" for _n in _sizes) + " |\n|---|"
                           + "---|" * len(_sizes) + "\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the barbell has about half the complete graph's edges (381 against 780 at
    $n = 40$), yet it mixes over a hundred times more slowly. why? (halving the edges can't
    explain a factor of 150.) (where is the
    bottleneck, and how long does the walk wait to cross it?) why does the cycle's mixing time
    grow like $n^2$, the time a random walk needs to travel a distance $n$? how well does
    $1/\gamma$ predict the mixing time across all three?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: time averages vs distributions

    two chains on four states. chain A is a generic chain. chain B is **periodic**: it
    alternates between the group $\{0, 1\}$ and the group $\{2, 3\}$. for each, we start in
    state 0 and plot (left) the *distribution* $p_t$ from your `evolve`, and (right) the
    fraction of time one simulated path (your `simulate_markov_chain`) has spent in each state
    so far. dotted lines: your `stationary_distribution`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for the periodic chain B: does p_t converge? do the time fractions along one path converge? "
        "to what?"
    )
    p2
    return (p2,)


@app.cell
def _():
    chain_choice = mo.ui.dropdown(options=["A (aperiodic)", "B (periodic)"], value="B (periodic)", label="chain")
    chain_choice
    return (chain_choice,)


@app.cell
def _(chain_choice, p2):
    ui.gate(p2)
    _chains = {
        "A (aperiodic)": np.array([[0.5, 0.2, 0.2, 0.1], [0.1, 0.6, 0.1, 0.2], [0.3, 0.3, 0.3, 0.1], [0.2, 0.2, 0.1, 0.5]]),
        "B (periodic)": np.array([[0.0, 0.0, 0.3, 0.7], [0.0, 0.0, 0.6, 0.4], [0.5, 0.5, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]]),
    }
    _P = _chains[chain_choice.value]
    _pi = ex.stationary_distribution(_P)
    _hist = ex.evolve(np.eye(4)[0], _P, 40)
    _path = ex.simulate_markov_chain(_P, 0, 20_000, rng=0)
    _frac = np.cumsum(np.eye(4)[_path[1:]], axis=0) / np.arange(1, len(_path))[:, None]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for _s in range(4):
        _a1.plot(_hist[:, _s], ".-", color=f"C{_s}", label=f"state {_s}")
        _a1.axhline(_pi[_s], color=f"C{_s}", ls=":", lw=0.8)
        _a2.semilogx(np.arange(1, len(_path)), _frac[:, _s], color=f"C{_s}")
        _a2.axhline(_pi[_s], color=f"C{_s}", ls=":", lw=0.8)
    _a1.set(xlabel="t", ylabel="P(X_t = state)", title="distribution p_t (your evolve)")
    _a2.set(xlabel="steps T", ylabel="fraction of time in state", title="one path (your simulate_markov_chain)")
    _a1.legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"stationary distribution: {np.round(_pi, 4).tolist()}. time fractions after 20,000 steps: "
        f"{np.round(_frac[-1], 4).tolist()}. spectral gap {ex.spectral_gap(_P):.4f}, mixing time "
        f"{ex.mixing_time(_P, 0.25, max_steps=2000)}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for chain B, why does $p_t$ keep oscillating (look at which states are
    possible at even and odd times), while the time fractions converge to $\pi$ anyway? what
    is $\lambda_2$ for chain B, and why does `mixing_time` return `None`? which of the two
    kinds of convergence does an MCMC sampler need?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how much time does Brownian motion spend above zero?

    20,000 Brownian paths on $[0, 1]$ (1,000 steps each, your `brownian_motion`). for each
    path, the fraction of time it's above 0, and its maximum.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what's the most likely value of the fraction of time above zero? what fraction of paths "
        "spend more than 90% of the time on one side (above or below)? what's P(max ≥ 1)?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _B = ex.brownian_motion(20_000, 1000, dt=0.001, rng=1)
    _pos = np.mean(_B[:, 1:] > 0, axis=1)
    _x = np.linspace(0.002, 0.998, 400)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    _a1.hist(_pos, bins=40, density=True, alpha=0.6, label="your Brownian paths")
    _a1.plot(_x, 1 / (np.pi * np.sqrt(_x * (1 - _x))), "k-", lw=1, label="arcsine density 1/(π√(x(1−x)))")
    _a1.set(xlabel="fraction of time above 0", ylabel="density", ylim=(0, 4))
    _a1.legend(fontsize=8)
    for _i in range(5):
        _a2.plot(np.linspace(0, 1, 1001), _B[_i], lw=0.8)
    _a2.axhline(0, color="k", lw=0.6)
    _a2.set(xlabel="t", ylabel="B_t", title="five of the paths")
    _fig.tight_layout()
    _one_sided = np.mean((_pos < 0.1) | (_pos > 0.9))
    _middle = np.mean((_pos > 0.4) & (_pos < 0.6))
    _pmax = np.mean(_B.max(axis=1) >= 1)
    mo.vstack([_fig, mo.md(
        f"paths spending over 90% of the time on one side: **{_one_sided:.3f}** (arcsine law: "
        f"{4 / np.pi * math.asin(math.sqrt(0.1)):.3f}). paths with the fraction between 0.4 and 0.6: "
        f"**{_middle:.3f}** (arcsine law: {4 / np.pi * math.asin(math.sqrt(0.6)) - 1:.3f}). "
        f"P(max over [0, 1] ≥ 1) = **{_pmax:.3f}**, while P(B_1 ≥ 1) = {0.5 * math.erfc(1 / math.sqrt(2)):.3f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a symmetric random process "should" split its time evenly, so why is 1/2
    the *least* likely fraction? (think about what happens after the path wanders away from
    0: how long until it comes back?) and why is P(max ≥ 1) exactly twice P(B₁ ≥ 1)? (reflect
    the path at the first moment it hits 1.)
    """)
    return


if __name__ == "__main__":
    app.run()
