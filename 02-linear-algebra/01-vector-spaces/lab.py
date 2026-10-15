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
    # vector spaces · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the angle between two random vectors

    draw two vectors with independent standard normal entries in $\mathbb{R}^d$ and measure the
    angle between them, $\arccos\big(u^\top v / (\|u\|\,\|v\|)\big)$. we repeat this 2,000 times
    for several dimensions $d$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "in d = 1000 dimensions, what's a typical angle between two random vectors? what's the "
        "smallest angle you'd expect to see in 2,000 tries?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _rows = []
    for _d in (2, 10, 100, 1000, 10000):
        _U = _rng.normal(size=(2000, _d))
        _V = _rng.normal(size=(2000, _d))
        _cos = np.sum(_U * _V, axis=1) / (np.linalg.norm(_U, axis=1) * np.linalg.norm(_V, axis=1))
        _ang = np.degrees(np.arccos(np.clip(_cos, -1, 1)))
        _ax.hist(_ang, bins=np.linspace(0, 180, 181), histtype="step", density=True, label=f"d = {_d}")
        _rows.append(f"| {_d} | {np.std(_cos):.4f} | {1 / np.sqrt(_d):.4f} | "
                     f"{np.min(np.minimum(_ang, 180 - _ang)):.1f}° |")
    _ax.set(xlabel="angle (degrees)", ylabel="density", xlim=(0, 180))
    _ax.legend()
    mo.vstack([_fig, mo.md("| d | std of cos(angle) | 1/√d | closest to parallel (of 2,000) |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** $u^\top v = \sum_i u_iv_i$ is a sum of $d$ independent terms with mean 0 and
    variance 1, while $\|u\|\,\|v\| \approx d$. what does that make the typical size of
    $\cos\theta$? why does this matter for nearest-neighbour search and for embeddings in
    high dimensions?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what is the rank of a noisy matrix?

    $A$ is a $30 \times 20$ matrix of rank exactly 3, plus a little noise $E$ with entries of
    size `noise`. we compute your `rank(A + E, tol=...)` for tolerances from $10^{-16}$ to $1$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with noise = 1e-8, what does rank(A + E) return? does the answer depend on tol, and how?"
    )
    p2
    return (p2,)


@app.cell
def _():
    noise = mo.ui.dropdown(["0", "1e-14", "1e-11", "1e-8", "1e-5", "1e-2"], value="1e-8", label="noise level")
    noise
    return (noise,)


@app.cell
def _(noise, p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _A = _rng.normal(size=(30, 3)) @ _rng.normal(size=(3, 20))
    _tols = np.logspace(-16, 0, 49)
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    for _lvl in ("0", "1e-14", "1e-11", "1e-8", "1e-5", "1e-2"):
        _E = float(_lvl) * np.random.default_rng(2).normal(size=(30, 20))
        _ranks = [ex.rank(_A + _E, tol=_t) for _t in _tols]
        _ax.semilogx(_tols, _ranks, "-", lw=3 if _lvl == noise.value else 1,
                     label=f"noise {_lvl}" + (" (selected)" if _lvl == noise.value else ""))
    _ax.axhline(3, color="gray", ls=":", lw=1)
    _ax.set(xlabel="tolerance tol", ylabel="rank(A + E, tol)", yticks=[0, 3, 5, 10, 15, 20])
    _ax.legend(fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each curve is a staircase. where does it step from 20 down to 3, and why
    there? so what *is* "the rank" of a measured data matrix? (topic 06 gives a better tool,
    the singular values, but the tolerance never goes away.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: when are random vectors a basis?

    $d$ random vectors in $\mathbb{R}^d$ with **continuous** (Gaussian) entries are a basis with
    probability 1. what if each entry is a coin flip, $\pm 1$? we ask your `is_basis` about 1,000
    random sign matrices for each $d$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what fraction of random ±1 matrices fail to be a basis at d = 2? at d = 4? at d = 10? "
        "does the fraction go up or down as d grows?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _ds = np.arange(1, 11)
    _fail_sign, _fail_gauss = [], []
    for _d in _ds:
        _fail_sign.append(np.mean([not ex.is_basis(_rng.choice([-1.0, 1.0], size=(_d, _d)))
                                   for _ in range(1000)]))
        _fail_gauss.append(np.mean([not ex.is_basis(_rng.normal(size=(_d, _d))) for _ in range(200)]))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    _ax.plot(_ds, _fail_sign, "o-", label="±1 entries")
    _ax.plot(_ds, _fail_gauss, "s-", label="Gaussian entries")
    _ax.set(xlabel="dimension d", ylabel="fraction NOT a basis", ylim=(-0.03, 1))
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at $d = 2$, list the ways two $\pm1$ vectors can be dependent. why can't that
    happen (with positive probability) for Gaussian vectors? the fraction eventually goes to
    zero, but how quickly does it get there?
    """)
    return


if __name__ == "__main__":
    app.run()
