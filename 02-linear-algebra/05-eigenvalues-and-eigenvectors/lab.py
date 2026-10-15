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

    PAGES = ["home", "blog", "about", "shop", "spam1", "spam2", "spam3", "spam4", "deals"]
    LINKS = {
        "home": ["blog", "about", "shop"],
        "blog": ["home", "shop"],
        "about": ["home"],
        "shop": ["home"],
        "spam1": ["deals"],
        "spam2": ["deals"],
        "spam3": ["deals"],
        "spam4": ["deals"],
        "deals": ["spam1"],
    }


@app.function
def google_matrix(alpha):
    """The PageRank matrix alpha * M + (1 - alpha)/n for the little web above (column-stochastic)."""
    n = len(PAGES)
    index = {p: i for i, p in enumerate(PAGES)}
    M = np.zeros((n, n))
    for src, outs in LINKS.items():
        for dst in outs:
            M[index[dst], index[src]] = 1.0 / len(outs)
    return alpha * M + (1.0 - alpha) / n


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # eigenvalues & eigenvectors · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: PageRank on a tiny web

    nine pages link to each other like this:

    | page | links to |
    |---|---|
    | home | blog, about, shop |
    | blog | home, shop |
    | about, shop | home |
    | spam1, spam2, spam3, spam4 | deals |
    | deals | spam1 |

    a random surfer follows a random outgoing link with probability $\alpha$ and jumps to a
    uniformly random page otherwise. the long-run fraction of time spent on each page is its
    PageRank: the eigenvector for eigenvalue 1 of $G = \alpha M + \frac{1-\alpha}{n}\mathbf{1}\mathbf{1}^\top$,
    where $M_{ij} = 1/\text{outdeg}(j)$ if $j$ links to $i$. your `power_iteration` finds it.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "rank the nine pages by PageRank, from first to last. what's your reasoning?"
    )
    p1
    return (p1,)


@app.cell
def _():
    alpha = mo.ui.slider(0.05, 0.99, value=0.85, step=0.01, label="α (probability of following a link)")
    alpha
    return (alpha,)


@app.cell
def _(alpha, p1):
    ui.gate(p1)
    _G = google_matrix(alpha.value)
    _lam, _v = ex.power_iteration(_G, rng=0)
    _rank = _v / _v.sum()  # fixes the sign and makes it a probability vector
    _indeg = {p: sum(p in outs for outs in LINKS.values()) for p in PAGES}
    _order = np.argsort(-_rank)
    _rows = "\n".join(f"| {k + 1} | {PAGES[i]} | {_rank[i]:.3f} | {_indeg[PAGES[i]]} |"
                      for k, i in enumerate(_order))
    mo.md(f"eigenvalue found: **{_lam:.10f}**\n\n| rank | page | PageRank | incoming links |\n"
          f"|---|---|---|---|\n{_rows}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does spam1 beat home? (PageRank is defined by $\pi = G\pi$: a page is
    important if important pages link to it.) what does the deals ↔ spam1 loop do to the
    random surfer, and how does α change that?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how fast does power iteration converge?

    we build symmetric matrices with eigenvalues $1, r, r/2, r/4, \dots$ and run your
    `power_iteration` for exactly $k$ steps (`max_iters=k`, `tol=0`), measuring two errors: the
    angle between $v_k$ and the true eigenvector, and $|\lambda_k - 1|$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with r = 0.9, about how many iterations until the eigenvector error is 1e-8? does the "
        "eigenvalue estimate converge faster, slower, or at the same rate as the vector?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ks = np.arange(1, 201, 2)
    for _r, _c in ((0.5, "C0"), (0.9, "C1"), (0.98, "C2")):
        _eigs = np.array([1.0, _r] + [_r / 2**j for j in range(1, 7)])
        _Q, _ = ex.qr(np.random.default_rng(1).normal(size=(8, 8)))
        _A = (_Q * _eigs) @ _Q.T
        _vec_err, _val_err = [], []
        for _k in _ks:
            _lam, _v = ex.power_iteration(_A, max_iters=int(_k), tol=0.0, rng=2)
            _q1 = _Q[:, 0]
            _vec_err.append(np.linalg.norm(_v - (_v @ _q1) * _q1) + 1e-17)  # sin of the angle
            _val_err.append(abs(_lam - 1) + 1e-17)
        _ax.semilogy(_ks, _vec_err, "-", color=_c, label=f"r = {_r}: eigenvector")
        _ax.semilogy(_ks, _val_err, "--", color=_c, label=f"r = {_r}: eigenvalue")
    _ax.axhline(1e-8, color="gray", lw=0.8)
    _ax.set(xlabel="iterations k", ylabel="error", ylim=(1e-17, 2))
    _ax.legend(fontsize=8, ncol=2)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** read off the slopes. the vector error shrinks like $r^k$. what does the
    eigenvalue error shrink like, and why? (write $v = \cos\theta\, q_1 + \sin\theta\, q_\perp$ and
    compute the Rayleigh quotient.) what plays the role of $r$ in PageRank?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: watching the QR algorithm, and breaking it

    the unshifted QR algorithm repeats $A_{k+1} = R_kQ_k$. below, the magnitude of every entry
    of $A_k$ for a symmetric matrix with eigenvalues $8, 4, 2, 1, 0.5$ (darker = larger), at the
    step chosen by the slider.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "now run it on the symmetric matrix [[0, 1], [1, 0]] (eigenvalues +1 and -1). how many "
        "iterations does the QR algorithm need?"
    )
    p3
    return (p3,)


@app.cell
def _():
    step = mo.ui.slider(0, 60, value=10, label="QR step k")
    step
    return (step,)


@app.cell
def _(p3, step):
    ui.gate(p3)
    _Q, _ = ex.qr(np.random.default_rng(3).normal(size=(5, 5)))
    _A = (_Q * np.array([8.0, 4.0, 2.0, 1.0, 0.5])) @ _Q.T
    for _ in range(step.value):
        _Qk, _Rk = ex.qr(_A)
        _A = _Rk @ _Qk
    _flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    _a1.imshow(np.log10(np.abs(_A) + 1e-16), cmap="Greys", vmin=-16, vmax=1)
    _a1.set_title(f"log10 |A_k|, k = {step.value}")
    for (_i, _j), _val in np.ndenumerate(_A):
        _a1.text(_j, _i, f"{_val:.1e}" if abs(_val) < 1e-2 else f"{_val:.2f}", ha="center",
                 va="center", fontsize=6, color="C3")
    _B = _flip.copy()
    _history = []
    for _ in range(25):
        _Qk, _Rk = ex.qr(_B)
        _B = _Rk @ _Qk
        _history.append(abs(_B[1, 0]))
    _a2.plot(_history, "o-")
    _a2.set(title="[[0,1],[1,0]]: |subdiagonal| per step", xlabel="step", ylim=(0, 1.2))
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"`qr_algorithm([[0,1],[1,0]])` returns **{ex.qr_algorithm(_flip).round(6).tolist()}**; "
              f"your `eigh` returns **{ex.eigh(_flip)[0].round(6).tolist()}**."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compute one step by hand for $\begin{pmatrix}0&1\\1&0\end{pmatrix}$: what are
    $Q$ and $R$, and what is $RQ$? the unshifted QR algorithm separates eigenvalues by their
    *magnitudes*. what goes wrong when two of them have equal magnitude? why doesn't Jacobi
    care?
    """)
    return


if __name__ == "__main__":
    app.run()
