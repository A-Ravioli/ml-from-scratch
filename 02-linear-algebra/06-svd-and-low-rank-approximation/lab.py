import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.linalg import eigh  # yours, from topic 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def square_image(angle_deg, size=64, half_width=16):
    """A filled square in the middle of a size x size image, rotated by angle_deg."""
    c = (size - 1) / 2
    y, x = np.mgrid[0:size, 0:size] - c
    th = np.deg2rad(angle_deg)
    u = np.cos(th) * x + np.sin(th) * y
    v = -np.sin(th) * x + np.cos(th) * y
    return ((np.abs(u) <= half_width) & (np.abs(v) <= half_width)).astype(float)


@app.function
def rank_for_energy(s, fraction):
    """Smallest k such that the top-k singular values hold `fraction` of sum(s**2)."""
    energy = np.cumsum(s**2) / np.sum(s**2)
    return int(np.searchsorted(energy, fraction) + 1)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # SVD & low-rank approximation · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: image compression, and what "low rank" means for a picture

    a grayscale image is a matrix. its best rank-$k$ approximation (your `low_rank_approx`)
    needs $k(m + n + 1)$ numbers instead of $mn$. first a face (Olivetti; a synthetic stand-in
    when offline), then two images of *the same square*, upright and rotated by 45°.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "(a) what rank makes the 64×64 face recognizable? (b) the upright square and the 45° "
        "square have the same area and the same pixel values. what rank does each need to keep "
        "99.9% of its energy?"
    )
    p1
    return (p1,)


@app.cell
def _():
    k = mo.ui.slider(1, 64, value=8, label="rank k")
    k
    return (k,)


@app.cell
def _(k, p1):
    ui.gate(p1)
    _faces, _ = datasets.olivetti_faces()
    _face = _faces[0]
    _approx = ex.low_rank_approx(_face, k.value)
    _err = np.linalg.norm(_face - _approx) / np.linalg.norm(_face)
    _store = k.value * (64 + 64 + 1) / (64 * 64)
    _sq0, _sq45 = square_image(0), square_image(45)
    _fig, _axes = plt.subplots(1, 5, figsize=(13, 2.9))
    _axes[0].imshow(_face, cmap="gray")
    _axes[0].set_title("original")
    _axes[1].imshow(_approx, cmap="gray")
    _axes[1].set_title(f"rank {k.value}: error {_err:.1%}\nstorage {_store:.0%}")
    _axes[2].imshow(ex.low_rank_approx(_sq0, min(k.value, 64)), cmap="gray")
    _axes[2].set_title(f"square 0°, rank {k.value}")
    _axes[3].imshow(ex.low_rank_approx(_sq45, min(k.value, 64)), cmap="gray")
    _axes[3].set_title(f"square 45°, rank {k.value}")
    for _a in _axes[:4]:
        _a.axis("off")
    _rows = []
    for _name, _img in (("face", _face), ("square 0°", _sq0), ("square 45°", _sq45)):
        _s = ex.svd(_img)[1]
        _axes[4].semilogy(_s / _s[0] + 1e-17, label=_name)
        _rows.append(f"| {_name} | {rank_for_energy(_s, 0.999)} | {int(np.sum(_s > 1e-10 * _s[0]))} |")
    _axes[4].set(title="singular values (scaled)", xlabel="index", ylim=(1e-17, 2))
    _axes[4].legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| image | rank for 99.9% energy | numerical rank |\n|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the upright square exactly rank 1? (write it as an outer product of
    two vectors.) what about rotation destroys that, even though the picture is "equally
    simple"? what does this tell you about using the SVD on images versus on data whose
    structure isn't aligned with the axes?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: why not just use the eigenvalues of $A^\top A$?

    the singular values of $A$ are the square roots of the eigenvalues of $A^\top A$. so you
    could compute an SVD with your topic-05 `eigh`: $\sigma_i = \sqrt{\lambda_i(A^\top A)}$. we
    build a $20 \times 16$ matrix whose singular values run from $1$ down to $10^{-15}$ and
    compare that recipe with your `svd`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "how many of the 16 singular values does each method get right to 3 significant digits?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _s_true = np.logspace(0, -15, 16)
    _U, _ = ex.qr(_rng.normal(size=(20, 16)))
    _V, _ = ex.qr(_rng.normal(size=(16, 16)))
    _A = (_U * _s_true) @ _V.T
    _s_svd = ex.svd(_A)[1]
    _lam = eigh(_A.T @ _A)[0][::-1]
    _s_gram = np.sqrt(np.abs(_lam))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.semilogy(_s_true, "k-", lw=3, alpha=0.3, label="true")
    _ax.semilogy(_s_svd, "o", label="your svd (one-sided Jacobi)")
    _ax.semilogy(_s_gram, "s", mfc="none", label=r"$\sqrt{\mathrm{eig}(A^\top A)}$")
    _ax.set(xlabel="index i", ylabel=r"$\sigma_i$")
    _ax.legend()
    _good_svd = int(np.sum(np.abs(_s_svd - _s_true) / _s_true < 1e-3))
    _good_gram = int(np.sum(np.abs(_s_gram - _s_true) / _s_true < 1e-3))
    _negs = int(np.sum(_lam < 0))
    mo.vstack([_fig, mo.md(
        f"correct to 3 digits: **your svd {_good_svd}/16**, **eig(AᵀA) {_good_gram}/16**. "
        f"eigh returned {_negs} *negative* eigenvalues for the positive semidefinite AᵀA.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the eigenvalues of $A^\top A$ are computed to an absolute accuracy of about
    $\varepsilon\,\|A^\top A\| = \varepsilon\sigma_1^2$. what does that mean for $\sigma_i$ near
    $10^{-8}$? and why does orthogonalizing the columns of $A$ directly avoid the problem?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: PCA is an SVD (of centered data)

    300 points from a tilted Gaussian cloud centered at $(5, 3)$. the first right singular
    vector of the data matrix $X$ (rows = points) is the first principal direction, *if* you
    center the data first. what if you forget?
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the cloud's long axis points along (1, -1). where does the first right singular vector "
        "point for centered data? for uncentered data?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(1)
    _z = _rng.normal(size=(300, 2)) * np.array([2.0, 0.4])
    _R = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)  # row 1 = long axis (1, -1)
    _X = _z @ _R + np.array([5.0, 3.0])
    _v_c = ex.svd(_X - _X.mean(axis=0))[2][0]
    _v_u = ex.svd(_X)[2][0]
    _fig, _ax = plt.subplots(figsize=(5.5, 4.5))
    _ax.scatter(_X[:, 0], _X[:, 1], s=6, alpha=0.5)
    _m = _X.mean(axis=0)
    for _v, _c, _lbl in ((_v_c, "C3", "centered"), (_v_u, "C2", "uncentered")):
        _v = _v * np.sign(_v[np.argmax(np.abs(_v))])
        _ax.annotate("", xy=_m + 3 * _v, xytext=_m, arrowprops=dict(color=_c, lw=2.5))
        _ax.plot([], [], color=_c, lw=2.5, label=f"{_lbl}: ({_v[0]:+.2f}, {_v[1]:+.2f})")
    _ax.plot([0, _m[0]], [0, _m[1]], "k:", lw=1, label="origin → mean")
    _ax.scatter([0], [0], color="k", s=15)
    _ax.set_aspect("equal")
    _ax.legend(fontsize=8, loc="lower left")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the first right singular vector maximizes $\|Xv\|^2 = \sum_i (x_i^\top v)^2$
    over unit $v$. for uncentered data, what dominates that sum? why does centering turn
    $\|Xv\|^2$ into ($n$ times) the variance along $v$?
    """)
    return


if __name__ == "__main__":
    app.run()
