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
def lu_no_pivot(A):
    """Gaussian elimination WITHOUT row swaps (for comparison only): returns (L, U)."""
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.eye(n)
    for k in range(n - 1):
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
    return L, U


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # solving linear systems · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: which matrices break elimination without pivoting?

    the system

    $$\begin{pmatrix}\delta & 1\\ 1 & 1\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix}
    = \begin{pmatrix}1\\2\end{pmatrix}$$

    has solution $x \approx (1, 1)$ for every small $\delta$. as $\delta \to 0$ the
    matrix approaches $\begin{pmatrix}0&1\\1&1\end{pmatrix}$, which is perfectly well behaved
    (its inverse has entries of size 1). we solve it two ways: plain elimination with no row
    swaps, and your pivoted `solve`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "as δ shrinks from 1e-2 to 1e-20, what happens to the error of elimination WITHOUT "
        "pivoting? and with your pivoted solve?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _eps = np.logspace(-1, -20, 40)
    _b = np.array([1.0, 2.0])
    _err_np, _err_p = [], []
    for _e in _eps:
        _A = np.array([[_e, 1.0], [1.0, 1.0]])
        _exact = np.array([1.0 / (1.0 - _e), (1.0 - 2.0 * _e) / (1.0 - _e)])
        _L, _U = lu_no_pivot(_A)
        _x_np = ex.solve_triangular(_U, ex.solve_triangular(_L, _b, unit_diagonal=True), lower=False)
        _x_p = ex.solve(_A, _b)
        _err_np.append(np.abs(_x_np - _exact).max() + 1e-18)
        _err_p.append(np.abs(_x_p - _exact).max() + 1e-18)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.5))
    _ax.loglog(_eps, _err_np, "o-", ms=3, label="no pivoting")
    _ax.loglog(_eps, _err_p, "s-", ms=3, label="partial pivoting (your solve)")
    _ax.invert_xaxis()
    _ax.set(xlabel="δ (shrinking →)", ylabel="max error in x", title="the tiny-pivot trap")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the matrix gets *nicer* as δ → 0, yet one method gets *worse*. write out
    the elimination by hand: what is the multiplier, what is $u_{22}$, and which number gets
    rounded away? why does swapping rows fix it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: can partial pivoting fail?

    partial pivoting keeps every multiplier $|\ell_{ik}| \le 1$. the danger left is **growth**:
    entries of $U$ becoming much larger than entries of $A$, so rounding errors (relative to
    $\max|U|$) swamp the answer. the *growth factor* is $\rho = \max|U| / \max|A|$.

    below we compare random Gaussian matrices with one specific matrix that Wilkinson wrote
    down in 1961: ones on the diagonal and in the last column, $-1$ everywhere below the
    diagonal.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for n = 10 … 60, how large does the growth factor get for random matrices? for "
        "Wilkinson's matrix? does your solve still get the right answer at n = 60?"
    )
    p2
    return (p2,)


@app.cell
def _():
    n_max = mo.ui.slider(20, 70, value=60, step=5, label="largest n")
    n_max
    return (n_max,)


@app.function
def wilkinson(n):
    """Wilkinson's growth matrix: 1 on the diagonal and last column, -1 below the diagonal."""
    W = np.eye(n) - np.tril(np.ones((n, n)), -1)
    W[:, -1] = 1.0
    return W


@app.cell
def _(n_max, p2):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _ns = np.arange(10, n_max.value + 1, 5)
    _g_rand, _g_wilk, _e_rand, _e_wilk = [], [], [], []
    for _n in _ns:
        _x_true = _rng.normal(size=_n)
        for _A, _g, _e in ((_rng.normal(size=(_n, _n)), _g_rand, _e_rand),
                           (wilkinson(_n), _g_wilk, _e_wilk)):
            _, _, _U = ex.lu(_A)
            _g.append(np.abs(_U).max() / np.abs(_A).max())
            _x = ex.solve(_A, _A @ _x_true)
            _e.append(np.linalg.norm(_x - _x_true) / np.linalg.norm(_x_true) + 1e-18)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    _a1.semilogy(_ns, _g_rand, "o-", label="random")
    _a1.semilogy(_ns, _g_wilk, "s-", label="Wilkinson")
    _a1.set(xlabel="n", ylabel="growth factor ρ", title="growth")
    _a1.legend()
    _a2.semilogy(_ns, _e_rand, "o-", label="random")
    _a2.semilogy(_ns, _e_wilk, "s-", label="Wilkinson")
    _a2.set(xlabel="n", ylabel="relative error of x", title="accuracy of your solve")
    _a2.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what is the growth factor of Wilkinson's matrix as a function of $n$
    (look at the last column of $U$)? if partial pivoting can fail this badly, why does
    every library still use it? (hint: how often did a random matrix come close?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how many symmetric matrices are positive definite?

    your `cholesky` doubles as a test: it succeeds exactly when a symmetric matrix is positive
    definite. take a random symmetric matrix $S = (G + G^\top)/2$ with $G$ full of standard
    normal entries, and ask your `cholesky` whether it's positive definite.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what fraction of random symmetric 1×1, 2×2, 4×4 and 8×8 matrices are positive definite?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(1)
    _trials = 2000
    _rows = []
    for _n in (1, 2, 3, 4, 6, 8):
        _hits = 0
        for _ in range(_trials):
            _G = _rng.normal(size=(_n, _n))
            try:
                ex.cholesky((_G + _G.T) / 2)
                _hits += 1
            except np.linalg.LinAlgError:
                pass
        _rows.append(f"| {_n} | {_hits / _trials:.4f} |")
    _covs = 0
    for _ in range(200):
        _X = _rng.normal(size=(50, 8))
        _C = np.cov(_X, rowvar=False)
        try:
            ex.cholesky(_C)
            _covs += 1
        except np.linalg.LinAlgError:
            pass
    mo.md("| n | fraction positive definite |\n|---|---|\n" + "\n".join(_rows)
          + f"\n\nfor comparison, sample covariance matrices of 50 points in 8-D: "
          f"**{_covs}/200** pass.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the fraction collapse so fast with $n$? (think about what has to
    be true of *every* eigenvalue.) and why do covariance matrices always pass? write
    $C = \frac{1}{m-1} X_c^\top X_c$ and compute $v^\top C v$.
    """)
    return


if __name__ == "__main__":
    app.run()
