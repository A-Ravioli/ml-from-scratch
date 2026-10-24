import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import logsumexp, pairwise_distances  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # entropic OT & sinkhorn · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `sinkhorn`: entropic OT by alternating scalings of $e^{-C/\varepsilon}$ |
    | 2 | `sinkhorn_log`: the same iterations on dual potentials, stable for small $\varepsilon$ |
    | 3 | `entropic_primal`, `entropic_dual`: the two objectives, and the gap between them |
    | 4 | `sinkhorn_divergence`: the debiased Sinkhorn divergence between point clouds |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def sinkhorn(a, b, C, eps, *, max_iters=1000, tol=1e-9):
    """stage 1: the (n, m) entropic plan diag(u) K diag(v), K = exp(-C / eps), by Sinkhorn's scalings from v = 1."""
    raise NotImplementedError("stage1: implement sinkhorn")


@app.cell(hide_code=True)
def _():
    checks.run(1, sinkhorn=sinkhorn)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sinkhorn_log(a, b, C, eps, *, max_iters=1000, tol=1e-9):
    """stage 2: (P, f, g): Sinkhorn on the potentials with log-sum-exp, from g = 0; P_ij = a_i b_j e^((f_i+g_j-C_ij)/eps)."""
    raise NotImplementedError("stage2: implement sinkhorn_log")


@app.cell(hide_code=True)
def _():
    checks.run(2, sinkhorn_log=sinkhorn_log)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def entropic_primal(P, a, b, C, eps):
    """stage 3: <C, P> + eps KL(P || a b^T) with 0 log 0 = 0, a float."""
    raise NotImplementedError("stage3: implement entropic_primal")


@app.function
def entropic_dual(f, g, a, b, C, eps):
    """stage 3: <f, a> + <g, b> - eps sum_ij a_i b_j (exp((f_i + g_j - C_ij) / eps) - 1), a float."""
    raise NotImplementedError("stage3: implement entropic_dual")


@app.cell(hide_code=True)
def _():
    checks.run(3, entropic_primal=entropic_primal, entropic_dual=entropic_dual, sinkhorn_log=sinkhorn_log)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def sinkhorn_divergence(X, Y, eps, *, a=None, b=None, p=2, max_iters=1000, tol=1e-9):
    """stage 4: OT_eps(a, b) - OT_eps(a, a) / 2 - OT_eps(b, b) / 2 for points X (n, d), Y (m, d), cost ||x - y||^p."""
    raise NotImplementedError("stage4: implement sinkhorn_divergence")


@app.cell(hide_code=True)
def _():
    checks.run(4, sinkhorn_divergence=sinkhorn_divergence, sinkhorn_log=sinkhorn_log, entropic_dual=entropic_dual)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it measures how the
    regularization strength trades blur for speed, races Sinkhorn against your exact LP from
    topic 15, and shows what goes wrong when you fit a point cloud with the *biased* entropic
    cost. topic 17 puts your `sinkhorn_log` to work inside flow matching and as a loss.
    """)
    return


if __name__ == "__main__":
    app.run()
