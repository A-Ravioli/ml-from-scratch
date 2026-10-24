import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment  # unlocked by your topic 15 solver

    from mlfs import checks, lesson, notebooks

    def _ot15_smoke(m):
        """touch the topic 15 functions this notebook uses."""
        m.kantorovich_lp(np.array([1.0]), np.array([1.0]), np.zeros((1, 1)))
        m.barycentric_map(np.ones((1, 1)), np.zeros((1, 2)))

    def _ot16_smoke(m):
        """touch the topic 16 function this notebook uses."""
        m.sinkhorn_log(np.array([1.0]), np.array([1.0]), np.zeros((1, 1)), 1.0)

    # your topic 15 and 16 code: the reference under pytest and verify.py (or with MLFS_REFERENCE=ot15,ot16), and also
    # whenever a stage it needs is still unfinished
    ot15_load = notebooks.load_topic("15-wasserstein-distances", key="ot15", smoke=_ot15_smoke)
    ot16_load = notebooks.load_topic("16-entropic-ot-and-sinkhorn", key="ot16", smoke=_ot16_smoke)
    ot15, ot16 = ot15_load.module, ot16_load.module


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # OT in machine learning · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. your topic 15 code is available as `ot15` (`kantorovich_lp`,
    `barycentric_map`, …) and your topic 16 code as `ot16` (`sinkhorn_log`, …).

    | stage | you build |
    |---|---|
    | 1 | `minibatch_ot_coupling`, `pairing_cost`: exact OT pairings of noise and data within a batch |
    | 2 | `sinkhorn_coupling`: pairs sampled from an entropic plan |
    | 3 | `sinkhorn_loss`: a differentiable (debiased) Sinkhorn loss, by the envelope theorem |
    | 4 | `ot_domain_adaptation`: carry labelled source data onto a shifted target domain |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    ot15_load.callout()
    return


@app.cell(hide_code=True)
def _():
    ot16_load.callout()
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def minibatch_ot_coupling(x0, x1):
    """stage 1: x0 (B, ...) reordered so that pairing x0[k] with x1[k] minimizes the total squared distance."""
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {tuple(x0.shape)} and {tuple(x1.shape)}")
    a = x0.detach().reshape(len(x0), -1).to("cpu", torch.float64)
    b = x1.detach().reshape(len(x1), -1).to("cpu", torch.float64)
    C = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)  # C[k, i]: cost of pairing data point k with noise i
    _, cols = linear_sum_assignment(C.numpy())
    return x0[torch.as_tensor(cols, device=x0.device)]


@app.function
def pairing_cost(x0, x1):
    """stage 1: mean over k of ||x0[k] - x1[k]||^2 (summed over the non-batch dims), a float."""
    d = (x0.detach() - x1.detach()).reshape(len(x0), -1).double()
    return float((d ** 2).sum(1).mean())


@app.cell(hide_code=True)
def _():
    checks.run(1, minibatch_ot_coupling=minibatch_ot_coupling, pairing_cost=pairing_cost)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sinkhorn_coupling(x0, x1, eps, *, generator=None, max_iters=1000, tol=1e-9):
    """stage 2: x0[idx] with idx[k] drawn from column k of the entropic plan (ot16.sinkhorn_log, squared cost)."""
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {tuple(x0.shape)} and {tuple(x1.shape)}")
    B = len(x0)
    a = x0.detach().reshape(B, -1).to("cpu", torch.float64)
    b = x1.detach().reshape(B, -1).to("cpu", torch.float64)
    C = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).numpy()  # C[i, k]: noise i, data k
    w = np.full(B, 1 / B)
    P, _, _ = ot16.sinkhorn_log(w, w, C, eps, max_iters=max_iters, tol=tol)
    Q = torch.as_tensor((P / P.sum(0, keepdims=True)).T)  # row k: the distribution of the noise index for data k
    idx = torch.multinomial(Q, 1, replacement=True, generator=generator).squeeze(1)
    return x0[idx.to(x0.device)]


@app.cell(hide_code=True)
def _():
    checks.run(2, sinkhorn_coupling=sinkhorn_coupling, minibatch_ot_coupling=minibatch_ot_coupling,
               pairing_cost=pairing_cost)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def sinkhorn_loss(x, y, eps, *, debiased=True, max_iters=1000, tol=1e-9):
    """stage 3: OT_eps(x, y) (debiased: minus half of each self-term) as a 0-dim tensor with envelope-theorem gradients."""

    def ot(u, v):
        C = ((u[:, None, :] - v[None, :, :]) ** 2).sum(-1)
        n, m = C.shape
        a, b = np.full(n, 1 / n), np.full(m, 1 / m)
        _, f, g = ot16.sinkhorn_log(a, b, C.detach().to("cpu", torch.float64).numpy(), eps,
                                    max_iters=max_iters, tol=tol)
        f, g = (torch.as_tensor(z, dtype=C.dtype, device=C.device) for z in (f, g))
        at, bt = (torch.as_tensor(z, dtype=C.dtype, device=C.device) for z in (a, b))
        mass = at[:, None] * bt[None, :] * (torch.exp((f[:, None] + g[None, :] - C) / eps) - 1)
        return (f * at).sum() + (g * bt).sum() - eps * mass.sum()

    out = ot(x, y)
    if debiased:
        out = out - ot(x, x) / 2 - ot(y, y) / 2
    return out


@app.cell(hide_code=True)
def _():
    checks.run(3, sinkhorn_loss=sinkhorn_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ot_domain_adaptation(Xs, Xt, *, eps=None, max_iters=1000):
    """stage 4: (n_s, d) barycentric images of the source points under the exact (eps=None) or entropic OT plan."""
    Xs, Xt = np.asarray(Xs, dtype=float), np.asarray(Xt, dtype=float)
    a, b = np.full(len(Xs), 1 / len(Xs)), np.full(len(Xt), 1 / len(Xt))
    C = ((Xs[:, None, :] - Xt[None, :, :]) ** 2).sum(-1)
    if eps is None:
        P, _ = ot15.kantorovich_lp(a, b, C)
    else:
        P, _, _ = ot16.sinkhorn_log(a, b, C, eps, max_iters=max_iters, tol=1e-9)
    return ot15.barycentric_map(P, Xt)


@app.cell(hide_code=True)
def _():
    checks.run(4, ot_domain_adaptation=ot_domain_adaptation)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it compares a WGAN critic
    with exact and entropic transport on minibatches, adapts a classifier across a domain
    shift, and ends with the block's **project**: does a minibatch OT coupling straighten
    your chapter 22 flow-matching paths, and what does that buy at few sampling steps?
    """)
    return


if __name__ == "__main__":
    app.run()
