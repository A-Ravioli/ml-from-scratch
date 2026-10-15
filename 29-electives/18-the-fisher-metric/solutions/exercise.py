import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson

    torch.set_default_dtype(torch.float64)  # curvature work needs float64


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the fisher metric · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. everything is float64 torch.

    | stage | you build |
    |---|---|
    | 1 | `categorical_fisher`, `gaussian_fisher`: the Fisher matrix of two common models, in closed form |
    | 2 | `fisher_exact`, `kl_hessian`: the Fisher matrix of any small model, two ways |
    | 3 | `true_fisher`, `empirical_fisher`: the Fisher matrix of a neural-network classifier, and its impostor |
    | 4 | `pullback_fisher`, `fisher_rao_length`: changing coordinates, and measuring curves on the manifold |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def categorical_fisher(logits):
    """stage 1: Fisher of Categorical(softmax(logits)) w.r.t. the logits; (..., K) -> (..., K, K)."""
    p = torch.softmax(logits, dim=-1)
    return torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)


@app.function
def gaussian_fisher(mean, log_std):
    """stage 1: Fisher of N(mean, exp(log_std)^2) w.r.t. (mean, log_std); a (2, 2) tensor."""
    log_std = torch.as_tensor(log_std, dtype=torch.float64)
    return torch.diag(torch.stack([torch.exp(-2 * log_std), torch.tensor(2.0)]))


@app.cell(hide_code=True)
def _():
    checks.run(1, categorical_fisher=categorical_fisher, gaussian_fisher=gaussian_fisher)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def fisher_exact(log_prob, theta, support):
    """stage 2: E_{x ~ p_theta}[s s^T] with s = grad_theta log_prob(theta, x), summing over a finite support; (P, P)."""
    theta = theta.detach().clone().requires_grad_(True)
    F = torch.zeros(theta.numel(), theta.numel())
    for x in support:
        lp = log_prob(theta, x)
        (s,) = torch.autograd.grad(lp, theta)
        s = s.reshape(-1)
        F += lp.detach().exp() * torch.outer(s, s)
    return F


@app.function
def kl_hessian(kl, theta0):
    """stage 2: the Hessian of theta -> kl(theta0, theta) at theta = theta0; (P, P)."""
    theta0 = theta0.detach().clone()
    return torch.autograd.functional.hessian(lambda t: kl(theta0, t), theta0.clone())


@app.cell(hide_code=True)
def _():
    checks.run(2, fisher_exact=fisher_exact, kl_hessian=kl_hessian, categorical_fisher=categorical_fisher)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def flat_params(model):
    """helper: all parameters of `model`, in model.parameters() order, flattened and concatenated (detached)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


@app.function
def logit_jacobian(model, X):
    """stage 3: (a suggested helper) d logits / d params for every example, (N, K, P), parameters() order."""
    params = list(model.parameters())
    logits = model(X)
    N, K = logits.shape
    J = torch.zeros(N, K, sum(p.numel() for p in params), dtype=logits.dtype)
    for n in range(N):
        for k in range(K):
            grads = torch.autograd.grad(logits[n, k], params, retain_graph=True, allow_unused=True)
            J[n, k] = torch.cat([torch.zeros(p.numel(), dtype=logits.dtype) if g is None else g.reshape(-1)
                                 for p, g in zip(params, grads)])
    return J


@app.function
def true_fisher(model, X):
    """stage 3: (1/N) sum_n E_{y ~ p(.|x_n)}[g g^T], g = grad_params log p(y | x_n); (P, P), parameters() order."""
    J = logit_jacobian(model, X)  # (N, K, P)
    with torch.no_grad():
        H = categorical_fisher(model(X))  # (N, K, K)
    return torch.einsum("nkp,nkl,nlq->pq", J, H, J) / X.shape[0]


@app.function
def empirical_fisher(model, X, y):
    """stage 3: (1/N) sum_n g_n g_n^T with g_n = grad_params log p(y_n | x_n) at the GIVEN labels; (P, P)."""
    J = logit_jacobian(model, X)  # (N, K, P)
    with torch.no_grad():
        p = torch.softmax(model(X), dim=-1)
    resid = torch.nn.functional.one_hot(y, p.shape[1]).to(p.dtype) - p  # d log p(y) / d logits
    G = torch.einsum("nk,nkp->np", resid, J)  # per-example score
    return G.T @ G / X.shape[0]


@app.cell(hide_code=True)
def _():
    checks.run(3, true_fisher=true_fisher, empirical_fisher=empirical_fisher)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def pullback_fisher(fisher, jacobian):
    """stage 4: the Fisher matrix in new coordinates psi, J^T F J with J = d theta / d psi (P_theta, P_psi)."""
    return jacobian.T @ fisher @ jacobian


@app.function
def fisher_rao_length(fisher_fn, path):
    """stage 4: sum over segments of sqrt(d^T F(midpoint) d) for a path (T, P), T >= 2; a float."""
    d = path[1:] - path[:-1]
    mid = 0.5 * (path[1:] + path[:-1])
    total = 0.0
    for m, step in zip(mid, d):
        q = step @ fisher_fn(m) @ step
        total += float(torch.sqrt(torch.clamp(q, min=0.0)))
    return total


@app.cell(hide_code=True)
def _():
    checks.run(4, pullback_fisher=pullback_fisher, fisher_rao_length=fisher_rao_length)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. topic 19 turns this metric
    into an optimizer.
    """)
    return


if __name__ == "__main__":
    app.run()
