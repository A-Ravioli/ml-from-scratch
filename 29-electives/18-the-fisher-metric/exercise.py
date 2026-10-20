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
    # the fisher metric · build

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
    raise NotImplementedError("stage1: implement categorical_fisher")


@app.function
def gaussian_fisher(mean, log_std):
    """stage 1: Fisher of N(mean, exp(log_std)^2) w.r.t. (mean, log_std); a (2, 2) tensor."""
    raise NotImplementedError("stage1: implement gaussian_fisher")


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
    raise NotImplementedError("stage2: implement fisher_exact")


@app.function
def kl_hessian(kl, theta0):
    """stage 2: the Hessian of theta -> kl(theta0, theta) at theta = theta0; (P, P)."""
    raise NotImplementedError("stage2: implement kl_hessian")


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
    raise NotImplementedError("stage3: implement logit_jacobian")


@app.function
def true_fisher(model, X):
    """stage 3: (1/N) sum_n E_{y ~ p(.|x_n)}[g g^T], g = grad_params log p(y | x_n); (P, P), parameters() order."""
    raise NotImplementedError("stage3: implement true_fisher")


@app.function
def empirical_fisher(model, X, y):
    """stage 3: (1/N) sum_n g_n g_n^T with g_n = grad_params log p(y_n | x_n) at the GIVEN labels; (P, P)."""
    raise NotImplementedError("stage3: implement empirical_fisher")


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
    raise NotImplementedError("stage4: implement pullback_fisher")


@app.function
def fisher_rao_length(fisher_fn, path):
    """stage 4: sum over segments of sqrt(d^T F(midpoint) d) for a path (T, P), T >= 2; a float."""
    raise NotImplementedError("stage4: implement fisher_rao_length")


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
