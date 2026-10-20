import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # K-FAC · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. every function works in the dtype of its inputs (the
    checks use float64, the lab float32).

    | stage | you build |
    |---|---|
    | 1 | `damped_inverses`, `kron_precondition`: inverting a Kronecker product one factor at a time |
    | 2 | `kfac_factors`: the two factors of every linear layer's Fisher block, from activations and backpropagated gradients |
    | 3 | `kfac_step`: one K-FAC step from given factors, with KL clipping |
    | 4 | `KFAC`: the optimizer, with running-average factors, sampled labels, periodic inverses and momentum |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def damped_inverses(A, G, damping):
    """stage 1: ((A + pi sqrt(damping) I)^-1, (G + sqrt(damping)/pi I)^-1), pi = sqrt((tr A / dim A) / (tr G / dim G))."""
    raise NotImplementedError("stage1: implement damped_inverses")


@app.function
def kron_precondition(grad, A_inv, G_inv):
    """stage 1: (G kron A)^-1 vec(grad) as a matrix, G_inv @ grad @ A_inv; grad is (out, in + 1)."""
    raise NotImplementedError("stage1: implement kron_precondition")


@app.cell(hide_code=True)
def _():
    checks.run(1, damped_inverses=damped_inverses, kron_precondition=kron_precondition)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def linear_layers(model):
    """helper: the nn.Linear modules of `model`, in model.modules() order."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


@app.function
def kfac_factors(model, X, labels=None):
    """stage 2: [(A, G) per nn.Linear]: A = mean of [a, 1][a, 1]^T, G = mean of g g^T, g = d log p(y|x) / d(layer output)."""
    raise NotImplementedError("stage2: implement kfac_factors")


@app.cell(hide_code=True)
def _():
    checks.run(2, kfac_factors=kfac_factors)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def layer_grad(layer):
    """helper: [dW | db] of an nn.Linear from its .grad fields, shape (out, in + 1)."""
    return torch.cat([layer.weight.grad, layer.bias.grad.unsqueeze(1)], dim=1)


@app.function
def kfac_step(model, factors, *, lr, damping, kl_clip=None):
    """stage 3: precondition each Linear's .grad by its factors and update in place; returns the KL-clip scale nu."""
    raise NotImplementedError("stage3: implement kfac_step")


@app.cell(hide_code=True)
def _():
    checks.run(3, kfac_step=kfac_step, kfac_factors=kfac_factors)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class KFAC:
    """K-FAC for MLPs built from nn.Linear layers (other layers must have no parameters)."""

    def __init__(self, model, lr=0.1, damping=1e-2, *, momentum=0.9, ema_decay=0.95, stat_every=1, inv_every=10,
                 kl_clip=1e-3, fisher="sample", seed=None):
        """stage 4: store the settings; factors, inverses and momentum buffers start empty, self.steps = 0."""
        raise NotImplementedError("stage4: implement KFAC.__init__")

    def step(self, X, y):
        """stage 4: one K-FAC step on the minibatch (X, y): mean cross-entropy, factors, inverses, update; returns the loss."""
        raise NotImplementedError("stage4: implement KFAC.step")


@app.cell(hide_code=True)
def _():
    checks.run(4, KFAC=KFAC, kfac_factors=kfac_factors, kfac_step=kfac_step)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it ends with the block's
    project, K-FAC against a tuned Adam on chapter 15's MNIST network.
    """)
    return


if __name__ == "__main__":
    app.run()
