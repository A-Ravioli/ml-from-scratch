import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch.func import functional_call, jacrev, vmap

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # regularization approaches · build

    spec-driven (L4): the **spec** boxes in the lesson define every function and class, the
    checks enforce them, and the internals are yours. models are `nn.Module`s mapping `(B, d)`
    inputs to `(B, n_classes)` logits; tasks are topic 12's dicts
    `{"X": float32 (n, d), "y": int64 (n,), "classes": tuple}`. importances and anchors are
    dicts `name -> tensor` over the model's trainable parameters (`model.named_parameters()`).

    | stage | you build |
    |---|---|
    | 1 | `fisher_diagonal`: the diagonal Fisher information, exact or empirical, from per-example gradients |
    | 2 | `EWC`: elastic weight consolidation (`consolidate`, `penalty`), plain or online |
    | 3 | `SI`: synaptic intelligence (`begin_task`, `after_step`, `consolidate`, `penalty`) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def fisher_diagonal(model, X, *, y=None, classes=None, mode="exact", chunk_size=64):
    """stage 1: dict name -> mean over examples of the squared per-example gradient of log p (exact: averaged over y ~ p)."""
    raise NotImplementedError("stage1: implement fisher_diagonal")


@app.cell(hide_code=True)
def _():
    checks.run(1, fisher_diagonal=fisher_diagonal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class EWC:
    """elastic weight consolidation: a quadratic pull towards old solutions, weighted by the Fisher."""

    def __init__(self, lam, *, mode="exact", n_samples=1000, online=False, gamma=1.0):
        """stage 2: store the settings; no consolidated tasks yet."""
        raise NotImplementedError("stage2: implement EWC.__init__")

    def consolidate(self, model, task, *, scenario="class"):
        """stage 2: Fisher on the task's first n_samples examples (classes restricted if scenario='task'); anchor = current weights."""
        raise NotImplementedError("stage2: implement EWC.consolidate")

    def penalty(self, model):
        """stage 2: (lam / 2) * sum over consolidated tasks of sum_i F_i (theta_i - anchor_i)^2, a differentiable scalar."""
        raise NotImplementedError("stage2: implement EWC.penalty")


@app.cell(hide_code=True)
def _():
    checks.run(2, EWC=EWC, fisher_diagonal=fisher_diagonal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class SI:
    """synaptic intelligence: importance = each weight's share of the loss decrease along the training path."""

    def __init__(self, c, *, xi=0.1):
        """stage 3: store c and xi; importances start at zero (created at the first begin_task)."""
        raise NotImplementedError("stage3: implement SI.__init__")

    def begin_task(self, model):
        """stage 3: remember the weights at the start of the task and reset the running path integral w."""
        raise NotImplementedError("stage3: implement SI.begin_task")

    def after_step(self, model):
        """stage 3: after an optimizer step, w += -grad * (theta - theta_prev), then theta_prev = theta."""
        raise NotImplementedError("stage3: implement SI.after_step")

    def consolidate(self, model):
        """stage 3: Omega += w / ((theta - theta_start)^2 + xi); the anchor becomes the current weights."""
        raise NotImplementedError("stage3: implement SI.consolidate")

    def penalty(self, model):
        """stage 3: c * sum_i Omega_i (theta_i - anchor_i)^2 (zero before the first consolidate), differentiable."""
        raise NotImplementedError("stage3: implement SI.penalty")


@app.cell(hide_code=True)
def _():
    checks.run(3, SI=SI)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it plugs your `EWC` and `SI`
    into the harness you built in topic 12 (`train_task(..., penalty=..., after_step=...)`).
    """)
    return


if __name__ == "__main__":
    app.run()
