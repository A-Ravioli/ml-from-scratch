import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.optim import minibatch_indices  # yours, from ch 06 · topic 04
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # optimization in practice · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. chapter 06 gave you optimizers; this
    topic gives you the controls practitioners actually turn: the learning rate *over time*
    (schedules and warmup), a quick experiment that finds a good learning rate before you
    train (the range test), and a safety valve for exploding gradients (clipping). they join
    **`mlfs.train`**, and the schedulers work unchanged with PyTorch's optimizers.

    | stage | you build |
    |---|---|
    | 1 | `get_lr` and `set_lr`: one interface to `mlfs.optim` and `torch.optim` |
    | 2 | `LRScheduler`, `StepLR`, `CosineSchedule` (with linear warmup) |
    | 3 | `clip_grad_norm`: rescale the gradients when their global norm is too large |
    | 4 | `lr_range_test` and `suggest_lr`: sweep the learning rate, read off a good one |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def get_lr(optimizer):
    """stage 1: the current learning rate: param_groups[0]['lr'] for a torch optimizer, else optimizer.lr."""
    raise NotImplementedError("stage1: implement get_lr")


@app.function
def set_lr(optimizer, lr):
    """stage 1: set the learning rate: every param_group's 'lr' for a torch optimizer, else optimizer.lr."""
    raise NotImplementedError("stage1: implement set_lr")


@app.cell(hide_code=True)
def _():
    checks.run(1, get_lr=get_lr, set_lr=set_lr)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class LRScheduler:
    """base scheduler: subclasses define lr_at(step); step() advances one optimizer step and sets that learning rate."""

    def __init__(self, optimizer):
        """stage 2: remember the optimizer, base_lr = get_lr(optimizer), last_step = 0, and set the lr to lr_at(0)."""
        raise NotImplementedError("stage2: implement LRScheduler.__init__")

    def lr_at(self, step):
        """stage 2: the learning rate for a given step; the base class keeps base_lr constant."""
        raise NotImplementedError("stage2: implement LRScheduler.lr_at")

    def step(self):
        """stage 2: last_step += 1, then set the optimizer's lr to lr_at(last_step)."""
        raise NotImplementedError("stage2: implement LRScheduler.step")


@app.class_definition
class StepLR(LRScheduler):
    """multiply the learning rate by gamma every step_size steps."""

    def __init__(self, optimizer, step_size, gamma=0.1):
        """stage 2: store step_size and gamma BEFORE calling the base __init__ (it calls lr_at(0))."""
        raise NotImplementedError("stage2: implement StepLR.__init__")

    def lr_at(self, step):
        """stage 2: base_lr * gamma ** (step // step_size)."""
        raise NotImplementedError("stage2: implement StepLR.lr_at")


@app.class_definition
class CosineSchedule(LRScheduler):
    """linear warmup from 0 to base_lr, then a half cosine down to min_lr at total_steps (held there afterwards)."""

    def __init__(self, optimizer, total_steps, *, warmup_steps=0, min_lr=0.0):
        """stage 2: store total_steps, warmup_steps, min_lr (ValueError unless 0 <= warmup_steps < total_steps), then the base __init__."""
        raise NotImplementedError("stage2: implement CosineSchedule.__init__")

    def lr_at(self, step):
        """stage 2: base_lr * step / warmup_steps during warmup; then min_lr + (base_lr - min_lr) * (1 + cos(pi * t)) / 2."""
        raise NotImplementedError("stage2: implement CosineSchedule.lr_at")


@app.cell(hide_code=True)
def _():
    checks.run(2, LRScheduler=LRScheduler, StepLR=StepLR, CosineSchedule=CosineSchedule, get_lr=get_lr)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def clip_grad_norm(params, max_norm):
    """stage 3: if the global 2-norm of all .grad (None skipped) exceeds max_norm, scale every grad IN PLACE by max_norm / (norm + 1e-6); return the norm before clipping."""
    raise NotImplementedError("stage3: implement clip_grad_norm")


@app.cell(hide_code=True)
def _():
    checks.run(3, clip_grad_norm=clip_grad_norm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def lr_range_test(model, loss_fn, optimizer, X, y, *, min_lr=1e-7, max_lr=10.0, n_steps=100, batch_size=64, rng=None):
    """stage 4: (mlfs.nn models only) one minibatch per step at lr min_lr * (max_lr/min_lr)^(i/(n_steps-1)); stop once loss > 4 x best; restore weights, lr and optimizer state; return (lrs, losses)."""
    raise NotImplementedError("stage4: implement lr_range_test")


@app.function
def suggest_lr(lrs, losses, *, beta=0.8):
    """stage 4: smooth the losses with a bias-corrected EMA(beta), find the lr with the lowest smoothed loss, return a tenth of it."""
    raise NotImplementedError("stage4: implement suggest_lr")


@app.cell(hide_code=True)
def _():
    checks.run(4, lr_range_test=lr_range_test, suggest_lr=suggest_lr, get_lr=get_lr)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.train import CosineSchedule, clip_grad_norm, lr_range_test` gives you these
    tools, and `CosineSchedule(torch_optimizer, ...)` works on PyTorch optimizers too.
    """)
    return


if __name__ == "__main__":
    app.run()
