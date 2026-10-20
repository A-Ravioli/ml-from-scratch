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
    # optimization in practice · build (reference solution)

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
    groups = getattr(optimizer, "param_groups", None)
    if groups is not None:
        return float(groups[0]["lr"])
    return float(optimizer.lr)


@app.function
def set_lr(optimizer, lr):
    """stage 1: set the learning rate: every param_group's 'lr' for a torch optimizer, else optimizer.lr."""
    groups = getattr(optimizer, "param_groups", None)
    if groups is not None:
        for group in groups:
            group["lr"] = float(lr)
    else:
        optimizer.lr = float(lr)


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
        self.optimizer = optimizer
        self.base_lr = get_lr(optimizer)
        self.last_step = 0
        set_lr(optimizer, self.lr_at(0))

    def lr_at(self, step):
        """stage 2: the learning rate for a given step; the base class keeps base_lr constant."""
        return self.base_lr

    def step(self):
        """stage 2: last_step += 1, then set the optimizer's lr to lr_at(last_step)."""
        self.last_step += 1
        set_lr(self.optimizer, self.lr_at(self.last_step))


@app.class_definition
class StepLR(LRScheduler):
    """multiply the learning rate by gamma every step_size steps."""

    def __init__(self, optimizer, step_size, gamma=0.1):
        """stage 2: store step_size and gamma BEFORE calling the base __init__ (it calls lr_at(0))."""
        self.step_size, self.gamma = int(step_size), float(gamma)
        super().__init__(optimizer)

    def lr_at(self, step):
        """stage 2: base_lr * gamma ** (step // step_size)."""
        return self.base_lr * self.gamma ** (step // self.step_size)


@app.class_definition
class CosineSchedule(LRScheduler):
    """linear warmup from 0 to base_lr, then a half cosine down to min_lr at total_steps (held there afterwards)."""

    def __init__(self, optimizer, total_steps, *, warmup_steps=0, min_lr=0.0):
        """stage 2: store total_steps, warmup_steps, min_lr (ValueError unless 0 <= warmup_steps < total_steps), then the base __init__."""
        if not 0 <= warmup_steps < total_steps:
            raise ValueError(f"need 0 <= warmup_steps < total_steps, got {warmup_steps} and {total_steps}")
        self.total_steps, self.warmup_steps, self.min_lr = int(total_steps), int(warmup_steps), float(min_lr)
        super().__init__(optimizer)

    def lr_at(self, step):
        """stage 2: base_lr * step / warmup_steps during warmup; then min_lr + (base_lr - min_lr) * (1 + cos(pi * t)) / 2."""
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr
        t = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


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
    grads = [p.grad for p in params if p.grad is not None]
    total = math.sqrt(sum(float(np.sum(np.square(g))) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g *= scale
    return total


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
    rng = as_rng(rng)
    saved_weights = model.state_dict()
    saved_lr = get_lr(optimizer)
    state = getattr(optimizer, "state", None)
    saved_state = copy.deepcopy(state) if type(state) is dict else None  # mlfs.optim's per-parameter state
    lrs, losses, best = [], [], math.inf
    batches = iter(())
    try:
        model.train()
        for i in range(n_steps):
            lr = min_lr * (max_lr / min_lr) ** (i / max(n_steps - 1, 1))
            idx = next(batches, None)
            if idx is None:  # start a new epoch of shuffled minibatches
                batches = iter(minibatch_indices(len(X), batch_size, shuffle=True, rng=rng))
                idx = next(batches)
            set_lr(optimizer, lr)
            optimizer.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            value = loss.item()
            lrs.append(lr)
            losses.append(value)
            best = min(best, value)
            if not math.isfinite(value) or value > 4 * best:
                break
    finally:
        model.load_state_dict(saved_weights)
        set_lr(optimizer, saved_lr)
        if saved_state is not None:
            optimizer.state = saved_state
        optimizer.zero_grad()
    return lrs, losses


@app.function
def suggest_lr(lrs, losses, *, beta=0.8):
    """stage 4: smooth the losses with a bias-corrected EMA(beta), find the lr with the lowest smoothed loss, return a tenth of it."""
    if len(lrs) == 0 or len(lrs) != len(losses):
        raise ValueError("need equally long, non-empty lrs and losses")
    avg, smooth = 0.0, []
    for k, value in enumerate(losses, start=1):
        avg = beta * avg + (1 - beta) * (value if math.isfinite(value) else 1e300)
        smooth.append(avg / (1 - beta ** k))
    return float(lrs[int(np.argmin(smooth))]) / 10.0


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
