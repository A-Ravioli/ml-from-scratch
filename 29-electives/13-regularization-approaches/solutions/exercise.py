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
    if mode not in ("exact", "empirical"):
        raise ValueError(f"mode must be 'exact' or 'empirical', got {mode!r}")
    if mode == "empirical" and y is None:
        raise ValueError("the empirical Fisher needs labels y")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p.detach() for n, p in model.named_parameters() if not p.requires_grad}
    buffers = dict(model.named_buffers())
    cols = None if classes is None else torch.tensor(list(classes))

    def log_probs(p, x):
        out = functional_call(model, ({**p, **frozen}, buffers), (x.unsqueeze(0),))[0]
        if cols is not None:
            out = out[cols]
        return F.log_softmax(out, dim=-1)

    fisher = {n: torch.zeros_like(params[n]) for n in names}
    for i in range(0, len(X), chunk_size):
        xb = X[i:i + chunk_size]
        jac = vmap(jacrev(log_probs), in_dims=(None, 0))(params, xb)  # name -> (b, C, *shape)
        with torch.no_grad():
            if mode == "exact":
                w = vmap(log_probs, in_dims=(None, 0))(params, xb).exp()  # (b, C)
            else:
                yb = y[i:i + chunk_size]
                if cols is not None:  # position of each label within the restricted classes
                    yb = (cols.unsqueeze(0) == yb.unsqueeze(1)).float().argmax(1)
                w = F.one_hot(yb, jac[names[0]].shape[1]).to(xb.dtype)
            for n in names:
                g2 = jac[n].detach() ** 2
                fisher[n] += torch.einsum("bc,bc...->...", w, g2)
    return {n: f / len(X) for n, f in fisher.items()}


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
        self.lam = float(lam)
        self.mode = mode
        self.n_samples = n_samples
        self.online = online
        self.gamma = float(gamma)
        self.anchors = []   # one dict per consolidated task (online: at most one)
        self.fishers = []

    def consolidate(self, model, task, *, scenario="class"):
        """stage 2: Fisher on the task's first n_samples examples (classes restricted if scenario='task'); anchor = current weights."""
        n = len(task["y"]) if self.n_samples is None else min(self.n_samples, len(task["y"]))
        classes = task["classes"] if scenario == "task" else None
        fisher = fisher_diagonal(model, task["X"][:n], y=task["y"][:n], classes=classes, mode=self.mode)
        anchor = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        if self.online and self.fishers:
            fisher = {k: self.gamma * self.fishers[0][k] + fisher[k] for k in fisher}
            self.anchors, self.fishers = [anchor], [fisher]
        else:
            self.anchors.append(anchor)
            self.fishers.append(fisher)

    def penalty(self, model):
        """stage 2: (lam / 2) * sum over consolidated tasks of sum_i F_i (theta_i - anchor_i)^2, a differentiable scalar."""
        params = dict(model.named_parameters())
        total = torch.zeros((), dtype=next(model.parameters()).dtype)
        for anchor, fisher in zip(self.anchors, self.fishers):
            for name, f in fisher.items():
                total = total + (f * (params[name] - anchor[name]) ** 2).sum()
        return 0.5 * self.lam * total


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
        self.c = float(c)
        self.xi = float(xi)
        self.omega = None    # accumulated importance Omega, name -> tensor
        self.anchor = None   # weights at the end of the last consolidated task
        self.w = None        # running path integral for the current task
        self.start = None
        self.prev = None

    def begin_task(self, model):
        """stage 3: remember the weights at the start of the task and reset the running path integral w."""
        params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if self.omega is None:
            self.omega = {n: torch.zeros_like(p) for n, p in params.items()}
        self.start = params
        self.prev = {n: p.clone() for n, p in params.items()}
        self.w = {n: torch.zeros_like(p) for n, p in params.items()}

    def after_step(self, model):
        """stage 3: after an optimizer step, w += -grad * (theta - theta_prev), then theta_prev = theta."""
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in self.w:
                    continue
                if p.grad is not None:
                    self.w[n] -= p.grad * (p.detach() - self.prev[n])
                self.prev[n] = p.detach().clone()

    def consolidate(self, model):
        """stage 3: Omega += w / ((theta - theta_start)^2 + xi); the anchor becomes the current weights."""
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in self.w:
                    continue
                delta = p.detach() - self.start[n]
                self.omega[n] += self.w[n] / (delta ** 2 + self.xi)
        self.anchor = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model):
        """stage 3: c * sum_i Omega_i (theta_i - anchor_i)^2 (zero before the first consolidate), differentiable."""
        total = torch.zeros((), dtype=next(model.parameters()).dtype)
        if self.anchor is None:
            return total
        for n, p in model.named_parameters():
            if n in self.anchor:
                total = total + (self.omega[n] * (p - self.anchor[n]) ** 2).sum()
        return self.c * total


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
