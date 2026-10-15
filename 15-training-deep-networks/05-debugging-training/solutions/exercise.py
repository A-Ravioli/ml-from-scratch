import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.nn import Module, ReLU  # yours, from ch 14 · topic 04


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # debugging training · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. a broken training run rarely crashes; it
    just doesn't learn, and the loss curve alone seldom says why. this topic builds the
    instrument panel: gradient norms, weight norms, update-to-weight ratios and dead-unit
    counts for every layer, logged at every step, with a summary that turns the classic
    failure signatures into plain-language warnings. it joins **`mlfs.train`**.

    | stage | you build |
    |---|---|
    | 1 | `grad_norm` and `named_modules`: the global gradient norm, and every sub-module by name |
    | 2 | `update_ratio` and `dead_fraction`: two per-layer health numbers |
    | 3 | `DiagnosticsLogger`: watch the ReLUs, and log everything after every step |
    | 4 | `DiagnosticsLogger.summary` and `.plot`: from numbers to a diagnosis |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def grad_norm(params):
    """stage 1: the global 2-norm of all .grad arrays (parameters whose grad is None are skipped), as a float."""
    return math.sqrt(sum(float(np.sum(np.square(p.grad))) for p in params if p.grad is not None))


@app.function
def named_modules(model):
    """stage 1: [(dotted name, module)] for the model ('' first) and every sub-module, depth first in assignment order."""
    out = []

    def visit(module, name):
        out.append((name, module))
        prefix = f"{name}." if name else ""
        for key, value in vars(module).items():
            if isinstance(value, Module):
                visit(value, prefix + key)
            elif isinstance(value, (list, tuple)):
                for i, m in enumerate(value):
                    if isinstance(m, Module):
                        visit(m, f"{prefix}{key}.{i}")

    visit(model, "")
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, grad_norm=grad_norm, named_modules=named_modules)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def update_ratio(before, after):
    """stage 2: ||after - before|| / ||after|| for two arrays (the relative size of one update); 0.0 if both are zero."""
    after = np.asarray(after, dtype=np.float64)
    step = float(np.linalg.norm(after - np.asarray(before, dtype=np.float64)))
    size = float(np.linalg.norm(after))
    if size == 0.0:
        return 0.0 if step == 0.0 else math.inf
    return step / size


@app.function
def dead_fraction(activations):
    """stage 2: the fraction of units (columns of a (batch, units, ...) array) that are exactly zero for every example."""
    a = np.asarray(activations)
    a = a.reshape(a.shape[0], -1)
    return float(np.mean(np.all(a == 0, axis=0)))


@app.cell(hide_code=True)
def _():
    checks.run(2, update_ratio=update_ratio, dead_fraction=dead_fraction)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class DiagnosticsLogger:
    """records per-step training health of an mlfs.nn model: loss, lr, gradient and weight norms, update ratios, dead-ReLU fractions."""

    def __init__(self, model):
        """stage 3: self.params (named_parameters), self.history, self._before (copies of the data), self._dead = {}; then _watch every ReLU."""
        self.model = model
        self.params = model.named_parameters()
        self.history = {"step": [], "loss": [], "lr": [], "grad_norm": []}
        self._before = {name: p.data.copy() for name, p in self.params}
        self._dead = {}
        for name, module in named_modules(model):
            if isinstance(module, ReLU):
                self._watch(name, module)

    def _watch(self, name, module):
        """replace this ReLU's forward with one that also records dead_fraction of its output (training mode only; given)."""
        original = module.forward

        def forward(x):
            out = original(x)
            if module.training:
                self._dead[name] = dead_fraction(out.data)
            return out

        module.forward = forward

    def log(self, step, loss, *, lr=None):
        """stage 3: append step, loss, lr, the global grad norm and, per parameter, grad/<name>, weight/<name>, update/<name>; dead/<relu>."""
        h = self.history
        h["step"].append(step)
        h["loss"].append(float(loss.item() if hasattr(loss, "item") else loss))
        h["lr"].append(math.nan if lr is None else float(lr))
        h["grad_norm"].append(grad_norm(p for _, p in self.params))
        for name, p in self.params:
            g = math.nan if p.grad is None else float(np.linalg.norm(p.grad))
            h.setdefault(f"grad/{name}", []).append(g)
            h.setdefault(f"weight/{name}", []).append(float(np.linalg.norm(p.data)))
            h.setdefault(f"update/{name}", []).append(update_ratio(self._before[name], p.data))
            self._before[name] = p.data.copy()
        for name, frac in self._dead.items():
            h.setdefault(f"dead/{name}", []).append(frac)

    def _weights(self):
        """names of the weight matrices (parameters with 2 or more dimensions; given)."""
        return [name for name, p in self.params if p.data.ndim >= 2]

    def summary(self):
        """stage 4: a list of warning strings, each starting with its tag: 'non-finite', 'exploding', 'vanishing', 'update ratio', 'dead relu', 'plateau'."""
        h, out = self.history, []
        losses = np.array(h["loss"], dtype=float)
        if len(losses) == 0:
            return out
        if not np.all(np.isfinite(losses)):
            first = int(np.argmax(~np.isfinite(losses)))
            out.append(f"non-finite loss: the loss became nan/inf at step {h['step'][first]}")
        norms = np.array(h["grad_norm"], dtype=float)
        for t in range(1, len(norms)):
            past = norms[:t][np.isfinite(norms[:t])]
            base = float(np.median(past)) if len(past) else 0.0
            if not np.isfinite(norms[t]) or norms[t] > 100 * max(base, 1e-12):
                out.append(f"exploding gradients: at step {h['step'][t]} the gradient norm was {norms[t]:.3g}, "
                           f"more than 100x its median so far ({base:.3g})")
                break
        weights = self._weights()
        med = {w: float(np.nanmedian(h[f"grad/{w}"])) for w in weights if np.any(np.isfinite(h[f"grad/{w}"]))}
        if len(med) >= 2:
            top = max(med.values())
            for w, m in med.items():
                if top > 0 and m < 1e-3 * top:
                    out.append(f"vanishing gradients: {w} gets {m:.3g}, under 1/1000 of the largest layer's {top:.3g}")
        for w in weights:
            r = np.array(h[f"update/{w}"], dtype=float)
            r = r[np.isfinite(r)]
            if len(r) == 0:
                continue
            m = float(np.median(r))
            if m > 1e-1:
                out.append(f"update ratio too large: {w} changes by {m:.2g} of its size per step (aim for ~1e-3)")
            elif m < 1e-5:
                out.append(f"update ratio too small: {w} changes by {m:.2g} of its size per step (aim for ~1e-3)")
        for key in [k for k in h if k.startswith("dead/")]:
            if h[key] and h[key][-1] > 0.5:
                out.append(f"dead relu: {h[key][-1]:.0%} of the units after layer {key[5:]} output 0 for the whole batch")
        n = len(losses)
        if n >= 10 and np.all(np.isfinite(losses)):
            q = max(n // 4, 1)
            first, last = losses[:q].mean(), losses[-q:].mean()
            if last > 0.95 * first:
                out.append(f"plateau: the loss went from {first:.3g} (first quarter) to {last:.3g} (last quarter)")
        return out

    def plot(self):
        """stage 4: a matplotlib Figure with four panels: loss, gradient norms, update ratios (log scale), dead fractions."""
        h = self.history
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
        steps = h["step"]
        axes[0].plot(steps, h["loss"])
        axes[0].set(title="loss", xlabel="step")
        for w in self._weights():
            axes[1].semilogy(steps, np.maximum(h[f"grad/{w}"], 1e-30), label=w)
            axes[2].semilogy(steps, np.maximum(h[f"update/{w}"], 1e-30), label=w)
        axes[1].set(title="gradient norm per weight", xlabel="step")
        axes[2].axhline(1e-3, color="gray", ls=":", label="1e-3")
        axes[2].set(title="update / weight ratio", xlabel="step")
        for key in [k for k in h if k.startswith("dead/")]:
            axes[3].plot(steps[-len(h[key]):], h[key], label=key[5:])
        axes[3].set(title="dead ReLU fraction", xlabel="step", ylim=(-0.02, 1.02))
        for ax in axes[1:]:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6)
        fig.tight_layout()
        return fig


@app.cell(hide_code=True)
def _():
    checks.run(3, DiagnosticsLogger=DiagnosticsLogger, grad_norm=grad_norm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, DiagnosticsLogger=DiagnosticsLogger)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.train import DiagnosticsLogger` gives you this instrument panel: create it
    after the model, call `logger.log(step, loss.item(), lr=...)` right after every
    `optimizer.step()`, and read `logger.summary()`.
    """)
    return


if __name__ == "__main__":
    app.run()
