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
    # debugging training · build

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
    raise NotImplementedError("stage1: implement grad_norm")


@app.function
def named_modules(model):
    """stage 1: [(dotted name, module)] for the model ('' first) and every sub-module, depth first in assignment order."""
    raise NotImplementedError("stage1: implement named_modules")


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
    raise NotImplementedError("stage2: implement update_ratio")


@app.function
def dead_fraction(activations):
    """stage 2: the fraction of units (columns of a (batch, units, ...) array) that are exactly zero for every example."""
    raise NotImplementedError("stage2: implement dead_fraction")


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
        raise NotImplementedError("stage3: implement DiagnosticsLogger.__init__")

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
        raise NotImplementedError("stage3: implement DiagnosticsLogger.log")

    def _weights(self):
        """names of the weight matrices (parameters with 2 or more dimensions; given)."""
        return [name for name, p in self.params if p.data.ndim >= 2]

    def summary(self):
        """stage 4: a list of warning strings, each starting with its tag: 'non-finite', 'exploding', 'vanishing', 'update ratio', 'dead relu', 'plateau'."""
        raise NotImplementedError("stage4: implement DiagnosticsLogger.summary")

    def plot(self):
        """stage 4: a matplotlib Figure with four panels: loss, gradient norms, update ratios (log scale), dead fractions."""
        raise NotImplementedError("stage4: implement DiagnosticsLogger.plot")


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
