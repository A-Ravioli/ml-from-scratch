import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # replay & architectural approaches · build

    spec-driven (L4): the **spec** boxes in the lesson define every class, the checks enforce
    them, and the internals are yours. examples are rows of a float32 tensor `x` `(b, d)` with
    int64 labels `y` `(b,)`; every memory reports what it stores with `n_floats()`, so you can
    compare strategies under one memory budget.

    | stage | you build |
    |---|---|
    | 1 | `ReservoirBuffer`: a uniform sample of everything seen so far, in fixed memory |
    | 2 | `ClassBalancedBuffer`: the same budget, split evenly over the classes seen |
    | 3 | `GaussianGenerator`: generative replay from one diagonal Gaussian per class |
    | 4 | `GatedMLP`: parameter isolation with a fixed random gate per task |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class ReservoirBuffer:
    """reservoir sampling: after n examples, each one is stored with probability capacity / n."""

    def __init__(self, capacity, *, seed=0):
        """stage 1: an empty buffer of `capacity` slots; all randomness from np.random.default_rng(seed)."""
        raise NotImplementedError("stage1: implement ReservoirBuffer.__init__")

    def add(self, x, y):
        """stage 1: stream the rows of (x, y) in order through reservoir sampling."""
        raise NotImplementedError("stage1: implement ReservoirBuffer.add")

    def sample(self, n):
        """stage 1: min(n, len(self)) distinct stored examples chosen uniformly at random, as (x, y)."""
        raise NotImplementedError("stage1: implement ReservoirBuffer.sample")

    def __len__(self):
        """stage 1: the number of examples stored (at most capacity)."""
        raise NotImplementedError("stage1: implement ReservoirBuffer.__len__")

    def n_floats(self):
        """stage 1: floats used for stored inputs: capacity * (features per example), 0 before the first add."""
        raise NotImplementedError("stage1: implement ReservoirBuffer.n_floats")


@app.cell(hide_code=True)
def _():
    checks.run(1, ReservoirBuffer=ReservoirBuffer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ClassBalancedBuffer:
    """class-balanced reservoir sampling (Chrysakis & Moens, 2020): equal shares for every class seen."""

    def __init__(self, capacity, *, seed=0):
        """stage 2: an empty buffer of `capacity` slots; all randomness from np.random.default_rng(seed)."""
        raise NotImplementedError("stage2: implement ClassBalancedBuffer.__init__")

    def counts(self):
        """stage 2: dict class -> number of stored examples of that class (classes with 0 left out)."""
        raise NotImplementedError("stage2: implement ClassBalancedBuffer.counts")

    def add(self, x, y):
        """stage 2: stream rows; when full, a new example evicts a random example of the largest class."""
        raise NotImplementedError("stage2: implement ClassBalancedBuffer.add")

    def sample(self, n):
        """stage 2: min(n, len(self)) distinct stored examples chosen uniformly at random, as (x, y)."""
        raise NotImplementedError("stage2: implement ClassBalancedBuffer.sample")

    def __len__(self):
        """stage 2: the number of examples stored."""
        raise NotImplementedError("stage2: implement ClassBalancedBuffer.__len__")

    def n_floats(self):
        """stage 2: floats used for stored inputs, as for ReservoirBuffer."""
        raise NotImplementedError("stage2: implement ClassBalancedBuffer.n_floats")


@app.cell(hide_code=True)
def _():
    checks.run(2, ClassBalancedBuffer=ClassBalancedBuffer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class GaussianGenerator:
    """the cheapest generative model: a diagonal Gaussian per class, sampled to replay old classes."""

    def __init__(self, *, min_std=0.01, seed=0):
        """stage 3: no classes yet; std floored at min_std; sampling randomness from torch.Generator seeded with seed."""
        raise NotImplementedError("stage3: implement GaussianGenerator.__init__")

    def fit(self, x, y):
        """stage 3: per class in y: mean and (biased) std of its rows, std clamped below at min_std; refits replace."""
        raise NotImplementedError("stage3: implement GaussianGenerator.fit")

    def sample(self, n):
        """stage 3: n examples: labels uniform over fitted classes, x = mean + std * standard normal noise."""
        raise NotImplementedError("stage3: implement GaussianGenerator.sample")

    def n_floats(self):
        """stage 3: floats stored: a mean and a std vector per fitted class."""
        raise NotImplementedError("stage3: implement GaussianGenerator.n_floats")


@app.cell(hide_code=True)
def _():
    checks.run(3, GaussianGenerator=GaussianGenerator)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class GatedMLP(nn.Module):
    """context-dependent gating (Masse et al., 2018): each task uses a fixed random subset of hidden units."""

    def __init__(self, in_features, hidden_sizes, out_features, n_tasks, *, keep=0.2, seed=0):
        """stage 4: Linear+ReLU hidden layers, a Linear head; per task and layer a fixed mask with round(keep * h) ones."""
        raise NotImplementedError("stage4: implement GatedMLP.__init__")

    def masks(self):
        """stage 4: list over hidden layers of the (n_tasks, h) float 0/1 mask tensors."""
        raise NotImplementedError("stage4: implement GatedMLP.masks")

    def set_task(self, task):
        """stage 4: select the gate that forward uses when no task is given."""
        raise NotImplementedError("stage4: implement GatedMLP.set_task")

    def forward(self, x, task=None):
        """stage 4: each hidden layer computes relu(linear(h)) * mask[task]; then the head."""
        raise NotImplementedError("stage4: implement GatedMLP.forward")


@app.cell(hide_code=True)
def _():
    checks.run(4, GatedMLP=GatedMLP)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it ends with the block's
    project: beat fine-tuning on split-MNIST with a fixed memory budget, using your buffers, your
    generator, your gated network and topic 12's harness.
    """)
    return


if __name__ == "__main__":
    app.run()
