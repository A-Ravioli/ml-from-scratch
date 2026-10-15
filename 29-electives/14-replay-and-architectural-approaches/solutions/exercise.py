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
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0
        self.x = None
        self.y = None
        self.size = 0

    def add(self, x, y):
        """stage 1: stream the rows of (x, y) in order through reservoir sampling."""
        if self.x is None:
            self.x = torch.zeros((self.capacity, *x.shape[1:]), dtype=x.dtype)
            self.y = torch.zeros(self.capacity, dtype=torch.int64)
        for i in range(len(y)):
            self.n_seen += 1
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                slot = int(self.rng.integers(0, self.n_seen))
                if slot >= self.capacity:
                    continue
            self.x[slot] = x[i]
            self.y[slot] = y[i]

    def sample(self, n):
        """stage 1: min(n, len(self)) distinct stored examples chosen uniformly at random, as (x, y)."""
        k = min(int(n), self.size)
        idx = torch.from_numpy(self.rng.choice(self.size, size=k, replace=False))
        return self.x[idx], self.y[idx]

    def __len__(self):
        """stage 1: the number of examples stored (at most capacity)."""
        return self.size

    def n_floats(self):
        """stage 1: floats used for stored inputs: capacity * (features per example), 0 before the first add."""
        return 0 if self.x is None else self.x.numel()


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
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.x = None
        self.y = None
        self.size = 0
        self.seen = {}  # class -> examples of that class streamed so far

    def counts(self):
        """stage 2: dict class -> number of stored examples of that class (classes with 0 left out)."""
        if self.size == 0:
            return {}
        labels, n = torch.unique(self.y[:self.size], return_counts=True)
        return {int(c): int(k) for c, k in zip(labels, n)}

    def add(self, x, y):
        """stage 2: stream rows; when full, a new example evicts a random example of the largest class."""
        if self.x is None:
            self.x = torch.zeros((self.capacity, *x.shape[1:]), dtype=x.dtype)
            self.y = torch.zeros(self.capacity, dtype=torch.int64)
        for i in range(len(y)):
            c = int(y[i])
            self.seen[c] = self.seen.get(c, 0) + 1
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                counts = self.counts()
                largest = max(counts.values())
                if counts.get(c, 0) < largest:
                    # evict a random member of a largest class
                    big = [k for k, v in counts.items() if v == largest]
                    victim = big[int(self.rng.integers(0, len(big)))]
                    members = torch.nonzero(self.y[:self.size] == victim).flatten()
                    slot = int(members[int(self.rng.integers(0, len(members)))])
                else:
                    # c is already a largest class: reservoir sampling within c
                    if self.rng.random() >= counts[c] / self.seen[c]:
                        continue
                    members = torch.nonzero(self.y[:self.size] == c).flatten()
                    slot = int(members[int(self.rng.integers(0, len(members)))])
            self.x[slot] = x[i]
            self.y[slot] = c

    def sample(self, n):
        """stage 2: min(n, len(self)) distinct stored examples chosen uniformly at random, as (x, y)."""
        k = min(int(n), self.size)
        idx = torch.from_numpy(self.rng.choice(self.size, size=k, replace=False))
        return self.x[idx], self.y[idx]

    def __len__(self):
        """stage 2: the number of examples stored."""
        return self.size

    def n_floats(self):
        """stage 2: floats used for stored inputs, as for ReservoirBuffer."""
        return 0 if self.x is None else self.x.numel()


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
        self.min_std = float(min_std)
        self.gen = torch.Generator().manual_seed(seed)
        self.means = {}
        self.stds = {}

    def fit(self, x, y):
        """stage 3: per class in y: mean and (biased) std of its rows, std clamped below at min_std; refits replace."""
        for c in torch.unique(y).tolist():
            xc = x[y == c]
            self.means[int(c)] = xc.mean(0)
            self.stds[int(c)] = xc.std(0, unbiased=False).clamp_min(self.min_std)

    def sample(self, n):
        """stage 3: n examples: labels uniform over fitted classes, x = mean + std * standard normal noise."""
        if not self.means:
            raise RuntimeError("fit at least one class before sampling")
        classes = torch.tensor(sorted(self.means))
        y = classes[torch.randint(len(classes), (n,), generator=self.gen)]
        mu = torch.stack([self.means[int(c)] for c in y])
        sd = torch.stack([self.stds[int(c)] for c in y])
        return mu + sd * torch.randn(mu.shape, generator=self.gen, dtype=mu.dtype), y

    def n_floats(self):
        """stage 3: floats stored: a mean and a std vector per fitted class."""
        return sum(m.numel() + s.numel() for m, s in zip(self.means.values(), self.stds.values()))


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
        super().__init__()
        sizes = [in_features, *hidden_sizes]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.head = nn.Linear(sizes[-1], out_features)
        g = torch.Generator().manual_seed(seed)
        for li, h in enumerate(hidden_sizes):
            k = int(round(keep * h))
            masks = torch.zeros(n_tasks, h)
            for t in range(n_tasks):
                masks[t, torch.randperm(h, generator=g)[:k]] = 1.0
            self.register_buffer(f"mask{li}", masks)
        self.n_tasks = n_tasks
        self.task = 0

    def masks(self):
        """stage 4: list over hidden layers of the (n_tasks, h) float 0/1 mask tensors."""
        return [getattr(self, f"mask{li}") for li in range(len(self.hidden))]

    def set_task(self, task):
        """stage 4: select the gate that forward uses when no task is given."""
        if not 0 <= task < self.n_tasks:
            raise ValueError(f"task must be in 0..{self.n_tasks - 1}, got {task}")
        self.task = int(task)

    def forward(self, x, task=None):
        """stage 4: each hidden layer computes relu(linear(h)) * mask[task]; then the head."""
        t = self.task if task is None else task
        h = x
        for layer, m in zip(self.hidden, self.masks()):
            h = torch.relu(layer(h)) * m[t]
        return self.head(h)


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
