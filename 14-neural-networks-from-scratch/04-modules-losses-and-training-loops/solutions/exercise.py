import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.optim import minibatch_indices  # yours, from ch 06 · topic 04
    from mlfs.tensor import Tensor, no_grad, stack  # yours, from topic 03
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # modules, losses & training loops · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. your `Tensor` can differentiate anything;
    this topic gives it the structure every deep learning framework has: **modules** that
    own their parameters, **losses** that are numerically safe, and a **training loop** you
    write once and reuse. everything here becomes **`mlfs.nn`**, with the same names and
    conventions as PyTorch's `torch.nn`, so chapter 15 can port it line by line.

    | stage | you build |
    |---|---|
    | 1 | `Parameter` and `Module`: parameter discovery, `train`/`eval`, `zero_grad`, `state_dict` |
    | 2 | `Linear` (torch layout and init), `ReLU`, `Tanh`, `Sigmoid`, `Sequential` |
    | 3 | `mse_loss`, `cross_entropy`, `binary_cross_entropy_with_logits`, `MSELoss`, `CrossEntropyLoss` |
    | 4 | `fit`, `evaluate`, `predict` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Parameter(Tensor):
    """a Tensor that is a model's trainable weight: always requires grad."""

    def __init__(self, data):
        """stage 1: a Tensor holding a float copy of data, with requires_grad=True."""
        super().__init__(data, requires_grad=True)


@app.class_definition
class Module:
    """base class: subclasses assign Parameters and sub-Modules as attributes and define forward."""

    def __init__(self):
        """stage 1: every module starts in training mode (self.training = True)."""
        self.training = True

    def __call__(self, *args, **kwargs):
        """stage 1: calling a module runs its forward."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must define forward(...)")

    def _children(self):
        """(name, module) for every sub-Module attribute, lists/tuples expanded as name.i (given)."""
        out = []
        for name, value in vars(self).items():
            if isinstance(value, Module):
                out.append((name, value))
            elif isinstance(value, (list, tuple)):
                out += [(f"{name}.{i}", m) for i, m in enumerate(value) if isinstance(m, Module)]
        return out

    def named_parameters(self):
        """stage 1: [(dotted name, Parameter)] in assignment order, recursing into sub-modules; each Parameter once."""
        out, seen = [], set()

        def visit(module, prefix):
            for name, value in vars(module).items():
                if isinstance(value, Parameter) and id(value) not in seen:
                    seen.add(id(value))
                    out.append((prefix + name, value))
                elif isinstance(value, Module):
                    visit(value, f"{prefix}{name}.")
                elif isinstance(value, (list, tuple)):
                    for i, m in enumerate(value):
                        if isinstance(m, Module):
                            visit(m, f"{prefix}{name}.{i}.")

        visit(self, "")
        return out

    def parameters(self):
        """stage 1: the Parameters of named_parameters(), as a list."""
        return [p for _, p in self.named_parameters()]

    def train(self, mode=True):
        """stage 1: set .training = mode on this module and every sub-module; return self."""
        self.training = bool(mode)
        for _, child in self._children():
            child.train(mode)
        return self

    def eval(self):
        """stage 1: train(False)."""
        return self.train(False)

    def zero_grad(self):
        """stage 1: set every parameter's .grad to None."""
        for p in self.parameters():
            p.grad = None

    def state_dict(self):
        """stage 1: {dotted name: a copy of the parameter's data}."""
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        """stage 1: copy state[name] into each parameter's data IN PLACE; KeyError for missing/extra names, ValueError for shapes."""
        params = dict(self.named_parameters())
        missing, extra = set(params) - set(state), set(state) - set(params)
        if missing or extra:
            raise KeyError(f"state_dict mismatch: missing {sorted(missing)}, unexpected {sorted(extra)}")
        for name, p in params.items():
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != p.data.shape:
                raise ValueError(f"{name}: expected shape {p.data.shape}, got {value.shape}")
            p.data[...] = value


@app.cell(hide_code=True)
def _():
    checks.run(1, Parameter=Parameter, Module=Module)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class Linear(Module):
    """y = x @ W.T + b, with W of shape (out_features, in_features) like torch.nn.Linear."""

    def __init__(self, in_features, out_features, bias=True, *, rng=None):
        """stage 2: weight (out, in) and bias (out,) Parameters drawn from U(-1/sqrt(in), 1/sqrt(in)), weight first; bias None if bias=False."""
        super().__init__()
        rng = as_rng(rng)
        self.in_features, self.out_features = int(in_features), int(out_features)
        bound = 1.0 / np.sqrt(self.in_features)
        self.weight = Parameter(rng.uniform(-bound, bound, size=(self.out_features, self.in_features)))
        self.bias = Parameter(rng.uniform(-bound, bound, size=self.out_features)) if bias else None

    def forward(self, x):
        """stage 2: x (n, in) Tensor or ndarray -> (n, out) Tensor."""
        x = x if isinstance(x, Tensor) else Tensor(x)
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


@app.class_definition
class ReLU(Module):
    """elementwise max(x, 0)."""

    def forward(self, x):
        """stage 2: x.relu() (wrap an ndarray first)."""
        return (x if isinstance(x, Tensor) else Tensor(x)).relu()


@app.class_definition
class Tanh(Module):
    """elementwise tanh."""

    def forward(self, x):
        """stage 2: x.tanh()."""
        return (x if isinstance(x, Tensor) else Tensor(x)).tanh()


@app.class_definition
class Sigmoid(Module):
    """elementwise logistic sigmoid."""

    def forward(self, x):
        """stage 2: x.sigmoid()."""
        return (x if isinstance(x, Tensor) else Tensor(x)).sigmoid()


@app.class_definition
class Sequential(Module):
    """layers applied in order; layer i is stored as attribute str(i), so names read '0.weight' like torch."""

    def __init__(self, *layers):
        """stage 2: store each layer as attribute str(i) (setattr), in order."""
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)

    def _layers(self):
        return [v for k, v in vars(self).items() if k.isdigit()]

    def forward(self, x):
        """stage 2: feed x through every layer in order."""
        for layer in self._layers():
            x = layer(x)
        return x

    def __getitem__(self, idx):
        """stage 2: an int gives that layer (negative ok); a slice gives a new Sequential of those layers."""
        layers = self._layers()
        if isinstance(idx, slice):
            return Sequential(*layers[idx])
        return layers[idx]

    def __len__(self):
        """stage 2: the number of layers."""
        return len(self._layers())


@app.cell(hide_code=True)
def _():
    checks.run(2, Parameter=Parameter, Module=Module, Linear=Linear, ReLU=ReLU, Tanh=Tanh, Sigmoid=Sigmoid,
               Sequential=Sequential)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def mse_loss(pred, target):
    """stage 3: mean of (pred - target)^2 over all entries; ValueError if the shapes differ (no silent broadcasting)."""
    target_shape = target.shape if isinstance(target, Tensor) else np.shape(target)
    if tuple(pred.shape) != tuple(target_shape):
        raise ValueError(f"mse_loss: pred has shape {pred.shape} but target has shape {target_shape}; "
                         "broadcasting them would compare every prediction with every target")
    return ((pred - target) ** 2).mean()


@app.function
def cross_entropy(logits, targets):
    """stage 3: mean over rows of logsumexp(logits_i) - logits_i[targets_i]; logits (n, k), targets int (n,)."""
    targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets).astype(np.int64)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError(f"cross_entropy: logits must be (n, k) and targets (n,); got {logits.shape}, {targets.shape}")
    n = logits.shape[0]
    return (logits.logsumexp(axis=1) - logits[np.arange(n), targets]).mean()


@app.function
def binary_cross_entropy_with_logits(logits, targets):
    """stage 3: mean of softplus(z) - y z for logits z (n,) and labels y in {0, 1} (n,), stable for any z."""
    y = np.asarray(targets.data if isinstance(targets, Tensor) else targets, dtype=np.float64)
    if tuple(logits.shape) != y.shape:
        raise ValueError(f"binary_cross_entropy_with_logits: logits {logits.shape} vs targets {y.shape}")
    softplus = stack([np.zeros(logits.shape), logits], axis=-1).logsumexp(axis=-1)  # log(e^0 + e^z)
    return (softplus - logits * y).mean()


@app.class_definition
class MSELoss(Module):
    """module form of mse_loss."""

    def forward(self, pred, target):
        """stage 3: mse_loss(pred, target)."""
        return mse_loss(pred, target)


@app.class_definition
class CrossEntropyLoss(Module):
    """module form of cross_entropy."""

    def forward(self, logits, targets):
        """stage 3: cross_entropy(logits, targets)."""
        return cross_entropy(logits, targets)


@app.cell(hide_code=True)
def _():
    checks.run(3, mse_loss=mse_loss, cross_entropy=cross_entropy,
               binary_cross_entropy_with_logits=binary_cross_entropy_with_logits,
               MSELoss=MSELoss, CrossEntropyLoss=CrossEntropyLoss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def evaluate(model, loss_fn, X, y, *, batch_size=256):
    """stage 4: the mean loss over (X, y) in batches, under no_grad and in eval mode (restoring the mode after)."""
    was_training = getattr(model, "training", True)
    model.eval()
    total, n = 0.0, len(X)
    try:
        with no_grad():
            for idx in minibatch_indices(n, batch_size, shuffle=False):
                total += loss_fn(model(X[idx]), y[idx]).item() * len(idx)
    finally:
        model.train(was_training)
    return total / n


@app.function
def predict(model, X, *, batch_size=256):
    """stage 4: the model's outputs as one ndarray (n, ...), in batches, under no_grad and in eval mode."""
    was_training = getattr(model, "training", True)
    model.eval()
    outs = []
    try:
        with no_grad():
            for idx in minibatch_indices(len(X), batch_size, shuffle=False):
                outs.append(np.asarray(model(X[idx]).data))
            if not outs:  # no rows: still return the right trailing shape
                outs.append(np.asarray(model(X[:0]).data))
    finally:
        model.train(was_training)
    return np.concatenate(outs, axis=0)


@app.function
def fit(model, loss_fn, optimizer, X, y, *, epochs=10, batch_size=32, X_val=None, y_val=None, shuffle=True,
        rng=None, callback=None):
    """stage 4: minibatch training (zero_grad, forward, loss, backward, step); returns {'train_loss': [...], 'val_loss': [...]} per epoch."""
    rng = as_rng(rng)
    history = {"train_loss": [], "val_loss": []}
    n = len(X)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in minibatch_indices(n, batch_size, shuffle=shuffle, rng=rng):
            optimizer.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history["train_loss"].append(total / n)
        if X_val is not None:
            history["val_loss"].append(evaluate(model, loss_fn, X_val, y_val))
        if callback is not None and callback(epoch, history):
            break
    return history


@app.cell(hide_code=True)
def _():
    checks.run(4, Module=Module, Linear=Linear, ReLU=ReLU, Tanh=Tanh, Sequential=Sequential,
               mse_loss=mse_loss, cross_entropy=cross_entropy, MSELoss=MSELoss, CrossEntropyLoss=CrossEntropyLoss,
               fit=fit, evaluate=evaluate, predict=predict)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs import nn` gives you these modules, and `from mlfs.optim import Adam` trains
    their `parameters()` unchanged.
    """)
    return


if __name__ == "__main__":
    app.run()
