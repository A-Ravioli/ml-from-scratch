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
    # modules, losses & training loops · build

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
        raise NotImplementedError("stage1: implement Parameter.__init__")


@app.class_definition
class Module:
    """base class: subclasses assign Parameters and sub-Modules as attributes and define forward."""

    def __init__(self):
        """stage 1: every module starts in training mode (self.training = True)."""
        raise NotImplementedError("stage1: implement Module.__init__")

    def __call__(self, *args, **kwargs):
        """stage 1: calling a module runs its forward."""
        raise NotImplementedError("stage1: implement Module.__call__")

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
        raise NotImplementedError("stage1: implement Module.named_parameters")

    def parameters(self):
        """stage 1: the Parameters of named_parameters(), as a list."""
        raise NotImplementedError("stage1: implement Module.parameters")

    def train(self, mode=True):
        """stage 1: set .training = mode on this module and every sub-module; return self."""
        raise NotImplementedError("stage1: implement Module.train")

    def eval(self):
        """stage 1: train(False)."""
        raise NotImplementedError("stage1: implement Module.eval")

    def zero_grad(self):
        """stage 1: set every parameter's .grad to None."""
        raise NotImplementedError("stage1: implement Module.zero_grad")

    def state_dict(self):
        """stage 1: {dotted name: a copy of the parameter's data}."""
        raise NotImplementedError("stage1: implement Module.state_dict")

    def load_state_dict(self, state):
        """stage 1: copy state[name] into each parameter's data IN PLACE; KeyError for missing/extra names, ValueError for shapes."""
        raise NotImplementedError("stage1: implement Module.load_state_dict")


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
        raise NotImplementedError("stage2: implement Linear.__init__")

    def forward(self, x):
        """stage 2: x (n, in) Tensor or ndarray -> (n, out) Tensor."""
        raise NotImplementedError("stage2: implement Linear.forward")


@app.class_definition
class ReLU(Module):
    """elementwise max(x, 0)."""

    def forward(self, x):
        """stage 2: x.relu() (wrap an ndarray first)."""
        raise NotImplementedError("stage2: implement ReLU.forward")


@app.class_definition
class Tanh(Module):
    """elementwise tanh."""

    def forward(self, x):
        """stage 2: x.tanh()."""
        raise NotImplementedError("stage2: implement Tanh.forward")


@app.class_definition
class Sigmoid(Module):
    """elementwise logistic sigmoid."""

    def forward(self, x):
        """stage 2: x.sigmoid()."""
        raise NotImplementedError("stage2: implement Sigmoid.forward")


@app.class_definition
class Sequential(Module):
    """layers applied in order; layer i is stored as attribute str(i), so names read '0.weight' like torch."""

    def __init__(self, *layers):
        """stage 2: store each layer as attribute str(i) (setattr), in order."""
        raise NotImplementedError("stage2: implement Sequential.__init__")

    def _layers(self):
        return [v for k, v in vars(self).items() if k.isdigit()]

    def forward(self, x):
        """stage 2: feed x through every layer in order."""
        raise NotImplementedError("stage2: implement Sequential.forward")

    def __getitem__(self, idx):
        """stage 2: an int gives that layer (negative ok); a slice gives a new Sequential of those layers."""
        raise NotImplementedError("stage2: implement Sequential.__getitem__")

    def __len__(self):
        """stage 2: the number of layers."""
        raise NotImplementedError("stage2: implement Sequential.__len__")


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
    raise NotImplementedError("stage3: implement mse_loss")


@app.function
def cross_entropy(logits, targets):
    """stage 3: mean over rows of logsumexp(logits_i) - logits_i[targets_i]; logits (n, k), targets int (n,)."""
    raise NotImplementedError("stage3: implement cross_entropy")


@app.function
def binary_cross_entropy_with_logits(logits, targets):
    """stage 3: mean of softplus(z) - y z for logits z (n,) and labels y in {0, 1} (n,), stable for any z."""
    raise NotImplementedError("stage3: implement binary_cross_entropy_with_logits")


@app.class_definition
class MSELoss(Module):
    """module form of mse_loss."""

    def forward(self, pred, target):
        """stage 3: mse_loss(pred, target)."""
        raise NotImplementedError("stage3: implement MSELoss.forward")


@app.class_definition
class CrossEntropyLoss(Module):
    """module form of cross_entropy."""

    def forward(self, logits, targets):
        """stage 3: cross_entropy(logits, targets)."""
        raise NotImplementedError("stage3: implement CrossEntropyLoss.forward")


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
    raise NotImplementedError("stage4: implement evaluate")


@app.function
def predict(model, X, *, batch_size=256):
    """stage 4: the model's outputs as one ndarray (n, ...), in batches, under no_grad and in eval mode."""
    raise NotImplementedError("stage4: implement predict")


@app.function
def fit(model, loss_fn, optimizer, X, y, *, epochs=10, batch_size=32, X_val=None, y_val=None, shuffle=True,
        rng=None, callback=None):
    """stage 4: minibatch training (zero_grad, forward, loss, backward, step); returns {'train_loss': [...], 'val_loss': [...]} per epoch."""
    raise NotImplementedError("stage4: implement fit")


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
