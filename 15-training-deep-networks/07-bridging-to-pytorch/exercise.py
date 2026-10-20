import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.nn import cross_entropy, mse_loss  # yours, from ch 14 · topic 04
    from mlfs.tensor import Tensor  # yours, from ch 14 · topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bridging to PyTorch · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you've built a tensor engine, modules,
    losses, initializers, normalization layers, schedulers and a training loop. PyTorch is
    the same ideas, written in C++ and CUDA, with a much larger library. this topic maps
    one onto the other: you'll convert your `mlfs.nn` networks into PyTorch modules and
    prove they compute the same gradients, then write the small PyTorch toolkit
    (**`mlfs.torchnn`**) that chapters 16–28 train everything with.

    | stage | you build |
    |---|---|
    | 1 | `to_torch`: an `mlfs.nn` network → the equivalent `torch.nn` module, weights copied |
    | 2 | `gradient_parity`: the same loss and gradients in both frameworks, to the last digit |
    | 3 | `MLP`, `RMSNorm`, `count_parameters`, `get_device`: PyTorch modules and helpers |
    | 4 | `make_loader`, `evaluate`, `fit`: data loading and the training loop, PyTorch-style |
    | 5 | `ema_update`: an exponential moving average of a model's weights |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def to_torch(module):
    """stage 1: an equivalent float64 torch.nn module (Sequential/Linear/ReLU/Tanh/Sigmoid/Dropout/LayerNorm/BatchNorm1d/RMSNorm), weights copied, same train/eval mode."""
    raise NotImplementedError("stage1: implement to_torch")


@app.cell(hide_code=True)
def _():
    checks.run(1, to_torch=to_torch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def gradient_parity(model, X, y):
    """stage 2: dict(loss, output, grad) of max |mlfs - torch| differences; cross-entropy for integer y, else MSE. no side effects."""
    raise NotImplementedError("stage2: implement gradient_parity")


@app.cell(hide_code=True)
def _():
    checks.run(2, gradient_parity=gradient_parity, to_torch=to_torch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2, last dim) + eps) * weight; float16/bfloat16 inputs are normalized in float32 and cast back.

    float32/float64 inputs are computed in their own dtype (so float64 parity with mlfs stays exact). a half or
    bfloat16 input times the float32 weight returns float32 (cast the module, or the result, as Llama-style code does).
    """

    def __init__(self, dim, eps=1e-6):
        """stage 3: store eps and a Parameter weight = ones(dim)."""
        raise NotImplementedError("stage3: implement RMSNorm.__init__")

    def forward(self, x):
        """stage 3: upcast float16/bfloat16 to float32 (others stay), normalize, cast back to x's dtype, times weight."""
        raise NotImplementedError("stage3: implement RMSNorm.forward")


@app.class_definition
class MLP(nn.Module):
    """Linear -> [norm] -> activation -> [dropout] for every hidden size, then a plain Linear output layer."""

    def __init__(self, in_features, hidden_sizes, out_features, *, activation="relu", dropout=0.0, norm=None):
        """stage 3: build self.net, an nn.Sequential of those layers; ValueError for an unknown activation or norm."""
        raise NotImplementedError("stage3: implement MLP.__init__")

    def forward(self, x):
        """stage 3: self.net(x)."""
        raise NotImplementedError("stage3: implement MLP.forward")


@app.function
def count_parameters(model, *, trainable_only=True):
    """stage 3: the total number of entries in the model's parameters (only those with requires_grad, by default)."""
    raise NotImplementedError("stage3: implement count_parameters")


@app.function
def get_device(prefer=None):
    """stage 3: torch.device(prefer) if given, else 'cuda' if available, else 'mps' if available, else 'cpu'."""
    raise NotImplementedError("stage3: implement get_device")


@app.cell(hide_code=True)
def _():
    checks.run(3, MLP=MLP, RMSNorm=RMSNorm, count_parameters=count_parameters, get_device=get_device)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def make_loader(X, y=None, *, batch_size=64, shuffle=True, seed=None):
    """stage 4: a DataLoader over TensorDataset(X float32[, y int64 if integer else float32]); `seed` fixes the shuffling."""
    raise NotImplementedError("stage4: implement make_loader")


@app.function
def evaluate(model, loader, loss_fn=None, *, device="cpu"):
    """stage 4: under no_grad in eval mode (restored after): {'loss': mean per example (if loss_fn)} and 'accuracy' for (B, C) outputs with integer targets."""
    raise NotImplementedError("stage4: implement evaluate")


@app.function
def fit(model, loss_fn, optimizer, train_loader, *, epochs=1, val_loader=None, scheduler=None, grad_clip=None,
        device="cpu", callback=None):
    """stage 4: per epoch: train mode, per batch zero_grad/forward/loss/backward/[clip]/step/[scheduler.step()]; history train_loss, val_loss, lr."""
    raise NotImplementedError("stage4: implement fit")


@app.cell(hide_code=True)
def _():
    checks.run(4, make_loader=make_loader, evaluate=evaluate, fit=fit, MLP=MLP)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def ema_update(target, source, decay=0.999):
    """stage 5: in place, under no_grad: every target parameter <- decay * target + (1 - decay) * source; buffers copied."""
    raise NotImplementedError("stage5: implement ema_update")


@app.cell(hide_code=True)
def _():
    checks.run(5, ema_update=ema_update, MLP=MLP)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.torchnn import MLP, fit, make_loader` gives you this toolkit, and from
    chapter 16 on you may use the PyTorch components you have built yourself (the chapter
    README lists them).
    """)
    return


if __name__ == "__main__":
    app.run()
