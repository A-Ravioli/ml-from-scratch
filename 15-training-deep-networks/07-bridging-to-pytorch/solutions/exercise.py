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
    # bridging to PyTorch · build (reference solution)

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
    kind = type(module).__name__

    def tensor(a):
        return torch.tensor(np.asarray(a.data if isinstance(a, Tensor) else a), dtype=torch.float64)

    if kind == "Sequential":
        out = nn.Sequential(*[to_torch(module[i]) for i in range(len(module))])
    elif kind == "Linear":
        n_out, n_in = module.weight.shape
        out = nn.Linear(n_in, n_out, bias=module.bias is not None, dtype=torch.float64)
        with torch.no_grad():
            out.weight.copy_(tensor(module.weight))
            if module.bias is not None:
                out.bias.copy_(tensor(module.bias))
    elif kind in ("ReLU", "Tanh", "Sigmoid"):
        out = getattr(nn, kind)()
    elif kind == "Dropout":
        out = nn.Dropout(module.p)
    elif kind == "LayerNorm":
        out = nn.LayerNorm(module.weight.shape[0], eps=module.eps, dtype=torch.float64)
        with torch.no_grad():
            out.weight.copy_(tensor(module.weight))
            out.bias.copy_(tensor(module.bias))
    elif kind == "BatchNorm1d":
        out = nn.BatchNorm1d(module.weight.shape[0], eps=module.eps, momentum=module.momentum, dtype=torch.float64)
        with torch.no_grad():
            out.weight.copy_(tensor(module.weight))
            out.bias.copy_(tensor(module.bias))
            out.running_mean.copy_(tensor(module.running_mean))
            out.running_var.copy_(tensor(module.running_var))
    elif kind == "RMSNorm":
        out = RMSNorm(module.weight.shape[0], eps=module.eps).double()
        with torch.no_grad():
            out.weight.copy_(tensor(module.weight))
    else:
        raise TypeError(f"to_torch doesn't know how to convert {kind}")
    return out.train(bool(getattr(module, "training", True)))


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
    mine, theirs = copy.deepcopy(model), to_torch(model)  # convert first: both start from the same state
    integer = np.issubdtype(np.asarray(y).dtype, np.integer)
    for p in mine.parameters():
        p.grad = None
    out = mine(Tensor(X))
    loss = cross_entropy(out, y) if integer else mse_loss(out, y)
    loss.backward()
    t_out = theirs(torch.tensor(np.asarray(X), dtype=torch.float64))
    t_y = torch.tensor(np.asarray(y))
    t_loss = (nn.functional.cross_entropy(t_out, t_y) if integer
              else nn.functional.mse_loss(t_out, t_y.to(torch.float64)))
    t_loss.backward()
    grads = dict(theirs.named_parameters())
    grad_diff = max(float(np.max(np.abs(p.grad - grads[name].grad.numpy()))) for name, p in mine.named_parameters())
    return {"loss": abs(loss.item() - t_loss.item()),
            "output": float(np.max(np.abs(out.data - t_out.detach().numpy()))),
            "grad": grad_diff}


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
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """stage 3: upcast float16/bfloat16 to float32 (others stay), normalize, cast back to x's dtype, times weight."""
        xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


@app.class_definition
class MLP(nn.Module):
    """Linear -> [norm] -> activation -> [dropout] for every hidden size, then a plain Linear output layer."""

    def __init__(self, in_features, hidden_sizes, out_features, *, activation="relu", dropout=0.0, norm=None):
        """stage 3: build self.net, an nn.Sequential of those layers; ValueError for an unknown activation or norm."""
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
        norms = {None: None, "layer": nn.LayerNorm, "batch": nn.BatchNorm1d}
        if activation not in acts:
            raise ValueError(f"activation must be one of {sorted(acts)}, got {activation!r}")
        if norm not in norms:
            raise ValueError(f"norm must be None, 'layer' or 'batch', got {norm!r}")
        layers, n_in = [], in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(n_in, h))
            if norm is not None:
                layers.append(norms[norm](h))
            layers.append(acts[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            n_in = h
        layers.append(nn.Linear(n_in, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """stage 3: self.net(x)."""
        return self.net(x)


@app.function
def count_parameters(model, *, trainable_only=True):
    """stage 3: the total number of entries in the model's parameters (only those with requires_grad, by default)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@app.function
def get_device(prefer=None):
    """stage 3: torch.device(prefer) if given, else 'cuda' if available, else 'mps' if available, else 'cpu'."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    tensors = [X]
    if y is not None:
        y_arr = np.asarray(y)
        dtype = torch.int64 if np.issubdtype(y_arr.dtype, np.integer) else torch.float32
        tensors.append(torch.as_tensor(y_arr, dtype=dtype))
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=batch_size,
                                       shuffle=shuffle, generator=generator)


@app.function
def evaluate(model, loader, loss_fn=None, *, device="cpu"):
    """stage 4: under no_grad in eval mode (restored after): {'loss': mean per example (if loss_fn)} and 'accuracy' for (B, C) outputs with integer targets."""
    was_training = model.training
    model.eval()
    total_loss, correct, n, classify = 0.0, 0, 0, True
    try:
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                out = model(x)
                n += len(x)
                if len(batch) < 2:
                    classify = False
                    continue
                y = batch[1].to(device)
                if loss_fn is not None:
                    total_loss += loss_fn(out, y).item() * len(x)
                if out.ndim == 2 and not torch.is_floating_point(y):
                    correct += (out.argmax(dim=1) == y).sum().item()
                else:
                    classify = False
    finally:
        model.train(was_training)
    result = {}
    if loss_fn is not None and n:
        result["loss"] = total_loss / n
    if classify and n:
        result["accuracy"] = correct / n
    return result


@app.function
def fit(model, loss_fn, optimizer, train_loader, *, epochs=1, val_loader=None, scheduler=None, grad_clip=None,
        device="cpu", callback=None):
    """stage 4: per epoch: train mode, per batch zero_grad/forward/loss/backward/[clip]/step/[scheduler.step()]; history train_loss, val_loss, lr."""
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "lr": []}
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * len(x)
            n += len(x)
        history["train_loss"].append(total / max(n, 1))
        if val_loader is not None:
            history["val_loss"].append(evaluate(model, val_loader, loss_fn, device=device)["loss"])
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        if callback is not None and callback(epoch, history):
            break
    return history


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
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(decay).add_(s.detach(), alpha=1.0 - decay)
        for t, s in zip(target.buffers(), source.buffers()):
            t.copy_(s)


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
