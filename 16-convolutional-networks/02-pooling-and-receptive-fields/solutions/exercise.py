import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.cnn import as_pair, col2im, im2col  # yours, from topic 01


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # pooling & receptive fields · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. a convolution sees a few pixels; a
    network needs to see a whole object. this topic is about how that happens: pooling
    layers (and their backward passes, built on your `im2col`/`col2im`), the arithmetic
    of how far each output can see, and a measurement of how far it *actually* looks.

    | stage | you build |
    |---|---|
    | 1 | `max_pool2d`, `avg_pool2d`: downsampling by taking the max or mean of each window (NumPy) |
    | 2 | `max_pool2d_backward`, `avg_pool2d_backward`: where pooling sends its gradients |
    | 3 | `receptive_field`, `rf_layers`: how many input pixels one output can depend on |
    | 4 | `effective_receptive_field`, `erf_radius`: how many it actually depends on (PyTorch autograd) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def max_pool2d(x, kernel_size, stride=None):
    """stage 1: the max of every kernel_size window, stride defaults to kernel_size, no padding: (N, C, H, W) -> (N, C, H_out, W_out)."""
    n, c, h, w = x.shape
    stride = kernel_size if stride is None else stride
    kh, kw = as_pair(kernel_size)
    sh, sw = as_pair(stride)
    h_out, w_out = (h - kh) // sh + 1, (w - kw) // sw + 1
    cols = im2col(x.reshape(n * c, 1, h, w), (kh, kw), stride=(sh, sw))  # (N*C, kh*kw, L)
    return cols.max(axis=1).reshape(n, c, h_out, w_out)


@app.function
def avg_pool2d(x, kernel_size, stride=None):
    """stage 1: the mean of every kernel_size window, stride defaults to kernel_size, no padding."""
    n, c, h, w = x.shape
    stride = kernel_size if stride is None else stride
    kh, kw = as_pair(kernel_size)
    sh, sw = as_pair(stride)
    h_out, w_out = (h - kh) // sh + 1, (w - kw) // sw + 1
    cols = im2col(x.reshape(n * c, 1, h, w), (kh, kw), stride=(sh, sw))
    return cols.mean(axis=1).reshape(n, c, h_out, w_out)


@app.cell(hide_code=True)
def _():
    checks.run(1, max_pool2d=max_pool2d, avg_pool2d=avg_pool2d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def max_pool2d_backward(dout, x, kernel_size, stride=None):
    """stage 2: dL/dx for max_pool2d: each window's gradient goes to its (first) max; overlapping windows add."""
    n, c, h, w = x.shape
    stride = kernel_size if stride is None else stride
    kh, kw = as_pair(kernel_size)
    cols = im2col(x.reshape(n * c, 1, h, w), (kh, kw), stride=stride)  # (N*C, K, L)
    winner = cols.argmax(axis=1)  # (N*C, L)
    dcols = np.zeros_like(cols)
    np.put_along_axis(dcols, winner[:, None, :], dout.reshape(n * c, 1, -1), axis=1)
    return col2im(dcols, (n * c, 1, h, w), (kh, kw), stride=stride).reshape(x.shape)


@app.function
def avg_pool2d_backward(dout, x_shape, kernel_size, stride=None):
    """stage 2: dL/dx for avg_pool2d: each window's gradient is shared equally by its kh*kw pixels."""
    n, c, h, w = x_shape
    stride = kernel_size if stride is None else stride
    kh, kw = as_pair(kernel_size)
    d = dout.reshape(n * c, 1, -1) / (kh * kw)
    dcols = np.broadcast_to(d, (n * c, kh * kw, d.shape[-1]))
    return col2im(dcols, (n * c, 1, h, w), (kh, kw), stride=stride).reshape(x_shape)


@app.cell(hide_code=True)
def _():
    checks.run(2, max_pool2d_backward=max_pool2d_backward, avg_pool2d_backward=avg_pool2d_backward,
               max_pool2d=max_pool2d, avg_pool2d=avg_pool2d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def receptive_field(layers):
    """stage 3: layers = [(kernel_size, stride, dilation), ...] in order -> (rf, jump) of the last layer's outputs, in input pixels."""
    rf, jump = 1, 1
    for k, s, d in layers:
        rf += d * (k - 1) * jump
        jump *= s
    return rf, jump


@app.function
def rf_layers(model):
    """stage 3: the (kernel_size, stride, dilation) of every Conv2d, MaxPool2d and AvgPool2d in model.modules(), in order."""
    out = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            k = as_pair(m.kernel_size)[0]
            s = as_pair(m.stride if m.stride is not None else m.kernel_size)[0]
            d = as_pair(getattr(m, "dilation", 1))[0]
            out.append((k, s, d))
    return out


@app.cell(hide_code=True)
def _():
    checks.run(3, receptive_field=receptive_field, rf_layers=rf_layers)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def effective_receptive_field(model, input_size, *, in_channels=1, n_samples=8, seed=0):
    """stage 4: mean |d(centre output, summed over channels) / d input| over n_samples Gaussian inputs, summed over input channels, normalized to sum 1 -> (H, W) ndarray."""
    h, w = as_pair(input_size)
    params = list(model.parameters())
    dtype = params[0].dtype if params else torch.float32
    device = params[0].device if params else torch.device("cpu")
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, in_channels, h, w, generator=gen, dtype=dtype).to(device).requires_grad_(True)
    was_training = model.training
    model.eval()
    try:
        out = model(x)
        oh, ow = out.shape[-2:]
        (grad,) = torch.autograd.grad(out[:, :, oh // 2, ow // 2].sum(), x)
    finally:
        model.train(was_training)
    erf = grad.abs().sum(dim=(0, 1)).double().cpu().numpy()
    return erf / erf.sum()


@app.function
def erf_radius(erf, mass=0.9):
    """stage 4: the smallest r such that the (2r+1) x (2r+1) square centred at (H // 2, W // 2) holds at least `mass` of the erf."""
    h, w = erf.shape
    ci, cj = h // 2, w // 2
    for r in range(max(h, w)):
        if erf[max(ci - r, 0):ci + r + 1, max(cj - r, 0):cj + r + 1].sum() >= mass - 1e-12:
            return r
    return max(h, w)


@app.cell(hide_code=True)
def _():
    checks.run(4, effective_receptive_field=effective_receptive_field, erf_radius=erf_radius)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.cnn import max_pool2d, receptive_field` gives you these tools, and you may use
    `torch.nn.MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` and their functional forms:
    you've built them.
    """)
    return


if __name__ == "__main__":
    app.run()
