import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # convolution · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll write the operation every vision
    model is built on, twice: first as loops you can read, then as one matrix product you
    can afford to train with, and finally its backward pass, so a convolutional layer can
    learn its own filters. everything here is NumPy, in PyTorch's conventions (`NCHW`
    images, `(C_out, C_in, kh, kw)` weights, cross-correlation), and checked against
    `torch.nn.functional.conv2d`.

    | stage | you build |
    |---|---|
    | 1 | `conv_output_size`, `conv2d_naive`: the definition, as loops over output pixels |
    | 2 | `im2col`, `col2im`: every receptive field as a column, and the adjoint that puts them back |
    | 3 | `conv2d`: convolution as one matrix product |
    | 4 | `conv2d_backward`: the gradients with respect to the input, the weights and the bias |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def as_pair(v):
    """an int or a pair -> a tuple (a, b) of ints (a helper for kernel_size, stride, padding, dilation)."""
    if isinstance(v, (tuple, list)):
        a, b = v
        return int(a), int(b)
    return int(v), int(v)


@app.function
def conv_output_size(n, kernel_size, *, stride=1, padding=0, dilation=1):
    """stage 1: the output length along one axis: floor((n + 2p - d(k - 1) - 1) / s) + 1."""
    return (n + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@app.function
def conv2d_naive(x, w, b=None, *, stride=1, padding=0, dilation=1):
    """stage 1: cross-correlation by loops over output pixels: x (N, C_in, H, W), w (C_out, C_in, kh, kw) -> (N, C_out, H_out, W_out)."""
    n, c_in, h, wd = x.shape
    c_out, c_in_w, kh, kw = w.shape
    if c_in != c_in_w:
        raise ValueError(f"x has {c_in} channels but w expects {c_in_w}")
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    dh, dw = as_pair(dilation)
    h_out = conv_output_size(h, kh, stride=sh, padding=ph, dilation=dh)
    w_out = conv_output_size(wd, kw, stride=sw, padding=pw, dilation=dw)
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((n, c_out, h_out, w_out), dtype=np.result_type(x, w))
    for i in range(h_out):
        for j in range(w_out):
            r, c = i * sh, j * sw
            patch = xp[:, :, r:r + dh * (kh - 1) + 1:dh, c:c + dw * (kw - 1) + 1:dw]  # (N, C_in, kh, kw)
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    if b is not None:
        out += np.asarray(b).reshape(1, -1, 1, 1)
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, conv_output_size=conv_output_size, conv2d_naive=conv2d_naive)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def im2col(x, kernel_size, *, stride=1, padding=0, dilation=1):
    """stage 2: x (N, C, H, W) -> cols (N, C*kh*kw, H_out*W_out), rows ordered (c, i, j); loop over kernel offsets only."""
    n, c, h, wd = x.shape
    kh, kw = as_pair(kernel_size)
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    dh, dw = as_pair(dilation)
    h_out = conv_output_size(h, kh, stride=sh, padding=ph, dilation=dh)
    w_out = conv_output_size(wd, kw, stride=sw, padding=pw, dilation=dw)
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    cols = np.empty((n, c, kh, kw, h_out, w_out), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            r, s = i * dh, j * dw
            cols[:, :, i, j] = xp[:, :, r:r + sh * (h_out - 1) + 1:sh, s:s + sw * (w_out - 1) + 1:sw]
    return cols.reshape(n, c * kh * kw, h_out * w_out)


@app.function
def col2im(cols, x_shape, kernel_size, *, stride=1, padding=0, dilation=1):
    """stage 2: the adjoint of im2col: scatter-add every column back to its pixels -> (N, C, H, W)."""
    n, c, h, wd = x_shape
    kh, kw = as_pair(kernel_size)
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    dh, dw = as_pair(dilation)
    h_out = conv_output_size(h, kh, stride=sh, padding=ph, dilation=dh)
    w_out = conv_output_size(wd, kw, stride=sw, padding=pw, dilation=dw)
    cols = cols.reshape(n, c, kh, kw, h_out, w_out)
    xp = np.zeros((n, c, h + 2 * ph, wd + 2 * pw), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            r, s = i * dh, j * dw
            xp[:, :, r:r + sh * (h_out - 1) + 1:sh, s:s + sw * (w_out - 1) + 1:sw] += cols[:, :, i, j]
    return xp[:, :, ph:ph + h, pw:pw + wd]


@app.cell(hide_code=True)
def _():
    checks.run(2, im2col=im2col, col2im=col2im)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def conv2d(x, w, b=None, *, stride=1, padding=0, dilation=1):
    """stage 3: the same result as conv2d_naive, computed as im2col followed by one matrix product."""
    n, _, h, wd = x.shape
    c_out, c_in, kh, kw = w.shape
    if x.shape[1] != c_in:
        raise ValueError(f"x has {x.shape[1]} channels but w expects {c_in}")
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    dh, dw = as_pair(dilation)
    h_out = conv_output_size(h, kh, stride=sh, padding=ph, dilation=dh)
    w_out = conv_output_size(wd, kw, stride=sw, padding=pw, dilation=dw)
    cols = im2col(x, (kh, kw), stride=stride, padding=padding, dilation=dilation)  # (N, K, L)
    out = w.reshape(c_out, -1) @ cols  # (C_out, K) @ (N, K, L) -> (N, C_out, L)
    if b is not None:
        out = out + np.asarray(b).reshape(1, -1, 1)
    return out.reshape(n, c_out, h_out, w_out)


@app.cell(hide_code=True)
def _():
    checks.run(3, conv2d=conv2d, conv2d_naive=conv2d_naive)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def conv2d_backward(dout, x, w, *, stride=1, padding=0, dilation=1):
    """stage 4: given dL/dout (N, C_out, H_out, W_out), return (dx, dw, db) with the shapes of x, w and (C_out,)."""
    n = x.shape[0]
    c_out, _, kh, kw = w.shape
    d = dout.reshape(n, c_out, -1)  # (N, C_out, L)
    cols = im2col(x, (kh, kw), stride=stride, padding=padding, dilation=dilation)  # (N, K, L)
    db = d.sum(axis=(0, 2))
    dw = np.einsum("nol,nkl->ok", d, cols).reshape(w.shape)
    dcols = w.reshape(c_out, -1).T @ d  # (K, C_out) @ (N, C_out, L) -> (N, K, L)
    dx = col2im(dcols, x.shape, (kh, kw), stride=stride, padding=padding, dilation=dilation)
    return dx, dw, db


@app.cell(hide_code=True)
def _():
    checks.run(4, conv2d_backward=conv2d_backward, conv2d=conv2d)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.cnn import conv2d, conv2d_backward` gives you this layer, and you may use
    `torch.nn.Conv2d` and `torch.nn.functional.conv2d` (and `unfold`/`fold`, PyTorch's
    `im2col`/`col2im`): you've built them.
    """)
    return


if __name__ == "__main__":
    app.run()
