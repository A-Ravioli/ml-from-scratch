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
    # convolution · build

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
    raise NotImplementedError("stage1: implement conv_output_size")


@app.function
def conv2d_naive(x, w, b=None, *, stride=1, padding=0, dilation=1):
    """stage 1: cross-correlation by loops over output pixels: x (N, C_in, H, W), w (C_out, C_in, kh, kw) -> (N, C_out, H_out, W_out)."""
    raise NotImplementedError("stage1: implement conv2d_naive")


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
    raise NotImplementedError("stage2: implement im2col")


@app.function
def col2im(cols, x_shape, kernel_size, *, stride=1, padding=0, dilation=1):
    """stage 2: the adjoint of im2col: scatter-add every column back to its pixels -> (N, C, H, W)."""
    raise NotImplementedError("stage2: implement col2im")


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
    raise NotImplementedError("stage3: implement conv2d")


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
    raise NotImplementedError("stage4: implement conv2d_backward")


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
