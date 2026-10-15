# pooling & receptive fields · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

set `stride = kernel_size if stride is None else stride`, then turn both into pairs with `as_pair`. treat every channel as its own one-channel image: `x.reshape(n * c, 1, h, w)`. `im2col` with the pooling kernel and stride (no padding) puts every window in a column of shape `(N*C, kh*kw, L)`; reduce over axis 1 and reshape to `(n, c, h_out, w_out)` with `h_out = (h - kh) // sh + 1`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
cols = im2col(x.reshape(n * c, 1, h, w), (kh, kw), stride=(sh, sw))  # (N*C, kh*kw, L)
return cols.max(axis=1).reshape(n, c, h_out, w_out)                    # .mean(axis=1) for avg_pool2d
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

build the gradient with respect to the columns, shape `(N*C, kh*kw, L)`, then `col2im(dcols, (n * c, 1, h, w), (kh, kw), stride=stride)` and reshape to `x`'s shape: col2im adds the contributions of overlapping windows and leaves never-read pixels at zero. for max pooling, `winner = cols.argmax(axis=1)` (shape `(N*C, L)`) says which row of each column gets the window's gradient. for average pooling, every row gets `dout / (kh * kw)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# max
cols = im2col(x.reshape(n * c, 1, h, w), (kh, kw), stride=stride)
winner = cols.argmax(axis=1)
dcols = np.zeros_like(cols)
np.put_along_axis(dcols, winner[:, None, :], dout.reshape(n * c, 1, -1), axis=1)
return col2im(dcols, (n * c, 1, h, w), (kh, kw), stride=stride).reshape(x.shape)

# average
d = dout.reshape(n * c, 1, -1) / (kh * kw)
dcols = np.broadcast_to(d, (n * c, kh * kw, d.shape[-1]))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

start from `rf, jump = 1, 1` and walk the layers in order: each one adds `d * (k - 1) * jump` to the field, and *then* multiplies the jump by its stride. for `rf_layers`, iterate over `model.modules()` (it walks nested containers in order) and keep the `nn.Conv2d`, `nn.MaxPool2d` and `nn.AvgPool2d` layers. their `kernel_size`, `stride` and `dilation` may be ints or pairs: `as_pair(v)[0]`. pooling layers may have `stride=None` (meaning the kernel size), and `nn.AvgPool2d` has no `dilation` attribute (`getattr(m, "dilation", 1)`).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
rf, jump = 1, 1
for k, s, d in layers:
    rf += d * (k - 1) * jump
    jump *= s
return rf, jump
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

take the dtype and device from `next(model.parameters())`; draw `x` with `torch.randn(n_samples, in_channels, h, w, generator=torch.Generator().manual_seed(seed), dtype=dtype)`, move it to the device and call `.requires_grad_(True)`. remember `model.training`, switch to `model.eval()`, run the model, select `out[:, :, oh // 2, ow // 2]`, sum it, and get `torch.autograd.grad(total, x)`. restore the mode in a `finally:`. then `grad.abs().sum(dim=(0, 1))`, converted to a float64 NumPy array and divided by its sum. `erf_radius` tries `r = 0, 1, 2, ...` and returns the first whose centred square's sum reaches `mass`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
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

# erf_radius
ci, cj = h // 2, w // 2
for r in range(max(h, w)):
    if erf[max(ci - r, 0):ci + r + 1, max(cj - r, 0):cj + r + 1].sum() >= mass - 1e-12:
        return r
```

</details>
