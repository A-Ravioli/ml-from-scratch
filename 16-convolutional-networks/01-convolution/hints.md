# convolution · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`conv_output_size` is one line of integer arithmetic: `//` is floor division. for `conv2d_naive`: unpack the shapes, turn `stride`, `padding` and `dilation` into pairs with `as_pair`, compute `H_out` and `W_out`, pad once with `np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))`, then loop over `i in range(H_out)` and `j in range(W_out)`. the window for output `(i, j)` starts at padded pixel `(i * sh, j * sw)` and takes every `dh`-th row for `kh` rows. raise `ValueError` if the channel counts disagree.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for i in range(h_out):
    for j in range(w_out):
        r, c = i * sh, j * sw
        patch = xp[:, :, r:r + dh * (kh - 1) + 1:dh, c:c + dw * (kw - 1) + 1:dw]  # (N, C_in, kh, kw)
        out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))      # (N, C_out)
if b is not None:
    out += np.asarray(b).reshape(1, -1, 1, 1)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

allocate `cols = np.empty((N, C, kh, kw, H_out, W_out))` and fill it one kernel offset `(i, j)` at a time: the slice you need is every window's tap `(i, j)`, which starts at padded pixel `(i * dh, j * dw)` and steps by the stride: `xp[:, :, i*dh : i*dh + sh*(H_out-1)+1 : sh, j*dw : ... : sw]` has shape `(N, C, H_out, W_out)`. reshape to `(N, C*kh*kw, H_out*W_out)` at the end: that reshape gives exactly the `(c, i, j)` row order. `col2im` reshapes the columns back to six axes and runs the same loop with `+=` into a zero padded image of shape `(N, C, H + 2ph, W + 2pw)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# im2col
for i in range(kh):
    for j in range(kw):
        r, s = i * dh, j * dw
        cols[:, :, i, j] = xp[:, :, r:r + sh * (h_out - 1) + 1:sh, s:s + sw * (w_out - 1) + 1:sw]
return cols.reshape(n, c * kh * kw, h_out * w_out)

# col2im
cols = cols.reshape(n, c, kh, kw, h_out, w_out)
xp = np.zeros((n, c, h + 2 * ph, wd + 2 * pw), dtype=cols.dtype)
for i in range(kh):
    for j in range(kw):
        r, s = i * dh, j * dw
        xp[:, :, r:r + sh * (h_out - 1) + 1:sh, s:s + sw * (w_out - 1) + 1:sw] += cols[:, :, i, j]
return xp[:, :, ph:ph + h, pw:pw + wd]
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`cols = im2col(x, (kh, kw), stride=..., padding=..., dilation=...)` has shape `(N, K, L)` with `K = C_in*kh*kw`, `L = H_out*W_out`. `w.reshape(C_out, -1)` has shape `(C_out, K)`, and `@` broadcasts a 2-D matrix against a stack of matrices. add the bias as `b.reshape(1, -1, 1)` and reshape the result to `(N, C_out, H_out, W_out)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
cols = im2col(x, (kh, kw), stride=stride, padding=padding, dilation=dilation)  # (N, K, L)
out = w.reshape(c_out, -1) @ cols                                              # (N, C_out, L)
if b is not None:
    out = out + np.asarray(b).reshape(1, -1, 1)
return out.reshape(n, c_out, h_out, w_out)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

reshape `dout` to `G` of shape `(N, C_out, L)` and rebuild `X = im2col(x, ...)` of shape `(N, K, L)`. then `db` sums `G` over axes 0 and 2; `dw` is $\sum_n G_n X_n^\top$, one `np.einsum` call (or `G @ X.transpose(0, 2, 1)` summed over the batch), reshaped to `w.shape`; the columns' gradient is `W.T @ G` with `W = w.reshape(C_out, -1)`; and `dx` is `col2im` of that, with the same kernel size, stride, padding and dilation as the forward pass.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
d = dout.reshape(n, c_out, -1)
cols = im2col(x, (kh, kw), stride=stride, padding=padding, dilation=dilation)
db = d.sum(axis=(0, 2))
dw = np.einsum("nol,nkl->ok", d, cols).reshape(w.shape)
dcols = w.reshape(c_out, -1).T @ d
dx = col2im(dcols, x.shape, (kh, kw), stride=stride, padding=padding, dilation=dilation)
return dx, dw, db
```

</details>
