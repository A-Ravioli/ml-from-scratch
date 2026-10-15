# normalization · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`standardize`: `mean` and `var` with `axis=axis, keepdims=True` (NumPy's `var` is the biased one by default), `inv_std = 1 / np.sqrt(var + eps)`, return `((x - mean) * inv_std, inv_std)`. `standardize_backward` is the boxed formula, one line: every mean is over `axis` with `keepdims=True` so it broadcasts back.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def standardize_backward(g, xhat, inv_std, axis):
    return inv_std * (g - g.mean(axis=axis, keepdims=True)
                      - xhat * (g * xhat).mean(axis=axis, keepdims=True))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

in `forward`, wrap an ndarray input as a `Tensor`, then branch on `self.training`. training: `xhat, inv_std = standardize(x.data, axis=0, eps=self.eps)`, update the running statistics in place (`self.running_mean[...] = ...`, with `x.data.var(axis=0, ddof=1)` for the variance), and define a `vjp(g)` returning three gradients: `standardize_backward(g * w.data, xhat, inv_std, axis=0)`, `(g * xhat).sum(axis=0)` and `g.sum(axis=0)`. eval: compute `xhat` from the running statistics; the input gradient is `g * w.data * inv_std`. both branches end with `Tensor.from_op(xhat * w.data + b.data, (x, w, b), vjp)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if self.training:
    if x.shape[0] < 2:
        raise ValueError("need more than 1 example per batch in training mode")
    xhat, inv_std = standardize(x.data, axis=0, eps=self.eps)
    m = self.momentum
    self.running_mean[...] = (1 - m) * self.running_mean + m * x.data.mean(axis=0)
    self.running_var[...] = (1 - m) * self.running_var + m * x.data.var(axis=0, ddof=1)

    def vjp(g):
        return (standardize_backward(g * w.data, xhat, inv_std, axis=0),
                (g * xhat).sum(axis=0), g.sum(axis=0))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

the same as batch norm's training branch with `axis=-1`, no running statistics and no train/eval branch. the parameter gradients sum over every axis except the last: `lead = tuple(range(x.ndim - 1))`, then `(g * xhat).sum(axis=lead)` and `g.sum(axis=lead)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
xhat, inv_std = standardize(x.data, axis=-1, eps=self.eps)
lead = tuple(range(x.ndim - 1))

def vjp(g):
    return (standardize_backward(g * w.data, xhat, inv_std, axis=-1),
            (g * xhat).sum(axis=lead), g.sum(axis=lead))

return Tensor.from_op(xhat * w.data + b.data, (x, w, b), vjp)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`r = 1 / np.sqrt((X * X).mean(axis=-1, keepdims=True) + self.eps)` with `X = x.data`; the output is `X * r * w.data`. the parents are `(x, w)` only (there's no bias), so the `vjp` returns two gradients: with `u = g * w.data`, `r * u - X * r**3 * (u * X).mean(axis=-1, keepdims=True)` and `(g * X * r).sum(axis=lead)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def vjp(g):
    u = g * w.data
    dx = r * u - X * r ** 3 * (u * X).mean(axis=-1, keepdims=True)
    return dx, (g * X * r).sum(axis=lead)

return Tensor.from_op(X * r * w.data, (x, w), vjp)
```

</details>
