# a tensor autograd engine · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

two steps. first, `extra = grad.ndim - len(shape)` leading axes were prepended by broadcasting: sum them away (`grad.sum(axis=tuple(range(extra)))`). now `grad` has as many axes as `shape`. second, every axis `i` with `shape[i] == 1` (but `grad.shape[i] != 1`) was stretched: sum it with `keepdims=True`. a final `reshape(shape)` also covers the scalar case `shape == ()`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
grad = np.asarray(grad, dtype=np.float64)
extra = grad.ndim - len(shape)
if extra > 0:
    grad = grad.sum(axis=tuple(range(extra)))
axes = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
if axes:
    grad = grad.sum(axis=axes, keepdims=True)
return grad.reshape(shape)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`from_op`: build `out = Tensor(data)`; if `is_grad_enabled()` and any parent has `requires_grad`, set `out.requires_grad = True`, `out._parents = tuple(parents)`, `out._vjp = vjp`. every operator is then three lines: lift the other operand (`Tensor._lift`), compute the forward array, and pass a `lambda g: (...)` that returns one gradient per parent, each through `unbroadcast(·, parent_shape)`. capture the *arrays* you need (`x, y = self.data, other.data`) in the lambda, not `self.data` lazily.

`backward`: (1) check `requires_grad`, default `grad` to ones for one-element outputs (`ValueError` otherwise); (2) build a topological order with an explicit stack of `(node, expanded)` pairs: when you pop an unexpanded node, push it back as expanded, then push its parents; when you pop an expanded node, append it to the order; (3) walk the order reversed with a dict `id(node) -> running gradient`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
grads = {id(self): grad}
for node in reversed(order):
    g = grads.pop(id(node), None)
    if g is None:
        continue
    if not node._parents:                       # a leaf you created
        node.grad = np.array(g) if node.grad is None else node.grad + g
        continue
    for p, gp in zip(node._parents, node._vjp(g)):
        if gp is not None and p.requires_grad:
            grads[id(p)] = grads[id(p)] + gp if id(p) in grads else gp
```

and for multiplication: `return Tensor.from_op(x * y, (self, other), lambda g: (unbroadcast(g * y, x.shape), unbroadcast(g * x, y.shape)))`. `sum` is `Tensor.from_op(self.data.sum(axis=axis, keepdims=keepdims), (self,), lambda g: (Tensor._expand(g, shape, axis, keepdims),))`.

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

compute the forward output once and reuse it in the VJP (`exp`, `tanh` and `sigmoid` derivatives are functions of the output). for `mean`, divide `_expand(g, …)` by `x.size / out.size`. for `max` and `logsumexp`, compute the max with `keepdims=True` (so it broadcasts against `x`), then squeeze the reduced axes for the output if `keepdims` is False. for `logsumexp`, replace a non-finite max by 0 before shifting (all `-inf` rows).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# max
m = x.max(axis=axis, keepdims=True)
mask = (x == m).astype(float)
mask /= mask.sum(axis=axis, keepdims=True)           # split ties evenly
out = m if keepdims else (m.reshape(()) if axis is None else np.squeeze(m, axis=axis))
vjp = lambda g: (Tensor._expand(g, x.shape, axis, keepdims) * mask,)

# logsumexp: lse has keepdims=True shape; soft = np.exp(x - lse) is the softmax
vjp = lambda g: (Tensor._expand(g, x.shape, axis, keepdims) * soft,)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

matmul: in the VJP, promote 1-D operands (`A[None, :]`, `B[:, None]`) and expand `g` to match (`np.expand_dims(g, -2)` or `-1`), compute `gA = G @ swapaxes(B2, -1, -2)` and `gB = swapaxes(A2, -1, -2) @ G`, drop the promoted axis again, and `unbroadcast` both to the original shapes (that sums any batch axes). indexing: `full = np.zeros_like(x); np.add.at(full, idx, g)`. `concatenate`: remember the sizes along `axis`, and `np.split(g, np.cumsum(sizes)[:-1], axis=axis)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# transpose
inverse = tuple(np.argsort(axes))
vjp = lambda g: (np.transpose(g, inverse),)

# stack
ax = axis % out.ndim
vjp = lambda g: tuple(np.take(g, i, axis=ax) for i in range(len(ts)))

# where
vjp = lambda g: (unbroadcast(np.where(cond, g, 0.0), x.shape), unbroadcast(np.where(cond, 0.0, g), y.shape))
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`no_grad`: in `__enter__`, save `no_grad._enabled` on `self` and set the class attribute to False; in `__exit__`, restore the saved value and return False (don't swallow exceptions). `gradcheck`: compute `out = f(*inputs)`, draw fixed weights `W = np.random.default_rng(0).normal(size=out.shape)`, clear the inputs' grads (after saving them), backpropagate `(out * W).sum()`, then for each input with `requires_grad` compare its `.grad` with `numerical_gradient(scalar, x.data, h=eps)`, where `scalar(arr)` temporarily sets `x.data = arr`, evaluates `float((f(*inputs).data * W).sum())` under `no_grad`, and restores `x.data`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for i, x in enumerate(inputs):
    if not (isinstance(x, Tensor) and x.requires_grad):
        continue
    analytic = x.grad if x.grad is not None else np.zeros_like(x.data)
    def scalar(arr, x=x):
        old, x.data = x.data, arr
        try:
            with no_grad():
                return float((f(*inputs).data * W).sum())
        finally:
            x.data = old
    numeric = numerical_gradient(scalar, x.data, h=eps)
    if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
        ...  # print the worst entry, ok = False
# finally: restore every input's saved .grad, return ok
```

</details>
