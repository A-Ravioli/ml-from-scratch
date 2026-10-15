# backprop in depth · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`vjp`: wrap `x` in a `Tensor` with `requires_grad=True`, run `out = f(t)`, then `out.backward(v)` (reshape `v` to `out.shape`). the input's `.grad` is $J^\top v$; return `(out.data.copy(), t.grad)` (zeros if `t.grad` is `None`: the output didn't depend on `x`). `jacobian_reverse`: one forward pass, then for each output index `i`, reset `t.grad = None` and backpropagate the basis vector $e_i$; row `i` of the Jacobian is `t.grad.ravel()`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
t = Tensor(x, requires_grad=True)
out = f(t)
J = np.zeros((out.size, t.size))
for i in range(out.size):
    t.grad = None
    e = np.zeros(out.size)
    e[i] = 1.0
    out.backward(e.reshape(out.shape))
    if t.grad is not None:
        J[i] = t.grad.ravel()
return J
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

walk the graph with a stack, starting from `out`. pop a node; skip it if you've seen its `id`, or if it's a leaf (`not node._parents`: leaves and constants hold no graph); otherwise mark it seen, add `node.data.nbytes`, and push its `_parents`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
total, seen, stack = 0, set(), [out]
while stack:
    node = stack.pop()
    if id(node) in seen or not node._parents:
        continue
    seen.add(id(node))
    total += node.data.nbytes
    stack.extend(node._parents)
return total
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

three parts. (1) lift non-Tensor inputs, then run `out = fn(*inputs)` inside `with no_grad():` so nothing inside is recorded. (2) if grad is disabled (`not is_grad_enabled()`), return `Tensor(out.data)` as a constant. (3) otherwise build the node by hand: `result = Tensor(out.data)`, `result.requires_grad = True`, `result._parents = tuple(inputs)`, `result._vjp = recompute`. `recompute(g)` makes fresh copies of the inputs (`Tensor(x.data, requires_grad=x.requires_grad)`), runs `fn` on them with recording on, backpropagates `g` (if the result requires grad), and returns one gradient per input (`None` for inputs that don't require grad).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def recompute(g):
    fresh = [Tensor(x.data, requires_grad=x.requires_grad) for x in inputs]
    y = fn(*fresh)                     # weights closed over by fn get their .grad here
    if y.requires_grad:
        y.backward(g)
    return tuple(d.grad if d.requires_grad else None for d in fresh)
```

don't use `from_op` for the node: it drops the graph when no input requires grad, and the first block of a network usually gets plain data.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

clip `segments` to `[1, len(functions)]`, split the layer indices with `np.array_split(np.arange(len(functions)), segments)`, and for each chunk build a function that applies that chunk's layers in order. then `x = checkpoint(segment_fn, x)` chunk by chunk. careful with closures in a loop: bind the chunk as a default argument or through a helper function, or every segment will run the last chunk.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def run(chunk):
    def segment(h):
        for i in chunk:
            h = functions[i](h)
        return h
    return segment

for chunk in np.array_split(np.arange(len(functions)), segments):
    x = checkpoint(run(list(chunk)), x)
return x
```

</details>
