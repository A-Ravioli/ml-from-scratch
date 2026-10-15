# scalar reverse-mode autodiff · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`__init__`: `self.data = float(data)`, `self.grad = 0.0`, `self._prev = tuple(_prev)`, `self._op = _op`, `self._backward = lambda: None`. each operator: wrap `other` if it isn't a `Value`, create `out = Value(result, (self, other), "+")`, define a closure `_backward()` that adds into `self.grad` and `other.grad` using `out.grad`, attach it with `out._backward = _backward`, and return `out`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

write `__neg__`, `__sub__` and `__truediv__` in terms of operations you already have (`self * -1`, `self + (-other)`, `self * other ** -1`), so they need no closures of their own. `__pow__`, `exp`, `log`, `tanh` and `relu` each create one node with one parent, and a closure `self.grad += (local derivative) * out.grad`. compute the forward value with `math.exp`, `math.log`, `math.tanh` on `self.data`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")

    def _backward():
        self.grad += (1.0 - t * t) * out.grad

    out._backward = _backward
    return out
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`topological_order`: a stack of `(node, expanded)` pairs, starting with `(root, False)`, a `seen` set of ids, and an output list. pop; if expanded, append the node; else, if not seen, mark it seen, push `(node, True)` and then `(p, False)` for each parent. `backward`: get the order, set `grad = 0.0` on every node that has parents, set `self.grad = 1.0`, then call `_backward()` on the nodes in reverse order.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
order, seen, stack = [], set(), [(root, False)]
while stack:
    node, expanded = stack.pop()
    if expanded:
        order.append(node)
    elif id(node) not in seen:
        seen.add(id(node))
        stack.append((node, True))
        stack.extend((p, False) for p in node._prev)
return order
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`value_and_grad`: `xs = [Value(v) for v in np.asarray(x, dtype=float).ravel()]`, `out = f(xs)`, `out.backward()`, then return `out.data` and `np.array([v.grad for v in xs])`. `graph_size` is the length of the topological order.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def graph_size(root):
    return len(topological_order(root))
```

</details>
