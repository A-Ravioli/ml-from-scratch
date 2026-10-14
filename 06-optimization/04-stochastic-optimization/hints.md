# stochastic optimization · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`Param` and `Optimizer.__init__` are a few assignments; the details that matter are *copying* the data (with a float64 dtype) and turning `params` into a list exactly once. `minibatch_indices` is a generator: build the order of indices for the whole epoch first, then `yield` slices of it.

</details>

<details><summary>hint 2: the approach</summary>

`self.data = np.array(data, dtype=np.float64)` always copies (unlike `np.asarray`). `self.params = list(params)`, `self.lr = lr`, `self.state = {}`. `zero_grad` loops over `self.params` setting `p.grad = None`. for the batches: `order = as_rng(rng).permutation(n) if shuffle else np.arange(n)`; the last start index is `n` normally, or `n - n % batch_size` with `drop_last`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
order = as_rng(rng).permutation(n) if shuffle else np.arange(n)
stop = n - n % batch_size if drop_last else n
for start in range(0, stop, batch_size):
    yield order[start:start + batch_size]
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

loop over `enumerate(self.params)` so you have the index for `self.state`. skip parameters whose grad is `None`. build the effective gradient as a new array, then apply momentum, then update `p.data` in place.

</details>

<details><summary>hint 2: the approach</summary>

`g = p.grad + self.weight_decay * p.data`. if `self.momentum != 0`: get `st = self.state.setdefault(i, {})`; if it has no `"momentum_buffer"` yet, store a copy of `g`, else replace it with `self.momentum * buf + g`. then `g = g + self.momentum * buf` for Nesterov, else `g = buf`. finally `p.data -= self.lr * g`. in `__init__`, check `nesterov and momentum <= 0` before calling `super().__init__`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for i, p in enumerate(self.params):
    if p.grad is None:
        continue
    g = p.grad + self.weight_decay * p.data
    if self.momentum != 0:
        st = self.state.setdefault(i, {})
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = np.array(g, dtype=float)
        else:
            st["momentum_buffer"] = self.momentum * st["momentum_buffer"] + g
        b = st["momentum_buffer"]
        g = g + self.momentum * b if self.nesterov else b
    p.data -= self.lr * g
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

both optimizers keep one array per parameter, the same shape as the gradient, starting at zeros. update that array, then divide the gradient by its square root (plus `eps`, outside the square root) elementwise.

</details>

<details><summary>hint 2: the approach</summary>

`st = self.state.setdefault(i, {"sum": np.zeros_like(g)})` creates the state on the first real step. Adagrad: `st["sum"] = st["sum"] + g * g`. RMSProp: `st["square_avg"] = alpha * st["square_avg"] + (1 - alpha) * g * g`. then `p.data -= self.lr * g / (np.sqrt(...) + self.eps)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# Adagrad.step
for i, p in enumerate(self.params):
    if p.grad is None:
        continue
    g = np.asarray(p.grad, dtype=float)
    st = self.state.setdefault(i, {"sum": np.zeros_like(g)})
    st["sum"] = st["sum"] + g * g
    p.data -= self.lr * g / (np.sqrt(st["sum"]) + self.eps)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

per parameter you need three things in `self.state[i]`: a step counter and two moving averages. increment the counter first, so the bias corrections use $t = 1$ on the first step. AdamW is the same code with two changes: the decay is a multiplication of `p.data` before the update, and it's not added to the gradient.

</details>

<details><summary>hint 2: the approach</summary>

`st = self.state.setdefault(i, {"step": 0, "exp_avg": np.zeros_like(g), "exp_avg_sq": np.zeros_like(g)})`, `st["step"] += 1`, update `exp_avg` with `beta1` and `exp_avg_sq` with `beta2`, then `m_hat = exp_avg / (1 - beta1**t)`, `v_hat = exp_avg_sq / (1 - beta2**t)` and `p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)`. for Adam, `g = p.grad + wd * p.data`; for AdamW, `g = p.grad` and first `p.data *= 1 - lr * wd`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
b1, b2 = self.betas
for i, p in enumerate(self.params):
    if p.grad is None:
        continue
    g = p.grad + self.weight_decay * p.data      # AdamW: g = p.grad; p.data *= 1 - self.lr * self.weight_decay
    st = self.state.setdefault(i, {"step": 0, "exp_avg": np.zeros_like(g), "exp_avg_sq": np.zeros_like(g)})
    st["step"] += 1
    t = st["step"]
    st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
    st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * g * g
    m_hat = st["exp_avg"] / (1 - b1**t)
    v_hat = st["exp_avg_sq"] / (1 - b2**t)
    p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

both are direct transcriptions of formulas from idea 5. for the variance you need the spread $S$: the mean over examples of the squared distance of each per-example gradient from their mean.

</details>

<details><summary>hint 2: the approach</summary>

`S = np.mean(np.sum((G - G.mean(axis=0)) ** 2, axis=1))`, then multiply by `(n - b) / (b * (n - 1))`. for the floor, convert both inputs to arrays and sum `lr * s / (2 * (2 - lr * a))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
G = np.asarray(per_example_grads, dtype=float)
n = G.shape[0]
S = float(np.mean(np.sum((G - G.mean(axis=0)) ** 2, axis=1)))
return (n - batch_size) / (batch_size * (n - 1)) * S
```

</details>
