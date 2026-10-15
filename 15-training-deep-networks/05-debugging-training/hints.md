# debugging training · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`grad_norm`: `math.sqrt(sum(np.sum(p.grad ** 2) for p in params if p.grad is not None))`, as a Python float. `named_modules`: a recursive `visit(module, name)` that appends `(name, module)` first, then loops over `vars(module).items()`, recursing into `Module` values with `prefix + key` and into each `Module` in a list or tuple with `f"{prefix}{key}.{i}"`, where `prefix` is `name + "."` (or `""` at the root).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
out = []

def visit(module, name):
    out.append((name, module))
    prefix = f"{name}." if name else ""
    for key, value in vars(module).items():
        if isinstance(value, Module):
            visit(value, prefix + key)
        elif isinstance(value, (list, tuple)):
            for i, m in enumerate(value):
                if isinstance(m, Module):
                    visit(m, f"{prefix}{key}.{i}")

visit(model, "")
return out
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`update_ratio`: two `np.linalg.norm` calls; if the norm of `after` is 0, return 0.0 when the step is 0 too (and `inf` otherwise). `dead_fraction`: reshape to `(batch, -1)` so extra axes become units, then `np.all(a == 0, axis=0)` marks the dead units; return its mean.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
a = np.asarray(activations)
a = a.reshape(a.shape[0], -1)
return float(np.mean(np.all(a == 0, axis=0)))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`__init__`: store `self.params = model.named_parameters()`, `self.history = {"step": [], "loss": [], "lr": [], "grad_norm": []}`, `self._before = {name: p.data.copy() ...}` and `self._dead = {}`, then call `self._watch(name, module)` for every `(name, module)` in `named_modules(model)` that is a `ReLU`. `log`: append the four scalars, then for each parameter use `history.setdefault(f"grad/{name}", []).append(...)` (and the same for `weight/` and `update/`), refresh `self._before[name]` with a copy, and finally append the latest `self._dead` values under `dead/<name>`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
h = self.history
h["step"].append(step)
h["loss"].append(float(loss.item() if hasattr(loss, "item") else loss))
h["lr"].append(math.nan if lr is None else float(lr))
h["grad_norm"].append(grad_norm(p for _, p in self.params))
for name, p in self.params:
    h.setdefault(f"grad/{name}", []).append(math.nan if p.grad is None else float(np.linalg.norm(p.grad)))
    h.setdefault(f"weight/{name}", []).append(float(np.linalg.norm(p.data)))
    h.setdefault(f"update/{name}", []).append(update_ratio(self._before[name], p.data))
    self._before[name] = p.data.copy()
for name, frac in self._dead.items():
    h.setdefault(f"dead/{name}", []).append(frac)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

write `summary` as six independent blocks, one per row of the lesson's table, each appending an f-string that starts with its tag. `self._weights()` (given) lists the weight matrices. use `np.nanmedian` for the per-layer gradient medians (a `nan` from a missing gradient shouldn't poison them), and compare quarter means for the plateau rule: `q = max(n // 4, 1)`. `plot`: `fig, axes = plt.subplots(1, 4)`, one panel per group of series, `semilogy` for gradient norms and update ratios; return `fig`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for w in self._weights():
    r = np.array(h[f"update/{w}"], dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        continue
    m = float(np.median(r))
    if m > 1e-1:
        out.append(f"update ratio too large: {w} changes by {m:.2g} of its size per step")
    elif m < 1e-5:
        out.append(f"update ratio too small: {w} changes by {m:.2g} of its size per step")
```

</details>
