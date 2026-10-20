# modules, losses & training loops · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`vars(self)` is the module's attribute dict, in assignment order. write `named_parameters` as a recursive helper `visit(module, prefix)` that loops over `vars(module).items()` and handles three cases: a `Parameter` (append `(prefix + name, value)` unless its `id` is already in a `seen` set), a `Module` (recurse with `prefix + name + "."`), and a list/tuple (recurse into each `Module` element with `prefix + f"{name}.{i}."`). check `Parameter` before anything else. `train` sets the flag and calls `child.train(mode)` for every `(name, child)` in `self._children()`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def load_state_dict(self, state):
    params = dict(self.named_parameters())
    missing, extra = set(params) - set(state), set(state) - set(params)
    if missing or extra:
        raise KeyError(f"missing {sorted(missing)}, unexpected {sorted(extra)}")
    for name, p in params.items():
        value = np.asarray(state[name], dtype=np.float64)
        if value.shape != p.data.shape:
            raise ValueError(f"{name}: expected {p.data.shape}, got {value.shape}")
        p.data[...] = value          # in place: same array object afterwards
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`Linear.__init__`: call `super().__init__()` first, then `rng = as_rng(rng)`, `bound = 1 / np.sqrt(in_features)`, and two `Parameter(rng.uniform(-bound, bound, size=...))` draws, weight `(out, in)` first. `forward`: wrap an ndarray in `Tensor`, return `x @ self.weight.T + self.bias` (or without the bias). `Sequential.__init__`: `setattr(self, str(i), layer)` for each layer; the given `_layers()` lists them back in order.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def __getitem__(self, idx):
    layers = self._layers()
    if isinstance(idx, slice):
        return Sequential(*layers[idx])
    return layers[idx]

def forward(self, x):
    for layer in self._layers():
        x = layer(x)
    return x
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`cross_entropy`: turn the targets into an int array, then `(logits.logsumexp(axis=1) - logits[np.arange(n), targets]).mean()`; your engine does the gradient. `mse_loss`: compare `pred.shape` with the target's shape (a Tensor's `.shape` or `np.shape(target)`) and raise `ValueError` if they differ. `binary_cross_entropy_with_logits`: softplus(z) is the logsumexp of 0 and z.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
y = np.asarray(targets, dtype=float)
softplus = stack([np.zeros(logits.shape), logits], axis=-1).logsumexp(axis=-1)
return (softplus - logits * y).mean()
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`fit`: `rng = as_rng(rng)` once (so every epoch draws a new permutation from the same stream), then for each epoch: `model.train()`, loop over `minibatch_indices(n, batch_size, shuffle=shuffle, rng=rng)` doing the five lines, and add `loss.item() * len(idx)` to a running total; append `total / n`; if validation data was given, append `evaluate(model, loss_fn, X_val, y_val)`; finally call the callback and `break` if it returns True. `evaluate`/`predict`: remember `model.training`, call `model.eval()`, loop over batches in order under `no_grad()`, and restore the mode in a `finally`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def evaluate(model, loss_fn, X, y, *, batch_size=256):
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with no_grad():
            for idx in minibatch_indices(len(X), batch_size, shuffle=False):
                total += loss_fn(model(X[idx]), y[idx]).item() * len(idx)
    finally:
        model.train(was_training)
    return total / len(X)
```

</details>
