# regularization · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

start from `Tensor(0.0)` and add `(p * p).sum()` for every parameter, then multiply by `0.5 * weight_decay`. using your `Tensor` operations (not `p.data`) is what makes the penalty differentiable.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
total = Tensor(0.0)
for p in params:
    total = total + (p * p).sum()
return total * (0.5 * weight_decay)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`__init__`: `super().__init__()`, check $0 \le p \le 1$, store `self.p` and `self.rng = as_rng(rng)`. `forward`: wrap ndarrays as `Tensor`; return `x` itself when `not self.training` or `p == 0`; return `x * np.zeros(x.shape)` when `p == 1`; otherwise build a boolean keep-mask with `self.rng.random(x.shape) >= self.p` and multiply `x` by the constant array `keep / (1 - p)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def forward(self, x):
    x = x if isinstance(x, Tensor) else Tensor(x)
    if not self.training or self.p == 0.0:
        return x
    if self.p == 1.0:
        return x * np.zeros(x.shape)
    keep = self.rng.random(x.shape) >= self.p
    return x * (keep / (1.0 - self.p))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

expand the loss: $-\sum_j q_j\log p_j = (1 - s)(\operatorname{lse} - z_y) + s(\operatorname{lse} - \bar z)$ per row, with `lse = logits.logsumexp(axis=1)`, `z_y = logits[np.arange(n), targets]` and `zbar = logits.mean(axis=1)`. then take the mean over rows. targets may arrive as any integer array: convert with `np.asarray(...).astype(np.int64)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
n = logits.shape[0]
lse = logits.logsumexp(axis=1)
nll = lse - logits[np.arange(n), targets]
uniform = lse - logits.mean(axis=1)
return ((1.0 - smoothing) * nll + smoothing * uniform).mean()
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`random_crop`: `np.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)))`, then draw `tops` and `lefts` (in that order) with `rng.integers(0, Hp - size + 1, size=n)`. a loop over the $n$ images that slices `padded[i, :, t:t + size, l:l + size]` and stacks the results is perfectly fine; fancy indexing can do it without a loop. `random_horizontal_flip`: `flip = rng.random(n) < p`, copy the images, and replace the flipped ones with `[..., ::-1]`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
rng = as_rng(rng)
tops = rng.integers(0, hp - size + 1, size=n)
lefts = rng.integers(0, wp - size + 1, size=n)
return np.stack([padded[i, :, tops[i]:tops[i] + size, lefts[i]:lefts[i] + size] for i in range(n)])

# flip
out = images.copy()
out[flip] = out[flip][..., ::-1]
return out
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

keep a step counter. in `step(value)`: `value = float(value)`; decide `improved` (the first value always improves; then compare with `best - min_delta` or `best + min_delta` depending on the mode); on improvement record `best`, `best_step = self.steps` and reset the bad-step counter, otherwise increment it; increment `self.steps`; set `should_stop = True` once the counter reaches `patience`; return `should_stop`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if self.best is None:
    improved = True
elif self.mode == "min":
    improved = value < self.best - self.min_delta
else:
    improved = value > self.best + self.min_delta
if improved:
    self.best, self.best_step, self.num_bad_steps = value, self.steps, 0
else:
    self.num_bad_steps += 1
self.steps += 1
if self.num_bad_steps >= self.patience:
    self.should_stop = True
return self.should_stop
```

</details>
