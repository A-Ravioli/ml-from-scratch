# online learning · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

store the cumulative losses, not the weights. in `predict`, form `z = -lr * cum_loss` and return `np.exp(z - logsumexp(z))` (`logsumexp` is imported from `mlfs.utils`): subtracting the log-normalizer keeps every exponent at most 0, so nothing overflows or underflows to all-zeros. `update` checks the shape and adds.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def predict(self):
    z = -self.lr * self.cum_loss
    return np.exp(z - logsumexp(z))

def update(self, losses):
    losses = np.asarray(losses, dtype=float)
    if losses.shape != (self.n_experts,):
        raise ValueError("wrong shape")
    self.cum_loss = self.cum_loss + losses
    return self
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

keep `x` (a float array) and the round counter `t`. `update` increments `t` *first*, so the first step uses $\eta_1 = $ `lr`. compute the unprojected point `x - eta * grad`, then pass it through `project` if one was given. `predict` returns `self.x.copy()`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
self.t += 1
eta = self.lr / math.sqrt(self.t) if self.decay == "sqrt" else self.lr
x = self.x - eta * np.asarray(grad, dtype=float)
self.x = np.asarray(self.project(x), dtype=float) if self.project is not None else x
return self
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

for FTL you need the cumulative losses *before* each round: a row of zeros followed by `np.cumsum(L, axis=0)[:-1]`. `np.argmin(..., axis=1)` gives each round's pick (ties to the lowest index automatically), and fancy indexing `L[np.arange(T), picks]` gives the losses. the regret curve is one `cumsum` minus the row-wise `min` of another.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
cum_before = np.vstack([np.zeros(L.shape[1]), np.cumsum(L, axis=0)[:-1]])
picks = np.argmin(cum_before, axis=1)
learner = L[np.arange(len(L)), picks]
regret = np.cumsum(learner) - np.min(np.cumsum(L, axis=0), axis=1)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

create one `Hedge(N, lr)` and loop over the rows: record `h.predict() @ losses`, *then* `h.update(losses)`. the two formulas are one-liners with `math.log` and `math.sqrt`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
h = Hedge(L.shape[1], lr)
out = np.empty(len(L))
for t, losses in enumerate(L):
    out[t] = h.predict() @ losses
    h.update(losses)
return out
```

with `math.sqrt(8 * math.log(n_experts) / T)` and `math.log(n_experts) / lr + lr * T / 8`.

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

build `OnlineGradientDescent(d, lr, project=lambda w: project_l2_ball(w, radius), decay=decay)`. each round: `w = ogd.predict()`, residual `r = w @ x_t - y_t`, record `0.5 * r**2`, then `ogd.update(r * x_t)` (the gradient of $\frac12(w^\top x - y)^2$ with respect to $w$).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
ogd = OnlineGradientDescent(X.shape[1], lr, project=lambda w: project_l2_ball(w, radius), decay=decay)
losses = np.empty(len(y))
for t in range(len(y)):
    r = ogd.predict() @ X[t] - y[t]
    losses[t] = 0.5 * r * r
    ogd.update(r * X[t])
return losses, ogd.predict()
```

</details>
