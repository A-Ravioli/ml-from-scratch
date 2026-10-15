# universal approximation · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`set_linear`: `layer.weight.data[...] = np.asarray(W, dtype=float).reshape(layer.weight.shape)`, the same for the bias, and return the layer. `relu_interpolant`: sort the knots (and the values with them), compute the segment slopes `s = np.diff(y) / np.diff(x)`, and build the three weight vectors of idea 1: the first two units are $\operatorname{relu}(x - x_0)$ and $\operatorname{relu}(x_0 - x)$ with output weights $s_0$ and $-s_0$; unit $i \ge 2$ is $\operatorname{relu}(x - x_{i-1})$ with output weight $s_{i-1} - s_{i-2}$ (`np.diff(s)`); the output bias is $y_0$.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
W1 = np.concatenate([[1.0, -1.0], np.ones(k - 2)])
b1 = np.concatenate([[-x[0], x[0]], -x[1:-1]])
W2 = np.concatenate([[s[0], -s[0]], np.diff(s)])
net = Sequential(Linear(1, k), ReLU(), Linear(k, 1))
set_linear(net[0], W1[:, None], b1)
set_linear(net[2], W2[None, :], [y[0]])
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the first layer computes the first tent's two units from $x$: weights $(1, 1)^\top$, biases $(0, -\tfrac12)$. every later tent reads $t = 2h_1 - 4h_2$ from the previous ReLU layer and computes $\operatorname{relu}(t)$ and $\operatorname{relu}(t - \tfrac12)$: as one layer, both rows of $W$ are $(2, -4)$ and the bias is $(0, -\tfrac12)$. the last layer outputs $2h_1 - 4h_2$.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
layers = [set_linear(Linear(1, 2), [[1.0], [1.0]], [0.0, -0.5]), ReLU()]
for _ in range(depth - 1):
    layers += [set_linear(Linear(2, 2), [[2.0, -4.0], [2.0, -4.0]], [0.0, -0.5]), ReLU()]
layers.append(set_linear(Linear(2, 1), [[2.0, -4.0]], [0.0]))
return Sequential(*layers)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

sort `x`, make it a column `(n, 1)`, then loop over `range(len(model))`: apply `model[i]` (under `no_grad`), and whenever the layer is a `ReLU`, keep the boolean array `output > 0`. concatenate the kept arrays along axis 1 to get one pattern row per point, and count the rows that differ from the previous row.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
P = np.concatenate(patterns, axis=1)            # (n, total ReLU units)
changes = np.any(P[1:] != P[:-1], axis=1)       # (n - 1,)
return int(1 + changes.sum())
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`fit_function`: `X = np.linspace(lo, hi, n_points)[:, None]`, `y = f(X[:, 0])[:, None]` (a column, so it matches the model's `(n, 1)` output in `mse_loss`), then return `fit(model, mse_loss, Adam(model.parameters(), lr=lr), X, y, epochs=epochs, batch_size=batch_size, rng=rng)`. `max_error`: evaluate with `predict` on a fine grid and take the largest absolute difference.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
x = np.linspace(lo, hi, n_points)
return float(np.max(np.abs(predict(model, x[:, None])[:, 0] - f(x))))
```

</details>
