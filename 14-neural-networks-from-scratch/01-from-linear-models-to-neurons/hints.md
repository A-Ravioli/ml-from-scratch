# from linear models to neurons · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`linear_layer` is one line: with $W$ of shape (out, in), the batch product is `X @ W.T`, and `+ b` broadcasts the bias over the rows. `relu` is `np.maximum(z, 0)`. for `sigmoid`, $1/(1 + e^{-z}) = e^{-\log(1 + e^{-z})}$, and $\log(1 + e^{-z})$ is `np.logaddexp(0, -z)`, which never overflows.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`mlp_forward`: loop over `enumerate(params)`, apply `linear_layer`, and apply `activation` only if it isn't the last layer (`i < len(params) - 1`). `collapse_linear`: start from the first layer's `(W, b)` and fold in each later layer: the new layer multiplies the running `W` and `b` from the left, then adds its own bias.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
W, b = params[0]
for W2, b2 in params[1:]:
    W, b = W2 @ W, W2 @ b + b2
return W, b
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

both hidden units can look at the same thing, $x_1 + x_2$ (so both rows of $W_1$ are $(1, 1)$). pick their biases so that one unit is the count of ones and the other is positive only when the count is 2. then choose the output weights so that $(\text{count}) - c \cdot (\text{second unit})$ is 0 at count 2.

</details>

<details><summary>hint 2: nearly the answer</summary>

$b_1 = (0, -1)$ gives $h = (s, \operatorname{relu}(s - 1))$ with $s = x_1 + x_2$, i.e. $(0, 0)$, $(1, 0)$, $(2, 1)$ for $s = 0, 1, 2$. the output weights $W_2 = (1, -2)$, $b_2 = 0$ give $0, 1, 0$.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

do the forward pass keeping every intermediate: `z1 = linear_layer(X, W1, b1)`, `h = np.tanh(z1)` (or `z1` for identity), `z2 = linear_layer(h, W2, b2)[:, 0]` (a vector of n logits). the loss is `np.mean(np.logaddexp(0, z2) - y * z2)`. then go backwards line by line from idea 4, keeping shapes straight: make `dz2` a column `(n, 1)` so that `dz2.T @ h` is `(1, hidden)` like `W2`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
dz2 = (sigmoid(z2) - y)[:, None] / n      # (n, 1)
dW2, db2 = dz2.T @ h, dz2.sum(axis=0)     # (1, hidden), (1,)
dh = dz2 @ W2                             # (n, hidden)
dz1 = dh * (1 - h ** 2)                   # tanh; just dh for identity
dW1, db1 = dz1.T @ X, dz1.sum(axis=0)     # (hidden, d), (hidden,)
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`rng = as_rng(rng)`, then draw `rng.normal(size=(hidden, d))`, `rng.normal(size=hidden)`, `rng.normal(size=(1, hidden))`, `rng.normal(size=1)` in that order. each step: compute the loss and gradients, append the loss, then update every parameter with `p - lr * grad`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for _ in range(n_steps):
    loss, grads = two_layer_loss_and_grads(X, y, params, activation)
    losses.append(loss)
    params = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]
return params, losses
```

</details>
