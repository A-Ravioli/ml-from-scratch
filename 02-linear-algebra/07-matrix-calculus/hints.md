# matrix calculus · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`np.ndindex(X.shape)` yields every index tuple of an array of any shape, `(0,)`, `(1,)`, … for a vector or `(0, 0)`, `(0, 1)`, … for a matrix, so one loop handles every case.

</details>

<details><summary>hint 2: the approach</summary>

make a float working copy `Xw = np.array(X, dtype=float)` and `G = np.zeros_like(Xw)`. for each index: remember the old value, set `Xw[idx] = old + h` and evaluate `f`, set `Xw[idx] = old - h` and evaluate again, then **restore** `Xw[idx] = old` and store the central difference in `G[idx]`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for idx in np.ndindex(Xw.shape):
    old = Xw[idx]
    Xw[idx] = old + h
    f_plus = f(Xw)
    Xw[idx] = old - h
    f_minus = f(Xw)
    Xw[idx] = old
    G[idx] = (f_plus - f_minus) / (2 * h)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

two gradients, one formula. `np.linalg.norm` of an array of any shape is its Frobenius norm.

</details>

<details><summary>hint 2: the approach</summary>

`g = np.asarray(grad_f(X))`, `g_fd = numerical_gradient(f, X, h=h)`, then `norm(g - g_fd) / max(norm(g) + norm(g_fd), 1e-12)`. pass `grad_f` a copy of `X`, in case it modifies its input.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
X = np.array(X, dtype=float)
g = np.asarray(grad_f(X.copy()), dtype=float)
g_fd = numerical_gradient(f, X, h=h)
return float(np.linalg.norm(g - g_fd) / max(np.linalg.norm(g) + np.linalg.norm(g_fd), 1e-12))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

for each function, write $f(x + dx) - f(x)$, keep only the terms linear in $dx$, and rearrange them into the form $g^\top dx$. then $g$ is the gradient.

</details>

<details><summary>hint 2: the approach</summary>

for the quadratic, $dx^\top Ax$ is a scalar, so it equals its transpose $x^\top A^\top dx$. for least squares, write $r = Ax - b$, then $d(r^\top r) = 2r^\top dr$ and $dr = A\,dx$. for the norm, $\|x\| = (x^\top x)^{1/2}$, so use the 1-D chain rule on the square root.

</details>

<details><summary>hint 3: nearly the answer</summary>

the answers are $a$, $(A + A^\top)x$, $2A^\top(Ax - b)$ and $x/\|x\|$. each is one line of NumPy. check the shapes: every result is `(n,)`.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

write $df$ as $\operatorname{tr}(M\,dX)$. the gradient is $M^\top$. use the cyclic rule to move $dX$ to the end of the product.

</details>

<details><summary>hint 2: the approach</summary>

$\operatorname{tr}(A\,dX)$ gives $A^\top$. $a^\top dX\,b = \operatorname{tr}(b\,a^\top dX)$ gives $(ba^\top)^\top = ab^\top$. for $\|XW - Y\|_F^2$, see on paper 2. for $\log|\det X|$, $df = \operatorname{tr}(X^{-1}dX)$, so the gradient is $(X^{-1})^\top$.

</details>

<details><summary>hint 3: nearly the answer</summary>

`A.T`, `np.outer(a, b)`, `2 * X.T @ (X @ W - Y)`, and for the log-determinant, solve $X^\top G = I$: `solve(X.T, np.eye(n))`.

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

both gradients have the form "predicted probabilities minus targets". the targets are a one-hot vector for softmax and the labels `y` for logistic regression.

</details>

<details><summary>hint 2: the approach</summary>

softmax: `e = np.exp(z - z.max())`, `p = e / e.sum()`, then subtract 1 from `p[y]`. logistic: `p = sigmoid(X @ w)` computed stably (for example `np.exp(-np.logaddexp(0, -z))`), then `X.T @ (p - y) / n`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# softmax cross-entropy
e = np.exp(z - np.max(z))
p = e / e.sum()
p[y] -= 1.0
return p

# logistic
z = X @ w
p = np.exp(-np.logaddexp(0.0, -z))
return X.T @ (p - y) / X.shape[0]
```

</details>
