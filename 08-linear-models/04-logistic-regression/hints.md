# logistic regression · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: the approach</summary>

compute `z = A @ theta` once. the data term is `-np.mean(y * log_sigmoid(z) + (1 - y) * log_sigmoid(-z))`, the penalty `0.5 * np.sum(penalty * theta**2)`. the gradient is `A.T @ (sigmoid(z) - y) / n + penalty * theta`. wrap the loss in `float`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def logistic_loss(theta, A, y, penalty):
    z = A @ theta
    nll = -np.mean(y * log_sigmoid(z) + (1 - y) * log_sigmoid(-z))
    return float(nll + 0.5 * np.sum(penalty * theta**2))

def logistic_gradient(theta, A, y, penalty):
    return A.T @ (sigmoid(A @ theta) - y) / len(y) + penalty * theta
```

</details>

## stage 2

<details><summary>hint 1: the approach</summary>

with `s = sigmoid(A @ theta)` and weights `v = s * (1 - s)`, $A^\top\operatorname{diag}(v)A$ is `(A.T * v) @ A`: broadcasting multiplies column $i$ of `A.T` by `v[i]`. divide by `n` and add `np.diag(penalty)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
s = sigmoid(A @ theta)
return (A.T * (s * (1 - s))) @ A / len(A) + np.diag(penalty)
```

</details>

## stage 3

<details><summary>hint 1: the approach</summary>

`fit`: validate `solver` and `alpha`, `check_X_y`, `classes_ = np.unique(y)` (two or `ValueError`), `t = (y == classes_[1]).astype(float)`, then `A = with_intercept_column(X, fit_intercept)` and `pen = penalty_weights(d, alpha, fit_intercept)`. call `gradient_descent` with lambdas of `theta`, `np.zeros(p)`, and `lr`, `max_iters`, `tol`. unpack `theta[0]` / `theta[1:]` when there's an intercept. `predict_proba` stacks `sigmoid(-z)` and `sigmoid(z)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
res = gradient_descent(lambda th: logistic_loss(th, A, t, pen), lambda th: logistic_gradient(th, A, t, pen),
                       np.zeros(A.shape[1]), lr=self.lr, max_iters=self.max_iters, tol=self.tol)
theta, self.n_iter_, self.converged_ = res.x, res.n_iters, res.converged
if self.fit_intercept:
    self.intercept_, self.coef_ = float(theta[0]), theta[1:]
else:
    self.intercept_, self.coef_ = 0.0, theta
```

and `predict_proba`: `z = self.decision_function(X); return np.column_stack([sigmoid(-z), sigmoid(z)])`.

</details>

## stage 4

<details><summary>hint 1: the approach</summary>

keep `theta`, the current loss `f` and gradient `g`. while `||g|| > tol` and you've taken fewer than `max_iters` steps: solve `H d = g` (`np.linalg.solve`, falling back to `np.linalg.lstsq` on `LinAlgError`), then try `theta - t * d` for `t = 1, 1/2, 1/4, ...` (at most ~30 tries) until the loss is no larger than `f + 1e-12 * (1 + abs(f))` (a strict test rejects good steps near the optimum, where rounding swamps the decrease). if no try works, stop. otherwise accept, update `f` and `g`, count the step.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
t = 1.0
for _ in range(31):
    candidate = theta - t * direction
    f_new = logistic_loss(candidate, A, y, penalty)
    if f_new <= f + 1e-12 * (1 + abs(f)):  # "no increase", up to rounding
        break
    t /= 2
else:
    break  # the for loop never hit `break`: no acceptable step
theta, f = candidate, f_new
g = logistic_gradient(theta, A, y, penalty)
steps += 1
```

this sits inside `while np.linalg.norm(g) > tol and steps < max_iters:`; return `theta, steps, bool(np.linalg.norm(g) <= tol)`. in `fit`, `solver="irls"` calls `irls(A, t, pen, max_iters=..., tol=...)`.

</details>
