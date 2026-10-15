# softmax regression · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: the approach</summary>

`Z = A @ Theta` is `(n, k)`. the data term is `-np.sum(Y * log_softmax(Z, axis=1)) / n` (only the true class's log-probability survives the multiplication by `Y`). the penalty is `0.5 * np.sum(penalty[:, None] * Theta**2)`: `penalty[:, None]` is a column, so each row of `Theta` gets its own strength. the gradient is `A.T @ (softmax(Z, axis=1) - Y) / n` plus `penalty[:, None] * Theta`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def softmax_loss(Theta, A, Y, penalty):
    nll = -np.sum(Y * log_softmax(A @ Theta, axis=1)) / len(A)
    return float(nll + 0.5 * np.sum(penalty[:, None] * Theta**2))

def softmax_gradient(Theta, A, Y, penalty):
    return A.T @ (softmax(A @ Theta, axis=1) - Y) / len(A) + penalty[:, None] * Theta
```

</details>

## stage 2

<details><summary>hint 1: the approach</summary>

`self.classes_, idx = np.unique(y, return_inverse=True)` gives the classes and each row's class index in one call; `Y = one_hot(idx, k)`. build `A` with a ones column when `fit_intercept`, and `pen = np.full(p, alpha)` with `pen[0] = 0` in that case. full batch: `gradient_descent(f, grad, np.zeros((p, k)), lr=..., max_iters=..., tol=...)` with `f` and `grad` closing over `A`, `Y`, `pen`. then split `Theta` into `intercept_ = Theta[0]` and `coef_ = Theta[1:]` (or zeros and `Theta` without an intercept).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
Theta0 = np.zeros((A.shape[1], k))
if self.batch_size is None:
    res = gradient_descent(lambda T: softmax_loss(T, A, Y, pen), lambda T: softmax_gradient(T, A, Y, pen),
                           Theta0, lr=self.lr, max_iters=self.max_iters, tol=self.tol)
    Theta, self.n_iter_, self.converged_ = res.x, res.n_iters, res.converged
else:
    Theta, self.n_iter_, self.converged_ = self._sgd(A, Y, pen, Theta0)
```

`decision_function` is `X @ self.coef_ + self.intercept_` after checking the number of columns; `predict_proba` is `softmax(self.decision_function(X), axis=1)`.

</details>

## stage 3

<details><summary>hint 1: the approach</summary>

`param = Param(Theta0)` and `opt = SGD([param], lr=self.lr)`. make `self.rng_ = as_rng(self.rng)` once. each epoch: `for rows in minibatch_indices(n, self.batch_size, shuffle=True, rng=self.rng_)`, set `param.grad = softmax_gradient(param.data, A[rows], Y[rows], pen)` and call `opt.step()`. after the epoch, compute the full gradient's norm and stop if it's `<= tol`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
epochs, converged = 0, False
while epochs < self.max_iters:
    for rows in minibatch_indices(len(A), self.batch_size, shuffle=True, rng=self.rng_):
        param.grad = softmax_gradient(param.data, A[rows], Y[rows], pen)
        opt.step()
    epochs += 1
    if np.linalg.norm(softmax_gradient(param.data, A, Y, pen)) <= self.tol:
        converged = True
        break
return param.data, epochs, converged
```

</details>

## stage 4

<details><summary>hint 1: the approach</summary>

`fit`: `self.classes_ = np.unique(y)`, then a list comprehension over the classes, each element `clone(self.estimator).fit(X, (y == c).astype(int))`. with labels 0/1, each binary model's `predict_proba(X)[:, 1]` is the probability of "this class". `predict_proba` stacks those columns and divides each row by its sum.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def predict_proba(self, X):
    P = np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])
    return P / P.sum(axis=1, keepdims=True)
```

</details>
