# regularization · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

it's topic 01's `LinearRegression.fit` with `"normal"`, plus $\alpha I$ added to $X^\top X$. the given `center` does the centering and returns the means you need for the intercept.

</details>

<details><summary>hint 2: the approach</summary>

`Xc, yc, x_mean, y_mean = center(X, y, self.fit_intercept)`, then `A = Xc.T @ Xc + self.alpha * np.eye(d)` and `coef = cho_solve(cholesky(A), Xc.T @ yc)`. the intercept is `float(y_mean - x_mean @ coef)`. check `alpha >= 0` before anything else.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if self.alpha < 0:
    raise ValueError("alpha must be >= 0")
Xc, yc, x_mean, y_mean = center(X, y, self.fit_intercept)
A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
self.coef_ = np.asarray(cho_solve(cholesky(A), Xc.T @ yc), dtype=float)
self.intercept_ = float(y_mean - x_mean @ self.coef_)
return self
```

`predict` is `return linear_predict(self, X)`.

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

one SVD, then broadcasting: put the alphas in a column `(k, 1)` and the singular values in a row `(r,)`, and `s / (s**2 + alphas)` is a `(k, r)` table of shrinkage factors, one row per alpha.

</details>

<details><summary>hint 2: the approach</summary>

`U, s, Vt = np.linalg.svd(X, full_matrices=False)`. each row of the answer is $\sum_j f_j (u_j^\top y) v_j^\top$, i.e. `(f * (U.T @ y)) @ Vt` with `f` the row of factors. do all rows at once with the `(k, r)` table. for the cutoff, `keep = s > 1e-12 * s.max()` and use `np.where(keep, ..., 0.0)`; make sure the discarded entries can't divide by zero first (add 1 to their denominator, say). `ridge_dof` sums `s**2 / (s**2 + alphas)` over the kept singular values along axis 1.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
U, s, Vt = np.linalg.svd(np.asarray(X, float), full_matrices=False)
keep = s > 1e-12 * s.max()
a = np.asarray(alphas, float).reshape(-1, 1)
f = np.where(keep, s / (s**2 + a + ~keep), 0.0)
return (f * (U.T @ y)) @ Vt
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`soft_threshold` is one line with `np.sign`, `np.abs` and `np.maximum`. for coordinate descent, write one sweep first (a `for j in range(d)` loop), make sure a single sweep decreases the objective, then wrap it in the outer loop with the stopping rule.

</details>

<details><summary>hint 2: the approach</summary>

before the loop: `w` (a copy of `w0`, or zeros), `r = y - X @ w` and `z = (X**2).sum(axis=0) / n`. for coordinate `j`: `rho = X[:, j] @ r / n + z[j] * w[j]`, `new = soft_threshold(rho, l1) / (z[j] + l2)` (or 0 if `z[j] == 0`), then `r -= X[:, j] * (new - w[j])` and `w[j] = new`. track the largest `abs(new - old)` in the sweep; stop when it's `<= tol` or after `max_iters` sweeps.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
sweeps = 0
while sweeps < max_iters:
    sweeps += 1
    biggest = 0.0
    for j in range(d):
        old = w[j]
        new = 0.0 if z[j] == 0 else float(soft_threshold(X[:, j] @ r / n + z[j] * old, l1)) / (z[j] + l2)
        if new != old:
            r -= X[:, j] * (new - old)
            w[j] = new
            biggest = max(biggest, abs(new - old))
    if biggest <= tol:
        break
return w, sweeps
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

both estimators are `Ridge.fit` with the solve replaced by a call to `coordinate_descent` on the centered data. `coordinate_descent` returns a pair: the coefficients and the number of sweeps.

</details>

<details><summary>hint 2: the approach</summary>

`Lasso`: `l1 = alpha`, `l2 = 0`. `ElasticNet`: `l1 = alpha * l1_ratio`, `l2 = alpha * (1 - l1_ratio)`. pass `max_iters` and `tol` through, store `coef_, n_iter_`, then the intercept as in stage 1. validate `alpha` (and `l1_ratio`) first.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
Xc, yc, x_mean, y_mean = center(X, y, self.fit_intercept)
l1, l2 = self.alpha * self.l1_ratio, self.alpha * (1 - self.l1_ratio)
self.coef_, self.n_iter_ = coordinate_descent(Xc, yc, l1, l2, max_iters=self.max_iters, tol=self.tol)
self.intercept_ = float(y_mean - x_mean @ self.coef_)
return self
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`lasso_alpha_max` is one line. for the path, visit the alphas in decreasing order but remember where each came from: `np.argsort(-alphas)` gives the positions in that order.

</details>

<details><summary>hint 2: the approach</summary>

allocate `coefs = np.zeros((len(alphas), d))` and `w = np.zeros(d)`. loop over `i in np.argsort(-alphas)`: `w, _ = coordinate_descent(X, y, alphas[i], w0=w, max_iters=..., tol=...)` and `coefs[i] = w`. because each call starts from the previous `w`, that's the warm start.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def lasso_alpha_max(X, y):
    return float(np.max(np.abs(X.T @ y)) / len(X))

def lasso_path(X, y, alphas, *, max_iters=1000, tol=1e-06):
    alphas = np.asarray(alphas, float)
    coefs = np.zeros((len(alphas), X.shape[1]))
    w = np.zeros(X.shape[1])
    for i in np.argsort(-alphas, kind="stable"):
        w, _ = coordinate_descent(X, y, float(alphas[i]), w0=w, max_iters=max_iters, tol=tol)
        coefs[i] = w
    return coefs
```

</details>
