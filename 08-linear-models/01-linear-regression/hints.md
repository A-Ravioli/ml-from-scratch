# linear regression · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

compute the residual vector `r = X @ w - y` once. the loss is its squared length divided by `2n`; the gradient multiplies it by `X.T` and divides by `n`.

</details>

<details><summary>hint 2: the approach</summary>

`r @ r` is the squared length of a 1-D array. wrap the loss in `float(...)` so it's a Python float, not a 0-d array. the gradient `X.T @ r / len(y)` has shape `(d,)` because `X.T` is `(d, n)` and `r` is `(n,)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def mse_loss(w, X, y):
    r = X @ w - y
    return float(r @ r) / (2 * len(y))

def mse_gradient(w, X, y):
    return X.T @ (X @ w - y) / len(y)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

both solvers are one or two lines once you know which of your chapter 02 functions to call. `cho_solve(L, b)` solves $(LL^\top)x = b$ given the lower-triangular `L` that `cholesky` returns.

</details>

<details><summary>hint 2: the approach</summary>

normal equations: `L = cholesky(X.T @ X)`, then `cho_solve(L, X.T @ y)`. QR: `Q, R = qr(X)` gives the reduced factors; $R$ is upper triangular, so solve $Rw = Q^\top y$ with `solve_triangular(..., lower=False)`. don't wrap anything in `try`: a `LinAlgError` from `cholesky` should reach the caller.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def solve_normal_equations(X, y):
    return cho_solve(cholesky(X.T @ X), X.T @ y)

def solve_qr(X, y):
    Q, R = qr(X)
    return solve_triangular(R, Q.T @ y, lower=False)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`fit` has three parts: validate (solver name, then `check_X_y`), center (or not), solve and store. `predict` validates `X`, checks its number of columns against `len(self.coef_)`, and returns `X @ coef_ + intercept_`.

</details>

<details><summary>hint 2: the approach</summary>

with `fit_intercept`, `x_mean = X.mean(axis=0)` and `y_mean = y.mean()`, solve on `X - x_mean` and `y - y_mean`. without it, use `x_mean = np.zeros(d)` and `y_mean = 0.0` so the same last line gives `intercept_ = 0.0`: `self.intercept_ = float(y_mean - x_mean @ self.coef_)`. set `self.n_iter_ = 0` for the exact solvers.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if self.solver not in ("normal", "qr", "gd"):
    raise ValueError(f"unknown solver {self.solver!r}")
X, y = check_X_y(X, y)
y = y.astype(np.float64)
if self.fit_intercept:
    x_mean, y_mean = X.mean(axis=0), float(y.mean())
else:
    x_mean, y_mean = np.zeros(X.shape[1]), 0.0
Xc, yc = X - x_mean, y - y_mean
self.n_iter_ = 0
if self.solver == "normal":
    coef = solve_normal_equations(Xc, yc)
elif self.solver == "qr":
    coef = solve_qr(Xc, yc)
else:
    res = solve_gd(Xc, yc, lr=self.lr, max_iters=self.max_iters, tol=self.tol)
    coef, self.n_iter_ = res.x, res.n_iters
self.coef_ = np.asarray(coef, dtype=np.float64)
self.intercept_ = float(y_mean - x_mean @ self.coef_)
return self
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`gradient_descent` wants functions of one argument, `f(w)` and `grad(w)`. your stage 1 functions take `(w, X, y)`: close over `X` and `y` with a `lambda`.

</details>

<details><summary>hint 2: the approach</summary>

start from `np.zeros(X.shape[1])` when `w0` is `None`, and pass `lr`, `max_iters` and `tol` straight through as keyword arguments. return the `OptimResult` itself: `LinearRegression` reads `.x` and `.n_iters` from it.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
w0 = np.zeros(X.shape[1]) if w0 is None else np.asarray(w0, dtype=float)
return gradient_descent(lambda w: mse_loss(w, X, y), lambda w: mse_gradient(w, X, y), w0,
                        lr=lr, max_iters=max_iters, tol=tol)
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

both functions start from the residuals `y - y_pred`. convert both inputs with `np.asarray(..., dtype=float)` first so lists work too.

</details>

<details><summary>hint 2: the approach</summary>

the mean NLL is the constant `0.5 * np.log(2 * np.pi * sigma2)` plus `np.mean(r**2) / (2 * sigma2)`. `noise_variance` is `r @ r / (n - n_params)`, after raising `ValueError` when `n_params >= n`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def gaussian_nll(y, y_pred, sigma2):
    r = np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(0.5 * np.log(2 * np.pi * sigma2) + np.mean(r**2) / (2 * sigma2))

def noise_variance(y, y_pred, n_params=0):
    r = np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)
    if n_params >= len(r):
        raise ValueError("no degrees of freedom left")
    return float(r @ r) / (len(r) - n_params)
```

</details>

## stage 6

<details><summary>hint 1: a nudge</summary>

both are one broadcast. turn `x` into a column `x[:, None]` (shape `(n, 1)`) and combine it with a row (shape `(m,)`): the result is `(n, m)`.

</details>

<details><summary>hint 2: the approach</summary>

for the polynomial, the row is the exponents `np.arange(1, degree + 1)`; `x[:, None] ** exponents` raises each `x` to each power. for the bumps, the row is `centers`: `x[:, None] - centers[None, :]` is the `(n, m)` table of differences.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def polynomial_basis(x, degree):
    x = np.asarray(x, dtype=float).ravel()
    return x[:, None] ** np.arange(1, degree + 1)

def rbf_basis(x, centers, width):
    x = np.asarray(x, dtype=float).ravel()
    c = np.asarray(centers, dtype=float).ravel()
    return np.exp(-((x[:, None] - c[None, :]) ** 2) / (2 * width**2))
```

</details>
