# generalized linear models · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: the approach</summary>

most methods are one line: the table in idea 1 gives every link, inverse link and variance. for the logit use `np.log(mu) - np.log1p(-mu)`. the deviances need $0\log 0 = 0$: compute `y * np.log(y / mu)` only where `y > 0` and use 0 elsewhere. `np.where` evaluates both branches, so feed it a safe value (`np.where(y > 0, y, 1.0)`) inside the log, and silence warnings with `np.errstate`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# Poisson.deviance
with np.errstate(divide="ignore", invalid="ignore"):
    ylog = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
return 2.0 * (ylog - (y - mu))
```

the Bernoulli deviance is the same trick twice: one term for `y > 0` with `y / mu`, one for `y < 1` with `(1 - y) / (1 - mu)`.

</details>

## stage 2

<details><summary>hint 1: the approach</summary>

all three functions start with `eta = A @ theta` and `mu = family.inverse_link(eta)`. the loss is `0.5 * np.mean(family.deviance(y, mu))` plus the penalty; the gradient is `A.T @ (mu - y) / n + penalty * theta`. for the step, `w = family.variance(mu)`, the left side is `(A.T * w) @ A + n * np.diag(penalty)` and the right side `A.T @ (w * eta + (y - mu))`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def irls_step(theta, A, y, family, penalty):
    eta = A @ theta
    mu = family.inverse_link(eta)
    w = family.variance(mu)
    lhs = (A.T * w) @ A + len(y) * np.diag(penalty)
    rhs = A.T @ (w * eta + (y - mu))
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]
```

</details>

## stage 3

<details><summary>hint 1: the approach</summary>

set up as in topic 04 (design matrix, penalty vector with a 0 for the intercept), plus the family and the checks on `y`. initialize `theta[0] = family.link(clipped mean of y)`. the loop: `target = irls_step(...)`, then try `candidate = theta + t * (target - theta)` for `t = 1, 1/2, ...` (at most ~30 times) until the loss is finite and at most `f + 1e-12 * (1 + abs(f))` (not larger, up to rounding); accept, recompute the gradient, count the step. `predict` returns `family_.inverse_link(X @ coef_ + intercept_)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
while np.linalg.norm(g) > self.tol and steps < self.max_iters:
    target = irls_step(theta, A, y, self.family_, pen)
    t = 1.0
    for _ in range(31):
        candidate = theta + t * (target - theta)
        with np.errstate(over="ignore"):
            f_new = glm_loss(candidate, A, y, self.family_, pen)
        if np.isfinite(f_new) and f_new <= f + 1e-12 * (1 + abs(f)):
            break
        t /= 2
    else:
        break
    theta, f = candidate, f_new
    g = glm_gradient(theta, A, y, self.family_, pen)
    steps += 1
```

</details>

## stage 4

<details><summary>hint 1: the approach</summary>

$D^2$ needs two sums of unit deviances: one at the fitted means, one at the constant `np.full_like(y, y.mean())`. the dispersion is `np.sum((y - mu)**2 / family.variance(mu)) / (n - n_params)`, after checking `n_params < n`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def d2_score(y, mu, family):
    y = np.asarray(y, float)
    null = np.sum(family.deviance(y, np.full_like(y, y.mean())))
    return float(1 - np.sum(family.deviance(y, mu)) / null)
```

</details>
