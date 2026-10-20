# exponential families & fisher information · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`__init__` only stores things. `log_prob` is the canonical form, evaluated for every data point at once: `T(x)` has shape `(n, k)` and `eta` has shape `(k,)`, so a matrix–vector product gives the `(n,)` linear term.

</details>

<details><summary>hint 2: the approach</summary>

in `__init__`: `self.T, self.A, self.log_h = sufficient_stats, log_partition, log_base_measure`, `self.eta0 = np.atleast_1d(np.asarray(eta0, dtype=float))` and `self.dim = len(self.eta0)`. in `log_prob`: convert `x` and `eta` to float arrays, then `self.log_h(x) + self.T(x) @ eta - self.A(eta)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def log_prob(self, x, eta):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    return np.asarray(self.log_h(x), dtype=float) + np.asarray(self.T(x), dtype=float) @ eta - float(self.A(eta))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

idea 2 derives every ingredient. `T` must return shape `(n, k)`: for a one-parameter family, `x[:, None]`. write `A` in terms of `e[0]` (and `e[1]`) using only `+ - * / **`, `np.exp`, `np.log` and `np.log1p`, so that your `Dual` numbers can flow through it.

</details>

<details><summary>hint 2: the approach</summary>

Bernoulli: `A = np.log1p(np.exp(e[0]))`, `log_h = 0`. Poisson: `A = np.exp(e[0])`, `log_h = -lgamma(x + 1)` (with `math.lgamma` per element). exponential: `A = -np.log(-e[0])`, start at `[-1.0]`. Gaussian: `T = np.stack([x, x**2], axis=1)`, `A = -e[0]**2 / (4 * e[1]) - 0.5 * np.log(-2 * e[1])`, `log_h = -0.5 * log(2 pi)` for every point, start at `[0.0, -0.5]`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def gaussian_family():
    return ExponentialFamily(
        lambda x: np.stack([np.asarray(x, dtype=float), np.asarray(x, dtype=float) ** 2], axis=1),
        lambda e: -e[0] ** 2 / (4 * e[1]) - 0.5 * np.log(-2 * e[1]),
        lambda x: np.full(len(x), -0.5 * math.log(2 * math.pi)),
        [0.0, -0.5],
    )
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`grad(self.A, eta)` from `mlfs.autodiff` gives $\nabla A$ exactly, using dual numbers. the Fisher information is the Jacobian of that gradient: pass your `self.mean_params` itself to `numerical_jacobian`.

</details>

<details><summary>hint 2: the approach</summary>

`mean_params`: `grad(self.A, np.atleast_1d(np.asarray(eta, dtype=float)))`. `fisher_information`: `J = numerical_jacobian(self.mean_params, eta)`, then return `(J + J.T) / 2`, because finite differences leave a tiny asymmetry.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def fisher_information(self, eta):
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    J = np.asarray(numerical_jacobian(self.mean_params, eta), dtype=float)
    return 0.5 * (J + J.T)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

you need $\bar T$ (the column means of `self.T(x)`), a helper for the average log-likelihood $\eta^\top\bar T - A(\eta)$, and a loop of at most `max_iters` Newton steps. inside each step, an inner loop of at most 50 halvings.

</details>

<details><summary>hint 2: the approach</summary>

each iteration: `g = tbar - self.mean_params(eta)`; stop if `np.linalg.norm(g) <= tol`. `step = solve(self.fisher_information(eta), g)` (your `mlfs.linalg.solve`). then halve `step` until `avg_loglik(eta + step)` is finite and `>= ` the current value; if 50 halvings don't get there, stop. wrap the evaluation of `A` in `np.errstate(invalid="ignore", divide="ignore")`, since stepping outside the parameter space makes `np.log` complain.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
current = avg_loglik(eta)
for _ in range(max_iters):
    g = tbar - self.mean_params(eta)
    if np.linalg.norm(g) <= tol:
        break
    step = np.asarray(solve(self.fisher_information(eta), g), dtype=float)
    for _ in range(50):
        trial = avg_loglik(eta + step)
        if np.isfinite(trial) and trial >= current:
            break
        step = step / 2
    else:
        break
    eta, current = eta + step, trial
return eta
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

the score of an exponential family is $T(x) - \nabla A(\eta)$, and the Cramér–Rao bound is the inverse Fisher information divided by $n$.

</details>

<details><summary>hint 2: the approach</summary>

`score`: `self.T(x) - self.mean_params(eta)` (broadcasting a `(k,)` vector against `(n, k)`). `cramer_rao_bound`: solve `fisher_information(eta) @ M = I` for `M` (e.g. `solve(info, np.eye(self.dim))`), then divide by `n`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def cramer_rao_bound(self, eta, n):
    info = self.fisher_information(eta)
    return np.asarray(solve(info, np.eye(self.dim)), dtype=float) / n
```

</details>
