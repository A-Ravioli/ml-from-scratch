# variational inference · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

build all the samples at once, `z = mean + np.exp(log_std) * eps` (shape `(S, d)` by broadcasting), evaluate `log_prob` on each row and average, then add the entropy `np.sum(log_std) + d/2 * (1 + log 2 pi)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
z = mean + np.exp(log_std) * eps
entropy = float(np.sum(log_std)) + 0.5 * d * (1.0 + math.log(2 * math.pi))
return float(np.mean([log_prob(zs) for zs in z])) + entropy
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

vectorize each update over points and components. the logits are the `(n, K)` matrix `x[:, None] * m[None, :] - 0.5 * (s2 + m**2)[None, :]`; subtract `logsumexp(logits, axis=1)[:, None]` and exponentiate to get `phi`. the soft counts are `phi.sum(0)`; then `s2 = 1 / (1/prior_var + counts)` and `m = s2 * (phi * x[:, None]).sum(0)`. after the updates, append `gmm_elbo(x, m, s2, phi, prior_var)` and test the change against `tol`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
m = x[rng.choice(n, size=K, replace=False)].copy()
s2 = np.ones(K)
for _ in range(max_iters):
    logits = x[:, None] * m[None, :] - 0.5 * (s2 + m**2)[None, :]
    phi = np.exp(logits - logsumexp(logits, axis=1)[:, None])
    s2 = 1.0 / (1.0 / prior_var + phi.sum(axis=0))
    m = s2 * (phi * x[:, None]).sum(axis=0)
    elbo.append(gmm_elbo(x, m, s2, phi, prior_var))
    if len(elbo) > 1 and abs(elbo[-1] - elbo[-2]) < tol:
        break
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

recompute `z` from the same `eps`, stack the gradients `G = np.array([grad_log_prob(zs) for zs in z])` into `(S, d)`, and apply the chain rule: $\partial z/\partial\mu = 1$, $\partial z/\partial\log\sigma = \varepsilon\sigma$. don't forget the entropy's contribution to the second one.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
std = np.exp(log_std)
z = mean + std * eps
G = np.array([grad_log_prob(zs) for zs in z])
return G.mean(axis=0), (G * eps * std).mean(axis=0) + 1.0
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

two `Param`s (zeros of shape `(dim,)`), one `Adam([mean, scale], lr=lr)`. in the loop: draw `eps`, record the ELBO estimate (you can inline `elbo_diag`'s formula with the same `z`), get the gradients from `elbo_grad_diag`, set `mean.grad = -g_mean` and `scale.grad = -g_log_std`, call `opt.step()`. at the end, build `MultivariateNormal(mean.data, np.diag(np.exp(2 * scale.data)))`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if grad_log_prob is None:
    grad_log_prob = lambda z: grad(log_prob, z)
mean, scale = Param(np.zeros(dim)), Param(np.zeros(dim))
opt = Adam([mean, scale], lr=lr)
for _ in range(n_iters):
    eps = rng.standard_normal((n_samples, dim))
    z = mean.data + np.exp(scale.data) * eps
    history.append(float(np.mean([log_prob(zs) for zs in z])) + float(np.sum(scale.data)) + const)
    g_mean, g_scale = elbo_grad_diag(grad_log_prob, mean.data, scale.data, eps)
    mean.grad, scale.grad = -g_mean, -g_scale
    opt.step()
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`elbo_grad_full`: `z = mean + eps @ L.T`, `G` as before; the $L$ gradient is the lower triangle of `G.T @ eps / S` (an average of outer products $\nabla\log p(z_s)\,\varepsilon_s^\top$) plus `np.diag(1 / np.diag(L))`. in `reparam_vi`, the scale `Param` becomes `(dim, dim)`; build `L = np.tril(R, -1) + np.diag(np.exp(np.diag(R)))` each iteration, and convert the gradient back to `R`: same off the diagonal, times `L_ii` on it.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# elbo_grad_full
z = mean + eps @ L.T
G = np.array([grad_log_prob(zs) for zs in z])
g_L = np.tril(G.T @ eps / len(eps)) + np.diag(1.0 / np.diag(L))
return G.mean(axis=0), g_L

# in reparam_vi, family == "full"
L = np.tril(scale.data, -1) + np.diag(np.exp(np.diag(scale.data)))
g_mean, g_L = elbo_grad_full(grad_log_prob, mean.data, L, eps)
g_scale = np.tril(g_L, -1) + np.diag(np.diag(g_L) * np.diag(L))
```

</details>
