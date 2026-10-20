# hamiltonian monte carlo · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`hamiltonian` is one line: `-log_prob(q) + 0.5 * np.sum(m * p * p)` with `m = 1.0` when `inv_mass` is None. for `leapfrog`, copy the inputs into new float arrays, do one half kick before the loop, then inside the loop a drift followed by a kick that is a *full* kick on every step except the last, where it's a half kick. that's `n_steps + 1` gradient evaluations.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if n_steps == 0:
    return q, p
p = p + 0.5 * step_size * grad_log_prob(q)
for i in range(n_steps):
    q = q + step_size * m * p
    g = grad_log_prob(q)
    p = p + (step_size if i < n_steps - 1 else 0.5 * step_size) * g
return q, p
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

per iteration: `p = rng.standard_normal(d) / np.sqrt(m)`, `h0 = hamiltonian(q, p, ...)`, run `leapfrog`, compute `h1` (use `math.inf` if the new position has any non-finite entry), then `u = rng.random()` and accept if `np.isfinite(h1)` and `log(u) < h0 - h1`. only the position is kept; the new momentum is thrown away. record `q` once you're past the burn-in.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for it in range(burn_in + n_samples):
    p = rng.standard_normal(d) / np.sqrt(m)
    h0 = hamiltonian(q, p, log_prob, m)
    q_new, p_new = leapfrog(q, p, grad_log_prob, step_size, n_leapfrog, m)
    h1 = hamiltonian(q_new, p_new, log_prob, m) if np.all(np.isfinite(q_new)) else math.inf
    u = rng.random()
    if np.isfinite(h1) and math.log(max(u, 1e-300)) < h0 - h1:
        q = q_new
        accepted += 1
    if it >= burn_in:
        out[it - burn_in] = q
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

at the top of `hmc`, if `grad_log_prob is None`, replace it by a function of `x` that returns `grad(log_prob, x)`; `grad` is imported from `mlfs.autodiff` in the setup cell. everything else stays the same, so with the same seed the samples match the exact-gradient run.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if grad_log_prob is None:
    grad_log_prob = lambda x: grad(log_prob, x)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`no_u_turn`: compute `dq = q_plus - q_minus` and return whether both dot products `dq @ p_minus` and `dq @ p_plus` are non-negative. `u_turn_steps`: remember `q0`, then loop `s = 1 .. max_steps`, each time calling your `leapfrog` with `n_steps=1`, and return `s` as soon as `(q - q0) @ p < 0`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
q_start = q.copy()
for s in range(1, max_steps + 1):
    q, p = leapfrog(q, p, grad_log_prob, step_size, 1)
    if (q - q_start) @ p < 0:
        return s
return max_steps
```

</details>
