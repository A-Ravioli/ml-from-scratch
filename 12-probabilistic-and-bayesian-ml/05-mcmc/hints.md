# markov chain monte carlo · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

keep two pieces of state, the current `x` and its `lp = log_prob(x)`. one loop over all `burn_in + n_samples * thin` steps: propose, evaluate the proposal once, accept or not (updating *both* `x` and `lp` on acceptance), then decide whether this step is one you record: past the burn-in, and `(it - burn_in + 1) % thin == 0`. the state is recorded whether or not the proposal was accepted.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for it in range(burn_in + n_samples * thin):
    prop = x + proposal_scale * rng.standard_normal(d)
    lp_prop = log_prob(prop)
    if math.log(max(rng.random(), 1e-300)) < lp_prop - lp:
        x, lp = prop, lp_prop
        accepted += 1
    if it >= burn_in and (it - burn_in + 1) % thin == 0:
        out[k] = x
        k += 1
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the same skeleton as stage 1 without the accept step. copy `x0` into a float working array first. one sweep is `for i in range(d): x[i] = conditionals[i](x, rng)`: because you write into `x` immediately, coordinate `i + 1` sees the new value of coordinate `i`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
for it in range(burn_in + n_samples * thin):
    for i in range(d):
        x[i] = conditionals[i](x, rng)
    if it >= burn_in and (it - burn_in + 1) % thin == 0:
        out[k] = x
        k += 1
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

centre the chain, take `f = np.fft.rfft(y, size)` with `size` a power of two at least `2n - 1` (`1 << (2 * n - 1).bit_length()`), and the autocovariance at every lag is `np.fft.irfft(f * np.conj(f), size)`. keep the first `max_lag + 1` entries and divide by entry 0 (unless it's 0).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
y = x - x.mean()
size = 1 << (2 * n - 1).bit_length()
f = np.fft.rfft(y, size)
acov = np.fft.irfft(f * np.conj(f), size)[: max_lag + 1]
if acov[0] <= 0:
    return np.ones(max_lag + 1)
return acov / acov[0]
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

get `rho = autocorrelation(chain)`, start `tau = -1.0`, and loop `m = 0, 1, ...`: the pair sum is `rho[2m] + rho[2m + 1]`; stop (without adding it) as soon as it's `<= 0` or `2m + 1` runs past the end; otherwise add twice the pair sum. floor `tau` at `1 / log10(n)`. for a 2-D input, call yourself on each column.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
tau = -1.0
for m in range(n // 2):
    if 2 * m + 1 >= n:
        break
    pair = rho[2 * m] + rho[2 * m + 1]
    if pair <= 0:
        break
    tau += 2.0 * pair
tau = max(tau, 1.0 / math.log10(n))
return float(n / tau)
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`half = n // 2`; the split chains are `np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)`, shape `(2m, half)`. then `W` is the mean of the row variances (`ddof=1`), `B` is `half` times the variance of the row means (`ddof=1`), and the rest is the formula with `half` in place of `n`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
half = chains.shape[1] // 2
split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
W = np.mean(np.var(split, axis=1, ddof=1))
B = half * np.var(np.mean(split, axis=1), ddof=1)
var_plus = (half - 1) / half * W + B / half
return float(np.sqrt(var_plus / W))
```

</details>
