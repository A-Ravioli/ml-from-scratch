# privacy & secure aggregation · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

convert tensors with `.detach().cpu().numpy()`, then `np.mod(np.round(x * scale).astype(np.int64), modulus)`; `np.mod` always returns a non-negative result for a positive modulus, which is exactly the wrap-around you want. to decode, `np.where(z >= modulus // 2, z - modulus, z) / scale`.

</details>

## stage 2

<details><summary>hint</summary>

Python's three-argument `pow(base, exp, mod)` does modular exponentiation on arbitrarily large ints; convert NumPy integers with `int()` first. in `mask`, loop over `publics.items()`, skip `j == i`, compute `m = prg(shared_secret(my_secret, pub), len(z))`, and do `out = np.mod(out + m, modulus)` if `j > i` else `np.mod(out - m, modulus)`.

</details>

## stage 3

<details><summary>hint</summary>

for sharing, evaluate the polynomial with Python ints (Horner: `y = (y * x + c) % prime`), never NumPy, which overflows past $2^{63}$. for reconstruction, for each share accumulate `num *= x_j` and `den *= (x_j - x_i)` over the other shares (mod prime), and add `y_i * num * pow(den, -1, prime)`. in `secure_aggregate`: draw every client's secret, publish keys, share every secret among all clients, sum the survivors' `mask(...)` outputs mod $M$, then for each dropped `d` reconstruct its secret from `threshold` survivors' shares and, for every survivor `j`, subtract `prg(shared_secret(s_d, publics[j]))` if `d > j`, add it if `d < j`. decode at the end. if `uploads` is a dict, set `uploads[i]` to each survivor's masked vector as you add it; if `revealed` is a dict, set `revealed[d] = {j: share}` for the survivors `j` whose shares you used.

</details>

## stage 4

<details><summary>hint</summary>

`delta * torch.clamp(clip / (delta.norm() + 1e-12), max=1.0)` clips without a branch and keeps a zero vector at zero. for the aggregate, stack the clipped deltas, sum over the first axis, add `torch.randn(total.shape, generator=generator, dtype=total.dtype) * noise_multiplier * clip`, and divide by `expected_clients`.

</details>

## stage 5

<details><summary>hint</summary>

for each order `a`, build the list of `a + 1` log-terms `lgamma(a+1) - lgamma(k+1) - lgamma(a-k+1) + (a-k)*log1p(-q) + k*log(q) + (k*k - k) / (2*sigma**2)`, take `m = max(terms)`, and `log_A = m + log(sum(exp(t - m) for t in terms))`; the RDP is `log_A / (a - 1)`. handle `q == 0`, `sigma == 0` and `q == 1` before the general case (`log(0)` and division by zero). conversion: `eps = rdp + log(1/delta) / (orders - 1)`, then `argmin`.

</details>

## stage 6

<details><summary>hint</summary>

compute the per-round RDP once (`rdp_sampled_gaussian(q, sigma, orders)`), and each round's ε as `rdp_to_dp(r * rdp_step, orders, delta)[0]`. snapshot the flat weights `w` at the start of the round and restore them with `torch.nn.utils.vector_to_parameters(w, model.parameters())` before every client. after the client loop, `vector_to_parameters(w + server_lr * dp_aggregate(...), ...)`. keep the order of random draws in the spec (generator seed first, then the sampling coin flips, then one seed per chosen client).

</details>
