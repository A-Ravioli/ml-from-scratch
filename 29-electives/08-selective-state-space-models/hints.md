# selective state-space models · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`h = torch.zeros_like(b[:, 0]) if h0 is None else h0`, then for each `t`: `h = a[:, t] * h + b[:, t]`, appending `h` to a list; `torch.stack(out, dim=1)` at the end. don't write into a preallocated tensor in place if you want gradients to flow cleanly; stacking a list is simplest.

</details>

## stage 2

<details><summary>hint</summary>

`combine((a1, b1), (a2, b2))` is `(a1 * a2, a2 * b1 + b2)`. for the scan: fold `h0` into the first `b` with `torch.cat` (not in place), then loop `step = 1, 2, 4, …` while `step < L`. each round, build the shifted copies `a_prev = cat([ones_like(a[:, :step]), a[:, :-step]], dim=1)` and `b_prev = cat([zeros_like(b[:, :step]), b[:, :-step]], dim=1)` (the identity map pads the front), and set `a, b = combine((a_prev, b_prev), (a, b))`. return `b`.

</details>

## stage 3

<details><summary>hint</summary>

`a = torch.exp(delta.unsqueeze(-1) * A)` broadcasts `(batch, L, D, 1) * (D, N)` to `(batch, L, D, N)`; `b = (delta * u).unsqueeze(-1) * B.unsqueeze(2)` broadcasts `(batch, L, D, 1) * (batch, L, 1, N)`. scan along dimension 1 with the chosen scan, then `y = torch.einsum("bldn,bln->bld", h, C)` and add `D * u` if given.

</details>

## stage 4

<details><summary>hint</summary>

`A_log = log(arange(1, N + 1)).repeat(D, 1)`; `D = ones(D)`; `x_proj = Linear(D, R + 2N, bias=False)`; `dt_proj = Linear(R, D)`, then under `torch.no_grad()` sample `dt = exp(log(dt_min) + rand(D) * (log(dt_max) - log(dt_min)))` and copy `dt + log(-expm1(-dt))` into its bias. `params(x)`: `dt_raw, B, C = x_proj(x).split([R, N, N], dim=-1)` and `delta = softplus(dt_proj(dt_raw))`. `step`: compute `params` on `x_t[:, None]` and take position 0, then one update `h = exp(delta[..., None] * A) * h + (delta * x_t)[..., None] * B[:, None]` and `y = einsum("bdn,bn->bd", h, C) + D * x_t`.

</details>

## stage 5

<details><summary>hint</summary>

`xi, z = in_proj(norm(x)).chunk(2, dim=-1)`. for the causal convolution, either call `conv1d` on `xi.transpose(1, 2)` and keep the first `L` outputs (`nn.Conv1d(..., padding=d_conv - 1)` pads both sides, so the extra outputs at the end are the future-peeking ones), or pad `d_conv - 1` zeros at the front of the time axis and sum `d_conv` shifted copies times `conv1d.weight[:, 0, j]`, plus `conv1d.bias`. the second is much faster to backpropagate through on a CPU. then `x + out_proj(ssm(silu(xi)) * silu(z))`.

</details>
