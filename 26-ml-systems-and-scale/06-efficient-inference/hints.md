# efficient inference · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

reduce over every dimension but `axis`: `dims = [d for d in range(w.dim()) if d != axis % w.dim()]`, `amax = w.abs().amax(dim=dims, keepdim=True)`. then `scale = torch.where(amax > 0, amax / 127, torch.ones_like(amax))` and `q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)`. `fake_quantize` is the same with `qmax = 2 ** (bits - 1) - 1`, a scalar `amax` for `axis=None`, and, for groups, `w.reshape(-1, group_size)` with `amax` over `dim=1`, reshaped back at the end.

</details>

## stage 2

<details><summary>hint</summary>

in `__init__`, `self.register_buffer("weight_q", ...)`, the same for `scale` and `bias` (`register_buffer("bias", None)` is allowed). `from_linear` is `@classmethod def from_linear(cls, linear)` returning `cls(*quantize_int8(linear.weight), linear.bias)`. for `quantize_linears`, deep-copy, then loop over `list(model.named_modules())`; for each module, loop over `named_children()`, build the qualified name (`f"{name}.{child_name}"`, or just `child_name` at the top), and `setattr(module, child_name, Int8Linear.from_linear(child))` for `nn.Linear` children not in `skip`.

</details>

## stage 3

<details><summary>hint</summary>

the threshold for a set of values is `torch.kthvalue(values.flatten(), k).values` with `k = round(sparsity * n)` (skip when `k == 0`). for `'global'`, compute it once on `torch.cat` of every layer's `|w|`; for `'layer'`, per layer. the mask is `w.abs() > threshold`; apply it with `w.mul_(mask)` under `torch.no_grad()`.

</details>

## stage 4

<details><summary>hint</summary>

`F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t.detach() / T, -1), log_target=True, reduction="batchmean")` is $\mathrm{KL}(p_t \| p_s)$ averaged over the batch (note the argument order: input is the student's log-probabilities, target the teacher's). multiply by `alpha * T * T` and add `(1 - alpha) * F.cross_entropy(s, targets)`.

</details>

## stage 5

<details><summary>hint</summary>

`batch_size / max(n_params * bytes_per_param / bandwidth, 2 * n_params * batch_size / peak_flops)`.

</details>
