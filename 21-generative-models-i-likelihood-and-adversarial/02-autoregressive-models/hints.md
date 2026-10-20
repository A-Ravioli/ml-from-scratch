# autoregressive models · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

subclass `nn.Linear`, call `super().__init__(in_features, out_features, bias)`, then `self.register_buffer("mask", torch.ones(out_features, in_features))`. `set_mask` is `self.mask.copy_(torch.as_tensor(np.asarray(mask), dtype=self.mask.dtype))` (copying in place keeps the buffer registered and on its device). `forward` is `F.linear(x, self.weight * self.mask, self.bias)`.

</details>

## stage 2

<details><summary>hint</summary>

keep a list of degree arrays, starting with `np.arange(1, D + 1)`. for each hidden size `h`: `lo = min(prev.min(), D - 1)` and `rng.integers(lo, D - 1, size=h, endpoint=True)`. the mask between consecutive layers is the boolean matrix `deg_next[:, None] >= deg_prev[None, :]` (rows: outputs of the layer, columns: inputs), and for the output layer `out_deg[:, None] > deg_last[None, :]` with `out_deg = np.tile(np.arange(1, D + 1), n_outputs_per_dim)`. put the `MaskedLinear`s with ReLUs in an `nn.Sequential`, and reshape the output with `.view(B, n_outputs_per_dim, D)`.

</details>

## stage 3

<details><summary>hint</summary>

the nll is `F.binary_cross_entropy_with_logits(model(x)[:, 0, :], x, reduction="none").sum(1)` (stable for large logits, unlike `log(sigmoid(...))`). to sample: `x = torch.zeros(n, model.in_dim)`; under `torch.no_grad()`, for `i` in `range(D)`, compute `p = torch.sigmoid(model(x)[:, 0, i])` and set `x[:, i] = torch.bernoulli(p, generator=generator)`.

</details>

## stage 4

<details><summary>hint</summary>

call `super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)`. with `c = kernel_size // 2`, build `mask = torch.ones(out, in, k, k)`, then `mask[:, :, c, c + (mask_type == "B"):] = 0` (the centre row from the centre, or just after it, to the right edge) and `mask[:, :, c + 1:, :] = 0` (every row below). `forward` is `F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)`.

</details>

## stage 5

<details><summary>hint</summary>

`self.first = MaskedConv2d("A", C, h, 7)`, then an `nn.ModuleList` of `MaskedConv2d("B", h, h, 3)` used as `h = h + block(relu(h))`, then `ReLU, Conv1x1(h, h), ReLU, Conv1x1(h, L * C)`; reshape the `(B, L*C, H, W)` output with `.view(B, L, C, H, W)`. the nll is `F.cross_entropy(logits, levels, reduction="none").sum(dim=(1, 2, 3))` with `levels = torch.round(x * (L - 1)).long()` (cross-entropy takes the class axis as dim 1, which is why the levels go there). to sample, loop over `i`, `j`: take `logits[:, :, :, i, j]` (shape `(n, L, C)`), softmax over dim 1, permute to `(n, C, L)`, reshape to `(n*C, L)`, `torch.multinomial(probs.cpu(), 1, generator=generator)`, and write `level / (L - 1)` into `x[:, :, i, j]`.

</details>
