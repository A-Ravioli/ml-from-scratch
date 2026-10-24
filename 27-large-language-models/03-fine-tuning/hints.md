# fine-tuning · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

freeze with `for p in base.parameters(): p.requires_grad_(False)`. create `self.lora_A = nn.Parameter(torch.empty(r, in_f))` followed by `nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))`, and `self.lora_B = nn.Parameter(torch.zeros(out_f, r))` (pass `device=` and `dtype=` from `base.weight`). `forward`: `self.base(x) + self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)`. in `merge`, under `torch.no_grad()`, build an `nn.Linear(in_f, out_f, bias=base.bias is not None)`, `copy_` in `base.weight + scaling * lora_B @ lora_A` and the bias, then copy each parameter's `requires_grad` from the base.

</details>

## stage 2

<details><summary>hint</summary>

first pass: collect the names to wrap (`isinstance(m, nn.Linear)`, the target rule, and not `name.startswith(w + ".")` for any `w` that is already a `LoRALinear`). then freeze all parameters. second pass: for each name, `parent_name, _, child = name.rpartition(".")`, `parent = model.get_submodule(parent_name) if parent_name else model`, and `setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha=alpha, dropout=dropout))`. finally set `requires_grad_(True)` on every `lora_A` and `lora_B` (so adapters from an earlier call stay trainable).

</details>

## stage 3

<details><summary>hint</summary>

loop over `list(model.named_modules())` (a list, because you're about to change the tree), and for each `LoRALinear` do the same `rpartition` / `get_submodule` / `setattr` dance with `module.merge()`.

</details>

## stage 4

<details><summary>hint</summary>

`seq = head + tail`; `input_ids = seq[:-1]`; `targets = [-100] * (len(head) - 1) + tail`; slice both with `[:max_len]` when it's given. `collate_sft`: `torch.full((B, T), pad_id)` and `torch.full((B, T), -100)`, then fill each row's prefix. `sft_loss` is `F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)`.

</details>

## stage 5

<details><summary>hint</summary>

with `from scipy.stats import norm`: `pos = norm.ppf(np.linspace(0.9677083, 0.5, 9)[:-1])`, `neg = -norm.ppf(np.linspace(0.9677083, 0.5, 8)[:-1])`, sort `[*pos, 0.0, *neg]` and divide by the largest absolute value. in `nf4_quantize`: `flat = w.reshape(-1, block_size)`, `s = flat.abs().amax(1, keepdim=True)` (replace zeros by 1), `idx = ((flat / s)[..., None] - levels).abs().argmin(-1)`, return `(levels[idx] * s).reshape(w.shape)`.

</details>
