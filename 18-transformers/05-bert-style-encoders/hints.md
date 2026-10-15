# BERT-style encoders · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

```python
special = torch.zeros_like(idx, dtype=torch.bool)
for s in special_ids:
    special |= idx == s
selected = (torch.rand(idx.shape, generator=generator) < mask_prob) & ~special
action = torch.rand(idx.shape, generator=generator)        # < 0.8 mask, 0.8-0.9 random, >= 0.9 keep
random_tokens = torch.randint(0, vocab_size, idx.shape, generator=generator)
```

start from `inputs = idx.clone()`, write `mask_token_id` where `selected & (action < 0.8)` and `random_tokens` where `selected & (0.8 <= action < 0.9)`; `labels = torch.where(selected, idx, -100)`.

</details>

## stage 2

<details><summary>hint</summary>

`forward`: embed tokens, add `self.pos_emb(torch.arange(T))`, dropout, then `block(x, mask=attention_mask[:, None, None, :])` for each block (or `mask=None`), then `self.ln_f`. for the head keep `mlm_dense` (`Linear(d, d)`), `mlm_norm` (`LayerNorm(d)`) and `mlm_bias` (`nn.Parameter(torch.zeros(V))`), and compute `mlm_norm(F.gelu(mlm_dense(h))) @ self.tok_emb.weight.T + self.mlm_bias`. initialize like GPT: loop over `self.modules()`.

</details>

## stage 3

<details><summary>hint</summary>

one generator drives everything: `rows = torch.randint(0, len(data), (batch_size,), generator=g)`, then `mask_tokens(data[rows], ..., generator=g)`, and the attention mask rows `attention_mask[rows]` if there is one. `mlm_loss` is a reshape to `(-1, V)` and `F.cross_entropy(..., ignore_index=-100)`.

</details>

## stage 4

<details><summary>hint</summary>

`h = self.encoder(idx, attention_mask)`; `'cls'` pools `h[:, 0]`; `'mean'` computes `(h * m[..., None]).sum(1) / m.sum(1, keepdim=True)` with `m = attention_mask.float()` (all ones without a mask). in `finetune`, loop over epochs, `perm = torch.randperm(len(X), generator=g)`, take slices of `batch_size`, and weight each batch's loss by its size when averaging the epoch loss.

</details>
