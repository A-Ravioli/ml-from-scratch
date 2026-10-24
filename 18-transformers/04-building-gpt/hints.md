# building GPT · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

convert with `torch.as_tensor(data, dtype=torch.long)`, then `starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)` (the upper bound is exclusive) and index with `starts[:, None] + torch.arange(block_size)`; `y` uses the same indices plus one.

</details>

## stage 2

<details><summary>hint</summary>

write one private helper that computes the stream and use it in both `forward` and `hidden_states`:

```python
offset = cache.seq_len if cache is not None else 0
pos = torch.arange(offset, offset + T, device=idx.device)       # check offset + T <= block_size first
x = self.drop(self.tok_emb(idx) + position_vectors(pos))
for i, block in enumerate(self.blocks):
    x = block(x, is_causal=True, cache=None if cache is None else cache[i])
return self.ln_f(x)
```

for the sinusoidal option, register `sinusoidal_positions(block_size, d)` as a buffer and index it with `pos`. initialize by looping over `self.modules()` (`nn.init.normal_(w, std=0.02)`, zero biases), then re-initialize each block's `attn.out_proj.weight` and `mlp.fc2.weight` with `std=0.02 / math.sqrt(2 * n_layers)`, and tie last (`self.lm_head.weight = self.tok_emb.weight`).

</details>

## stage 3

<details><summary>hint</summary>

top-k: `kth = torch.topk(logits, k).values[:, -1:]`, then `masked_fill(logits < kth, -inf)`. top-p: `sorted, order = torch.sort(logits, descending=True)`, `probs = softmax(sorted)`, `before = probs.cumsum(-1) - probs`, mask `before >= top_p` with `-inf`, then put the values back in vocabulary order with `torch.full_like(logits, -inf).scatter(-1, order, sorted)`. finish with `torch.multinomial(softmax(logits), 1, generator=generator).squeeze(-1)`.

</details>

## stage 4

<details><summary>hint</summary>

remember `was_training = self.training`, call `self.eval()`, and in a `try/finally` loop `max_new_tokens` times: `logits, _ = self(idx[:, -self.block_size:])`, pick from `logits[:, -1]` with `sample_next_token`, `idx = torch.cat([idx, nxt[:, None]], 1)`. restore with `self.train(was_training)`.

</details>

## stage 5

<details><summary>hint</summary>

parameter groups: `[{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay}, {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]` into `torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))`. each step: batch, forward with targets, `zero_grad`, `backward`, `nn.utils.clip_grad_norm_`, `opt.step()`, remember `opt.param_groups[0]['lr']` (the rate this step used), `sched.step()`. evaluate when `step % eval_interval == 0 or step == max_steps` with your own `estimate_loss` (which builds its own seed-0 generator, so it doesn't disturb the training batches).

</details>
