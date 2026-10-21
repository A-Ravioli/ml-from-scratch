# modern LLM architecture · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`d_ff = 8 * math.ceil(int(8 * d_model / 3) / 8)` when it's `None`; three `nn.Linear(..., bias=False)`; `forward` is one line: `self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))`.

</details>

## stage 2

<details><summary>hint</summary>

reshape the projections to heads: `q = q_proj(x).view(B, T, H, dh).transpose(1, 2)` and the same for `k`, `v` with `H_kv`. `offset = cache.seq_len if cache is not None else 0`, then `q, k = self.rope(q, offset), self.rope(k, offset)`, then `k, v = cache.update(k, v)`, then `repeat_kv(k, H // H_kv)` (and `v`). for the attention, `F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)` works when the query and key lengths are equal; otherwise pass `attn_mask=torch.ones(T, Tk, dtype=torch.bool).tril(diagonal=Tk - T)`. `p = self.dropout if self.training else 0.0`. merge heads with `out.transpose(1, 2).reshape(B, T, H * dh)` and apply `o_proj`.

</details>

## stage 3

<details><summary>hint</summary>

flatten to `(N, d)`. `probs = F.softmax(self.router(flat).float(), -1)`, `top_p, top_i = probs.topk(self.top_k, -1)`, `gates = top_p / top_p.sum(-1, keepdim=True)`. for each expert `e`: `rows, slot = (top_i == e).nonzero(as_tuple=True)` lists the tokens (and which of their k slots) chose it; `out.index_add_(0, rows, gates[rows, slot, None] * expert(flat[rows]))`. counts: `torch.bincount(top_i.flatten(), minlength=E)`; `f = counts / top_i.numel()`; `aux = E * (f * probs.mean(0)).sum()`.

</details>

## stage 4

<details><summary>hint</summary>

`h = x + self.attn(self.norm1(x), cache=cache)`, `m = self.mlp(self.norm2(h))`. an MoE returns a tuple: `if isinstance(m, tuple): m, self.aux_loss = m` (else `self.aux_loss = 0.0`). return `h + m`.

</details>

## stage 5

<details><summary>hint</summary>

copy the structure of chapter 18's `GPT` without `pos_emb`. initialize every `nn.Linear` and `nn.Embedding` weight with `nn.init.normal_(w, std=0.02)`, then loop over `named_parameters()` and re-initialize names ending in `attn.o_proj.weight` or `w_down.weight` with `std=0.02 / math.sqrt(2 * n_layers)`; tie the head last. in `forward`, check `cache.seq_len + T <= block_size` before running the blocks, pass `cache[i]` to block `i`, and add `self.aux_loss_coef * sum(b.aux_loss for b in self.blocks)` to the cross-entropy. `generate` is chapter 18's `GPT.generate` unchanged.

</details>
