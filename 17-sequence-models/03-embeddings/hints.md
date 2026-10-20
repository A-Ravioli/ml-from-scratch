# embeddings · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

offsets `[-w, ..., -1, 1, ..., w]`; `j = np.arange(n)[:, None] + offsets[None, :]` is an `(n, 2w)` array of context positions whose rows are already in the right order. `valid = (j >= 0) & (j < n)` masks the ones inside the text. boolean indexing a 2-D array flattens it row by row, which keeps "by center, then by context" order.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
centers = np.broadcast_to(ids[:, None], j.shape)[valid]
contexts = ids[j[valid]]
return np.stack([centers, contexts], axis=1)
```

convert `ids` with `np.asarray(ids, dtype=np.int64)` first.

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

weights `w = counts ** power` as a float64 tensor, set to 0 wherever `counts == 0`, normalized; then `torch.multinomial(probs, n, replacement=True, generator=generator)` already returns int64.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
c = torch.as_tensor(np.asarray(counts, dtype=np.float64))
w = torch.where(c > 0, c ** power, torch.zeros_like(c))
return torch.multinomial(w / w.sum(), n, replacement=True, generator=generator)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`__init__`: two `nn.Embedding(vocab_size, embed_dim)`, then under `torch.no_grad()`, `self.in_embed.weight.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)` and `self.out_embed.weight.zero_()`. `forward`: `v = self.in_embed(center)`, `u_pos = self.out_embed(context)`, `u_neg = self.out_embed(negatives)`; positive scores `(u_pos * v).sum(-1)`, negative scores `torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
pos = F.logsigmoid((u_pos * v).sum(-1))                                      # (B,)
neg = F.logsigmoid(-torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)).sum(-1)  # (B,)
return -(pos + neg).mean()
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

turn the input into a float64 NumPy array (`.detach().cpu().double().numpy()` for a tensor), divide every row by its norm (guard against 0), and the cosines to row `idx` are `E @ E[idx]`. set the excluded entries to `-np.inf` before ranking; `np.argsort(-sims, kind="stable")[:k]` gives the top $k$, ties to the smaller index.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
target = E[b] - E[a] + E[c]      # E already row-normalized
sims = E @ target                # dividing by |target| wouldn't change the ranking
sims[[a, b, c]] = -np.inf
return np.argsort(-sims, kind="stable")[:k]
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`g = torch.Generator().manual_seed(seed)`; `pairs = torch.as_tensor(skipgram_pairs(ids, window))`; `counts = np.bincount(ids, minlength=vocab_size)`; build the model inside `with torch.random.fork_rng(): torch.manual_seed(seed); ...`. each epoch: `order = torch.randperm(len(pairs), generator=g)`, then slices of `batch_size`; negatives `sample_negatives(counts, len(batch) * n_negatives, generator=g).view(len(batch), n_negatives)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
loss = model(batch[:, 0].to(device), batch[:, 1].to(device), neg.to(device))
opt.zero_grad()
loss.backward()
opt.step()
losses.append(loss.item())
```

with `opt = torch.optim.Adam(model.parameters(), lr=lr)` created once, before the loop.

</details>
