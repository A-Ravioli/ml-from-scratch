# two-tower retrieval · hints

open a hint only after you've been stuck for a while. this block is spec-driven: there's one hint per stage, and it points at the idea, not the code.

## stage 1

<details><summary>hint</summary>

call `super().__init__()` first, then build `self.embedding = nn.Embedding(n_ids, dim)` (re-initialize it with `nn.init.normal_(self.embedding.weight, 0.0, 0.05)`) and `self.mlp = MLP(n_features, [hidden], dim)`. `forward` is one line: `F.normalize(self.embedding(ids) + self.mlp(features), dim=-1)`. `TwoTower.forward` embeds the batch's users and items with their towers and returns `u @ v.T / self.temperature`.

</details>

## stage 2

<details><summary>hint</summary>

`np.bincount(item_ids, minlength=n_items)`, clip at 1, divide by `len(item_ids)`, log, wrap in a float32 tensor. in the loss: `logits - log_q[None, :]` (one value per column), then a mask `dup = (item_ids[None, :] == item_ids[:, None]) & ~torch.eye(B, dtype=torch.bool)` and `logits.masked_fill(dup, float("-inf"))`, then `F.cross_entropy(logits, torch.arange(B))`.

</details>

## stage 3

<details><summary>hint</summary>

convert everything to tensors once at the top (`torch.as_tensor`, int64 ids, float32 features), compute `log_freq = item_log_frequency(item_ids, n_items)` once, create the generator once (not per epoch), and inside the batch loop index the feature tables with the batch's ids: `model(u, user_feats[u], i, item_feats[i])`, `log_q=log_freq[i]`. record `loss.item()` so you don't keep the graph alive. `embed_all`: `model.eval()`, `with torch.no_grad():`, `model.user_tower(torch.arange(n_users), user_feats)`, same for items, `.numpy()`.

</details>

## stage 4

<details><summary>hint</summary>

`exact_search`: `scores = queries @ vectors.T`, `np.argsort(-scores, axis=1, kind="stable")[:, :k]`, and `np.take_along_axis` for the scores. `build`: fit `KMeans(...)`, keep `cluster_centers_`, and `lists_ = [np.flatnonzero(labels_ == c) for c in range(n_lists)]`. `search`: squared distances from each query to every centroid, the `n_probe` smallest per query, then per query concatenate those lists, score them, and fill the first `min(k, n_candidates)` slots of preallocated `-1` / `-inf` arrays. count candidates as you go for `last_n_scored_`.

</details>

## stage 5

<details><summary>hint</summary>

`ranker_features` is one `np.stack([...], axis=1)` of `(u * v).sum(1)`, `price - budget`, `price`, `budget`. the ranker is `MLP(n_features, [hidden, hidden], 1)` with `.squeeze(-1)`, which works for any leading shape. `train_ranker` is `train_two_tower`'s loop with `F.cross_entropy(ranker(X[b]), torch.zeros(len(b), dtype=torch.int64))`: the scores of a list are its logits and the positive is class 0. in `two_stage`, build a boolean `bad = cand < 0`, OR in `exclude[user_ids[:, None], np.maximum(cand, 0)]`, set those scores to `-np.inf`, then a stable argsort and `np.take_along_axis(cand, order, axis=1)`.

</details>
