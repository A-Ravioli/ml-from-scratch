# prototypical networks · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

`split_classes`: `ids = sorted({int(c) for c in classes})` (call `.tolist()` first if it's an array), `perm = torch.randperm(len(ids), generator=generator).tolist()`, the first `n_test` of the permuted ids are the test classes; sort both tuples. `sample_episode`: convert `y` to a tensor, check `n_way` and every allowed class's count first, then `chosen = classes[torch.randperm(len(classes), generator=generator)[:n_way]]`; for each chosen class, `idx = torch.nonzero(y == c).flatten()`, `pick = idx[torch.randperm(len(idx), generator=generator)[:k_shot + n_query]]`, support `pick[:k_shot]`, query the rest. labels are `torch.arange(n_way).repeat_interleave(k_shot)` (and `n_query`).

</details>

## stage 2

<details><summary>hint</summary>

`prototypes`: `onehot = F.one_hot(labels, n_way).to(z.dtype)` is `(S, N)`; `onehot.T @ z` sums each class's embeddings and `onehot.sum(0)` counts them. `proto_logits`: `((zq[:, None, :] - protos[None, :, :]) ** 2).sum(-1)` is the `(Q, N)` matrix of squared distances; for cosine, `F.normalize(zq, dim=-1) @ F.normalize(protos, dim=-1).T`. multiply by `scale` (and negate the distances).

</details>

## stage 3

<details><summary>hint</summary>

`z = embed(X[torch.cat([s, q])])`, then `zs, zq = z[:len(s)], z[len(s):]`; `logits = proto_logits(zq, prototypes(zs, episode["support_labels"], len(episode["classes"])), metric=metric, scale=scale)`; `F.cross_entropy(logits, episode["query_labels"])`; accuracy `(logits.argmax(1) == labels).float().mean().item()`.

</details>

## stage 4

<details><summary>hint</summary>

training: `generator = torch.Generator().manual_seed(seed)`, `opt = torch.optim.Adam(embed.parameters(), lr=lr)`, `embed.train()`, then per episode `sample_episode(..., generator=generator)`, `episode_loss`, `zero_grad`/`backward`/`step`, record `loss.item()`. evaluation: `embed.eval()`, `with torch.no_grad():` the same loop collecting accuracies; return `accs.mean()` and `1.96 * accs.std(ddof=1) / np.sqrt(len(accs))` as floats.

</details>
