import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # prototypical networks · build

    spec-driven (L4): the **spec** boxes in the lesson define every function, the checks
    enforce them, and the internals are yours. a dataset is a tensor `X` (any shape after the
    first axis) with integer labels `y (n,)`; an **episode** is a dict of index tensors into it,
    so that the same episode can be replayed with any embedding. `embed` is any `nn.Module`
    mapping `(B, ...)` inputs to `(B, D)` embeddings.

    | stage | you build |
    |---|---|
    | 1 | `split_classes`, `sample_episode`: disjoint meta-train/meta-test classes and N-way K-shot episodes |
    | 2 | `prototypes`, `proto_logits`: class means, and negative squared distances (or scaled cosines) as logits |
    | 3 | `episode_loss`: the prototypical loss and accuracy of one episode |
    | 4 | `train_protonet`, `evaluate_protonet`: episodic training, and few-shot accuracy with a 95% interval |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def split_classes(classes, n_test, *, generator=None):
    """stage 1: shuffle the distinct class ids, return (train, test) as sorted tuples of ints, len(test) = n_test."""
    ids = sorted({int(c) for c in (classes.tolist() if hasattr(classes, "tolist") else classes)})
    if not 0 < n_test < len(ids):
        raise ValueError(f"n_test must be between 1 and {len(ids) - 1}, got {n_test}")
    perm = torch.randperm(len(ids), generator=generator).tolist()
    test = tuple(sorted(ids[i] for i in perm[:n_test]))
    train = tuple(sorted(ids[i] for i in perm[n_test:]))
    return train, test


@app.function
def sample_episode(y, classes, n_way, k_shot, n_query, *, generator=None):
    """stage 1: dict of int64 tensors support (N*K,), query (N*Q,) indices into y, support_labels, query_labels in 0..N-1, classes (N,)."""
    y = torch.as_tensor(np.asarray(y)) if not torch.is_tensor(y) else y
    classes = torch.as_tensor(list(classes), dtype=torch.int64)
    if n_way > len(classes):
        raise ValueError(f"n_way={n_way} but only {len(classes)} classes are allowed")
    counts = torch.stack([(y == c).sum() for c in classes])
    if int(counts.min()) < k_shot + n_query:
        raise ValueError(f"every class needs k_shot + n_query = {k_shot + n_query} examples; one has {int(counts.min())}")
    chosen = classes[torch.randperm(len(classes), generator=generator)[:n_way]]
    support, query = [], []
    for c in chosen:
        idx = torch.nonzero(y == c).flatten()
        pick = idx[torch.randperm(len(idx), generator=generator)[:k_shot + n_query]]
        support.append(pick[:k_shot])
        query.append(pick[k_shot:])
    return {
        "support": torch.cat(support).long(),
        "query": torch.cat(query).long(),
        "support_labels": torch.arange(n_way).repeat_interleave(k_shot),
        "query_labels": torch.arange(n_way).repeat_interleave(n_query),
        "classes": chosen,
    }


@app.cell(hide_code=True)
def _():
    checks.run(1, split_classes=split_classes, sample_episode=sample_episode)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def prototypes(z, labels, n_way):
    """stage 2: (N, D) mean embedding of each class 0..N-1; labels in any order, counts may differ."""
    onehot = F.one_hot(labels.long(), n_way).to(z.dtype)  # (S, N)
    counts = onehot.sum(0)
    if bool((counts == 0).any()):
        raise ValueError("every class 0..n_way-1 needs at least one support example")
    return onehot.T @ z / counts[:, None]


@app.function
def proto_logits(zq, protos, *, metric="sqeuclidean", scale=1.0):
    """stage 2: (Q, N) logits: -scale * squared Euclidean distance, or scale * cosine similarity."""
    if metric == "sqeuclidean":
        return -scale * ((zq[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
    if metric == "cosine":
        return scale * F.normalize(zq, dim=-1) @ F.normalize(protos, dim=-1).T
    raise ValueError(f"metric must be 'sqeuclidean' or 'cosine', got {metric!r}")


@app.cell(hide_code=True)
def _():
    checks.run(2, prototypes=prototypes, proto_logits=proto_logits)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def episode_loss(embed, X, episode, *, metric="sqeuclidean", scale=1.0):
    """stage 3: (cross-entropy of the query logits as a scalar tensor, query accuracy as a float); one embed call on support then query."""
    s, q = episode["support"], episode["query"]
    z = embed(X[torch.cat([s, q])])
    zs, zq = z[:len(s)], z[len(s):]
    n_way = len(episode["classes"])
    logits = proto_logits(zq, prototypes(zs, episode["support_labels"], n_way), metric=metric, scale=scale)
    loss = F.cross_entropy(logits, episode["query_labels"])
    acc = (logits.argmax(1) == episode["query_labels"]).float().mean().item()
    return loss, acc


@app.cell(hide_code=True)
def _():
    checks.run(3, prototypes=prototypes, proto_logits=proto_logits, episode_loss=episode_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_protonet(embed, X, y, classes, *, n_way, k_shot, n_query, n_episodes, lr=1e-3, metric="sqeuclidean",
                   scale=1.0, seed=0):
    """stage 4: Adam on episode_loss over n_episodes episodes drawn from `classes` (train mode); returns the list of losses."""
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(embed.parameters(), lr=lr)
    embed.train()
    history = []
    for _ in range(n_episodes):
        ep = sample_episode(y, classes, n_way, k_shot, n_query, generator=generator)
        loss, _ = episode_loss(embed, X, ep, metric=metric, scale=scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


@app.function
def evaluate_protonet(embed, X, y, classes, *, n_way, k_shot, n_query=15, n_episodes=600, metric="sqeuclidean",
                      scale=1.0, seed=0):
    """stage 4: (mean query accuracy over episodes, 95% half-width 1.96 * sd / sqrt(n_episodes)), eval mode, no gradients."""
    generator = torch.Generator().manual_seed(seed)
    embed.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = sample_episode(y, classes, n_way, k_shot, n_query, generator=generator)
            accs.append(episode_loss(embed, X, ep, metric=metric, scale=scale)[1])
    accs = np.array(accs)
    return float(accs.mean()), float(1.96 * accs.std(ddof=1) / np.sqrt(len(accs)))


@app.cell(hide_code=True)
def _():
    checks.run(4, sample_episode=sample_episode, prototypes=prototypes, proto_logits=proto_logits,
               episode_loss=episode_loss, train_protonet=train_protonet, evaluate_protonet=evaluate_protonet)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it trains your prototypical
    networks on the faces of 28 people and tests them on 12 people they've never seen.
    """)
    return


if __name__ == "__main__":
    app.run()
