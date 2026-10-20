import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from types import SimpleNamespace

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07
    from mlfs.unsupervised import KMeans  # yours, from ch 11 · topic 01
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # two-tower retrieval · build (reference solution)

    this block is spec-driven (L4): each stage's exact specification is in the lesson's
    **spec** box next to it, the checks below enforce it, and how you structure the code is up
    to you. you'll build the retrieval half of a production recommender: two neural towers
    trained with an in-batch sampled softmax, an approximate nearest-neighbour index to search
    their embeddings, and a second-stage ranker. the lab assembles them into the block's
    project, a retrieve-then-rank system.

    | stage | you build |
    |---|---|
    | 1 | `Tower`, `TwoTower`: user and item encoders from ids and features |
    | 2 | `item_log_frequency`, `in_batch_softmax_loss`: in-batch negatives with the logQ correction |
    | 3 | `train_two_tower`, `embed_all`: the training loop and the embedding export |
    | 4 | `exact_search`, `IVFIndex`: brute-force and inverted-file nearest-neighbour search |
    | 5 | `ranker_features`, `Ranker`, `train_ranker`, `two_stage`: a listwise ranker over retrieved candidates |

    the data come from `make_catalog` (given, below), the simulator of topics 21–22 with
    user and item features added. you can also run the checks from a terminal:
    `pytest -k stage1`.
    """)
    return


@app.function
def make_catalog(n_users=1500, n_items=3000, rank=8, *, n_features=12, mean_interactions=25,
                 popularity_exponent=0.8, selection_strength=2.0, price_penalty=3.0, rng=None):
    """given: a synthetic catalogue with user and item features (see the lesson's "the data" box).

    tastes U (n_users, rank), V (n_items, rank), biases and power-law exposure as in topics 21-22, plus a
    budget per user and a price per item (both N(0, 1)): pref[u, i] = 3 + b_u + b_i + U[u] . V[i]
    - price_penalty * [price[i] > budget[u]]. users consume items with probability proportional to
    pop[i] * exp(selection_strength * pref[u, i]), without replacement.

    features: user_features = [U A + noise (n_features columns), budget] and item_features =
    [V B + noise (n_features columns), price], float32: noisy views of the tastes, and the exact budget
    and price (last columns). returns a SimpleNamespace with users, items (int64 interaction pairs),
    n_users, n_items, user_features, item_features, preference (float32 (n_users, n_items)), popularity.
    """
    rng = as_rng(rng)
    scale = (0.8 / rank) ** 0.25
    U = rng.normal(0.0, scale, (n_users, rank))
    V = rng.normal(0.0, scale, (n_items, rank))
    log_pop = -popularity_exponent * np.log1p(rng.permutation(n_items))
    item_bias = 0.25 * (log_pop - log_pop.mean()) / log_pop.std() + rng.normal(0.0, 0.3, n_items)
    user_bias = rng.normal(0.0, 0.4, n_users)
    budget = rng.normal(0.0, 1.0, n_users)
    price = rng.normal(0.0, 1.0, n_items)
    pref = 3.0 + user_bias[:, None] + item_bias[None, :] + U @ V.T - price_penalty * (price[None, :] > budget[:, None])
    n_per = np.clip(np.round(rng.lognormal(np.log(mean_interactions) - 0.5, 1.0, n_users)), 5, n_items // 4).astype(int)
    logits = log_pop[None, :] + selection_strength * (pref - pref.mean(axis=1, keepdims=True))
    keys = logits + rng.gumbel(size=logits.shape)
    users = np.concatenate([np.full(n, u) for u, n in enumerate(n_per)]).astype(np.int64)
    items = np.concatenate([np.argpartition(-keys[u], n)[:n] for u, n in enumerate(n_per)]).astype(np.int64)
    A = rng.normal(0.0, 1.0, (rank, n_features)) / np.sqrt(rank * scale ** 2)
    B = rng.normal(0.0, 1.0, (rank, n_features)) / np.sqrt(rank * scale ** 2)
    user_features = np.c_[U @ A + rng.normal(0.0, 0.5, (n_users, n_features)), budget].astype(np.float32)
    item_features = np.c_[V @ B + rng.normal(0.0, 0.5, (n_items, n_features)), price].astype(np.float32)
    return SimpleNamespace(users=users, items=items, n_users=n_users, n_items=n_items, user_features=user_features,
                           item_features=item_features, preference=pref.astype(np.float32), popularity=np.exp(log_pop))


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Tower(nn.Module):
    """one tower: normalize(Embedding(ids) + MLP(features)), a unit vector per row."""

    def __init__(self, n_ids, n_features, *, dim=32, hidden=64):
        """stage 1: an nn.Embedding(n_ids, dim) (init N(0, 0.05^2)) and an MLP n_features -> hidden -> dim."""
        super().__init__()
        self.embedding = nn.Embedding(n_ids, dim)
        nn.init.normal_(self.embedding.weight, 0.0, 0.05)
        self.mlp = MLP(n_features, [hidden], dim)

    def forward(self, ids, features):
        """stage 1: ids int64 (B,), features float (B, n_features) -> float (B, dim) with unit rows."""
        return F.normalize(self.embedding(ids) + self.mlp(features), dim=-1)


@app.class_definition
class TwoTower(nn.Module):
    """a user tower and an item tower; logits[b, b'] = u_b . v_b' / temperature."""

    def __init__(self, n_users, n_items, user_dim, item_dim, *, dim=32, hidden=64, temperature=0.1):
        """stage 1: self.user_tower, self.item_tower (Tower), self.temperature."""
        super().__init__()
        self.user_tower = Tower(n_users, user_dim, dim=dim, hidden=hidden)
        self.item_tower = Tower(n_items, item_dim, dim=dim, hidden=hidden)
        self.temperature = temperature

    def forward(self, user_ids, user_feats, item_ids, item_feats):
        """stage 1: (B, B) logits between every user and every item of the batch."""
        u = self.user_tower(user_ids, user_feats)
        v = self.item_tower(item_ids, item_feats)
        return u @ v.T / self.temperature


@app.cell(hide_code=True)
def _():
    checks.run(1, Tower=Tower, TwoTower=TwoTower)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def item_log_frequency(item_ids, n_items):
    """stage 2: log(count_i / total) for every item, float32 tensor (n_items,), counts clipped at 1."""
    counts = np.bincount(np.asarray(item_ids), minlength=n_items).astype(np.float64)
    return torch.tensor(np.log(np.maximum(counts, 1.0) / len(item_ids)), dtype=torch.float32)


@app.function
def in_batch_softmax_loss(logits, item_ids, *, log_q=None):
    """stage 2: mean cross-entropy of the diagonal; subtract log_q (B,) per column; mask other copies of a row's item."""
    if log_q is not None:
        logits = logits - log_q[None, :]
    dup = (item_ids[None, :] == item_ids[:, None]) & ~torch.eye(len(item_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(dup, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(item_ids), device=logits.device))


@app.cell(hide_code=True)
def _():
    checks.run(2, item_log_frequency=item_log_frequency, in_batch_softmax_loss=in_batch_softmax_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def train_two_tower(model, user_ids, item_ids, user_feats, item_feats, *, epochs=5, batch_size=512, lr=3e-3,
                    logq_correction=True, seed=0):
    """stage 3: Adam over shuffled (user, item) pairs with the in-batch loss; returns the mean loss of each epoch."""
    user_ids = torch.as_tensor(np.asarray(user_ids), dtype=torch.int64)
    item_ids = torch.as_tensor(np.asarray(item_ids), dtype=torch.int64)
    user_feats = torch.as_tensor(np.asarray(user_feats), dtype=torch.float32)
    item_feats = torch.as_tensor(np.asarray(item_feats), dtype=torch.float32)
    n_items = item_feats.shape[0]
    log_freq = item_log_frequency(item_ids.numpy(), n_items) if logq_correction else None
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    history = []
    for _ in range(epochs):
        perm = torch.randperm(len(user_ids), generator=gen)
        total, n_batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            u, i = user_ids[b], item_ids[b]
            logits = model(u, user_feats[u], i, item_feats[i])
            loss = in_batch_softmax_loss(logits, i, log_q=None if log_freq is None else log_freq[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        history.append(total / n_batches)
    return history


@app.function
def embed_all(model, user_feats, item_feats):
    """stage 3: (user embeddings (n_users, dim), item embeddings (n_items, dim)) as float32 ndarrays, eval mode, no grad."""
    model.eval()
    uf = torch.as_tensor(np.asarray(user_feats), dtype=torch.float32)
    itf = torch.as_tensor(np.asarray(item_feats), dtype=torch.float32)
    with torch.no_grad():
        u = model.user_tower(torch.arange(len(uf)), uf)
        v = model.item_tower(torch.arange(len(itf)), itf)
    return u.numpy().astype(np.float32), v.numpy().astype(np.float32)


@app.cell(hide_code=True)
def _():
    checks.run(3, train_two_tower=train_two_tower, embed_all=embed_all, TwoTower=TwoTower)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def exact_search(queries, vectors, k):
    """stage 4: top-k inner products by brute force: (ids int64 (Q, k), scores (Q, k)), best first, ties to the lower id."""
    scores = np.asarray(queries) @ np.asarray(vectors).T
    ids = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return ids.astype(np.int64), np.take_along_axis(scores, ids, axis=1)


@app.class_definition
class IVFIndex:
    """inverted-file index: k-means cells; a query scans only the n_probe cells with the nearest centroids."""

    def __init__(self, n_lists=32, *, n_probe=4, rng=None):
        """stage 4: store n_lists, n_probe, rng."""
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.rng = rng

    def build(self, vectors):
        """stage 4: KMeans(n_lists, n_init=1, max_iters=25) on the vectors; centroids_, lists_ (ids per cell); returns self."""
        self.vectors_ = np.asarray(vectors, dtype=np.float64)
        km = KMeans(self.n_lists, n_init=1, max_iters=25, rng=self.rng).fit(self.vectors_)
        self.centroids_ = km.cluster_centers_
        self.lists_ = [np.flatnonzero(km.labels_ == c) for c in range(self.n_lists)]
        return self

    def search(self, queries, k):
        """stage 4: per query, scan the n_probe cells (centroids_, lists_) nearest (L2) to it; exact top-k inner products, ties to the lower id (pad -1 / -inf)."""
        queries = np.asarray(queries, dtype=np.float64)
        d2 = ((queries[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(-1)
        probe = np.argsort(d2, axis=1, kind="stable")[:, : self.n_probe]
        ids = np.full((len(queries), k), -1, dtype=np.int64)
        scores = np.full((len(queries), k), -np.inf)
        scanned = 0
        for q in range(len(queries)):
            cand = np.sort(np.concatenate([self.lists_[c] for c in probe[q]]))
            scanned += len(cand)
            s = self.vectors_[cand] @ queries[q]
            order = np.argsort(-s, kind="stable")[:k]
            ids[q, : len(order)] = cand[order]
            scores[q, : len(order)] = s[order]
        self.last_n_scored_ = scanned / max(len(queries), 1)
        return ids, scores


@app.cell(hide_code=True)
def _():
    checks.run(4, exact_search=exact_search, IVFIndex=IVFIndex)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def ranker_features(user_emb, item_emb, user_feats, item_feats):
    """stage 5: float32 (B, 4) = [u . v, price - budget, price, budget] for B (user, item) pairs."""
    u, v = np.asarray(user_emb, dtype=np.float64), np.asarray(item_emb, dtype=np.float64)
    budget = np.asarray(user_feats)[:, -1].astype(np.float64)
    price = np.asarray(item_feats)[:, -1].astype(np.float64)
    return np.stack([np.sum(u * v, axis=1), price - budget, price, budget], axis=1).astype(np.float32)


@app.class_definition
class Ranker(nn.Module):
    """second stage: an MLP from pair features to one score per pair."""

    def __init__(self, n_features=4, *, hidden=64):
        """stage 5: an MLP n_features -> hidden -> hidden -> 1."""
        super().__init__()
        self.mlp = MLP(n_features, [hidden, hidden], 1)

    def forward(self, x):
        """stage 5: float (..., n_features) -> scores (...)."""
        return self.mlp(x).squeeze(-1)


@app.function
def train_ranker(ranker, X, *, epochs=5, batch_size=256, lr=2e-3, seed=0):
    """stage 5: X (n, L, n_features), list item 0 the positive; Adam on the softmax loss over each list; mean loss per epoch."""
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(ranker.parameters(), lr=lr)
    ranker.train()
    history = []
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=gen)
        total = 0.0
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            scores = ranker(X[b])
            loss = F.cross_entropy(scores, torch.zeros(len(b), dtype=torch.int64))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        history.append(total / len(X))
    ranker.eval()
    return history


@app.function
def two_stage(user_ids, user_emb, index, score_fn, *, n_candidates=100, k=10, exclude=None):
    """stage 5: retrieve n_candidates per user with index.search, rescore with score_fn, drop excluded/padding; top-k ids (Q, k)."""
    user_ids = np.asarray(user_ids)
    cand, _ = index.search(np.asarray(user_emb)[user_ids], n_candidates)
    scores = np.array(score_fn(user_ids, cand), dtype=np.float64)
    bad = cand < 0
    if exclude is not None:
        bad |= np.asarray(exclude)[user_ids[:, None], np.maximum(cand, 0)]
    scores[bad] = -np.inf
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(cand, order, axis=1)


@app.cell(hide_code=True)
def _():
    checks.run(5, ranker_features=ranker_features, Ranker=Ranker, train_ranker=train_ranker, two_stage=two_stage)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it trains your towers on
    the simulated catalogue, asks what the logQ correction is for, how much an approximate
    index costs you, what features buy for items nobody has seen, and ends with the block's
    project: a two-stage retrieve-then-rank recommender.
    """)
    return


if __name__ == "__main__":
    app.run()
