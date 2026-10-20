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
    # two-tower retrieval · build

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
        raise NotImplementedError("stage1: implement Tower.__init__")

    def forward(self, ids, features):
        """stage 1: ids int64 (B,), features float (B, n_features) -> float (B, dim) with unit rows."""
        raise NotImplementedError("stage1: implement Tower.forward")


@app.class_definition
class TwoTower(nn.Module):
    """a user tower and an item tower; logits[b, b'] = u_b . v_b' / temperature."""

    def __init__(self, n_users, n_items, user_dim, item_dim, *, dim=32, hidden=64, temperature=0.1):
        """stage 1: self.user_tower, self.item_tower (Tower), self.temperature."""
        raise NotImplementedError("stage1: implement TwoTower.__init__")

    def forward(self, user_ids, user_feats, item_ids, item_feats):
        """stage 1: (B, B) logits between every user and every item of the batch."""
        raise NotImplementedError("stage1: implement TwoTower.forward")


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
    raise NotImplementedError("stage2: implement item_log_frequency")


@app.function
def in_batch_softmax_loss(logits, item_ids, *, log_q=None):
    """stage 2: mean cross-entropy of the diagonal; subtract log_q (B,) per column; mask other copies of a row's item."""
    raise NotImplementedError("stage2: implement in_batch_softmax_loss")


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
    raise NotImplementedError("stage3: implement train_two_tower")


@app.function
def embed_all(model, user_feats, item_feats):
    """stage 3: (user embeddings (n_users, dim), item embeddings (n_items, dim)) as float32 ndarrays, eval mode, no grad."""
    raise NotImplementedError("stage3: implement embed_all")


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
    raise NotImplementedError("stage4: implement exact_search")


@app.class_definition
class IVFIndex:
    """inverted-file index: k-means cells; a query scans only the n_probe cells with the nearest centroids."""

    def __init__(self, n_lists=32, *, n_probe=4, rng=None):
        """stage 4: store n_lists, n_probe, rng."""
        raise NotImplementedError("stage4: implement IVFIndex.__init__")

    def build(self, vectors):
        """stage 4: KMeans(n_lists, n_init=1, max_iters=25) on the vectors; centroids_, lists_ (ids per cell); returns self."""
        raise NotImplementedError("stage4: implement IVFIndex.build")

    def search(self, queries, k):
        """stage 4: per query, scan the n_probe cells (centroids_, lists_) nearest (L2) to it; exact top-k inner products, ties to the lower id (pad -1 / -inf)."""
        raise NotImplementedError("stage4: implement IVFIndex.search")


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
    raise NotImplementedError("stage5: implement ranker_features")


@app.class_definition
class Ranker(nn.Module):
    """second stage: an MLP from pair features to one score per pair."""

    def __init__(self, n_features=4, *, hidden=64):
        """stage 5: an MLP n_features -> hidden -> hidden -> 1."""
        raise NotImplementedError("stage5: implement Ranker.__init__")

    def forward(self, x):
        """stage 5: float (..., n_features) -> scores (...)."""
        raise NotImplementedError("stage5: implement Ranker.forward")


@app.function
def train_ranker(ranker, X, *, epochs=5, batch_size=256, lr=2e-3, seed=0):
    """stage 5: X (n, L, n_features), list item 0 the positive; Adam on the softmax loss over each list; mean loss per epoch."""
    raise NotImplementedError("stage5: implement train_ranker")


@app.function
def two_stage(user_ids, user_emb, index, score_fn, *, n_candidates=100, k=10, exclude=None):
    """stage 5: retrieve n_candidates per user with index.search, rescore with score_fn, drop excluded/padding; top-k ids (Q, k)."""
    raise NotImplementedError("stage5: implement two_stage")


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
