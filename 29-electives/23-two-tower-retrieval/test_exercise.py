"""Checks for exercise.py (two-tower retrieval).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The towers are
checked for shapes, unit norms and gradient flow; the in-batch loss against a brute-force
cross-entropy (with the logQ correction and duplicate masking); both training loops against an
exact replay of the spec on a copy of the same model; the index against brute-force search; and
the two-stage pipeline with a fake index and scorer. Learning checks use tiny clustered data with
fixed seeds, under `mlfs.testing.bounded`.
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _clusters(n_users=96, n_items=48, n_groups=4, per_user=8, seed=0):
    """users and items in groups; every user interacts with per_user random items of their group."""
    g = np.random.default_rng(seed)
    ug, ig = np.arange(n_users) % n_groups, np.arange(n_items) % n_groups
    users, items = [], []
    for u in range(n_users):
        pool = np.flatnonzero(ig == ug[u])
        users += [u] * per_user
        items += g.choice(pool, size=per_user, replace=False).tolist()
    uf = (np.eye(n_groups)[ug] + g.normal(0, 0.3, (n_users, n_groups))).astype(np.float32)
    itf = (np.eye(n_groups)[ig] + g.normal(0, 0.3, (n_items, n_groups))).astype(np.float32)
    return np.array(users), np.array(items), uf, itf, ug, ig


# ---------------------------------------------------------------- stage 1: towers


def test_stage1_tower_shapes_and_norms(nb):
    torch.manual_seed(0)
    t = nb.Tower(10, 3, dim=8, hidden=16)
    ids = torch.tensor([0, 1, 1, 9])
    f = torch.randn(4, 3)
    out = t(ids, f)
    assert out.shape == (4, 8), f"Tower(10, 3, dim=8) should map 4 rows to (4, 8), got {tuple(out.shape)}"
    assert torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-5), (
        f"every output row must have unit length (F.normalize); norms {out.norm(dim=1).tolist()}")
    n_params = sum(p.numel() for p in t.parameters())
    assert n_params >= 10 * 8 + 3 * 16 + 16 * 8, (
        "the tower needs an id embedding (n_ids x dim) and an MLP on the features (n_features -> hidden -> dim)")
    same_f = t(torch.tensor([2, 3]), torch.zeros(2, 3))
    assert not torch.allclose(same_f[0], same_f[1]), "two different ids with the same features must embed differently"
    same_id = t(torch.tensor([4, 4]), torch.tensor([[0.0, 0, 0], [3.0, -1, 2]]))
    assert not torch.allclose(same_id[0], same_id[1]), "the same id with different features must embed differently"
    out.sum().backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in t.parameters()]
    assert all(grads), "gradients must reach every parameter (the embedding AND the MLP)"


def test_stage1_id_embedding_init(nb):
    torch.manual_seed(0)
    t = nb.Tower(4000, 3, dim=16, hidden=8)
    W = [p for p in t.parameters() if p.dim() == 2 and p.shape == (4000, 16)]
    assert len(W) == 1, "the tower needs one embedding table of shape (n_ids, dim)"
    std, mean = W[0].detach().std().item(), W[0].detach().mean().item()
    assert abs(std - 0.05) < 0.003 and abs(mean) < 0.003, (
        f"initialise the id embedding N(0, 0.05^2) (nn.init.normal_(weight, 0.0, 0.05)); its entries have std "
        f"{std:.4f}. nn.Embedding's default is N(0, 1), which would swamp the feature MLP for items never trained")


def test_stage1_two_tower_logits(nb):
    torch.manual_seed(1)
    m = nb.TwoTower(20, 30, 5, 7, dim=8, hidden=16, temperature=0.05)
    assert isinstance(m.user_tower, torch.nn.Module) and isinstance(m.item_tower, torch.nn.Module)
    assert m.temperature == 0.05
    u, i = torch.tensor([0, 5, 19]), torch.tensor([29, 0, 3])
    uf, itf = torch.randn(3, 5), torch.randn(3, 7)
    logits = m(u, uf, i, itf)
    assert logits.shape == (3, 3), f"logits are (B, B): every user of the batch against every item; got {tuple(logits.shape)}"
    want = m.user_tower(u, uf) @ m.item_tower(i, itf).T / 0.05
    assert torch.allclose(logits, want, atol=1e-5), "logits[b, b'] = user_tower(b) . item_tower(b') / temperature"
    assert logits.abs().max() <= 1 / 0.05 + 1e-4, "unit vectors: |logit| can't exceed 1 / temperature"


# ---------------------------------------------------------------- stage 2: the in-batch loss


def test_stage2_item_log_frequency(nb):
    got = nb.item_log_frequency(np.array([0, 0, 1, 3]), 5)
    assert isinstance(got, torch.Tensor) and got.shape == (5,) and got.dtype == torch.float32, (
        "return a float32 tensor with one entry per item")
    want = np.log(np.array([2, 1, 1, 1, 1]) / 4)
    assert np.allclose(got.numpy(), want), (
        f"log(count / total) with counts clipped at 1 (items never seen count as 1): {np.round(want, 4).tolist()}, "
        f"got {np.round(got.numpy(), 4).tolist()}")


def test_stage2_loss_matches_cross_entropy(nb):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 6, generator=g)
    ids = torch.tensor([10, 11, 12, 13, 14, 15])
    got = nb.in_batch_softmax_loss(logits, ids)
    want = F.cross_entropy(logits, torch.arange(6))
    assert torch.isclose(got, want, atol=1e-6), (
        f"with distinct items the loss is the mean cross-entropy with the diagonal as targets: {want:.5f}, got {float(got):.5f}")
    log_q = torch.log(torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
    got_q = nb.in_batch_softmax_loss(logits, ids, log_q=log_q)
    want_q = F.cross_entropy(logits - log_q[None, :], torch.arange(6))
    assert torch.isclose(got_q, want_q, atol=1e-6), (
        f"the logQ correction subtracts log_q[j] from COLUMN j (the item side): expected {want_q:.5f}, got "
        f"{float(got_q):.5f}. (subtracting it per row changes nothing: a constant per row cancels in the softmax)")


def test_stage2_duplicates_are_masked(nb):
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(4, 4, generator=g, requires_grad=True)
    ids = torch.tensor([5, 7, 5, 2])
    got = nb.in_batch_softmax_loss(logits, ids)
    masked = logits.detach().clone()
    masked[0, 2] = masked[2, 0] = float("-inf")
    want = F.cross_entropy(masked, torch.arange(4))
    assert torch.isclose(got, want, atol=1e-6), (
        f"rows 0 and 2 have the same positive item (5), so column 2 must not count as a negative for row 0, nor "
        f"column 0 for row 2 (mask them to -inf, but never the diagonal): expected {want:.5f}, got {float(got):.5f}")
    got.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), "the loss must be differentiable, with finite gradients"


# ---------------------------------------------------------------- stage 3: training and export


def _replay(model, users, items, uf, itf, epochs, batch_size, lr, logq, seed):
    users, items = torch.as_tensor(users), torch.as_tensor(items)
    uf, itf = torch.as_tensor(uf), torch.as_tensor(itf)
    counts = np.bincount(items.numpy(), minlength=itf.shape[0]).astype(float)
    lq = torch.tensor(np.log(np.maximum(counts, 1) / len(items)), dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    history = []
    for _ in range(epochs):
        perm = torch.randperm(len(users), generator=gen)
        losses = []
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            u, i = users[b], items[b]
            logits = model.user_tower(u, uf[u]) @ model.item_tower(i, itf[i]).T / model.temperature
            if logq:
                logits = logits - lq[i][None, :]
            dup = (i[None, :] == i[:, None]) & ~torch.eye(len(i), dtype=torch.bool)
            loss = F.cross_entropy(logits.masked_fill(dup, float("-inf")), torch.arange(len(i)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    return history


def test_stage3_training_follows_the_spec(nb):
    users, items, uf, itf, _, _ = _clusters(n_users=24, n_items=12, per_user=3, seed=1)
    users, items = np.r_[users, np.arange(12)], np.r_[items, np.zeros(12, dtype=int)]  # item 0 is popular
    for logq in (True, False):
        torch.manual_seed(0)
        a = nb.TwoTower(24, 12, 4, 4, dim=8, hidden=16)
        b = copy.deepcopy(a)
        hist = bounded(nb.train_two_tower, a, users, items, uf, itf, epochs=2, batch_size=42, lr=1e-2,
                       logq_correction=logq, seed=3, budget=5_000_000, what="train_two_tower")
        want_hist = _replay(b, users, items, uf, itf, 2, 42, 1e-2, logq, 3)
        dev = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
        assert dev < 1e-5, (
            f"logq_correction={logq}: after 2 epochs your parameters differ from the spec's replay by {dev:.2e}. the spec: "
            "Adam(model.parameters(), lr); each epoch torch.randperm(n_pairs, generator=torch.Generator().manual_seed(seed)) "
            "(one generator for the whole run); consecutive batches; the in-batch loss with log_q = "
            "item_log_frequency(all training items)[batch items] when logq_correction")
        assert np.allclose(hist, want_hist, atol=1e-5), (
            f"return the mean batch loss of each epoch: expected {np.round(want_hist, 4).tolist()}, got {np.round(hist, 4).tolist()}")


def test_stage3_learns_the_clusters(nb):
    wins = []
    for seed in range(3):
        users, items, uf, itf, ug, ig = _clusters(seed=10 + seed)
        torch.manual_seed(seed)
        m = nb.TwoTower(96, 48, 4, 4, dim=16, hidden=32)
        hist = bounded(nb.train_two_tower, m, users, items, uf, itf, epochs=15, batch_size=32, lr=1e-2,
                       logq_correction=True, seed=seed, budget=40_000_000, what="train_two_tower")
        U, V = nb.embed_all(m, uf, itf)
        same = (ug[:, None] == ig[None, :])
        S = U @ V.T
        # fraction of each user's top-5 items that come from the user's own group (random: 0.25)
        top = np.argsort(-S, axis=1)[:, :5]
        wins.append(np.take_along_axis(same, top, axis=1).mean())
        assert hist[-1] < hist[0], f"the training loss should fall; got {np.round(hist, 3).tolist()}"
    assert np.median(wins) > 0.8, (
        f"after training, most of each user's top-5 items should come from their own cluster (random would be 25%); "
        f"got {np.round(wins, 3).tolist()}")


def test_stage3_embed_all(nb):
    users, items, uf, itf, _, _ = _clusters(n_users=10, n_items=8, per_user=2)
    torch.manual_seed(0)
    m = nb.TwoTower(10, 8, 4, 4, dim=8, hidden=16)
    m.train()
    U, V = nb.embed_all(m, uf, itf)
    assert isinstance(U, np.ndarray) and U.shape == (10, 8) and V.shape == (8, 8) and U.dtype == np.float32, (
        f"embed_all returns float32 ndarrays (n_users, dim) and (n_items, dim); got {type(U)}, {getattr(U, 'shape', None)}")
    assert np.allclose(np.linalg.norm(V, axis=1), 1, atol=1e-5), "item embeddings are unit vectors"
    with torch.no_grad():
        want = m.item_tower(torch.arange(8), torch.as_tensor(itf)).numpy()
    assert np.allclose(V, want, atol=1e-6), "row i of the item embeddings is item_tower(i, item_feats[i])"
    assert not m.training, "embed_all should leave the model in eval mode"


# ---------------------------------------------------------------- stage 4: nearest-neighbour search


def test_stage4_exact_search(nb):
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = np.array([[0.5, 0.5], [1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [-1.0, 0.0]])
    ids, scores = nb.exact_search(q, v, 3)
    assert ids.dtype == np.int64 and ids.shape == (2, 3)
    assert ids.tolist() == [[1, 2, 0], [0, 3, 2]], (
        f"top-3 inner products, best first, ties to the lower id: expected [[1, 2, 0], [0, 3, 2]], got {ids.tolist()}")
    assert np.allclose(scores, [[1.0, 0.9, 0.5], [0.5, 0.5, 0.1]]), f"scores are the inner products; got {scores.tolist()}"


def test_stage4_full_probe_is_exact(nb):
    g = np.random.default_rng(0)
    v = g.normal(size=(400, 8))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = g.normal(size=(20, 8))
    idx = bounded(nb.IVFIndex(10, n_probe=10, rng=0).build, v, budget=5_000_000, what="IVFIndex.build")
    assert idx.centroids_.shape == (10, 8), f"centroids_ is (n_lists, dim); got {idx.centroids_.shape}"
    assert sorted(np.concatenate(idx.lists_).tolist()) == list(range(400)), "lists_ must partition the item ids"
    ids, scores = bounded(idx.search, q, 5, budget=5_000_000, what="IVFIndex.search")
    want_ids, want_scores = nb.exact_search(q, v, 5)
    assert np.array_equal(ids, want_ids) and np.allclose(scores, want_scores), (
        "probing every list scans every item, so the result must equal exact_search")
    assert idx.last_n_scored_ == 400, f"last_n_scored_ is the mean number of items scored per query: 400, got {idx.last_n_scored_}"


def test_stage4_few_probes_on_clustered_data(nb):
    g = np.random.default_rng(1)
    centers = g.normal(size=(8, 16)) * 3
    v = centers[np.arange(800) % 8] + g.normal(size=(800, 16))
    q = centers[np.arange(40) % 8] + g.normal(size=(40, 16))
    idx = bounded(nb.IVFIndex(8, n_probe=2, rng=0).build, v, budget=5_000_000, what="IVFIndex.build")
    ids, _ = idx.search(q, 10)
    want, _ = nb.exact_search(q, v, 10)
    recall = np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(ids, want)])
    assert recall > 0.85, (
        f"on 8 well-separated clusters, probing the 2 nearest of 8 lists should find > 85% of the true top 10; got "
        f"{recall:.2f}. probe the lists whose centroids are CLOSEST to the query")
    assert 100 <= idx.last_n_scored_ <= 400, (
        f"two probes out of 8 lists should score about 200 of 800 items per query; got {idx.last_n_scored_}")


def test_stage4_build_uses_kmeans(nb):
    from mlfs.unsupervised import KMeans
    g = np.random.default_rng(3)
    v = g.normal(size=(300, 6))
    idx = nb.IVFIndex(7, n_probe=2, rng=5).build(v)
    km = KMeans(7, n_init=1, max_iters=25, rng=5).fit(v)
    assert np.allclose(idx.centroids_, km.cluster_centers_), (
        "build runs KMeans(n_lists, n_init=1, max_iters=25, rng=self.rng) on the vectors; centroids_ = cluster_centers_")
    assert all(np.array_equal(np.sort(l), np.flatnonzero(km.labels_ == c)) for c, l in enumerate(idx.lists_)), (
        "lists_[c] holds the ids of the vectors k-means put in cell c")


def test_stage4_probes_by_distance_and_breaks_ties_by_id(nb):
    v = np.array([[1.0, 0.0], [0.6, 0.8], [0.6, 0.8], [0.0, 1.0], [0.8, 0.6]])
    idx = nb.IVFIndex(2, n_probe=1, rng=0).build(v)
    # replace the cells by hand: a far, large centroid with items {0, 3} and a near, small one with {4, 2, 1}
    idx.centroids_ = np.array([[3.0, 0.0], [0.7, 0.6]])
    idx.lists_ = [np.array([3, 0]), np.array([4, 2, 1])]
    ids, scores = idx.search(np.array([[0.8, 0.6]]), 3)
    assert set(ids[0].tolist()) == {1, 2, 4}, (
        f"the query (0.8, 0.6) has the larger inner product with centroid (3, 0) but is much CLOSER (Euclidean) to "
        f"(0.7, 0.6): probe by distance, so the candidates are items 4, 2, 1; got {ids.tolist()}")
    assert ids[0].tolist() == [4, 1, 2], (
        f"items 1 and 2 tie (same vector): ties go to the lower id whatever order the list stores them in; "
        f"expected [4, 1, 2], got {ids[0].tolist()}")


def test_stage4_padding(nb):
    v = np.eye(4)
    idx = nb.IVFIndex(4, n_probe=1, rng=0).build(v)
    ids, scores = idx.search(np.array([[1.0, 0, 0, 0]]), 3)
    assert ids[0, 0] == 0 and (ids[0, 1:] == -1).all() and np.isneginf(scores[0, 1:]).all(), (
        f"with 4 single-item lists and one probe there is one candidate: the rest of the k slots are id -1, score -inf; "
        f"got ids {ids.tolist()}, scores {scores.tolist()}")


# ---------------------------------------------------------------- stage 5: the second stage


def test_stage5_ranker_features(nb):
    u = np.array([[1.0, 0.0], [0.6, 0.8]])
    v = np.array([[0.0, 1.0], [0.6, 0.8]])
    uf = np.array([[9.0, 0.5], [9.0, -1.0]])   # last column: budget
    itf = np.array([[7.0, 7.0, 1.0], [7.0, 7.0, -2.0]])  # last column: price
    got = nb.ranker_features(u, v, uf, itf)
    want = np.array([[0.0, 0.5, 1.0, 0.5], [1.0, -1.0, -2.0, -1.0]])
    assert got.shape == (2, 4) and got.dtype == np.float32, f"(B, 4) float32, got {got.shape} {got.dtype}"
    assert np.allclose(got, want), (
        f"columns [u . v, price - budget, price, budget] with budget = user_feats[:, -1], price = item_feats[:, -1]: "
        f"expected {want.tolist()}, got {got.tolist()}")


def test_stage5_ranker_shapes(nb):
    torch.manual_seed(0)
    r = nb.Ranker(4, hidden=8)
    assert r(torch.randn(5, 4)).shape == (5,), "Ranker maps (B, 4) to scores (B,)"
    assert r(torch.randn(3, 7, 4)).shape == (3, 7), "and (n, L, 4) lists to (n, L)"
    n = sum(p.numel() for p in r.parameters())
    assert n == (4 * 8 + 8) + (8 * 8 + 8) + (8 + 1), f"an MLP 4 -> 8 -> 8 -> 1 has 129 parameters; got {n} (two hidden layers)"


def test_stage5_train_ranker_follows_the_spec_and_learns(nb):
    g = np.random.default_rng(0)
    X = g.normal(size=(300, 6, 4)).astype(np.float32)
    X[:, 0, 1] -= 3.0  # the positive (list item 0) is much more affordable
    torch.manual_seed(0)
    a = nb.Ranker(4, hidden=16)
    b = copy.deepcopy(a)
    hist = bounded(nb.train_ranker, a, X, epochs=3, batch_size=64, lr=1e-2, seed=4, budget=5_000_000, what="train_ranker")
    gen = torch.Generator().manual_seed(4)
    opt = torch.optim.Adam(b.parameters(), lr=1e-2)
    Xt = torch.tensor(X)
    want_hist = []
    for _ in range(3):
        perm = torch.randperm(300, generator=gen)
        tot = 0.0
        for s in range(0, 300, 64):
            bb = perm[s:s + 64]
            loss = F.cross_entropy(b(Xt[bb]), torch.zeros(len(bb), dtype=torch.int64))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bb)
        want_hist.append(tot / 300)
    dev = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
    assert dev < 1e-5, (
        f"your ranker's parameters differ from the spec's replay by {dev:.2e}: Adam(lr), one torch.Generator seeded "
        "with seed, torch.randperm(n_lists) per epoch, consecutive batches, cross-entropy of each list's scores "
        "with target 0 (the positive)")
    assert np.allclose(hist, want_hist, atol=1e-5), (
        f"return each epoch's mean loss weighted by batch size (the last batch is smaller): expected "
        f"{np.round(want_hist, 4).tolist()}, got {np.round(hist, 4).tolist()}")
    assert hist[-1] < hist[0], f"the loss should fall; got {np.round(hist, 3).tolist()}"
    assert not a.training, "train_ranker must leave the ranker in eval mode (call ranker.eval() at the end)"
    with torch.no_grad():
        acc = (a(Xt).argmax(dim=1) == 0).float().mean().item()
    assert acc > 0.6, f"the positive should be the top-scored item in most lists after training (random: 1/6); got {acc:.2f}"


class _FakeIndex:
    def __init__(self, cand):
        self.cand, self.calls = cand, []

    def search(self, queries, k):
        self.calls.append((np.asarray(queries).copy(), k))
        return self.cand[:, :k], -np.arange(k, dtype=float)[None].repeat(len(self.cand), 0)


def test_stage5_two_stage_ties_keep_candidate_order(nb):
    cand = np.array([np.random.default_rng(0).permutation(40)])
    top = nb.two_stage(np.array([0]), np.zeros((1, 2)), _FakeIndex(cand), lambda u, c: np.zeros(c.shape),
                       n_candidates=40, k=20)
    assert top[0].tolist() == cand[0, :20].tolist(), (
        "when the ranker's scores tie, keep the retrieval order: use a stable sort (NumPy's default argsort "
        "reorders ties in rows longer than about 16)")


def test_stage5_two_stage(nb):
    user_emb = np.arange(12, dtype=float).reshape(4, 3)
    cand = np.array([[5, 3, 8, -1, 9], [1, 2, 3, 4, 6]])
    index = _FakeIndex(cand)
    exclude = np.zeros((4, 10), dtype=bool)
    exclude[2, 8] = True   # user 2 already has item 8
    exclude[3, 2] = True

    def score_fn(users, c):
        assert c.shape == (2, 4), f"score_fn should receive the (Q, n_candidates) candidate ids; got {c.shape}"
        # item 3 is the ranker's favourite, then padding (which must be dropped anyway), then larger ids
        return np.where(c == 3, 100.0, np.where(c < 0, 50.0, c.astype(float)))

    top = nb.two_stage(np.array([2, 3]), user_emb, index, score_fn, n_candidates=4, k=2, exclude=exclude)
    q, k = index.calls[0]
    assert k == 4 and np.allclose(q, user_emb[[2, 3]]), (
        "retrieve with index.search(user_emb[user_ids], n_candidates)")
    assert top.tolist() == [[3, 5], [3, 4]], (
        f"rescore the candidates with score_fn, drop excluded items and padding (-1), and return the top k by score: "
        f"user 2's candidates (5, 3, 8, -1) minus excluded 8 and padding give [3, 5]; user 3's (1, 2, 3, 4) minus 2 give "
        f"[3, 4]. got {top.tolist()}")
