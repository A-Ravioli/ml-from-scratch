"""Checks for exercise.py (implicit feedback).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The metrics are
checked on hand examples and against brute-force loops; implicit ALS against a dense solve with
the full confidence matrix (and for not building one, by peak memory); BPR's step by hand and its
`fit` against the exact sequence the spec describes. The learning checks use a small clustered
interaction matrix with fixed seeds and wide margins, under `mlfs.testing.bounded`.
"""
import tracemalloc
from itertools import product

import numpy as np

from mlfs.testing import bounded


def _clustered(n_users=120, n_items=48, n_groups=4, p_in=0.5, p_out=0.03, seed=0):
    """users and items in groups; users mostly interact with their group's items. returns (train, test) counts."""
    g = np.random.default_rng(seed)
    ug, ig = np.arange(n_users) % n_groups, np.arange(n_items) % n_groups
    prob = np.where(ug[:, None] == ig[None, :], p_in, p_out)
    R = (g.random((n_users, n_items)) < prob) * g.integers(1, 5, (n_users, n_items))
    hide = (g.random(R.shape) < 0.25) & (R > 0)
    return np.where(hide, 0, R).astype(float), np.where(hide, R, 0).astype(float)


def _mean_auc(scores, train, test):
    """mean per-user AUC of held-out items against the other unseen items (ties 1/2)."""
    from scipy.stats import rankdata
    out = []
    for u in range(len(scores)):
        keep = train[u] == 0
        rel = test[u][keep] > 0
        if rel.all() or not rel.any():
            continue
        r = rankdata(scores[u][keep])
        n_pos, n_neg = rel.sum(), (~rel).sum()
        out.append((r[rel].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(out))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- stage 1: ranking metrics


def test_stage1_top_k(nb):
    s = np.array([[0.1, 0.5, 0.5, 0.9], [3.0, 2.0, 1.0, 0.0]])
    top = nb.top_k(s, 3)
    assert top.shape == (2, 3) and top.dtype == np.int64, f"top_k returns int64 (n_users, k); got {top.shape}, {top.dtype}"
    assert top.tolist() == [[3, 1, 2], [0, 1, 2]], (
        f"best first, ties to the lower item id: expected [[3, 1, 2], [0, 1, 2]], got {top.tolist()}")
    ex = np.array([[False, False, False, True], [True, False, False, False]])
    assert nb.top_k(s, 3, exclude=ex).tolist() == [[1, 2, 0], [1, 2, 3]], (
        "excluded items (exclude True, e.g. already seen in training) must never be recommended ahead of others")


def test_stage1_top_k_ties_on_long_rows(nb):
    g = np.random.default_rng(2)
    s = g.integers(0, 3, (5, 40)).astype(float)  # 40 items, many ties
    got = nb.top_k(s, 25)
    want = np.argsort(-s, axis=1, kind="stable")[:, :25]
    assert np.array_equal(got, want), (
        "with many tied scores, ties must go to the lower item id. NumPy's default argsort is only stable for short "
        "rows (below about 16 items): pass kind='stable'")


def test_stage1_recall_and_ndcg_by_hand(nb):
    top = np.array([[3, 1, 2], [0, 1, 2], [0, 1, 2]])
    rel = np.zeros((3, 5), dtype=bool)
    rel[0, [1, 2]] = True        # hits at positions 2 and 3
    rel[1, [0, 1, 2, 3, 4]] = True  # 5 relevant, k = 3, all hits
    rec = nb.recall_at_k(top, rel)
    assert np.allclose(rec[:2], [1.0, 1.0]) and np.isnan(rec[2]), (
        f"recall@k = hits / min(k, #relevant): user 0 has 2 of 2, user 1 has 3 hits out of min(3, 5); user 2 has "
        f"nothing relevant -> nan. got {rec.tolist()}")
    rel2 = rel.copy()
    rel2[0, 4] = True  # user 0 now has 3 relevant, 2 hits
    assert np.isclose(nb.recall_at_k(top, rel2)[0], 2 / 3), "2 hits, 3 relevant, k = 3 -> 2/3"
    nd = nb.ndcg_at_k(top, rel)
    want = (1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert np.isclose(nd[0], want) and np.isclose(nd[1], 1.0) and np.isnan(nd[2]), (
        f"user 0: hits at positions 2 and 3 give DCG = 1/log2(3) + 1/log2(4) = 1.131, the ideal with 2 relevant is "
        f"1 + 1/log2(3) = 1.631, so NDCG = {want:.4f}; user 1 is perfect (1.0); user 2 nan. got {nd.tolist()}")


def test_stage1_ndcg_matches_a_loop(nb):
    g = np.random.default_rng(0)
    scores = g.normal(size=(30, 40))
    rel = g.random((30, 40)) < 0.1
    top = np.argsort(-scores, axis=1)[:, :7]
    got = nb.ndcg_at_k(top, rel)
    for u in range(30):
        dcg = sum(1 / np.log2(p + 2) for p, it in enumerate(top[u]) if rel[u, it])
        idcg = sum(1 / np.log2(p + 2) for p in range(min(7, rel[u].sum())))
        want = dcg / idcg if idcg > 0 else np.nan
        assert np.isclose(got[u], want, equal_nan=True), f"user {u}: expected NDCG@7 {want:.4f}, got {got[u]:.4f}"


def test_stage1_auc(nb):
    s = np.array([[0.9, 0.1, 0.5, 0.5]])
    rel = np.array([[True, False, True, False]])
    got = nb.user_auc(s, rel)
    assert np.isclose(got[0], 0.875), (
        f"positives (0.9, 0.5) vs negatives (0.1, 0.5): 3 of 4 pairs correctly ordered plus a tie worth 1/2 -> "
        f"0.875; got {got[0]}")
    got2 = nb.user_auc(s, rel, exclude=np.array([[False, False, False, True]]))
    assert np.isclose(got2[0], 1.0), "with item 3 excluded the only negative is 0.1, below both positives: AUC 1"
    none = nb.user_auc(np.zeros((2, 3)), np.array([[False] * 3, [True] * 3]))
    assert np.isnan(none).all(), "AUC is undefined (nan) with no relevant or no irrelevant items"


def test_stage1_auc_matches_pair_counting(nb):
    g = np.random.default_rng(1)
    s = g.integers(0, 4, (12, 15)).astype(float)  # many ties on purpose
    rel = g.random((12, 15)) < 0.3
    ex = g.random((12, 15)) < 0.2
    got = nb.user_auc(s, rel, exclude=ex)
    for u in range(12):
        pos = [s[u, i] for i in range(15) if rel[u, i] and not ex[u, i]]
        neg = [s[u, i] for i in range(15) if not rel[u, i] and not ex[u, i]]
        if not pos or not neg:
            assert np.isnan(got[u])
            continue
        want = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in product(pos, neg)])
        assert np.isclose(got[u], want), f"user {u}: pair counting gives {want:.4f}, got {got[u]:.4f} (ties count 1/2)"


# ---------------------------------------------------------------- stage 2: implicit ALS


def test_stage2_loss_formula(nb):
    m = nb.ImplicitALS(rank=1, alpha=2.0, lam=0.5)
    m.user_factors_ = np.array([[1.0], [0.0]])
    m.item_factors_ = np.array([[0.5], [2.0]])
    R = np.array([[3.0, 0.0], [0.0, 1.0]])
    # predictions [[0.5, 2], [0, 0]], preferences [[1, 0], [0, 1]], confidences [[7, 1], [1, 3]]
    want = 7 * 0.25 + 1 * 4 + 1 * 0 + 3 * 1 + 0.5 * (1 + 0.25 + 4)
    got = m.loss(R)
    assert np.isclose(got, want), (
        f"loss sums c (p - x.y)^2 over EVERY cell, zeros included, plus lam (|X|^2 + |Y|^2): {want}; got {got}")
    assert np.allclose(m.scores(), [[0.5, 2.0], [0.0, 0.0]]), "scores() = X Y^T"


def test_stage2_init_order(nb):
    train, _ = _clustered(seed=4)
    m = nb.ImplicitALS(rank=3, n_iters=0, rng=6).fit(train)
    g = np.random.default_rng(6)
    P, Q = g.normal(0, 0.1, (120, 3)), g.normal(0, 0.1, (48, 3))
    assert np.allclose(m.user_factors_, P) and np.allclose(m.item_factors_, Q), (
        "with n_iters=0, fit only initialises: user_factors_ = rng.normal(0, 0.1, (n_users, rank)) FIRST, then "
        "item_factors_, from rng = as_rng(self.rng)")
    assert m.history_ == [], "no iterations, empty history_"


def test_stage2_monotone_and_exact_items(nb):
    train, _ = _clustered(seed=2)
    alpha, lam = 5.0, 0.8
    m = bounded(nb.ImplicitALS(rank=4, alpha=alpha, lam=lam, n_iters=5, rng=0).fit, train,
                budget=3_000_000, what="ImplicitALS.fit")
    h = np.asarray(m.history_)
    assert h.shape == (5,) and np.all(np.diff(h) <= 1e-8 * h[0]), (
        f"each half-step solves its block exactly, so the loss can't go up: history_ = {np.round(h, 3).tolist()}")
    assert np.isclose(h[-1], m.loss(train)), "history_[-1] equals loss(R) after fit"
    X = m.user_factors_
    worst = 0.0
    for i in (0, 13, 47):
        c = 1 + alpha * train[:, i]
        p = (train[:, i] > 0).astype(float)
        y = np.linalg.solve(X.T @ np.diag(c) @ X + lam * np.eye(4), X.T @ (c * p))
        worst = max(worst, np.abs(y - m.item_factors_[i]).max())
    assert worst < 1e-8, (
        f"after fit each item vector must solve (X^T C_i X + lam I) y = X^T C_i p_i exactly, with C_i the confidences "
        f"of ALL users (users first, items last in each iteration); max deviation {worst:.2e}")


def test_stage2_no_dense_confidence_matrix(nb):
    g = np.random.default_rng(3)
    R = (g.random((30, 3000)) < 0.01) * 2.0
    tracemalloc.start()
    try:
        bounded(nb.ImplicitALS(rank=8, n_iters=1, rng=0).fit, R, budget=3_000_000, what="ImplicitALS.fit")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 30e6, (
        f"fitting 30 users x 3,000 items peaked at {peak / 1e6:.0f} MB. building np.diag(c) for a user is a "
        "3000 x 3000 matrix (72 MB); use Y^T C_u Y = Y^T Y + Y^T (C_u - I) Y, where C_u - I is nonzero only on the "
        "user's few items")


def test_stage2_learns_the_clusters(nb):
    aucs = []
    for seed in range(3):
        train, test = _clustered(seed=10 + seed)
        m = bounded(nb.ImplicitALS(rank=4, alpha=10.0, lam=1.0, n_iters=8, rng=seed).fit, train,
                    budget=6_000_000, what="ImplicitALS.fit")
        aucs.append(_mean_auc(m.scores(), train, test))
    assert np.median(aucs) > 0.75, (
        f"on 4 user/item clusters implicit ALS should rank held-out items above the other unseen items (mean AUC "
        f"> 0.75; random is 0.5, knowing the clusters exactly gives about 0.85); got {np.round(aucs, 3).tolist()}")


# ---------------------------------------------------------------- stage 3: BPR


def _bpr_model(nb, P, Q, b, lr=0.1, lam=0.01):
    m = nb.BPR(rank=P.shape[1], lr=lr, lam=lam)
    m.user_factors_, m.item_factors_, m.item_bias_ = P.copy(), Q.copy(), b.copy()
    return m


def test_stage3_one_step_by_hand(nb):
    P = np.array([[1.0, 0.0]])
    Q = np.array([[9.0, 9.0], [0.5, 0.5], [-0.5, 1.0]])
    b = np.array([0.0, 0.2, -0.1])
    m = _bpr_model(nb, P, Q, b, lr=0.1, lam=0.01)
    ll = m.step(np.array([0]), np.array([1]), np.array([2]))
    x = 1.0 * (0.5 + 0.5) + 0.0 + 0.2 + 0.1  # = 1.3
    g = _sigmoid(-x)
    assert np.isclose(ll, np.log(_sigmoid(x))), f"step returns mean log sigmoid(x_uij) at the OLD values: x = 1.3 -> {np.log(_sigmoid(x)):.4f}; got {ll}"
    assert np.allclose(m.user_factors_[0], P[0] + 0.1 * (g * (Q[1] - Q[2]) - 0.01 * P[0])), (
        f"p_u += lr (g (q_i - q_j) - lam p_u) with g = sigmoid(-x) = {g:.4f}; got {m.user_factors_[0]}")
    assert np.allclose(m.item_factors_[1], Q[1] + 0.1 * (g * P[0] - 0.01 * Q[1])), "q_i += lr (g p_u - lam q_i), with the OLD p_u"
    assert np.allclose(m.item_factors_[2], Q[2] + 0.1 * (-g * P[0] - 0.01 * Q[2])), "q_j += lr (-g p_u - lam q_j)"
    assert np.allclose(m.item_bias_, [0.0, 0.2 + 0.1 * (g - 0.002), -0.1 + 0.1 * (-g + 0.001)]), (
        f"b_i += lr (g - lam b_i), b_j += lr (-g - lam b_j); got {m.item_bias_}")
    assert np.allclose(m.item_factors_[0], Q[0]), "items not in the batch must not change"


def test_stage3_repeated_ids_add_up(nb):
    g = np.random.default_rng(4)
    P, Q, b = g.normal(size=(2, 3)), g.normal(size=(5, 3)), g.normal(size=5)
    u, i, j = np.array([0, 0, 1]), np.array([1, 2, 1]), np.array([3, 3, 4])
    m = _bpr_model(nb, P, Q, b, lr=0.05, lam=0.1)
    m.step(u, i, j)
    dP, dQ, db = np.zeros_like(P), np.zeros_like(Q), np.zeros_like(b)
    for a, p, n in zip(u, i, j):
        gg = _sigmoid(-(P[a] @ (Q[p] - Q[n]) + b[p] - b[n]))
        dP[a] += gg * (Q[p] - Q[n]) - 0.1 * P[a]
        dQ[p] += gg * P[a] - 0.1 * Q[p]
        dQ[n] += -gg * P[a] - 0.1 * Q[n]
        db[p] += gg - 0.1 * b[p]
        db[n] += -gg - 0.1 * b[n]
    assert np.allclose(m.user_factors_, P + 0.05 * dP) and np.allclose(m.item_factors_, Q + 0.05 * dQ), (
        "when a user or item appears several times in a batch, its gradient contributions must ADD UP (all computed "
        "from the old values). `P[u] += ...` with repeated u keeps only one of them: use np.add.at")
    assert np.allclose(m.item_bias_, b + 0.05 * db), "item biases accumulate the same way"


def test_stage3_fit_follows_the_spec(nb):
    g0 = np.random.default_rng(5)
    R = (g0.random((6, 8)) < 0.3) * 1.0
    m = bounded(nb.BPR(rank=2, lr=0.1, lam=0.02, n_epochs=2, batch_size=4, rng=7).fit, R, budget=1_000_000, what="BPR.fit")
    g = np.random.default_rng(7)
    P, Q, b = g.normal(0, 0.1, (6, 2)), g.normal(0, 0.1, (8, 2)), np.zeros(8)
    pu, pi = np.nonzero(R > 0)
    hist = []
    for _ in range(2):
        perm = g.permutation(len(pu))
        tot = 0.0
        for s in range(0, len(perm), 4):
            bb = perm[s:s + 4]
            u, i, j = pu[bb], pi[bb], g.integers(8, size=len(bb))
            Pu, Qi, Qj, Bi, Bj = P[u].copy(), Q[i].copy(), Q[j].copy(), b[i].copy(), b[j].copy()
            x = np.sum(Pu * (Qi - Qj), 1) + Bi - Bj
            gg = _sigmoid(-x)[:, None]
            np.add.at(P, u, 0.1 * (gg * (Qi - Qj) - 0.02 * Pu))
            np.add.at(Q, i, 0.1 * (gg * Pu - 0.02 * Qi))
            np.add.at(Q, j, 0.1 * (-gg * Pu - 0.02 * Qj))
            np.add.at(b, i, 0.1 * (gg[:, 0] - 0.02 * Bi))
            np.add.at(b, j, 0.1 * (-gg[:, 0] - 0.02 * Bj))
            tot += np.sum(np.log(_sigmoid(x)))
        hist.append(tot / len(perm))
    dev = max(np.abs(m.user_factors_ - P).max(), np.abs(m.item_factors_ - Q).max())
    assert dev < 1e-10, (
        f"with the same seed your parameters should match the spec's sequence exactly (max deviation {dev:.2e}): "
        "user then item factors from rng.normal(0, 0.1, ...), item_bias_ zeros, positives from np.nonzero(R > 0); "
        "per epoch rng.permutation(n_positives), batches of batch_size in that order, and for each batch "
        "j = rng.integers(n_items, size=batch length)")
    assert np.allclose(m.history_, hist), (
        f"history_[e] is the mean log sigmoid(x_uij) over all of epoch e's triples (at the values used in each step); "
        f"expected {np.round(hist, 4).tolist()}, got {np.round(m.history_, 4).tolist()}")


def test_stage3_learns_to_rank(nb):
    aucs = []
    for seed in range(3):
        train, test = _clustered(seed=20 + seed)
        m = bounded(nb.BPR(rank=8, lr=0.05, lam=0.01, n_epochs=30, batch_size=64, rng=seed).fit, train,
                    budget=20_000_000, what="BPR.fit")
        aucs.append(_mean_auc(m.scores(), train, test))
    assert np.median(aucs) > 0.75, (
        f"on 4 user/item clusters BPR should rank held-out items above the other unseen items (mean AUC > 0.75; "
        f"random is 0.5, knowing the clusters exactly gives about 0.85); got {np.round(aucs, 3).tolist()}")


# ---------------------------------------------------------------- stage 4: baseline and protocol


def test_stage4_popularity(nb):
    R = np.array([[1.0, 0, 5], [0, 0, 2], [3, 0, 1]])
    s = nb.popularity_scores(R)
    assert s.shape == (3, 3) and np.allclose(s, [[2, 0, 3]] * 3), (
        f"popularity = number of USERS who interacted with each item (not the total count), the same for every user: "
        f"expected rows [2, 0, 3], got {s.tolist()}")


def test_stage4_evaluate_by_hand(nb):
    train = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=float)
    test = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=float)
    scores = np.array([[9, 8, 1, 2, 3], [5, 4, 3, 2, 1], [1, 9, 2, 3, 0]], dtype=float)
    out = nb.evaluate_topk(scores, train, test, k=2)
    # user 0: item 0 is seen -> top [1, 4], one hit at position 1; user 1 has no test items (skipped)
    # user 2: item 1 is seen -> top [3, 2], one hit of 2 at position 1
    assert out["n_users"] == 2, f"only users with at least one test item count: 2, got {out['n_users']}"
    assert np.isclose(out["recall"], (1.0 + 0.5) / 2), (
        f"user 0 finds 1 of 1, user 2 finds 1 of 2 (training items excluded first): mean recall 0.75, got {out['recall']}")
    want_ndcg = (1.0 + 1.0 / (1 + 1 / np.log2(3))) / 2
    assert np.isclose(out["ndcg"], want_ndcg), f"mean NDCG@2 should be {want_ndcg:.4f}, got {out['ndcg']}"
    assert np.isclose(out["coverage"], 4 / 5), (
        f"coverage = distinct items in the evaluated users' top-k lists / n_items = {{1, 4, 3, 2}} / 5 = 0.8; got {out['coverage']}")
