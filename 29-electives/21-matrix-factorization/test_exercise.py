"""Checks for exercise.py (matrix factorization).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The data are small
synthetic low-rank rating matrices built here with fixed seeds. The closed-form pieces (bias
updates, ALS block solves, one SGD step, fold-in) are checked exactly against hand computations
or NumPy; the learning checks compare test RMSE with the bias baseline with wide margins, and
every training call runs under `mlfs.testing.bounded` so a runaway loop fails instead of hanging.
"""
import numpy as np

from mlfs.testing import bounded


def _lowrank(n_users=60, n_items=50, rank=3, density=0.3, noise=0.1, seed=0):
    """observed triples from mu + b_u + b_i + U V^T + noise, every user and item seen at least twice."""
    g = np.random.default_rng(seed)
    U, V = g.normal(0, 0.7, (n_users, rank)), g.normal(0, 0.7, (n_items, rank))
    bu, bi = g.normal(0, 0.3, n_users), g.normal(0, 0.3, n_items)
    full = 3.0 + bu[:, None] + bi[None, :] + U @ V.T
    mask = g.random((n_users, n_items)) < density
    mask[np.arange(n_users), g.integers(n_items, size=n_users)] = True
    mask[np.arange(n_users), g.integers(n_items, size=n_users)] = True
    mask[g.integers(n_users, size=n_items), np.arange(n_items)] = True
    mask[g.integers(n_users, size=n_items), np.arange(n_items)] = True
    u, i = np.nonzero(mask)
    r = full[u, i] + g.normal(0, noise, len(u))
    return u, i, r, full


def _holdout(u, i, r, seed=1, frac=0.2):
    g = np.random.default_rng(seed)
    t = g.random(len(r)) < frac
    # keep at least one training rating per user and item
    for key in (u, i):
        for k in np.unique(key):
            idx = np.flatnonzero(key == k)
            if t[idx].all():
                t[idx[0]] = False
    return (u[~t], i[~t], r[~t]), (u[t], i[t], r[t])


def _bias_rmse(train, test, lam=5.0, n_iters=30):
    (u, i, r), (tu, ti, tr) = train, test
    nu, ni = max(u.max(), tu.max()) + 1, max(i.max(), ti.max()) + 1
    mu, bu, bi = r.mean(), np.zeros(nu), np.zeros(ni)
    for _ in range(n_iters):
        bi = np.bincount(i, r - mu - bu[u], ni) / (lam + np.bincount(i, minlength=ni))
        bu = np.bincount(u, r - mu - bi[i], nu) / (lam + np.bincount(u, minlength=nu))
    return np.sqrt(np.mean((mu + bu[tu] + bi[ti] - tr) ** 2))


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---------------------------------------------------------------- stage 1: split, metric, baseline


def test_stage1_split_is_a_partition(nb):
    n = 2000
    users, items = np.repeat(np.arange(40), 50), np.tile(np.arange(50), 40)
    ratings = (users * 7 + items * 3) % 5 + 1.0  # a rating we can recompute from (u, i)
    train, test = nb.split_ratings(users, items, ratings, test_frac=0.25, rng=0)
    assert len(train) == 3 and len(test) == 3, "return two tuples, (users, items, ratings) each"
    assert len(train[0]) + len(test[0]) == n, (
        f"train and test together must hold all {n} ratings exactly once; got {len(train[0])} + {len(test[0])}")
    pairs = set(zip(train[0].tolist(), train[1].tolist())) | set(zip(test[0].tolist(), test[1].tolist()))
    assert len(pairs) == n, "every (user, item) pair must land in exactly one of train and test"
    for u, i, r in (train, test):
        assert np.allclose(r, (u * 7 + i * 3) % 5 + 1.0), (
            "the three arrays in each part must stay aligned: index the users, items and ratings with the SAME mask")
    frac = len(test[0]) / n
    assert abs(frac - 0.25) < 5 * np.sqrt(0.25 * 0.75 / n), f"test_frac=0.25 should put about 25% in test, got {frac:.3f}"


def test_stage1_split_is_seeded(nb):
    u, i, r = np.arange(500), np.arange(500), np.ones(500)
    a = nb.split_ratings(u, i, r, test_frac=0.3, rng=4)[1][0]
    b = nb.split_ratings(u, i, r, test_frac=0.3, rng=4)[1][0]
    c = nb.split_ratings(u, i, r, test_frac=0.3, rng=5)[1][0]
    assert np.array_equal(a, b), "the same seed must give the same split (use as_rng(rng), not global state)"
    assert not np.array_equal(a, c), "different seeds should give different splits"


def test_stage1_rmse(nb):
    got = nb.rmse(np.array([3.0, 4.0, 5.0]), np.array([4.0, 4.0, 3.0]))
    assert np.isclose(got, np.sqrt(5 / 3)), f"rmse((3,4,5), (4,4,3)) = sqrt((1+0+4)/3) = 1.291, got {got}"
    assert isinstance(got, float), "return a plain float"


def test_stage1_bias_hand_example(nb):
    # the lesson's worked example: u0 rates i0 = 5, i1 = 3; u1 rates i0 = 4. mu = 4, lam = 1, one sweep
    m = nb.BiasBaseline(lam=1.0, n_iters=1).fit(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([5.0, 3.0, 4.0]))
    assert np.isclose(m.mu_, 4.0), f"mu_ is the mean training rating, 4.0; got {m.mu_}"
    assert np.allclose(m.item_bias_, [1 / 3, -0.5]), (
        f"items first: b_i0 = ((5-4) + (4-4)) / (1 + 2) = 1/3 and b_i1 = (3-4) / (1 + 1) = -0.5; got {m.item_bias_}. "
        "(did you update users first, or forget lam in the denominator?)")
    assert np.allclose(m.user_bias_, [(2 / 3 - 0.5) / 3, -1 / 6]), (
        f"then users, using the NEW item biases: b_u0 = ((5-4-1/3) + (3-4+0.5)) / 3 = 0.0556, "
        f"b_u1 = (4-4-1/3) / 2 = -0.1667; got {m.user_bias_}")
    assert np.allclose(m.predict(np.array([1]), np.array([1])), [4 - 1 / 6 - 0.5]), "predict = mu + b_u + b_i"


def test_stage1_bias_converges_to_ridge_solution(nb):
    u, i, r, _ = _lowrank(n_users=30, n_items=25, seed=3)
    lam = 2.0
    m = bounded(nb.BiasBaseline(lam=lam, n_iters=300).fit, u, i, r, budget=3_000_000, what="BiasBaseline.fit")
    # the exact minimiser of sum (r - mu - b_u - b_i)^2 + lam (|b_u|^2 + |b_i|^2), mu fixed at the mean
    X = np.zeros((len(r), 30 + 25))
    X[np.arange(len(r)), u] = 1
    X[np.arange(len(r)), 30 + i] = 1
    w = np.linalg.solve(X.T @ X + lam * np.eye(55), X.T @ (r - r.mean()))
    assert np.allclose(m.user_bias_, w[:30], atol=1e-6) and np.allclose(m.item_bias_, w[30:], atol=1e-6), (
        "after many sweeps the alternating updates must reach the exact ridge solution; max deviation "
        f"{max(np.abs(m.user_bias_ - w[:30]).max(), np.abs(m.item_bias_ - w[30:]).max()):.2e}. each update must "
        "subtract the CURRENT other-side biases and divide by (lam + count)")


def test_stage1_bias_lam_zero_unseen(nb):
    m = nb.BiasBaseline(lam=0.0, n_iters=2).fit(np.array([0, 1]), np.array([0, 1]), np.array([4.0, 2.0]),
                                                n_users=3, n_items=3)
    assert np.all(np.isfinite(m.user_bias_)) and np.all(np.isfinite(m.item_bias_)), (
        "with lam = 0, users/items with no ratings would divide 0 by 0: their bias must still be 0, not nan")
    assert m.user_bias_[2] == 0 and m.item_bias_[2] == 0


def test_stage1_bias_sizes_and_unseen(nb):
    m = nb.BiasBaseline(lam=1.0, n_iters=3).fit(np.array([0, 1]), np.array([0, 1]), np.array([4.0, 2.0]),
                                                n_users=5, n_items=4)
    assert m.user_bias_.shape == (5,) and m.item_bias_.shape == (4,), (
        "n_users / n_items set the array sizes (users and items that never appear in training exist too)")
    assert np.allclose(m.user_bias_[2:], 0) and np.allclose(m.item_bias_[2:], 0), "unseen users/items have bias 0"
    assert np.allclose(m.predict(np.array([4]), np.array([3])), [3.0]), "an unseen user and item get exactly mu"


# ---------------------------------------------------------------- stage 2: ALS


def test_stage2_attributes_and_history(nb):
    u, i, r, _ = _lowrank(seed=0)
    m = bounded(nb.ALS(rank=4, lam=1.0, n_iters=6, rng=0).fit, u, i, r, budget=5_000_000, what="ALS.fit")
    assert m.user_factors_.shape == (60, 4) and m.item_factors_.shape == (50, 4), "factors are (n_users, rank), (n_items, rank)"
    assert m.user_bias_.shape == (60,) and m.item_bias_.shape == (50,) and np.isclose(m.mu_, r.mean())
    h = np.asarray(m.history_)
    assert h.shape == (6,), f"history_ holds one objective value per iteration (6), got {h.shape}"
    assert np.all(np.diff(h) <= 1e-8 * h[0]), (
        f"each ALS half-step minimises the objective exactly over one block, so it can never go up; history_ = "
        f"{np.round(h, 3).tolist()}. (regularizing with a different lam than the objective, or a stale bias in "
        "the targets, breaks this)")
    assert np.isclose(h[-1], m.loss(u, i, r)), "history_[-1] must equal loss() on the training data after fitting"


def test_stage2_loss_formula(nb):
    m = nb.ALS(rank=2, lam=0.5)
    m.mu_ = 3.0
    m.user_factors_, m.item_factors_ = np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 1.0]])
    m.user_bias_, m.item_bias_ = np.array([0.5, 0.0]), np.array([-1.0])
    # predictions: user 0 -> 3 + 0.5 - 1 + 1 = 3.5, user 1 -> 3 + 0 - 1 + 1 = 3
    got = m.loss(np.array([0, 1]), np.array([0, 0]), np.array([4.0, 3.0]))
    want = 0.25 + 0.0 + 0.5 * (2 + 2 + 0.25 + 1)
    assert np.isclose(got, want), (
        f"loss = SSE + lam (|P|^2 + |Q|^2 + |b_u|^2 + |b_i|^2) = 0.25 + 0.5 * 5.25 = {want}; got {got}")
    assert np.allclose(m.predict(np.array([0, 1]), np.array([0, 0])), [3.5, 3.0]), "predict = mu + b_u + b_i + p.q"


def test_stage2_users_are_exact_ridge_solutions(nb):
    u, i, r, _ = _lowrank(seed=1)
    lam = 0.7
    m = bounded(nb.ALS(rank=3, lam=lam, n_iters=4, rng=1).fit, u, i, r, budget=5_000_000, what="ALS.fit")
    worst = 0.0
    for user in (0, 7, 33):
        idx = u == user
        X = np.c_[m.item_factors_[i[idx]], np.ones(idx.sum())]
        y = r[idx] - m.mu_ - m.item_bias_[i[idx]]
        w = np.linalg.solve(X.T @ X + lam * np.eye(4), X.T @ y)
        worst = max(worst, np.abs(w[:3] - m.user_factors_[user]).max(), abs(w[3] - m.user_bias_[user]))
    assert worst < 1e-8, (
        f"after fit, each user's [p_u, b_u] must be the exact ridge solution given the final item factors and "
        f"biases (users are solved LAST in each iteration); max deviation {worst:.2e}. regress r - mu - b_i on "
        "[q_i, 1] with lam * I on all rank + 1 coefficients")


def test_stage2_init_order(nb):
    u, i, r, _ = _lowrank(seed=8)
    m = nb.ALS(rank=3, lam=1.0, n_iters=0, rng=11).fit(u, i, r)
    g = np.random.default_rng(11)
    P, Q = g.normal(0, 0.1, (60, 3)), g.normal(0, 0.1, (50, 3))
    assert np.allclose(m.user_factors_, P) and np.allclose(m.item_factors_, Q), (
        "with n_iters=0, fit only initialises: user_factors_ = rng.normal(0, 0.1, (n_users, rank)) FIRST, then "
        "item_factors_ the same way, from rng = as_rng(self.rng)")
    assert np.allclose(m.user_bias_, 0) and np.allclose(m.item_bias_, 0) and m.history_ == [], "biases start at 0, history_ empty"


def test_stage2_items_are_exact_ridge_solutions(nb):
    u, i, r, _ = _lowrank(seed=9)
    lam = 0.9
    prev = bounded(nb.ALS(rank=3, lam=lam, n_iters=2, rng=4).fit, u, i, r, budget=5_000_000, what="ALS.fit")
    m = bounded(nb.ALS(rank=3, lam=lam, n_iters=3, rng=4).fit, u, i, r, budget=5_000_000, what="ALS.fit")
    # the same seed makes the 2-iteration fit a prefix of the 3-iteration one: in iteration 3 the items are solved
    # against the users of `prev`
    worst = 0.0
    for item in (0, 11, 49):
        idx = i == item
        X = np.c_[prev.user_factors_[u[idx]], np.ones(idx.sum())]
        y = r[idx] - prev.mu_ - prev.user_bias_[u[idx]]
        w = np.linalg.solve(X.T @ X + lam * np.eye(4), X.T @ y)
        worst = max(worst, np.abs(w[:3] - m.item_factors_[item]).max(), abs(w[3] - m.item_bias_[item]))
    assert worst < 1e-8, (
        f"each item's [q_i, b_i] must be the exact ridge solution of r - mu - b_u on [p_u, 1] (lam I on all rank + 1 "
        f"coefficients), using the user factors and biases from the end of the previous iteration; max deviation "
        f"{worst:.2e}. (did you drop the item bias column, or forget to subtract b_u from the target?)")


def test_stage2_recovers_a_low_rank_matrix(nb):
    g = np.random.default_rng(2)
    full = 3 + g.normal(0, 1, (20, 2)) @ g.normal(0, 1, (2, 15))
    u, i = np.nonzero(np.ones((20, 15)))
    m = bounded(nb.ALS(rank=3, lam=1e-6, n_iters=60, rng=0).fit, u, i, full[u, i], budget=20_000_000, what="ALS.fit")
    err = _rmse(m.predict(u, i), full[u, i])
    assert err < 1e-3, f"a fully observed rank-2 matrix should be fitted almost exactly by rank-3 ALS; train RMSE {err:.2e}"


def test_stage2_beats_the_bias_baseline(nb):
    gaps = []
    for seed in range(3):
        u, i, r, _ = _lowrank(seed=10 + seed, noise=0.2)
        train, test = _holdout(u, i, r, seed=seed)
        m = bounded(nb.ALS(rank=3, lam=2.0, n_iters=10, rng=seed).fit, *train, n_users=60, n_items=50,
                    budget=8_000_000, what="ALS.fit")
        gaps.append(_bias_rmse(train, test) - _rmse(m.predict(test[0], test[1]), test[2]))
    assert np.median(gaps) > 0.15, (
        f"on rank-3 data ALS (rank 3, lam 2) should beat the bias-only baseline's test RMSE by well over 0.15; "
        f"improvements {np.round(gaps, 3).tolist()}")


# ---------------------------------------------------------------- stage 3: SGD


def test_stage3_one_step_by_hand(nb):
    m = nb.SGDMF(rank=2, lr=0.1, lam=0.5, n_epochs=0, rng=0).fit(np.array([0]), np.array([0]), np.array([4.0]))
    assert m.user_factors_.shape == (1, 2) and np.allclose(m.user_bias_, 0), (
        "fit with n_epochs=0 must still initialise mu_, factors and zero biases")
    m.mu_ = 3.0
    m.user_bias_[:] = 0.1
    m.item_bias_[:] = -0.2
    m.user_factors_[0] = [1.0, 2.0]
    m.item_factors_[0] = [0.5, -1.0]
    e = m.step(0, 0, 4.0)
    assert np.isclose(e, 2.6), f"r_hat = 3 + 0.1 - 0.2 + (0.5 - 2) = 1.4, so the error r - r_hat is 2.6; got {e}"
    assert np.isclose(m.user_bias_[0], 0.355) and np.isclose(m.item_bias_[0], 0.07), (
        f"b_u += lr (e - lam b_u) -> 0.355, b_i += lr (e - lam b_i) -> 0.07; got {m.user_bias_[0]}, {m.item_bias_[0]}")
    assert np.allclose(m.user_factors_[0], [1.08, 1.64]), (
        f"p += lr (e q - lam p) -> (1.08, 1.64); got {m.user_factors_[0]}")
    assert np.allclose(m.item_factors_[0], [0.735, -0.43]), (
        f"q += lr (e p_OLD - lam q) -> (0.735, -0.43); got {m.item_factors_[0]}. if you got (0.7558, -0.4762) you "
        "used the already-updated p")


def test_stage3_follows_the_spec_exactly(nb):
    u, i, r, _ = _lowrank(n_users=8, n_items=6, density=0.6, seed=4)
    m = bounded(nb.SGDMF(rank=2, lr=0.05, lam=0.1, n_epochs=2, rng=9).fit, u, i, r, budget=2_000_000, what="SGDMF.fit")
    g = np.random.default_rng(9)
    P, Q = g.normal(0, 0.1, (8, 2)), g.normal(0, 0.1, (6, 2))
    bu, bi, mu = np.zeros(8), np.zeros(6), r.mean()
    for _ in range(2):
        for t in g.permutation(len(r)):
            a, b = u[t], i[t]
            e = r[t] - (mu + bu[a] + bi[b] + P[a] @ Q[b])
            bu[a] += 0.05 * (e - 0.1 * bu[a])
            bi[b] += 0.05 * (e - 0.1 * bi[b])
            P[a], Q[b] = P[a] + 0.05 * (e * Q[b] - 0.1 * P[a]), Q[b] + 0.05 * (e * P[a] - 0.1 * Q[b])
    dev = np.abs(m.user_factors_ - P).max()
    assert dev < 1e-10 and np.allclose(m.item_bias_, bi), (
        f"with the same seed your factors should match the spec's sequence exactly (max deviation {dev:.2e}): draw "
        "user factors then item factors with rng.normal(0, 0.1, ...), then each epoch visit the ratings in "
        "rng.permutation(n_ratings) order, one step each")
    assert len(m.history_) == 2, "history_ holds one training RMSE per epoch"


def test_stage3_learns(nb):
    gaps = []
    for seed in range(3):
        u, i, r, _ = _lowrank(seed=20 + seed, noise=0.2)
        train, test = _holdout(u, i, r, seed=seed)
        m = bounded(nb.SGDMF(rank=3, lr=0.02, lam=0.05, n_epochs=40, rng=seed).fit, *train, n_users=60, n_items=50,
                    budget=20_000_000, what="SGDMF.fit")
        gaps.append(_bias_rmse(train, test) - _rmse(m.predict(test[0], test[1]), test[2]))
        assert m.history_[-1] < m.history_[0], "training RMSE should fall over the epochs"
        want = np.sqrt(np.mean((m.predict(train[0], train[1]) - train[2]) ** 2))
        assert np.isclose(m.history_[-1], want), (
            f"history_[-1] is the training RMSE after the last epoch ({want:.4f}), got {m.history_[-1]:.4f}: RMSE, not MSE")
    assert np.median(gaps) > 0.05, (
        f"SGD (rank 3, 40 epochs) should beat the bias baseline's test RMSE by over 0.05; improvements "
        f"{np.round(gaps, 3).tolist()}")


# ---------------------------------------------------------------- stage 4: cold start


def test_stage4_hand_example(nb):
    # the lesson's worked example: rank 1, item factors 1 and 2, item biases 0, mu = 3, ratings 4 and 5, lam = 1
    from types import SimpleNamespace
    model = SimpleNamespace(mu_=3.0, item_factors_=np.array([[1.0], [2.0], [-1.0]]), item_bias_=np.zeros(3), lam=1.0)
    p, b = nb.fold_in(model, np.array([0, 1]), np.array([4.0, 5.0]))
    assert np.allclose(p, [2 / 3]) and np.isclose(b, 1 / 3), (
        f"solve ([[5, 3], [3, 2]] + I) w = [5, 3]: p = 2/3, b = 1/3; got p = {np.round(p, 4)}, b = {b:.4f}. "
        "regress r - mu - b_i on [q_i, 1] and regularize BOTH coefficients with lam")
    pred = nb.predict_new_user(model, p, b, np.array([0, 1, 2]))
    assert np.allclose(pred, [4.0, 3 + 1 / 3 + 4 / 3, 3 + 1 / 3 - 2 / 3]), f"predictions should be 4, 4.667, 2.667; got {pred}"


def test_stage4_fold_in_reproduces_als_users(nb):
    u, i, r, _ = _lowrank(seed=5)
    m = bounded(nb.ALS(rank=3, lam=1.5, n_iters=3, rng=2).fit, u, i, r, budget=5_000_000, what="ALS.fit")
    for user in (2, 40):
        p, b = nb.fold_in(m, i[u == user], r[u == user])
        assert np.allclose(p, m.user_factors_[user], atol=1e-8) and np.isclose(b, m.user_bias_[user], atol=1e-8), (
            f"folding in user {user}'s own ratings must reproduce the ALS user factors exactly: it's the same ridge "
            f"solve, with lam defaulting to model.lam. got {np.round(p, 4)}, {b:.4f} vs "
            f"{np.round(m.user_factors_[user], 4)}, {m.user_bias_[user]:.4f}")
    assert isinstance(b, float), "the bias is a plain float"


def test_stage4_no_ratings_falls_back_to_item_bias(nb):
    u, i, r, _ = _lowrank(seed=6)
    m = bounded(nb.ALS(rank=3, lam=1.0, n_iters=2, rng=0).fit, u, i, r, budget=5_000_000, what="ALS.fit")
    p, b = nb.fold_in(m, np.array([], dtype=int), np.array([]))
    assert np.allclose(p, 0) and b == 0, f"a user with no ratings gets zero factors and zero bias; got {p}, {b}"
    items = np.arange(50)
    pred = nb.predict_new_user(m, p, b, items)
    assert np.allclose(pred, m.mu_ + m.item_bias_), "with zero factors the prediction is mu + b_i: a popularity-style fallback"


def test_stage4_works_for_sgd_models_and_lam(nb):
    u, i, r, _ = _lowrank(seed=7)
    m = bounded(nb.SGDMF(rank=3, lr=0.02, lam=0.05, n_epochs=3, rng=0).fit, u, i, r, budget=5_000_000, what="SGDMF.fit")
    its, rs = np.array([1, 4, 9, 16]), np.array([5.0, 2.0, 4.0, 3.0])
    for lam in (None, 3.0):
        p, b = nb.fold_in(m, its, rs, lam=lam)
        L = m.lam if lam is None else lam
        X = np.c_[m.item_factors_[its], np.ones(4)]
        w = np.linalg.solve(X.T @ X + L * np.eye(4), X.T @ (rs - m.mu_ - m.item_bias_[its]))
        assert np.allclose(p, w[:3]) and np.isclose(b, w[3]), (
            f"fold_in uses only model.mu_, item_factors_, item_bias_ (and model.lam unless lam is given); "
            f"lam={lam}: expected {np.round(w, 4)}, got {np.round(p, 4)}, {b:.4f}")
    pred = nb.predict_new_user(m, p, b, np.array([0, 3]))
    want = m.mu_ + b + m.item_bias_[[0, 3]] + m.item_factors_[[0, 3]] @ p
    assert np.allclose(pred, want), "predict_new_user = mu + bias + b_i + q_i . factors"
