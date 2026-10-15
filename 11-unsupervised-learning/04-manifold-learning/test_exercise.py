"""Checks for exercise.py (manifold learning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scikit-learn's Isomap
and TSNE are oracles only; the t-SNE gradient is checked by finite differences. TSNE.fit runs
under mlfs.testing.bounded, so a loop without a cap fails instead of hanging.
"""
import numpy as np
from sklearn.manifold import Isomap as SkIsomap
from sklearn.manifold import TSNE as SkTSNE

from mlfs.testing import bounded

TRIANGLE = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])


def _dist(A):
    return np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1))


def _swiss_roll(n=400, seed=0, noise=0.05, fixed=False):
    # fixed=True: a generator that seed sweeps don't touch. Isomap is deterministic, so a data set known to be
    # sampled densely enough (no short-circuit edges between layers) is a fair, fixed test case
    rng = np.random.Generator(np.random.PCG64(seed)) if fixed else np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n))
    h = rng.uniform(0, 15, n)
    X = np.c_[t * np.cos(t), h, t * np.sin(t)] + noise * rng.standard_normal((n, 3))
    return X, t, h


def _blobs(n_per=30, k=3, d=10, sep=10.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, (k, d))
    X = np.concatenate([c + rng.standard_normal((n_per, d)) for c in centers])
    return X, np.repeat(np.arange(k), n_per)


def _knn_accuracy(Y, y, k=5):
    D = _dist(Y)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :k]
    return float(np.mean([np.bincount(y[r]).argmax() == y[i] for i, r in enumerate(nn)]))


def _same_up_to_sign(A, B, atol=1e-6):
    return all(np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol) for a, b in zip(A.T, B.T))


def _P(nb, X, perplexity):
    return bounded(nb.joint_probabilities, X, perplexity, what="joint_probabilities",
                   hint="the perplexity bisection must stop after max_iters rounds.")


# ---------------------------------------------------------------- stage 1: classical MDS


def test_stage1_worked_example(nb):
    D = _dist(TRIANGLE)
    m = nb.MDS(2, dissimilarity="precomputed").fit(D)
    E = m.embedding_
    assert E.shape == (3, 2), f"embedding_ is (n, k) = (3, 2), got {E.shape}"
    assert np.allclose(_dist(E), D), (
        f"from the 3-4-5 distances alone, MDS must rebuild a 3-4-5 triangle: got side lengths {np.round(_dist(E)[[0, 0, 1], [1, 2, 2]], 4)}")
    assert np.isclose(m.eigenvalues_.sum(), 50 / 3), (
        f"the eigenvalues of B sum to the total squared distance from the centroid, (9 + 16 + 25) / 3 = 16.667, "
        f"got {m.eigenvalues_.sum():.4f}. B = -1/2 J D^2 J (square the distances, then double-centre)")
    assert np.allclose(E.mean(0), 0), "the embedding is centred"
    big = np.abs(E).argmax(0)
    assert np.all(E[big, np.arange(2)] > 0), "each embedding column's largest-magnitude entry is positive"


def test_stage1_recovers_linear_configuration(nb):
    rng = np.random.default_rng(1)
    Z = rng.normal(0, 1, (40, 2)) * [5, 1]
    Q, _ = np.linalg.qr(rng.normal(0, 1, (6, 6)))
    X = np.c_[Z, np.zeros((40, 4))] @ Q.T + 3.0  # a flat 2-D sheet rotated into 6-D
    E = nb.MDS(2).fit_transform(X)
    assert np.allclose(_dist(E), _dist(X)), "points that lie in a 2-D plane are embedded with every distance preserved"
    E2 = nb.MDS(2, dissimilarity="precomputed").fit_transform(_dist(X))
    assert np.allclose(E, E2), "dissimilarity='precomputed' on the Euclidean distances gives the same embedding"
    m = nb.MDS(4).fit(X)
    assert np.allclose(m.eigenvalues_[2:], 0, atol=1e-8) and np.all(np.diff(m.eigenvalues_) <= 1e-9), (
        "eigenvalues_ are in descending order; a 2-D configuration has only two nonzero ones")
    for bad in (np.arange(12.0).reshape(3, 4), np.array([[0.0, 1.0], [2.0, 0.0]])):
        try:
            nb.MDS(1, dissimilarity="precomputed").fit(bad)
        except ValueError:
            continue
        raise AssertionError(f"a precomputed dissimilarity must be a symmetric square matrix; {bad.tolist()} should raise ValueError")


# ---------------------------------------------------------------- stage 2: geodesic distances


def test_stage2_line(nb):
    X = np.array([[0.0], [1.0], [3.0], [6.0], [10.0]])
    G = bounded(nb.geodesic_distances, X, n_neighbors=1, what="geodesic_distances")
    assert np.allclose(G, _dist(X)), (
        "points on a line with 1 neighbour each: the symmetrized graph is the chain 0-1-3-6-10, so geodesic distances "
        "equal the ordinary ones. symmetrize: an edge exists if EITHER endpoint has the other among its neighbours")


def test_stage2_matches_sklearn(nb):
    X, _, _ = _swiss_roll(n=300, seed=2)
    G = bounded(nb.geodesic_distances, X, n_neighbors=8, what="geodesic_distances")
    sk = SkIsomap(n_neighbors=8).fit(X).dist_matrix_
    assert G.shape == (300, 300) and np.allclose(G, G.T), "geodesic distances are a symmetric (n, n) matrix"
    assert np.allclose(G, sk), (
        f"shortest paths on the 8-NN graph differ from scikit-learn's by up to {np.abs(G - sk).max():.3g}. edges weighted "
        "by Euclidean length; paths may go through any vertex")


def test_stage2_disconnected_is_inf(nb):
    X = np.r_[np.arange(5.0), 100 + np.arange(5.0)][:, None]
    G = bounded(nb.geodesic_distances, X, n_neighbors=2, what="geodesic_distances")
    assert np.isinf(G[0, 9]) and np.isinf(G[9, 0]) and np.isfinite(G[0, 4]), (
        "two separate groups: there is no path between them, so their geodesic distance is inf")


# ---------------------------------------------------------------- stage 3: Isomap


def test_stage3_unrolls_swiss_roll(nb):
    X, t, h = _swiss_roll(n=800, seed=3, fixed=True)
    iso = bounded(nb.Isomap(2, n_neighbors=8).fit, X, what="Isomap.fit")
    E = iso.embedding_
    arc = t**2  # arc length along the spiral r = t grows like t^2 / 2
    c = max(abs(np.corrcoef(E[:, 0], arc)[0, 1]), abs(np.corrcoef(E[:, 1], arc)[0, 1]))
    assert c > 0.99, f"Isomap should unroll the swiss roll: one coordinate should track the arc length (|corr| {c:.3f} <= 0.99)"
    sk = SkIsomap(n_neighbors=8, n_components=2).fit(X).embedding_
    assert _same_up_to_sign(E, sk, atol=1e-5), "the embedding should match scikit-learn's Isomap up to the sign of each column"
    assert np.allclose(bounded(nb.Isomap(2, n_neighbors=8).fit_transform, X, what="Isomap.fit_transform"), E), "fit_transform returns embedding_"


def test_stage3_disconnected_raises(nb):
    X = np.r_[np.arange(6.0), 50 + np.arange(6.0)][:, None]
    try:
        bounded(nb.Isomap(1, n_neighbors=2).fit, X, what="Isomap.fit")
    except ValueError:
        return
    raise AssertionError("a disconnected neighbourhood graph has infinite geodesics: raise ValueError (increase n_neighbors)")


# ---------------------------------------------------------------- stage 4: input affinities


def _entropy(P):
    P = np.where(P > 0, P, 1.0)
    return -np.sum(np.where(P < 1.0, P * np.log(P), 0.0), axis=1)


def test_stage4_perplexity_is_matched(nb):
    X, _ = _blobs(seed=4)
    D2 = _dist(X) ** 2
    for perp in (5.0, 20.0):
        P = bounded(nb.conditional_probabilities, D2, perp, what="conditional_probabilities")
        assert P.shape == (90, 90) and np.allclose(np.diag(P), 0) and np.allclose(P.sum(1), 1), (
            "p_{j|i}: each row is a distribution over the OTHER points (zero diagonal), summing to 1")
        H = _entropy(P)
        assert np.allclose(H, np.log(perp), atol=1e-3), (
            f"every row's entropy should be log(perplexity) = {np.log(perp):.4f} (natural log); got entropies from "
            f"{H.min():.4f} to {H.max():.4f}. bisect each row's precision beta_i")
        i = 0
        order = np.argsort(D2[i])[1:]
        assert np.all(np.diff(P[i, order]) <= 1e-12), "within a row, nearer points get larger probabilities"


def test_stage4_far_points_and_validation(nb):
    D2 = np.array([[0, 1, 4, 1e6], [1, 0, 1, 1e6], [4, 1, 0, 1e6], [1e6, 1e6, 1e6, 0]], float)
    P = bounded(nb.conditional_probabilities, D2, 1.5, what="conditional_probabilities")
    assert np.isfinite(P).all() and np.allclose(P.sum(1), 1), (
        "rows must stay finite when distances are huge: subtract each row's max logit before exponentiating")
    for bad in (1.0, 3.0):
        try:
            bounded(nb.conditional_probabilities, D2, bad, what="conditional_probabilities")
        except ValueError:
            continue
        raise AssertionError(f"perplexity {bad} with n = 4 points is impossible (needs 1 < perplexity < n - 1): ValueError")


def test_stage4_joint(nb):
    X, _ = _blobs(seed=5)
    P = _P(nb, X, 10.0)
    Pc = bounded(nb.conditional_probabilities, _dist(X) ** 2, 10.0, what="conditional_probabilities")
    assert np.allclose(P, P.T) and np.isclose(P.sum(), 1.0), "the joint P is symmetric and sums to 1"
    assert np.allclose(P, (Pc + Pc.T) / (2 * len(X))), "P = (P_cond + P_cond^T) / (2n)"


# ---------------------------------------------------------------- stage 5: KL and gradient


def test_stage5_two_points(nb):
    P = np.array([[0.0, 0.5], [0.5, 0.0]])
    kl, g = nb.tsne_kl_gradient(np.array([[0.0, 0.0], [3.0, 1.0]]), P)
    assert np.isclose(kl, 0) and np.allclose(g, 0), (
        "with two points, q_12 = q_21 = 1/2 whatever their distance, so Q = P: KL 0 and gradient 0")


def test_stage5_gradient_matches_finite_differences(nb):
    rng = np.random.default_rng(6)
    X, _ = _blobs(n_per=8, seed=6)
    P = _P(nb, X, 5.0)
    Y = rng.normal(0, 1, (len(X), 2))
    kl, g = nb.tsne_kl_gradient(Y, P)
    num = 1 / (1 + _dist(Y) ** 2)
    np.fill_diagonal(num, 0)
    Q = num / num.sum()
    m = P > 0
    assert np.isclose(kl, np.sum(P[m] * np.log(P[m] / Q[m]))), "KL(P || Q) = sum p_ij log(p_ij / q_ij) over p_ij > 0"
    fd = np.zeros_like(Y)
    for idx in np.ndindex(Y.shape):
        e = np.zeros_like(Y)
        e[idx] = 1e-6
        fd[idx] = (nb.tsne_kl_gradient(Y + e, P)[0] - nb.tsne_kl_gradient(Y - e, P)[0]) / 2e-6
    none, g2 = nb.tsne_kl_gradient(Y, P, kl=False)
    assert none is None and np.allclose(g2, g), "kl=False skips the KL (returns None) but gives the same gradient"
    assert np.allclose(g, fd, atol=1e-6), (
        f"gradient differs from finite differences by up to {np.abs(g - fd).max():.2e}: "
        "dC/dy_i = 4 sum_j (p_ij - q_ij)(y_i - y_j) / (1 + ||y_i - y_j||^2)")


# ---------------------------------------------------------------- stage 6: TSNE


def _fit_tsne(nb, X, **kw):
    t = nb.TSNE(**kw)
    return bounded(t.fit, X, budget=2_000_000, what="TSNE.fit", hint="run exactly n_iters iterations.")


def test_stage6_separates_blobs(nb):
    X, y = _blobs(seed=7)
    fits = [_fit_tsne(nb, X, perplexity=10.0, n_iters=500, rng=s) for s in range(3)]
    assert fits[0].embedding_.shape == (90, 2), f"embedding_ is (n, n_components), got {fits[0].embedding_.shape}"
    acc = np.median([_knn_accuracy(t.embedding_, y) for t in fits])
    assert acc > 0.9, (
        f"three well-separated blobs should stay separate in the embedding (median 5-NN accuracy over 3 seeds {acc:.2f})")
    P = _P(nb, X, 10.0)
    kl0 = nb.tsne_kl_gradient(1e-4 * np.random.default_rng(0).standard_normal((90, 2)), P)[0]
    kl = np.median([t.kl_divergence_ for t in fits])
    assert np.isfinite(kl) and kl < 0.6 * kl0, (
        f"kl_divergence_ (with the un-exaggerated P) should fall far below its value at the start ({kl0:.2f}); "
        f"median over 3 seeds {kl:.2f}")


def test_stage6_kl_comparable_to_sklearn(nb):
    X, _ = _blobs(n_per=40, k=4, d=8, sep=4.0, seed=8)
    ours = [_fit_tsne(nb, X, perplexity=15.0, n_iters=600, rng=s).kl_divergence_ for s in range(3)]
    sk = SkTSNE(perplexity=15.0, method="exact", init="random", learning_rate=200.0, random_state=0).fit(X).kl_divergence_
    assert min(ours) < 1.25 * sk + 0.05, (
        f"your t-SNE's final KL (best of 3 seeds {min(ours):.3f}) should be about as low as scikit-learn's exact "
        f"t-SNE ({sk:.3f}). check momentum (0.5 then 0.8), the gains, and that early exaggeration stops after 250 iterations")


def _tsne_spec(P, n_iters, lr=200.0, exaggeration=12.0, seed=0, k=2, gradient=None):
    """the update rule from the lesson, written out independently. pass gradient=nb.tsne_kl_gradient to use the same
    gradient arithmetic as the notebook (the sign-based gains turn last-digit differences into different runs)."""
    Y = 1e-4 * np.random.default_rng(seed).standard_normal((len(P), k))
    update, gains = np.zeros_like(Y), np.ones_like(Y)
    for it in range(n_iters):
        Pe = P * exaggeration if it < 250 else P
        if gradient is not None:
            grad = gradient(Y, Pe, kl=False)[1]
        else:
            num = 1 / (1 + _dist(Y) ** 2)
            np.fill_diagonal(num, 0)
            W = (Pe - num / num.sum()) * num
            grad = 4 * (W.sum(1)[:, None] * Y - W @ Y)
        gains = np.maximum(np.where(np.sign(grad) != np.sign(update), gains + 0.2, gains * 0.8), 0.01)
        update = (0.5 if it < 250 else 0.8) * update - lr * gains * grad
        Y = Y + update
        Y = Y - Y.mean(0)
    return Y


def test_stage6_follows_the_update_rule(nb):
    X, _ = _blobs(n_per=4, seed=10)
    P = _P(nb, X, 3.0)
    # at lr=200 the sign-based gains amplify round-off over hundreds of steps, so long runs use lr=1, where the
    # trajectory is smooth and the late phase (P un-exaggerated and momentum 0.8 from iteration 250 on) is testable
    for n_iters, lr in ((1, 200.0), (3, 200.0), (10, 200.0), (300, 1.0)):
        got = _fit_tsne(nb, X, perplexity=3.0, n_iters=n_iters, lr=lr, rng=0).embedding_
        want = _tsne_spec(P, n_iters, lr=lr, gradient=nb.tsne_kl_gradient if n_iters > 10 else None)
        assert np.allclose(got, want, rtol=1e-6, atol=1e-10), (
            f"after {n_iters} iterations (lr={lr}) your embedding differs from the lesson's update rule (max diff "
            f"{np.abs(got - want).max():.2e}). check, in order: Y0 = 1e-4 * rng.standard_normal((n, n_components)); P times "
            "early_exaggeration and momentum 0.5 while it < 250, then P and 0.8; gains + 0.2 where sign(grad) != sign(update) "
            "else * 0.8, floored at 0.01; update = momentum * update - lr * gains * grad; Y += update; Y -= Y.mean(0). "
            "the long (lr=1) run replays your loop with YOUR tsne_kl_gradient(..., kl=False), so compute the gradient "
            "by calling it, as the lesson says, rather than re-deriving it inline (same math, different last digits).")


def test_stage6_protocol_and_rng(nb):
    X, _ = _blobs(n_per=10, seed=9)
    t = nb.TSNE(perplexity=5.0, n_iters=260, rng=3)
    assert t.get_params()["early_exaggeration"] == 12.0 and t.get_params()["lr"] == 200.0, (
        "TSNE subclasses Estimator; __init__ stores its arguments")
    np.random.seed(0)
    a = _fit_tsne(nb, X, perplexity=5.0, n_iters=260, rng=3).embedding_
    np.random.seed(1)
    b = bounded(nb.TSNE(perplexity=5.0, n_iters=260, rng=3).fit_transform, X, budget=2_000_000, what="TSNE.fit_transform")
    assert np.allclose(a, b), "the same rng gives the same embedding (draw from as_rng(rng), never np.random); fit_transform = embedding_"
    assert np.allclose(a.mean(0), 0, atol=1e-6), "re-centre the embedding after every update"
