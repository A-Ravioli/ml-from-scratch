"""Checks for exercise.py (online learning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.

Updates are checked exactly by hand. The regret bounds are THEOREMS that hold for every loss
sequence, so they are checked on random, structured and adversarial sequences with no
statistical slack: a correct implementation can never fail them. Long runs go through
mlfs.testing.bounded.
"""
import math

import numpy as np

from mlfs.testing import bounded


def _hedge_losses(nb, L, lr):
    h = nb.Hedge(L.shape[1], lr)
    out = []
    for losses in L:
        p = np.asarray(h.predict())
        out.append(float(p @ losses))
        h.update(losses)
    return np.array(out)


# ---------------------------------------------------------------- stage 1: Hedge


def test_stage1_hand_updates(nb):
    h = nb.Hedge(3, math.log(2))
    p0 = np.asarray(h.predict())
    assert p0.shape == (3,) and np.allclose(p0, 1 / 3), f"before any update Hedge plays uniformly; got {p0}"
    h.update(np.array([1.0, 0.0, 0.0]))
    p1 = np.asarray(h.predict())
    assert np.allclose(p1, [1 / 5, 2 / 5, 2 / 5]), (
        f"lr = ln 2, losses (1, 0, 0): weights become (1/2, 1, 1), normalized (0.2, 0.4, 0.4); got {p1}. "
        "the weight of expert i is exp(-lr * its cumulative loss)")
    h.update(np.array([0.0, 1.0, 0.5]))
    want = np.exp(-math.log(2) * np.array([1.0, 1.0, 0.5]))
    assert np.allclose(h.predict(), want / want.sum()), "weights depend on the CUMULATIVE loss of each expert"
    p = h.predict()
    p[:] = 0
    assert np.isclose(np.sum(h.predict()), 1.0), "predict must return a new array (changing it can't change Hedge's state)"


def test_stage1_stable_and_invariant(nb):
    h = nb.Hedge(4, 1.0)
    for _ in range(3000):
        h.update(np.array([1.0, 1.0, 1.0, 0.9]))
    p = np.asarray(h.predict())
    assert np.all(np.isfinite(p)) and np.isclose(p.sum(), 1) and p[3] > 0.999, (
        f"after 3,000 rounds every weight exp(-lr * L) is below exp(-2700) and underflows to 0, so a plain normalization "
        f"gives 0/0; subtract the max log-weight (softmax / logsumexp). expert 3 should have almost all the weight; got {p}")
    try:
        nb.Hedge(3, 0.5).update(np.array([0.1]))
    except ValueError:
        pass
    else:
        raise AssertionError("update with losses of the wrong shape (1 loss for 3 experts, which numpy would silently broadcast) must raise ValueError")
    a, b = nb.Hedge(3, 0.7), nb.Hedge(3, 0.7)
    rng = np.random.default_rng(0)
    for _ in range(20):
        ell = rng.random(3)
        a.update(ell)
        b.update(np.clip(ell + 0.3, 0, None))
    assert np.allclose(a.predict(), b.predict()), "adding the same constant to every expert's loss can't change p"


# ---------------------------------------------------------------- stage 2: OGD


def test_stage2_hand_steps(nb):
    o = nb.OnlineGradientDescent(2, lr=1.0)
    assert np.allclose(o.predict(), [0, 0]), "OGD starts at the origin"
    o.update(np.array([1.0, 0.0]))
    assert np.allclose(o.predict(), [-1.0, 0.0]), f"round 1: eta_1 = lr / sqrt(1) = 1, x = -g; got {o.predict()}"
    o.update(np.array([0.0, 2.0]))
    assert np.allclose(o.predict(), [-1.0, -math.sqrt(2)]), (
        f"round 2: eta_2 = 1 / sqrt(2), x = (-1, 0) - (0, 2) / sqrt(2) = (-1, -1.414); got {o.predict()}")
    c = nb.OnlineGradientDescent(2, lr=0.5, decay=None)
    for g in ([1.0, 1.0], [1.0, -1.0], [2.0, 0.0]):
        c.update(np.array(g))
    assert np.allclose(c.predict(), [-2.0, 0.0]), f"decay=None: constant step 0.5; got {c.predict()}"
    x = c.predict()
    x[:] = 99
    assert np.allclose(c.predict(), [-2.0, 0.0]), "predict must return a copy"


def test_stage2_projection(nb):
    seen = []

    def proj(v):
        seen.append(np.array(v))
        n = np.linalg.norm(v)
        return v if n <= 1 else v / n

    o = nb.OnlineGradientDescent(3, lr=4.0, project=proj)
    o.update(np.array([1.0, 0.0, 0.0]))
    assert len(seen) == 1 and np.allclose(seen[0], [-4, 0, 0]), "project the point AFTER the gradient step"
    assert np.allclose(o.predict(), [-1, 0, 0]), f"the projected point is the new x; got {o.predict()}"
    o.update(np.array([0.0, 1.0, 0.0]))
    assert np.allclose(seen[1], [-1, -2 * math.sqrt(2), 0]), "round 2 steps from the PROJECTED point with eta = 4/sqrt(2)"
    try:
        nb.OnlineGradientDescent(2, decay="linear")
    except ValueError:
        pass
    else:
        raise AssertionError("decay must be 'sqrt' or None; anything else should raise ValueError")


# ---------------------------------------------------------------- stage 3: FTL and regret


def test_stage3_regret_curve(nb):
    L = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    learner = np.array([0.5, 0.5, 1.0, 0.0])
    got = np.asarray(nb.regret_curve(learner, L))
    want = np.array([0.5, 0.0, 1.0, 0.0])
    assert got.shape == (4,) and np.allclose(got, want), (
        f"cumulative learner loss (0.5, 1, 2, 2) minus the best expert's cumulative loss AT EACH TIME (0, 1, 1, 2); "
        f"got {got}. the best expert can change over time")


def test_stage3_follow_the_leader(nb):
    L = np.array([[0.2, 0.1, 0.9], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    got = np.asarray(nb.follow_the_leader(L))
    want = np.array([0.2, 0.0, 1.0, 1.0])
    assert np.allclose(got, want), (
        f"round 1: all cumulative losses are 0, ties go to expert 0 (loss 0.2). round 2: leader is expert 1 (0.1). "
        f"round 3: cumulative (1.2, 0.1, 0.9), leader 1 again (loss 1). round 4: (1.2, 1.1, 0.9), leader 2 (loss 1); "
        f"expected {want.tolist()}, got {got.tolist()}. use only losses from BEFORE the current round")
    T = 1000
    K = np.zeros((T, 2))
    K[0] = [0.5, 0.0]
    K[1::2] = [0.0, 1.0]
    K[2::2] = [1.0, 0.0]
    r = np.asarray(nb.regret_curve(nb.follow_the_leader(K), K))
    assert r[-1] > 0.45 * T, (
        f"on the alternating sequence (0.5, 0), (0, 1), (1, 0), ... FTL always jumps to the expert about to lose: "
        f"regret grows linearly (about T/2); got {r[-1]:.1f} after {T} rounds")


# ---------------------------------------------------------------- stage 4: Hedge's guarantee


def test_stage4_formulas(nb):
    assert np.isclose(nb.hedge_optimal_lr(800, 10), math.sqrt(8 * math.log(10) / 800)), "lr* = sqrt(8 ln N / T)"
    assert np.isclose(nb.hedge_regret_bound(800, 10, 0.2), math.log(10) / 0.2 + 0.2 * 800 / 8), "ln N / lr + lr T / 8"
    lr = nb.hedge_optimal_lr(5000, 7)
    assert np.isclose(nb.hedge_regret_bound(5000, 7, lr), math.sqrt(5000 * math.log(7) / 2)), (
        "at the optimal learning rate the bound is sqrt(T ln N / 2)")


def test_stage4_run_hedge_and_the_bound_holds_for_every_sequence(nb):
    rng = np.random.default_rng(1)
    L = rng.random((50, 4))
    got = np.asarray(nb.run_hedge(L, 0.3))
    assert got.shape == (50,) and np.allclose(got, _hedge_losses(nb, L, 0.3)), (
        "run_hedge returns p_t . l_t for each round, with p_t computed BEFORE seeing l_t")
    T = 2000
    seqs = {
        "uniform random": rng.random((T, 10)),
        "one good expert": np.clip(rng.random((T, 10)) + np.r_[-0.3, np.zeros(9)], 0, 1),
        "ftl killer": np.tile(np.array([[0.0, 1.0], [1.0, 0.0]]), (T // 2, 1)),
        "switching best": np.where((np.arange(T)[:, None] // 200 % 2) == np.arange(2)[None, :], 0.2, 0.8),
        "bernoulli": (rng.random((T, 5)) < np.linspace(0.3, 0.7, 5)).astype(float),
    }
    for name, S in seqs.items():
        lr = nb.hedge_optimal_lr(T, S.shape[1])
        learner = bounded(nb.run_hedge, S, lr, what="run_hedge", budget=400_000)
        reg = float(np.asarray(nb.regret_curve(learner, S))[-1])
        bound = nb.hedge_regret_bound(T, S.shape[1], lr)
        assert reg <= bound + 1e-9, (
            f"{name}: Hedge's regret {reg:.2f} exceeds its guarantee {bound:.2f}. the bound holds for EVERY sequence, so "
            "a violation means a bug in Hedge or run_hedge (exp(-lr * cumulative loss), losses added after playing)")
    reg_k = float(np.asarray(nb.regret_curve(nb.run_hedge(seqs["ftl killer"], nb.hedge_optimal_lr(T, 2)), seqs["ftl killer"]))[-1])
    assert reg_k < 0.1 * T, f"Hedge must not be fooled like FTL on the alternating sequence: regret {reg_k:.1f} of T = {T}"


# ---------------------------------------------------------------- stage 5: OGD's guarantee


def _best_in_ball(X, y, R):
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    if np.linalg.norm(w) <= R:
        return w
    lo, hi = 1e-12, 1e8
    for _ in range(200):
        lam = math.sqrt(lo * hi)
        w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
        lo, hi = (lam, hi) if np.linalg.norm(w) > R else (lo, lam)
    return w


def test_stage5_bound_and_linear_losses(nb):
    assert np.isclose(nb.ogd_regret_bound(400, 2.0, 3.0), 1.5 * 3 * 2 * 20), "(3/2) G D sqrt(T)"
    rng = np.random.default_rng(2)
    T, d, R = 1500, 3, 1.0
    for name, G in [("random", rng.normal(size=(T, d))), ("adversarial", np.tile([[1.0, 0, 0], [-1.0, 0, 0]], (T // 2, 1)))]:
        G = G / np.linalg.norm(G, axis=1, keepdims=True)  # ||g_t|| = 1
        o = nb.OnlineGradientDescent(d, lr=2 * R / 1.0, project=lambda v: v if np.linalg.norm(v) <= R else R * v / np.linalg.norm(v))
        xs = []
        for g in G:
            xs.append(o.predict())
            o.update(g)
        learner = np.sum(np.array(xs) * G)
        best = -R * np.linalg.norm(G.sum(axis=0))  # min over the ball of sum_t g_t . x
        assert learner - best <= nb.ogd_regret_bound(T, 2 * R, 1.0) + 1e-9, (
            f"{name} linear losses on the unit ball: OGD's regret {learner - best:.2f} must be at most (3/2) G D sqrt(T) = "
            f"{nb.ogd_regret_bound(T, 2 * R, 1.0):.2f} with lr = D / G and decay='sqrt'")


def test_stage5_online_least_squares(nb):
    rng = np.random.default_rng(3)
    T, d, R = 3000, 4, 2.0
    w_true = rng.normal(size=d)
    w_true *= 1.5 / np.linalg.norm(w_true)
    X = rng.normal(size=(T, d))
    X /= np.maximum(1, np.linalg.norm(X, axis=1, keepdims=True))
    y = X @ w_true + 0.1 * rng.normal(size=T)
    Gb = 1.0 * (R * 1.0 + np.abs(y).max())  # ||grad|| <= ||x|| (||w|| ||x|| + |y|)
    losses, w = bounded(nb.online_least_squares, X, y, lr=2 * R / Gb, radius=R, what="online_least_squares",
                        budget=1_000_000)
    losses = np.asarray(losses)
    assert losses.shape == (T,) and np.all(losses >= 0), "one squared loss (w_t . x_t - y_t)^2 / 2 per round"
    assert np.isclose(losses[0], 0.5 * y[0] ** 2), "round 1 predicts with w_1 = 0, BEFORE updating"
    assert np.linalg.norm(w) <= R + 1e-9, f"the iterates stay in the ball of radius {R}; final norm {np.linalg.norm(w):.3f}"
    wb = _best_in_ball(X, y, R)
    regret = losses.sum() - np.sum(0.5 * (X @ wb - y) ** 2)
    assert regret <= nb.ogd_regret_bound(T, 2 * R, Gb), "OGD's regret must respect (3/2) G D sqrt(T)"
    assert np.linalg.norm(np.asarray(w) - w_true) < 0.3, (
        f"after 3,000 rounds of a noisy linear stream the final w should be near the truth: |w - w*| = "
        f"{np.linalg.norm(np.asarray(w) - w_true):.3f}")
    l2, _ = nb.online_least_squares(X[:5], y[:5], lr=0.1, radius=R, decay=None)
    w_ = np.zeros(d)
    for t in range(5):
        assert np.isclose(l2[t], 0.5 * (w_ @ X[t] - y[t]) ** 2), "decay=None: constant steps of size lr"
        w_ = w_ - 0.1 * (w_ @ X[t] - y[t]) * X[t]
