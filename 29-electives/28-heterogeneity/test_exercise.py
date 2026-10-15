"""Checks for exercise.py (heterogeneity).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Partitions are
checked for coverage and for the skew alpha controls; client drift is checked exactly on
quadratics, where FedAvg's fixed point has a closed form (the lesson's worked example);
SCAFFOLD is checked to reach the true optimum; FedProx and fine-tuning are checked against
hand-computed full-batch steps.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mlfs.testing import bounded

torch.set_num_threads(1)  # tiny models: one thread is fastest, and avoids oversubscribing a busy machine


def _flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def _net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 3))


def _data(n, seed):
    g = np.random.default_rng(seed)
    y = g.integers(0, 3, n)
    X = g.normal(size=(n, 3)) + 2.0 * np.eye(3)[y]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)


# 1-D worked example: f1 = 0.5 (w - 0)^2, f2 = 0.5 * 3 (w - 4)^2
A1 = np.array([[[1.0]], [[3.0]]])
C1 = np.array([[0.0], [4.0]])
# 2-D: different curvatures and different optima
A2 = np.array([[[2.0, 0.0], [0.0, 0.5]], [[0.5, 0.0], [0.0, 2.0]]])
C2 = np.array([[1.0, 0.0], [0.0, 1.0]])


# ---------------------------------------------------------------- stage 1: label skew


def test_stage1_heterogeneity_hand_example(nb):
    y = np.array([0, 0, 1, 1])
    d = nb.label_distribution([np.array([0]), np.array([1, 2, 3])], y, 2)
    assert np.allclose(d, [[1, 0], [1 / 3, 2 / 3]]), f"label histograms normalized per client: expected [[1, 0], [1/3, 2/3]], got {d.tolist()}"
    h = nb.heterogeneity([np.array([0]), np.array([1, 2, 3])], y, 2)
    assert np.isclose(h, 0.25), (
        f"global distribution (0.5, 0.5); TVs 0.5 and 1/6; weighted by sizes 1/4 and 3/4 gives 0.25, got {h}. "
        "TV is HALF the L1 distance; weights are n_k / n")
    assert np.isclose(nb.heterogeneity([np.array([0, 1]), np.array([2, 3])], y, 2), 0.5), "two one-label clients: 0.5"
    assert np.isclose(nb.heterogeneity([np.array([0, 2]), np.array([1, 3])], y, 2), 0.0), "identical label mixes: 0"


def test_stage1_dirichlet_partition_covers(nb):
    y = np.repeat(np.arange(10), 100)
    parts = bounded(nb.partition_dirichlet, y, 20, 0.5, seed=0, what="partition_dirichlet")
    assert len(parts) == 20, f"expected 20 clients, got {len(parts)}"
    allidx = np.concatenate(parts)
    assert np.issubdtype(allidx.dtype, np.integer), "partitions are integer index arrays"
    assert sorted(allidx.tolist()) == list(range(1000)), "every index appears in exactly one client"
    assert min(len(p) for p in parts) >= 1, "the default min_size=1: no empty clients"
    again = nb.partition_dirichlet(y, 20, 0.5, seed=0)
    assert all(np.array_equal(p, q) for p, q in zip(parts, again)), "the same seed gives the same split"


def test_stage1_alpha_controls_skew(nb):
    y = np.repeat(np.arange(10), 200)
    hs = {a: np.median([nb.heterogeneity(bounded(nb.partition_dirichlet, y, 10, a, seed=s, what="partition_dirichlet"), y, 10)
                        for s in range(5)]) for a in (1e4, 1.0, 0.05)}
    assert hs[1e4] < 0.05, f"alpha = 1e4 is essentially IID: heterogeneity should be near 0, got {hs[1e4]:.3f}"
    assert 0.2 < hs[1.0] < 0.6, f"alpha = 1 is moderate skew (around 0.35 for 10 classes and clients); got {hs[1.0]:.3f}"
    assert hs[0.05] > 0.6, f"alpha = 0.05: most clients hold one or two classes (around 0.75); got {hs[0.05]:.3f}"
    parts = nb.partition_dirichlet(y, 10, 1e-3, seed=0, min_size=0)
    share = nb.label_distribution(parts, y, 10) * np.array([len(p) for p in parts])[:, None] / 200
    assert (share.max(axis=0) > 0.95).mean() >= 0.7, (
        f"with tiny alpha each CLASS should land almost entirely on one client (largest share per class: {np.round(share.max(axis=0), 2).tolist()}). draw one Dirichlet(alpha) vector of "
        "client proportions per class and split that class's indices by it")


def test_stage1_impossible_min_size_raises(nb):
    y = np.repeat(np.arange(4), 10)
    try:
        bounded(nb.partition_dirichlet, y, 8, 0.1, seed=0, min_size=6, what="partition_dirichlet",
                hint="cap the number of resampling attempts (max_tries)")
    except RuntimeError:
        pass
    else:
        raise AssertionError("40 examples can't give 8 clients >= 6 each: raise RuntimeError after max_tries attempts")


# ---------------------------------------------------------------- stage 2: client drift on quadratics


def test_stage2_global_optimum(nb):
    assert np.allclose(nb.global_optimum(A1, C1), [3.0]), "(1*0 + 3*4) / (1 + 3) = 3"
    assert np.allclose(nb.global_optimum(A2, C2), [0.8, 0.8]), "solve (sum p_k A_k) w = sum p_k A_k c_k"
    assert np.allclose(nb.global_optimum(A1, C1, weights=[3, 1]), [12 / 6]), (
        "weights [3, 1]: (3*1*0 + 1*3*4) / (3*1 + 1*3) = 2. normalize the weights")


def test_stage2_one_local_step_is_gradient_descent(nb):
    w0 = np.array([5.0, -3.0])
    tr = nb.quadratic_fedavg(A2, C2, w0, rounds=3, local_steps=1, lr=0.2)
    assert tr.shape == (4, 2), f"return the (rounds + 1, d) iterates, starting with w0; got shape {tr.shape}"
    assert np.allclose(tr[0], w0), "row 0 is w0"
    w = w0.copy()
    for t in range(3):
        g = 0.5 * (A2[0] @ (w - C2[0]) + A2[1] @ (w - C2[1]))
        w = w - 0.2 * g
        assert np.allclose(tr[t + 1], w), "with one local step, FedAvg is gradient descent on the average objective"


def test_stage2_worked_example_fixed_point(nb):
    fp = nb.quadratic_fixed_point(A1, C1, local_steps=2, lr=0.1)
    assert np.allclose(fp, [1.02 / 0.35]), (
        f"T_k = (1 - 0.1 a_k)^2 = (0.81, 0.49), s = (0, 0.51 * 4): w* = 0.5 * 2.04 / (1 - 0.5 * 1.3) = 2.914; got {fp}")
    tr = nb.quadratic_fedavg(A1, C1, np.array([0.0]), rounds=300, local_steps=2, lr=0.1)
    assert np.allclose(tr[-1], fp, atol=1e-8), "quadratic_fedavg should converge to quadratic_fixed_point"
    assert np.allclose(nb.quadratic_fixed_point(A1, C1, local_steps=1, lr=0.1), [3.0]), "one local step: no drift"
    far = nb.quadratic_fixed_point(A1, C1, local_steps=2000, lr=0.1)
    assert np.allclose(far, [2.0], atol=1e-6), (
        f"with many local steps each client reaches its own optimum, so FedAvg converges to their AVERAGE (0 + 4) / 2 "
        f"= 2, not to the global optimum 3; got {far}")


def test_stage2_drift_needs_heterogeneous_curvature(nb):
    A_same = np.array([[[2.0, 0.3], [0.3, 1.0]]] * 3)
    cs = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    for K in (1, 5, 50):
        fp = nb.quadratic_fixed_point(A_same, cs, local_steps=K, lr=0.1)
        assert np.allclose(fp, cs.mean(0)), (
            f"with identical curvatures, FedAvg's fixed point is the optimum for every K (here {cs.mean(0)}); got {fp} at K={K}")


def test_stage2_fedprox_fixed_point(nb):
    w0 = np.zeros(2)
    for K, mu in ((5, 0.5), (50, 2.0)):
        tr = nb.quadratic_fedavg(A2, C2, w0, rounds=600, local_steps=K, lr=0.1, mu=mu)
        fp = nb.quadratic_fixed_point(A2, C2, local_steps=K, lr=0.1, mu=mu)
        assert np.allclose(tr[-1], fp, atol=1e-7), (
            f"K={K}, mu={mu}: the iterates converge to {tr[-1]} but the closed form says {fp}. the proximal term "
            "pulls toward the round's STARTING point w^t (fixed during local steps), and it enters T_k too")
    opt = nb.global_optimum(A2, C2)
    e_avg = np.abs(nb.quadratic_fixed_point(A2, C2, local_steps=100, lr=0.1) - opt).max()
    e_prox = np.abs(nb.quadratic_fixed_point(A2, C2, local_steps=100, lr=0.1, mu=1.0) - opt).max()
    assert e_prox < 0.6 * e_avg, f"with K=100, mu=1 should cut the drift substantially: {e_prox:.3f} vs {e_avg:.3f}"
    wt = nb.quadratic_fixed_point(A1, C1, local_steps=1, lr=0.1, weights=[3, 1])
    assert np.allclose(wt, [2.0]), f"weights [3, 1] with one local step: the weighted optimum 2; got {wt}"


# ---------------------------------------------------------------- stage 3: SCAFFOLD


def test_stage3_scaffold_removes_drift(nb):
    opt = nb.global_optimum(A2, C2)
    for K in (5, 20):
        tr = bounded(nb.quadratic_scaffold, A2, C2, np.zeros(2), rounds=300, local_steps=K, lr=0.1,
                     what="quadratic_scaffold")
        assert tr.shape == (301, 2), f"return (rounds + 1, d) iterates; got {tr.shape}"
        assert np.allclose(tr[-1], opt, atol=1e-6), (
            f"K={K}: SCAFFOLD should converge to the global optimum {opt}, got {tr[-1]} (FedAvg's fixed point is "
            f"{nb.quadratic_fixed_point(A2, C2, local_steps=K, lr=0.1)}). check the correction -c_i + c in every "
            "local step and c_i+ = c_i - c + (x - y_i) / (K lr)")


def test_stage3_first_round_and_one_step(nb):
    w0 = np.array([2.0, -1.0])
    sc = nb.quadratic_scaffold(A2, C2, w0, rounds=1, local_steps=7, lr=0.1)
    fa = nb.quadratic_fedavg(A2, C2, w0, rounds=1, local_steps=7, lr=0.1)
    assert np.allclose(sc, fa), "control variates start at 0, so SCAFFOLD's first round is exactly FedAvg's"
    sc1 = nb.quadratic_scaffold(A2, C2, w0, rounds=5, local_steps=1, lr=0.1)
    fa1 = nb.quadratic_fedavg(A2, C2, w0, rounds=5, local_steps=1, lr=0.1)
    assert np.allclose(sc1, fa1), "with one local step the corrections cancel in the average: SCAFFOLD = gradient descent"
    trw = nb.quadratic_scaffold(A1, C1, np.array([0.0]), rounds=400, local_steps=10, lr=0.1, weights=[3, 1])
    assert np.allclose(trw[-1], [2.0], atol=1e-6), (
        f"weights [3, 1]: converge to the weighted optimum 2; got {trw[-1]}. the server's c is the WEIGHTED average of "
        "the c_i, and the server step uses the same weights")


# ---------------------------------------------------------------- stage 4: FedProx


def test_stage4_two_full_batch_steps(nb):
    X, y = _data(30, 0)
    m, ref = _net(1), _net(1)
    w0 = _flat(ref)

    def grad(model):
        model.zero_grad()
        F.cross_entropy(model(X), y).backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters()])

    g0 = grad(ref)
    w1 = w0 - 0.2 * g0
    torch.nn.utils.vector_to_parameters(w1, ref.parameters())
    w2 = w1 - 0.2 * (grad(ref) + 0.7 * (w1 - w0))
    w, steps = nb.fedprox_client_update(m, X, y, mu=0.7, epochs=2, batch_size=None, lr=0.2, seed=0)
    assert steps == 2, f"2 epochs with batch_size=None is 2 steps; got {steps}"
    assert torch.allclose(w, w2, atol=1e-6), (
        "two full-batch steps: w1 = w0 - lr g(w0) (the prox gradient is 0 at the start), "
        "w2 = w1 - lr (g(w1) + mu (w1 - w0)). the anchor w0 is the weights at ENTRY, held fixed")
    assert torch.allclose(_flat(m), w), "train the model in place and return its weights"


def test_stage4_mu_pulls_toward_the_entry_weights(nb):
    ratios = []
    for s in range(3):
        X, y = _data(40, 1 + s)
        dists = []
        for mu in (0.0, 1.0, 3.0):
            m = _net(2 + s)
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(3.0)  # far from zero, so "toward the anchor" and "toward zero" differ
            w0 = _flat(m)
            w, _ = nb.fedprox_client_update(m, X, y, mu=mu, epochs=5, batch_size=8, lr=0.1, seed=s)
            dists.append((w - w0).norm().item())
            if mu == 3.0:
                assert w.norm() > 0.8 * w0.norm(), "the proximal term pulls toward the starting weights, not toward zero"
        ratios.append((dists[1] / dists[0], dists[2] / dists[0]))
    r1, r3 = np.median(np.array(ratios), axis=0)
    assert r1 < 0.85 and r3 < 0.7, (
        f"the proximal term keeps the client closer to where it started: relative to mu = 0, the distance moved "
        f"is {r1:.2f} for mu = 1 and {r3:.2f} for mu = 3 (median of 3 runs; expected about 0.6 and 0.35). if mu "
        "changes nothing, the anchor moves with the weights: snapshot it at entry")


def test_stage4_fits_the_fedavg_slot(nb):
    X, y = _data(25, 2)
    seen = []

    def loss_fn(out, yb):
        seen.append(len(yb))
        return F.cross_entropy(out, yb)

    w, steps = bounded(nb.fedprox_client_update, _net(3), X, y, epochs=2, batch_size=10, lr=0.1, seed=4,
                       loss_fn=loss_fn, what="fedprox_client_update")
    assert steps == 6 and seen == [10, 10, 5] * 2, (
        f"2 epochs of minibatches 10, 10, 5 (6 steps), like client_update; got {seen}")
    a, _ = nb.fedprox_client_update(_net(3), X, y, epochs=2, batch_size=10, lr=0.1, seed=4)
    b, _ = nb.fedprox_client_update(_net(3), X, y, epochs=2, batch_size=10, lr=0.1, seed=4)
    assert torch.equal(a, b), "same seed, same result (shuffle with np.random.default_rng(seed).permutation)"


# ---------------------------------------------------------------- stage 5: personalization


def test_stage5_fine_tune_copies_and_steps(nb):
    X, y = _data(20, 3)
    m = _net(4)
    w0 = _flat(m)
    tuned = nb.fine_tune(m, X, y, steps=1, lr=0.3, batch_size=None, seed=0)
    assert tuned is not m and torch.equal(_flat(m), w0), "fine_tune returns a fine-tuned COPY; the global model is untouched"
    m.zero_grad()
    F.cross_entropy(m(X), y).backward()
    g = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
    assert torch.allclose(_flat(tuned), w0 - 0.3 * g, atol=1e-6), "one full-batch step: w - lr * grad"


def test_stage5_steps_and_head_only(nb):
    X, y = _data(7, 5)
    sizes = []

    def loss_fn(out, yb):
        sizes.append(len(yb))
        return F.cross_entropy(out, yb)

    m = _net(5)
    tuned = bounded(nb.fine_tune, m, X, y, steps=4, lr=0.1, batch_size=5, head_only=True, loss_fn=loss_fn, seed=0,
                    what="fine_tune")
    assert sizes == [5, 5, 5, 5], (
        f"`steps` counts SGD steps, each on batch_size examples (reshuffle when fewer than batch_size remain); "
        f"got batch sizes {sizes}")
    assert torch.equal(tuned[0].weight, m[0].weight) and torch.equal(tuned[0].bias, m[0].bias), (
        "head_only=True trains only the LAST nn.Linear; the first layer must not change")
    assert not torch.equal(tuned[2].weight, m[2].weight), "head_only=True should still train the last layer"


def test_stage5_per_client_accuracy_and_gain(nb):
    lin = nn.Linear(2, 2)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(2))
        lin.bias.zero_()
    data = [(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
            (torch.tensor([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]]), torch.tensor([1, 0, 0]))]
    acc = nb.per_client_accuracy(lin, data)
    assert isinstance(acc, np.ndarray) and np.allclose(acc, [1.0, 1 / 3]), (
        f"argmax of x (identity model): client 0 gets 2/2 right, client 1 gets 1/3; got {acc}")
    assert np.allclose(nb.per_client_accuracy([lin, lin], data), [1.0, 1 / 3]), "a list holds one model per client"
    # a client whose labels are rotated relative to everyone else: fine-tuning should fix it
    gains = []
    for s in range(3):
        X, y = _data(60, 10 + s)
        y_rot = (y + 1) % 3
        glob = _net(s)
        F_X, F_y = _data(300, 20 + s)
        opt = torch.optim.SGD(glob.parameters(), lr=0.5)
        for _ in range(100):
            opt.zero_grad()
            F.cross_entropy(glob(F_X), F_y).backward()
            opt.step()
        tuned = nb.fine_tune(glob, X, y_rot, steps=60, lr=0.3, batch_size=10, seed=s)
        gains.append(nb.per_client_accuracy(tuned, [(X, y_rot)])[0] - nb.per_client_accuracy(glob, [(X, y_rot)])[0])
    assert np.median(gains) > 0.5, f"fine-tuning on a client with rotated labels should lift its accuracy a lot; gains {gains}"


# ---------------------------------------------------------------- one test per spec clause


def _spec_dirichlet(y, n_clients, alpha, seed):
    """the spec's algorithm, one attempt (min_size=0)."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        p = rng.dirichlet(np.full(n_clients, float(alpha)))
        cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
        for k, chunk in enumerate(np.split(idx, cuts)):
            parts[k].append(chunk)
    return [np.concatenate(ps) for ps in parts]


def test_stage1_dirichlet_follows_the_spec(nb):
    y = np.repeat(np.arange(4), [7, 13, 20, 9])
    np.random.default_rng(0).shuffle(y)
    got = nb.partition_dirichlet(y, 6, 0.7, seed=3, min_size=0)
    want = _spec_dirichlet(y, 6, 0.7, 3)
    for k, (g, w) in enumerate(zip(got, want)):
        assert sorted(g.tolist()) == sorted(w.tolist()), (
            f"client {k}: expected indices {sorted(w.tolist())}, got {sorted(g.tolist())}. per class (sorted labels): "
            "shuffle the class's indices with rng.permutation, draw rng.dirichlet(alpha * ones), cut at "
            "(cumsum(p)[:-1] * n_c).astype(int) (truncation, not rounding)")


def test_stage1_min_size_boundary_and_resampling(nb):
    y = np.repeat(np.arange(5), 20)
    whole = bounded(nb.partition_dirichlet, y, 1, 1.0, seed=0, min_size=100, what="partition_dirichlet")
    assert len(whole) == 1 and len(whole[0]) == 100, (
        "one client gets all 100 examples, which meets min_size=100 exactly: the check is >= min_size, not >")
    for s in range(5):
        parts = bounded(nb.partition_dirichlet, y, 5, 1.0, seed=s, min_size=14, what="partition_dirichlet")
        assert min(map(len, parts)) >= 14, (
            f"seed {s}: every client must hold >= min_size = 14 examples; got sizes {[len(p) for p in parts]}. only "
            "~1 attempt in 4 succeeds here, so keep drawing new splits until one does")


def test_stage5_eval_mode_and_fresh_gradients(nb):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(3, 16), nn.Dropout(0.9), nn.Linear(16, 3))
    X, y = _data(200, 7)
    m.train()
    a = nb.per_client_accuracy(m, [(X, y)])
    m.train()
    b = nb.per_client_accuracy(m, [(X, y)])
    m.eval()
    with torch.no_grad():
        want = (m(X).argmax(1) == y).float().mean().item()
    assert a[0] == b[0] == want, (
        f"per_client_accuracy must put models in eval mode (dropout off): got {a[0]:.3f}, {b[0]:.3f}, expected {want:.3f}")
    X2, y2 = _data(20, 8)
    base = _net(9)
    w0 = _flat(base)

    def grad_at(w):
        tmp = _net(9)
        torch.nn.utils.vector_to_parameters(w, tmp.parameters())
        F.cross_entropy(tmp(X2), y2).backward()
        return torch.cat([p.grad.reshape(-1) for p in tmp.parameters()])

    w1 = w0 - 0.2 * grad_at(w0)
    w2 = w1 - 0.2 * grad_at(w1)
    tuned = nb.fine_tune(base, X2, y2, steps=2, lr=0.2, batch_size=None, seed=0)
    assert torch.allclose(_flat(tuned), w2, atol=1e-6), (
        "two full-batch steps should give w2 = w1 - lr g(w1) with w1 = w0 - lr g(w0). if the second step used "
        "g(w0) + g(w1), you forgot to zero the gradients between steps")


def test_stage2_more_hand_values(nb):
    fp = nb.quadratic_fixed_point(A1, C1, local_steps=2000, lr=0.1, mu=1.0)
    assert np.allclose(fp, [2.4], atol=1e-6), (
        f"FedProx, mu = 1, many local steps: clients return (a c + mu w) / (a + mu), fixed point 2.4; got {fp}. "
        "the proximal gradient is mu (w - w^t), not 2 mu (w - w^t)")
    tr = nb.quadratic_fedavg(A1, C1, np.array([0.0]), rounds=2, local_steps=2, lr=0.1, weights=[3, 1])
    assert np.allclose(tr.ravel(), [0, 0.51, 0.8823]), (
        f"weights [3, 1]: round 1 is 0.75 * 0 + 0.25 * 2.04 = 0.51, round 2 is 0.8823; got {tr.ravel()}. "
        "use the normalized weights in the server average")


def test_stage3_hand_values(nb):
    sc = nb.quadratic_scaffold(A1, C1, np.array([0.0]), rounds=3, local_steps=2, lr=0.1)
    assert np.allclose(sc.ravel(), [0, 1.02, 1.734, 2.19504]), (
        f"SCAFFOLD on the worked example, K = 2: expected [0, 1.02, 1.734, 2.19504], got {np.round(sc.ravel(), 5)}. "
        "use option II: c_i+ = c_i - c + (x - y_i) / (K lr) (not the gradient at x, option I)")
