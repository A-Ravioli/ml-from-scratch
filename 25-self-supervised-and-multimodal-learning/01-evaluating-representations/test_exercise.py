"""Checks for exercise.py (evaluating representations).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Feature extraction is
checked against direct forward passes with recording encoders; the probe against the course's own
SoftmaxRegression; kNN against a brute-force implementation and hand-built neighbourhoods where each
of k, the temperature and the cosine normalisation changes the answer; fine-tuning on tiny separable
data, with the frozen encoder's state checked bit for bit.
"""
import tracemalloc

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlfs.linear import SoftmaxRegression
from mlfs.testing import bounded

torch.set_num_threads(1)  # tiny networks: one thread is fastest, and avoids oversubscribing a busy machine


def _loader(X, y, batch_size=16, shuffle=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y)),
                      batch_size=batch_size, shuffle=shuffle, generator=g)


# ---------------------------------------------------------------- stage 1: extract_features

def test_stage1_matches_forward_pass(nb):
    torch.manual_seed(0)
    enc = nn.Sequential(nn.Linear(5, 7), nn.Tanh())
    X = np.random.default_rng(0).normal(size=(37, 5))
    y = np.arange(37) % 3
    feats, labels = nb.extract_features(enc, _loader(X, y, batch_size=10))
    assert isinstance(feats, np.ndarray) and isinstance(labels, np.ndarray), (
        f"return two NumPy arrays (feats, labels), got {type(feats).__name__}, {type(labels).__name__}")
    assert feats.shape == (37, 7), f"features should be (n, d) = (37, 7), got {feats.shape}"
    assert feats.dtype == np.float32, f"features should be float32, got {feats.dtype}"
    with torch.no_grad():
        want = enc(torch.as_tensor(X, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(feats, want, rtol=1e-5, atol=1e-6,
                               err_msg="features differ from encoder(x): are batches kept in loader order?")
    np.testing.assert_array_equal(labels, y, err_msg="labels must be the loader's y, in order")


def test_stage1_flattens_and_casts(nb):
    class Grid(nn.Module):
        def forward(self, x):
            return x.double().reshape(x.shape[0], 2, 3) * 2

    X = np.random.default_rng(1).normal(size=(9, 6))
    feats, _ = nb.extract_features(Grid(), _loader(X, np.zeros(9, dtype=np.int64), batch_size=4))
    assert feats.shape == (9, 6), f"an encoder output (B, 2, 3) should be flattened to (B, 6), got {feats.shape}"
    assert feats.dtype == np.float32, f"float64 encoder outputs should come back as float32, got {feats.dtype}"
    np.testing.assert_allclose(feats, (2 * X).astype(np.float32), rtol=1e-6)


def test_stage1_eval_mode_and_no_grad(nb):
    seen = []

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(3)
            self.drop = nn.Dropout(0.5)

        def forward(self, x):
            seen.append((self.training, self.bn.training, torch.is_grad_enabled()))
            return self.drop(self.bn(x))

    torch.manual_seed(0)
    enc = Spy()
    enc.bn.running_mean.fill_(1.0)
    enc.bn.running_var.fill_(4.0)
    X = np.random.default_rng(2).normal(size=(12, 3))
    feats, _ = nb.extract_features(enc, _loader(X, np.zeros(12, dtype=np.int64), batch_size=5))
    assert seen and all(not t and not b for t, b, _ in seen), (
        "the encoder (and its BatchNorm/Dropout submodules) must be in eval mode during extraction: call "
        "encoder.eval() first")
    assert all(not g for _, _, g in seen), "run the encoder under torch.no_grad(): features need no autograd graph"
    np.testing.assert_allclose(feats, ((X - 1.0) / np.sqrt(4.0 + 1e-5)).astype(np.float32), rtol=1e-5, atol=1e-6,
                               err_msg="in eval mode BatchNorm uses its running statistics and Dropout is off")
    assert enc.bn.running_mean[0].item() == 1.0, "extraction must not update BatchNorm running statistics"


def test_stage1_restores_every_mode(nb):
    enc = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Dropout(0.1))
    enc.train()
    enc[1].eval()  # a frozen BatchNorm inside a training model: a common fine-tuning setup
    X = np.random.default_rng(3).normal(size=(8, 3))
    nb.extract_features(enc, _loader(X, np.zeros(8, dtype=np.int64)))
    got = [m.training for m in enc.modules()]
    assert got == [True, True, False, True], (
        f"after extraction every submodule's mode must be what it was before (got {got}, want "
        "[True, True, False, True]): record each module's .training first and restore it; encoder.train() "
        "would wrongly un-freeze the BatchNorm")
    enc.eval()
    nb.extract_features(enc, _loader(X, np.zeros(8, dtype=np.int64)))
    assert not any(m.training for m in enc.modules()), "an encoder that was in eval mode must stay in eval mode"


def test_stage1_moves_batches_to_device(nb):
    devices = []

    class Recorder(nn.Module):
        def forward(self, x):
            devices.append(x.device.type)
            return torch.zeros(x.shape[0], 2)

    X = np.zeros((6, 2))
    feats, _ = nb.extract_features(Recorder(), _loader(X, np.zeros(6, dtype=np.int64), batch_size=3), device="meta")
    assert devices == ["meta", "meta"], (
        f"each batch x must be moved to `device` before the encoder sees it (saw {devices}): x.to(device)")
    assert feats.shape == (6, 2)


# ---------------------------------------------------------------- stage 2: linear_probe

def _oracle_probe(Xtr, ytr, Xte, yte, alpha, max_iters):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    clf = SoftmaxRegression(alpha=alpha, max_iters=max_iters).fit((Xtr - mu) / sd, ytr)
    return float(np.mean(clf.predict((Xte - mu) / sd) == yte))


def _blobs(n, rng, d=4, k=3, spread=1.0):
    centres = rng.normal(scale=2.0, size=(k, d))
    y = rng.integers(0, k, size=n)
    return centres[y] + spread * rng.normal(size=(n, d)), y


def test_stage2_matches_softmax_regression(nb):
    rng = np.random.default_rng(0)
    X, y = _blobs(300, rng, spread=2.0)
    Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]
    got = bounded(nb.linear_probe, Xtr, ytr, Xte, yte, what="linear_probe", budget=5_000_000)
    want = _oracle_probe(Xtr, ytr, Xte, yte, 1e-4, 1000)
    assert isinstance(got, float), f"return a Python float, got {type(got).__name__}"
    assert got == pytest.approx(want, abs=1e-9), (
        f"test accuracy {got:.4f}, expected {want:.4f} from SoftmaxRegression(alpha=1e-4, max_iters=1000) on "
        "features standardised with the training mean and std")


def test_stage2_alpha_and_max_iters_are_used(nb):
    rng = np.random.default_rng(1)
    n = 400
    y = (rng.random(n) < 0.2).astype(int)  # imbalanced: heavy regularisation predicts the majority class
    X = rng.normal(size=(n, 3)) + 1.2 * y[:, None]
    Xtr, ytr, Xte, yte = X[:250], y[:250], X[250:], y[250:]
    for alpha, iters in ((5.0, 1000), (1e-4, 2), (1e-4, 1000)):
        got = nb.linear_probe(Xtr, ytr, Xte, yte, alpha=alpha, max_iters=iters)
        want = _oracle_probe(Xtr, ytr, Xte, yte, alpha, iters)
        assert got == pytest.approx(want, abs=1e-9), (
            f"with alpha={alpha}, max_iters={iters} the accuracy should be {want:.4f}, got {got:.4f}: pass alpha "
            "and max_iters through to SoftmaxRegression")
    # (at these fixed seeds the three settings give three different accuracies, so the check has power)


def test_stage2_standardises_with_training_statistics(nb):
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 100)
    Xtr = np.column_stack([np.where(y == 1, 2.0, -2.0) + 0.5 * rng.normal(size=200), rng.normal(size=200)])
    Xte = np.column_stack([2.0 + 0.5 * rng.normal(size=60), rng.normal(size=60)])  # only class 1 at test time
    got = nb.linear_probe(Xtr, y, Xte, np.ones(60, dtype=int))
    assert got > 0.95, (
        f"accuracy {got:.2f} on a test set that contains only class 1: the probe must standardise the test features "
        "with the TRAINING mean and std. standardising the test set with its own statistics moves its points to the "
        "middle of the decision boundary")


def test_stage2_is_scale_invariant(nb):
    rng = np.random.default_rng(3)
    X, y = _blobs(300, rng, spread=1.5)
    Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]
    base = nb.linear_probe(Xtr, ytr, Xte, yte)
    s = np.array([1e4, 1.0, 1e-3, 1.0])
    with np.errstate(all="ignore"):
        scaled = nb.linear_probe(Xtr * s, ytr, Xte * s, yte)
    assert scaled == pytest.approx(base, abs=1e-9), (
        f"rescaling the features changed the probe's accuracy ({base:.3f} -> {scaled:.3f}): standardise every feature "
        "(subtract the training mean, divide by the training std) before fitting")


def test_stage2_no_test_statistics_at_all(nb):
    rng = np.random.default_rng(8)
    Xtr = rng.normal(size=(300, 3))
    ytr = (Xtr @ np.array([1.0, -0.5, 0.3]) + 0.5 * rng.normal(size=300) > 0).astype(int)
    Xte = rng.normal(size=(600, 3)) * 3 + 1.5  # a shifted, wider test distribution
    yte = (Xte @ np.array([1.0, -0.5, 0.3]) > 0).astype(int)
    got = nb.linear_probe(Xtr, ytr, Xte, yte, alpha=0.5)
    want = _oracle_probe(Xtr, ytr, Xte, yte, 0.5, 1000)
    assert got == pytest.approx(want, abs=1e-9), (
        f"accuracy {got:.4f}, expected {want:.4f}: the mean and the (population) std must come from the TRAINING "
        "features alone, not from the test set and not from train and test combined")
    rng = np.random.default_rng(1)  # four training points: ddof=0 vs ddof=1 rescales the features by 15%
    Xs, ys = rng.normal(size=(4, 2)), np.array([0, 0, 0, 1])
    Xd = rng.normal(size=(2000, 2)) * 2
    yd = (Xd[:, 0] > 0).astype(int)
    small = nb.linear_probe(Xs, ys, Xd, yd, alpha=0.5)
    assert small == pytest.approx(_oracle_probe(Xs, ys, Xd, yd, 0.5, 1000), abs=1e-9), (
        "with 4 training points: use the population std (ddof=0, numpy's default), as the oracle does")


def test_stage2_constant_feature_and_labels(nb):
    rng = np.random.default_rng(4)
    X, y = _blobs(200, rng, d=3, k=2, spread=1.0)
    names = np.array(["cat", "dog"])[y]
    Xc = np.column_stack([X, np.full(200, 7.0)])  # a dead unit: zero variance
    got = nb.linear_probe(Xc[:150], names[:150], Xc[150:], names[150:])
    want = _oracle_probe(X[:150], y[:150], X[150:], y[150:], 1e-4, 1000)
    assert np.isfinite(got) and got == pytest.approx(want, abs=1e-9), (
        f"a constant feature (a dead unit) should be harmless, got accuracy {got}: guard the division by its zero "
        "std (divide by 1 instead). labels may be strings")


# ---------------------------------------------------------------- stage 3: knn_eval

def _brute_knn(tr, ytr, te, yte, k, t):
    tr = tr / np.linalg.norm(tr, axis=1, keepdims=True)
    te = te / np.linalg.norm(te, axis=1, keepdims=True)
    classes = np.unique(ytr)
    sims = te @ tr.T
    preds = []
    for row in sims:
        idx = np.argsort(-row)[:k]
        votes = np.array([np.exp(row[idx][ytr[idx] == c] / t).sum() for c in classes])
        preds.append(classes[np.argmax(votes)])
    return float(np.mean(np.array(preds) == yte))


def test_stage3_worked_example(nb):
    tr = np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [-1.0, 0.0]])
    ytr = np.array([0, 1, 1, 0])
    q = np.array([[0.96, 0.28]])
    for k, t, want_label in ((1, 0.1, 0), (2, 0.1, 0), (3, 0.1, 0), (3, 1.0, 1), (4, 10.0, 1)):
        got = nb.knn_eval(tr, ytr, q, np.array([want_label]), k=k, temperature=t)
        assert got == 1.0, (
            f"lesson worked example, k={k}, temperature={t}: the query (0.96, 0.28) should be labelled {want_label}")
    got = nb.knn_eval(tr, ytr, np.array([[0.995, 0.0998]]), np.array([0]), k=3, temperature=0.05)
    assert got == 1.0, ("query (0.995, 0.0998) with k=3, T=0.05: the class-0 neighbour (sim 0.995) outweighs the two "
                        "class-1 neighbours (sims 0.856 and 0.677) once votes are weighted by exp(sim / T)")


def test_stage3_k_and_temperature_change_the_vote(nb):
    # two class-0 points very close to the query, three class-1 points a little farther away
    q = np.array([[1.0, 0.0]])
    ang = np.array([0.14, -0.2, 0.45, -0.45, 0.46])
    tr = np.column_stack([np.cos(ang), np.sin(ang)])
    ytr = np.array([0, 0, 1, 1, 1])
    cases = [(1, 1.0, 0), (5, 0.01, 0), (5, 1.0, 1), (5, 0.07, 0), (5, 0.3, 1)]
    for k, t, want in cases:
        got = nb.knn_eval(tr, ytr, q, np.array([want]), k=k, temperature=t)
        assert got == 1.0, (
            f"k={k}, temperature={t}: expected label {want}. the two class-0 neighbours have cosine "
            f"{np.cos(0.14):.3f} and {np.cos(0.2):.3f}, the three class-1 ones about {np.cos(0.45):.3f}; votes are "
            "weighted by exp(sim / temperature), so a small temperature lets the closest points dominate and a "
            "large one approaches a plain majority vote")


def test_stage3_uses_cosine_similarity(nb):
    rng = np.random.default_rng(5)
    tr = rng.normal(size=(60, 5))
    ytr = rng.integers(0, 3, size=60)
    te = rng.normal(size=(25, 5))
    yte = rng.integers(0, 3, size=25)
    base = nb.knn_eval(tr, ytr, te, yte, k=7)
    scale_tr = rng.uniform(0.01, 100, size=(60, 1))
    scale_te = rng.uniform(0.01, 100, size=(25, 1))
    got = nb.knn_eval(tr * scale_tr, ytr, te * scale_te, yte, k=7)
    assert got == base, (
        f"rescaling feature vectors changed the kNN accuracy ({base:.3f} -> {got:.3f}): neighbours are ranked by "
        "COSINE similarity, so L2-normalise both train and test features first")


def test_stage3_matches_brute_force(nb):
    rng = np.random.default_rng(6)
    tr, ytr = _blobs(300, rng, d=6, k=4, spread=2.5)
    te, yte = _blobs(120, np.random.default_rng(60), d=6, k=4, spread=2.5)
    for k, t in ((1, 0.07), (10, 0.07), (20, 0.5), (50, 2.0)):
        got = bounded(nb.knn_eval, tr, ytr, te, yte, k=k, temperature=t, what="knn_eval")
        want = _brute_knn(tr, ytr, te, yte, k, t)
        assert isinstance(got, float), f"return a Python float, got {type(got).__name__}"
        assert got == pytest.approx(want, abs=1e-9), f"k={k}, temperature={t}: accuracy {got:.4f}, expected {want:.4f}"


def test_stage3_ties_go_to_the_smallest_label(nb):
    tr = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    for labels, want in (([5, 2, 9], 2), ([2, 5, 9], 2), (["b", "a", "c"], "a")):
        got = nb.knn_eval(tr, np.array(labels), np.array([[1.0, 0.0]]), np.array([want]), k=2)
        assert got == 1.0, (
            f"training labels {labels}: the query's two neighbours are identical (an exact tie between "
            f"{labels[0]!r} and {labels[1]!r}); ties go to the smallest label, {want!r}")


def test_stage3_edge_cases(nb):
    tr = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    ytr = np.array([7, 3, 7])  # labels need not be 0..k-1
    te = np.array([[1.0, 0.05], [0.1, 1.0]])
    got = nb.knn_eval(tr, ytr, te, np.array([7, 3]), k=50, temperature=0.07)
    assert got == 1.0, "k larger than the training set must work (use every training point) and keep the labels"
    with np.errstate(over="raise"):
        try:
            got = nb.knn_eval(tr, ytr, te, np.array([7, 3]), k=3, temperature=1e-3)
        except FloatingPointError:
            raise AssertionError("exp(sim / temperature) overflowed at temperature=1e-3: subtract each query's "
                                 "largest similarity before exponentiating (it doesn't change the vote)") from None
    assert got == 1.0, f"with a tiny temperature the vote is the nearest neighbour's, got accuracy {got}"


def test_stage3_memory_is_bounded(nb):
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(10, 8))
    ytr = rng.integers(0, 10, size=20_000)
    tr = (centres[ytr] + 0.8 * rng.normal(size=(20_000, 8))).astype(np.float32)
    yte = rng.integers(0, 10, size=3_000)
    te = (centres[yte] + 0.8 * rng.normal(size=(3_000, 8))).astype(np.float32)
    tracemalloc.start()
    try:
        got = nb.knn_eval(tr, ytr, te, yte, k=20)
        peak = tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()
    assert peak < 64, (
        f"peak memory {peak:.0f} MB: the full (n_test, n_train) similarity matrix is 3,000 x 20,000 here (and "
        "10,000 x 50,000 on CIFAR-10). process the test points in chunks so each block stays a few MB")
    # an independent oracle, in this test's own chunks of 100 test points
    trn = tr / np.linalg.norm(tr, axis=1, keepdims=True)
    ten = te / np.linalg.norm(te, axis=1, keepdims=True)
    correct = 0
    for s in range(0, 3000, 100):
        sims = ten[s:s + 100].astype(np.float64) @ trn.T.astype(np.float64)
        idx = np.argpartition(-sims, 19, axis=1)[:, :20]
        w = np.exp((np.take_along_axis(sims, idx, 1) - sims.max(1, keepdims=True)) / 0.07)
        votes = np.zeros((len(idx), 10))
        np.add.at(votes, (np.arange(len(idx))[:, None], ytr[idx]), w)
        correct += int((votes.argmax(1) == yte[s:s + 100]).sum())
    want = correct / 3000
    assert abs(got - want) <= 2 / 3000, (
        f"accuracy {got:.4f}, expected {want:.4f} (allowing two float-precision flips): with many chunks, every "
        "chunk's predictions must be compared with THAT chunk's labels, and the accuracy is over all test points")


# ---------------------------------------------------------------- stage 4: fine_tune

def _separable(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, 4)) + 2.0 * np.where(y == 1, 1.0, -1.0)[:, None] * np.array([1.0, -1.0, 0.5, 0.0])
    return X, y


def _encoder(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU())


def _accuracy(model, X, y):
    with torch.no_grad():
        return float((model(torch.as_tensor(X, dtype=torch.float32)).argmax(1).numpy() == y).mean())


def test_stage4_learns_end_to_end(nb):
    accs = []
    for seed in range(3):
        X, y = _separable(256, seed)
        model = nb.fine_tune(_encoder(seed), 8, 2, _loader(X, y, 32, shuffle=True, seed=seed), epochs=5, lr=1e-2,
                             seed=seed)
        accs.append(_accuracy(model, X, y))
    assert np.median(accs) > 0.95, f"training accuracies over 3 seeds {np.round(accs, 3)}: expected > 0.95"
    assert isinstance(model, nn.Sequential) and len(model) == 2 and isinstance(model[1], nn.Linear), (
        "return nn.Sequential(encoder, nn.Linear(feature_dim, n_classes))")
    assert model[1].in_features == 8 and model[1].out_features == 2
    assert not model.training, "return the model in eval mode, ready to evaluate"


def test_stage4_freeze_leaves_encoder_untouched(nb):
    X, y = _separable(128, 10)
    enc = _encoder(10)
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    model = nb.fine_tune(enc, 8, 2, _loader(X, y, 32, shuffle=True), epochs=3, lr=1e-2, freeze=True, seed=0)
    after = enc.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), (
            f"freeze=True changed the encoder's {k!r}: only the head may train, and the encoder must stay in eval "
            "mode so its BatchNorm running statistics don't move either")
    assert model[0] is enc
    torch.manual_seed(0)
    fresh = nn.Linear(8, 2)  # the head fine_tune creates right after torch.manual_seed(seed)
    assert not torch.equal(model[1].weight, fresh.weight), "with freeze=True the head must still be trained"

    enc2 = _encoder(10)
    enc2.eval()  # e.g. left in eval mode by extract_features: fine-tuning must switch it back to train mode
    nb.fine_tune(enc2, 8, 2, _loader(X, y, 32, shuffle=True), epochs=1, lr=1e-2, freeze=False, seed=0)
    changed = [k for k, v in before.items() if not torch.equal(v, enc2.state_dict()[k])]
    assert "0.weight" in changed and "1.running_mean" in changed, (
        f"freeze=False must train the encoder too, in train mode (weights and BatchNorm statistics); changed only "
        f"{changed}. call model.train() before the training loop, even if the encoder arrived in eval mode")


def test_stage4_epochs_lr_and_seed(nb):
    X, y = _separable(64, 20)

    class Counting:
        def __init__(self, loader):
            self.loader, self.batches = loader, 0

        def __iter__(self):
            for b in self.loader:
                self.batches += 1
                yield b

        def __len__(self):
            return len(self.loader)

    cl = Counting(_loader(X, y, 16))
    enc = _encoder(0)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    nb.fine_tune(enc, 8, 2, cl, epochs=3, seed=0)
    assert cl.batches == 12, f"3 epochs over 4 batches should draw 12 batches, drew {cl.batches}"
    assert torch.equal(torch.get_rng_state(), state), (
        "fine_tune changed the caller's global torch RNG state: seed the head's initialisation inside "
        "torch.random.fork_rng(devices=[]) instead of calling torch.manual_seed directly")

    Xt = torch.as_tensor(X, dtype=torch.float32)
    kw = dict(freeze=True)  # a frozen encoder isolates the head: lr=0 must then change nothing at all
    m0 = nb.fine_tune(_encoder(0), 8, 2, _loader(X, y, 16), epochs=0, seed=3, **kw)
    m_lr0 = nb.fine_tune(_encoder(0), 8, 2, _loader(X, y, 16), epochs=2, lr=0.0, seed=3, **kw)
    m_lr = nb.fine_tune(_encoder(0), 8, 2, _loader(X, y, 16), epochs=2, lr=1e-2, seed=3, **kw)
    m_lr_again = nb.fine_tune(_encoder(0), 8, 2, _loader(X, y, 16), epochs=2, lr=1e-2, seed=3, **kw)
    m_other = nb.fine_tune(_encoder(0), 8, 2, _loader(X, y, 16), epochs=2, lr=1e-2, seed=4, **kw)
    torch.testing.assert_close(m_lr0(Xt), m0(Xt), msg="lr=0 must leave the model as initialised: is lr passed to Adam?")
    assert not torch.allclose(m_lr(Xt), m0(Xt)), "with lr=1e-2 training should change the model"
    torch.testing.assert_close(m_lr(Xt), m_lr_again(Xt), msg="the same seed (and loader) must give the same model")
    assert not torch.allclose(m_lr(Xt), m_other(Xt)), "a different seed should initialise the head differently"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a second device (MPS) to observe `device`")
def test_stage4_device(nb):
    X, y = _separable(32, 30)
    model = nb.fine_tune(_encoder(0), 8, 2, _loader(X, y, 16), epochs=1, device="mps", seed=0)
    assert all(p.device.type == "mps" for p in model.parameters()), "move the model (encoder + head) to `device`"
