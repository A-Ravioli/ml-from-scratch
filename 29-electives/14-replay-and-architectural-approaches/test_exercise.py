"""Checks for exercise.py (replay & architectural approaches).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. examples carry their
own id in their features, so the checks can tell exactly which examples a buffer kept. the
reservoir's uniformity is measured over 300 buffers with different seeds (each example should be
kept with probability capacity / n, early or late in the stream, whole batches or not); the
class-balanced buffer on imbalanced streams whose final counts are known exactly; the Gaussian
generator on hand-computed means and standard deviations; the gated network against a forward
pass computed here and the zero gradients that isolation implies.
"""
import numpy as np
import pytest
from scipy.stats import chisquare

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _stream(ids, labels=None, d=3):
    ids = torch.as_tensor(ids, dtype=torch.float32)
    x = ids.unsqueeze(1).repeat(1, d)
    x[:, 1] = 0.5
    y = torch.zeros(len(ids), dtype=torch.int64) if labels is None else torch.as_tensor(labels, dtype=torch.int64)
    return x, y


def _ids(buf_x):
    return [int(v) for v in buf_x[:, 0].tolist()]


def _add(buf, x, y):
    return bounded(buf.add, x, y, budget=3_000_000, what="add",
                   hint="each incoming example is handled once: stored, used to replace one slot, or dropped.")


def _contents(buf):
    x, y = buf.sample(10 ** 6)
    return x, y


# ---------------------------------------------------------------- stage 1: reservoir sampling


def test_stage1_fill_and_capacity(nb):
    buf = nb.ReservoirBuffer(5, seed=0)
    assert len(buf) == 0 and buf.n_floats() == 0, "a new buffer is empty and stores nothing"
    _add(buf, *_stream([0, 1]))
    assert buf.n_floats() == 15, (
        f"n_floats counts the preallocated slots: capacity 5 x 3 features = 15 even with only 2 stored, got "
        f"{buf.n_floats()}")
    _add(buf, *_stream([2]))
    _add(buf, *_stream([3, 4]))
    assert len(buf) == 5, f"after 5 examples a 5-slot buffer holds 5, got {len(buf)}"
    x, y = _contents(buf)
    assert sorted(_ids(x)) == [0, 1, 2, 3, 4], f"until it's full, the buffer keeps everything; got ids {sorted(_ids(x))}"
    assert torch.equal(x[:, 1], torch.full((5,), 0.5)), "store the rows unchanged"
    assert buf.n_floats() == 15, f"5 slots of 3 features = 15 floats, got {buf.n_floats()}"
    _add(buf, *_stream(range(5, 100)))
    assert len(buf) == 5, f"the buffer never holds more than its capacity: got {len(buf)}"
    ids = _ids(_contents(buf)[0])
    assert len(set(ids)) == 5 and all(0 <= i < 100 for i in ids), (
        f"the buffer holds 5 distinct examples from the stream, got ids {ids}")
    with pytest.raises(ValueError):
        nb.ReservoirBuffer(0)


def test_stage1_uniform_inclusion(nb):
    n, cap, runs = 40, 8, 300
    kept = np.zeros(n)
    for seed in range(runs):
        buf = nb.ReservoirBuffer(cap, seed=seed)
        for start in range(0, n, 7):  # batches of 7, the last one short
            _add(buf, *_stream(range(start, min(n, start + 7))))
        for i in _ids(_contents(buf)[0]):
            kept[i] += 1
    p = kept / runs
    early, late = p[:10].mean(), p[-10:].mean()
    assert abs(early - 0.2) < 0.05 and abs(late - 0.2) < 0.05, (
        f"after 40 examples, an 8-slot reservoir keeps each one with probability 8/40 = 0.2, early or late. measured "
        f"{early:.2f} for the first 10 and {late:.2f} for the last 10 over {runs} seeds. replace slot j = "
        "randint(0, n_seen) only if j < capacity, with n_seen counted per *example* (not per batch)")


def test_stage1_inclusion_chi_square(nb):
    # 3 slots, 10 examples in batches of 4, 4, 2: every position should be kept with probability 0.3
    runs, kept = 2000, np.zeros(10)
    for seed in range(runs):
        buf = nb.ReservoirBuffer(3, seed=seed)
        x, y = _stream(range(10))
        add = _add if seed == 0 else (lambda b, xx, yy: b.add(xx, yy))
        for start in (0, 4, 8):
            add(buf, x[start:start + 4], y[start:start + 4])
        for i in _ids(_contents(buf)[0]):
            kept[i] += 1
    p = chisquare(kept, np.full(10, kept.sum() / 10)).pvalue
    assert len(buf) == 3 and kept.sum() == 3 * runs, "every buffer should end with exactly 3 examples"
    assert p > 1e-4, (
        f"over {runs} seeds each of the 10 positions should be kept about {0.3 * runs:.0f} times; got "
        f"{kept.astype(int).tolist()} (chi-square p = {p:.1e}). draw j uniformly from 0..n_seen-1 (n_seen counting "
        "this example, one per row, not per batch) and replace slot j only if j < capacity")


def test_stage1_sample(nb):
    buf = nb.ReservoirBuffer(10, seed=1)
    _add(buf, *_stream(range(30), labels=[i % 4 for i in range(30)]))
    x, y = buf.sample(4)
    assert x.shape == (4, 3) and y.shape == (4,), f"sample(4) returns x (4, 3) and y (4,), got {x.shape}, {y.shape}"
    assert y.dtype == torch.int64, "labels are int64"
    ids = _ids(x)
    assert len(set(ids)) == 4, f"sampled examples should be distinct, got ids {ids}"
    assert all(int(lab) == i % 4 for i, lab in zip(ids, y.tolist())), "each label must stay with its example"
    stored = set(_ids(_contents(buf)[0]))
    assert set(ids) <= stored, "samples must come from the stored examples"
    drawn = set()
    for _ in range(60):
        drawn.update(_ids(buf.sample(3)[0]))
    assert drawn == stored, (
        f"60 random samples of 3 from 10 stored examples should eventually draw every one of them; drew "
        f"{len(drawn)} of 10. choose the examples at random on every call, not the first k slots")
    xa, _ = buf.sample(50)
    assert len(xa) == 10, f"asking for more than is stored returns everything stored (10), got {len(xa)}"
    b1, b2 = nb.ReservoirBuffer(6, seed=3), nb.ReservoirBuffer(6, seed=3)
    for b in (b1, b2):
        _add(b, *_stream(range(50)))
    assert sorted(_ids(_contents(b1)[0])) == sorted(_ids(_contents(b2)[0])), "the same seed gives the same contents"


# ---------------------------------------------------------------- stage 2: class-balanced buffer


def test_stage2_imbalanced_stream_ends_balanced(nb):
    buf = nb.ClassBalancedBuffer(30, seed=0)
    labels = [0] * 300 + [1] * 60 + [2] * 30
    x, y = _stream(range(len(labels)), labels)
    for start in range(0, len(labels), 50):
        _add(buf, x[start:start + 50], y[start:start + 50])
    counts = buf.counts()
    assert counts == {0: 10, 1: 10, 2: 10}, (
        f"300 zeros, then 60 ones, then 30 twos into 30 slots should end with 10 of each class, got {counts}. "
        "when the buffer is full, an example of a class that isn't the largest evicts a random example of the "
        "largest class")
    assert len(buf) == 30 and buf.n_floats() == 90, "len is the number stored; n_floats = capacity * features"
    xs, ys = _contents(buf)
    assert all(int(lab) == labels[i] for i, lab in zip(_ids(xs), ys.tolist())), "labels must stay with examples"


def test_stage2_interleaved_stream(nb):
    rng = np.random.default_rng(0)
    labels = rng.choice(3, size=1000, p=[0.7, 0.2, 0.1])
    buf = nb.ClassBalancedBuffer(30, seed=1)
    _add(buf, *_stream(range(1000), labels))
    counts = buf.counts()
    assert sum(counts.values()) == 30 and max(counts.values()) - min(counts.values()) <= 1, (
        f"a 70/20/10 stream should still fill 30 slots evenly (10 per class), got {counts}")


def test_stage2_within_class_on_mixed_stream(nb):
    # classes 0 and 1 alternate, 40 each, into 10 slots: 5 per class, each class a reservoir of its own examples
    runs, kept = 500, np.zeros(80)
    labels = [i % 2 for i in range(80)]
    for seed in range(runs):
        buf = nb.ClassBalancedBuffer(10, seed=seed)
        buf.add(*_stream(range(80), labels))
        for i in _ids(_contents(buf)[0]):
            kept[i] += 1
    p0 = kept[0::2] / runs   # class 0's 40 examples, in stream order
    early, late = p0[:10].mean(), p0[-10:].mean()
    assert abs(early - 0.125) < 0.035 and abs(late - 0.125) < 0.035, (
        f"each class keeps 5 of its 40 examples, so every class-0 example should be kept with probability 0.125; "
        f"measured {early:.3f} (first 10) and {late:.3f} (last 10). the within-class acceptance probability is "
        "(stored of that class) / (seen *of that class*), not / (seen in total)")


def test_stage2_reservoir_within_a_class(nb):
    kept = np.zeros(30)
    runs = 300
    for seed in range(runs):
        buf = nb.ClassBalancedBuffer(10, seed=seed)
        _add(buf, *_stream(range(30)))
        for i in _ids(_contents(buf)[0]):
            kept[i] += 1
    p = kept / runs
    assert abs(p[:10].mean() - 1 / 3) < 0.07 and abs(p[-10:].mean() - 1 / 3) < 0.07, (
        f"with one class only, the buffer is a reservoir: each of 30 examples kept with probability 10/30. measured "
        f"{p[:10].mean():.2f} (first 10) and {p[-10:].mean():.2f} (last 10). when the incoming class is already the "
        "largest, keep the example with probability (stored of that class) / (seen of that class), replacing a "
        "random stored example of the same class")


# ---------------------------------------------------------------- stage 3: Gaussian generative replay


def test_stage3_fit_hand_example(nb):
    gen = nb.GaussianGenerator(min_std=0.01, seed=0)
    x = torch.tensor([[0.0, 2.0, 5.0], [2.0, 2.0, 5.0], [1.0, 8.0, 5.0], [9.0, 9.0, 9.0]])
    y = torch.tensor([1, 1, 1, 4])
    gen.fit(x, y)
    assert sorted(gen.means) == [1, 4], f"means should be a dict class -> mean, got keys {sorted(gen.means)}"
    assert torch.allclose(gen.means[1], torch.tensor([1.0, 4.0, 5.0])), f"class 1's mean is (1, 4, 5), got {gen.means[1]}"
    want = torch.tensor([(2 / 3) ** 0.5, 8 ** 0.5, 0.01])
    assert torch.allclose(gen.stds[1], want, atol=1e-6), (
        f"class 1's std is (sqrt(2/3), sqrt(8), 0.01): the biased (divide by n) std, floored at min_std for the "
        f"constant feature; got {gen.stds[1]}")
    assert torch.allclose(gen.stds[4], torch.full((3,), 0.01)), "a class with one example has std min_std everywhere"
    assert gen.n_floats() == 12, f"2 classes x (mean + std) x 3 features = 12 floats, got {gen.n_floats()}"
    gen.fit(torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]), torch.tensor([4, 4]))
    assert torch.allclose(gen.means[4], torch.full((3,), 2.0)) and torch.allclose(gen.means[1], torch.tensor(
        [1.0, 4.0, 5.0])), "refitting class 4 replaces its Gaussian and leaves class 1 alone"


def test_stage3_sampling(nb):
    gen = nb.GaussianGenerator(seed=3)
    with pytest.raises(RuntimeError):
        gen.sample(5)
    g = torch.Generator().manual_seed(0)
    mus = {0: torch.tensor([0.0, 5.0]), 3: torch.tensor([-2.0, 1.0]), 7: torch.tensor([4.0, 4.0])}
    sds = {0: torch.tensor([1.0, 0.5]), 3: torch.tensor([2.0, 0.1]), 7: torch.tensor([0.3, 3.0])}
    for c in mus:
        xc = mus[c] + sds[c] * torch.randn(4000, 2, generator=g)
        xc = (xc - xc.mean(0)) / xc.std(0, unbiased=False) * sds[c] + mus[c]  # exact moments
        gen.fit(xc, torch.full((4000,), c))
    x, y = gen.sample(6000)
    assert x.shape == (6000, 2) and y.shape == (6000,) and y.dtype == torch.int64, (
        f"sample(6000) returns x (6000, 2) and int64 y (6000,), got {tuple(x.shape)}, {tuple(y.shape)}, {y.dtype}")
    assert set(y.tolist()) == {0, 3, 7}, f"labels come from the fitted classes only, got {sorted(set(y.tolist()))}"
    for c in mus:
        sel = x[y == c]
        assert 1700 < len(sel) < 2300, f"labels should be uniform over the 3 classes; class {c} got {len(sel)} of 6000"
        assert torch.allclose(sel.mean(0), mus[c], atol=0.25) and torch.allclose(sel.std(0), sds[c], rtol=0.12), (
            f"class {c}'s samples should have mean {mus[c].tolist()} and std {sds[c].tolist()}, got "
            f"{sel.mean(0).tolist()} and {sel.std(0).tolist()}")
    g1, g2 = nb.GaussianGenerator(seed=5), nb.GaussianGenerator(seed=5)
    for gg in (g1, g2):
        gg.fit(torch.randn(10, 2, generator=torch.Generator().manual_seed(1)), torch.tensor([0, 1] * 5))
    assert torch.equal(g1.sample(7)[0], g2.sample(7)[0]), "the same seed gives the same samples"


# ---------------------------------------------------------------- stage 4: context-dependent gating


def test_stage4_masks(nb):
    net = nb.GatedMLP(6, [50, 20], 4, n_tasks=3, keep=0.2, seed=0)
    masks = net.masks()
    assert len(masks) == 2, f"one mask tensor per hidden layer (2), got {len(masks)}"
    for m, h in zip(masks, (50, 20)):
        assert tuple(m.shape) == (3, h), f"each mask is (n_tasks, h) = (3, {h}), got {tuple(m.shape)}"
        assert set(m.unique().tolist()) <= {0.0, 1.0}, "masks hold 0s and 1s"
        assert (m.sum(1) == round(0.2 * h)).all(), (
            f"every task keeps exactly round(keep * h) = {round(0.2 * h)} units of a {h}-unit layer, got "
            f"{m.sum(1).tolist()}")
    assert not torch.equal(masks[0][0], masks[0][1]), "different tasks should get different random gates"
    names = dict(net.named_buffers())
    assert sum(b.numel() for b in names.values()) >= 3 * 70, (
        "register the masks as buffers (register_buffer), so they're saved with the model and not trained")
    other = nb.GatedMLP(6, [50, 20], 4, n_tasks=3, keep=0.2, seed=0)
    assert all(torch.equal(a, b) for a, b in zip(masks, other.masks())), "the same seed gives the same gates"
    assert not any(p.requires_grad and p.shape in ((3, 50), (3, 20)) for p in net.parameters()), (
        "the gates are fixed, not trainable parameters")


def test_stage4_forward_and_isolation(nb):
    torch.manual_seed(0)
    net = nb.GatedMLP(5, [16, 8], 3, n_tasks=4, keep=0.5, seed=1)
    assert isinstance(net.hidden, nn.ModuleList) and all(isinstance(m, nn.Linear) for m in net.hidden), (
        "net.hidden should be an nn.ModuleList of the hidden nn.Linear layers")
    assert isinstance(net.head, nn.Linear), "net.head should be the output nn.Linear"
    x = torch.randn(7, 5, generator=torch.Generator().manual_seed(2))
    m = net.masks()
    for t in (0, 2):
        h = torch.relu(net.hidden[0](x)) * m[0][t]
        h = torch.relu(net.hidden[1](h)) * m[1][t]
        want = net.head(h)
        assert torch.allclose(net(x, t), want, atol=1e-6), (
            f"task {t}: each hidden layer computes relu(linear(h)) * mask[task], then the head")
    net.set_task(2)
    assert torch.allclose(net(x), net(x, 2)), "with no task argument, forward uses the task chosen by set_task"
    with pytest.raises(ValueError):
        net.set_task(4)
    net.zero_grad()
    net(x, 1).pow(2).sum().backward()
    off = m[0][1] == 0
    assert (net.hidden[0].weight.grad[off] == 0).all() and (net.hidden[0].weight.grad[~off] != 0).any(), (
        "a unit gated off for task 1 must get no gradient from task 1's loss: that's the isolation")
