"""Checks for exercise.py (prototypical networks).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the class split and
the episode sampler are checked for disjointness and leakage, and with chi-squared tests that
classes, examples and positions are drawn uniformly; prototypes and logits against explicit
formulas (shuffled labels, unequal shots); the episode loss against a hand-built cross-entropy;
training and evaluation against a reference loop replaying the same episodes, with probe modules
that record train/eval mode, gradient mode and which examples were embedded.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import stats  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

# the first optimizer step in a process runs ~3M lines of torch's lazy imports; do it here, untraced, so the
# budgets below only count your loops
_w = torch.zeros(1, requires_grad=True)
_w.grad = torch.zeros(1)
torch.optim.Adam([_w]).step()


def _train(nb, *args, budget=200_000, **kwargs):
    """train_protonet, stopped with a teaching message if its loop doesn't end (about 700 lines per episode)."""
    return bounded(nb.train_protonet, *args, budget=budget, what="train_protonet",
                   hint="the loop must run exactly n_episodes episodes.", **kwargs)


def _eval(nb, *args, budget=200_000, **kwargs):
    """evaluate_protonet, bounded the same way (about 160 lines per episode)."""
    return bounded(nb.evaluate_protonet, *args, budget=budget, what="evaluate_protonet",
                   hint="the loop must run exactly n_episodes episodes.", **kwargs)


def _labels(counts):
    """y with counts[c] examples of class c, shuffled deterministically."""
    y = np.concatenate([np.full(n, c) for c, n in enumerate(counts)])
    return np.random.default_rng(0).permutation(y).astype(np.int64)


def _sample(nb, *args, **kw):
    return bounded(nb.sample_episode, *args, what="sample_episode", **kw)


# ---------------------------------------------------------------- stage 1: classes and episodes


def test_stage1_split_classes_is_a_disjoint_partition(nb):
    labels = np.array([3, 7, 7, 1, 9, 3, 12, 1, 5, 5, 0])
    train, test = nb.split_classes(labels, 2, generator=torch.Generator().manual_seed(0))
    assert isinstance(train, tuple) and isinstance(test, tuple), "return two tuples (train, test)"
    assert all(type(c) is int for c in train + test), "class ids must be plain Python ints"
    assert len(test) == 2, f"n_test=2 classes in the test split, got {len(test)}"
    assert set(train) | set(test) == {0, 1, 3, 5, 7, 9, 12}, "together the splits cover every distinct class once"
    assert not set(train) & set(test), f"meta-train and meta-test classes must be disjoint, both contain {set(train) & set(test)}"
    assert list(train) == sorted(train) and list(test) == sorted(test), "each tuple is sorted"
    for bad in (0, 7):
        with pytest.raises(ValueError):
            nb.split_classes(labels, bad)


def test_stage1_split_classes_is_random_and_seeded(nb):
    hits = np.zeros(10)
    for s in range(1000):
        _, test = nb.split_classes(range(10), 3, generator=torch.Generator().manual_seed(s))
        hits[list(test)] += 1
    p = stats.chisquare(hits).pvalue
    assert p > 1e-4, f"every class should be equally likely to land in the test split; counts {hits.astype(int)}, p = {p:.2g}"
    a = nb.split_classes(range(10), 3, generator=torch.Generator().manual_seed(5))
    b = nb.split_classes(range(10), 3, generator=torch.Generator().manual_seed(5))
    assert a == b, "the same generator seed must give the same split"


def test_stage1_episode_structure_and_labels(nb):
    y = _labels([9, 8, 10, 7, 9, 12])
    ep = _sample(nb, y, (0, 2, 3, 5), 3, 2, 4, generator=torch.Generator().manual_seed(1))
    want = {"support": (6,), "query": (12,), "support_labels": (6,), "query_labels": (12,), "classes": (3,)}
    for k, s in want.items():
        assert k in ep, f"the episode dict needs key {k!r}"
        assert tuple(ep[k].shape) == s, f"{k} should have shape {s}, got {tuple(ep[k].shape)}"
        assert ep[k].dtype == torch.int64, f"{k} should be int64, got {ep[k].dtype}"
    assert torch.equal(ep["support_labels"], torch.tensor([0, 0, 1, 1, 2, 2])), (
        "support_labels: the first k_shot support examples have label 0, the next label 1, ...")
    assert torch.equal(ep["query_labels"], torch.arange(3).repeat_interleave(4)), "query_labels: blocks of n_query"
    cls = ep["classes"].tolist()
    assert len(set(cls)) == 3 and set(cls) <= {0, 2, 3, 5}, f"N distinct classes from the allowed ones, got {cls}"
    yt = torch.as_tensor(y)
    assert torch.equal(yt[ep["support"]], ep["classes"][ep["support_labels"]]), (
        "y[support[i]] must be classes[support_labels[i]]: label j means the episode's j-th class")
    assert torch.equal(yt[ep["query"]], ep["classes"][ep["query_labels"]]), "same for the query set"
    allidx = torch.cat([ep["support"], ep["query"]]).tolist()
    assert len(set(allidx)) == len(allidx), "no example may appear twice in an episode (support and query are disjoint)"


def test_stage1_classes_are_uniform_and_in_random_order(nb):
    y = _labels([6] * 7)
    allowed = (0, 1, 2, 4, 5, 6)  # class 3 is held out
    g = torch.Generator().manual_seed(2)
    picked, first = np.zeros(7), np.zeros(7)
    for _ in range(3000):
        ep = _sample(nb, y, allowed, 2, 1, 1, generator=g)
        c = ep["classes"].tolist()
        assert c[0] != c[1], f"the episode's classes must be distinct (drawn without replacement), got {c}"
        picked[c] += 1
        first[c[0]] += 1
    assert picked[3] == 0, "a class outside `classes` was sampled: that's class leakage between meta-train and meta-test"
    p = stats.chisquare(picked[list(allowed)]).pvalue
    assert p > 1e-4, f"each allowed class should be chosen equally often; counts {picked.astype(int)}, p = {p:.2g}"
    p0 = stats.chisquare(first[list(allowed)]).pvalue
    assert p0 > 1e-4, (f"the class given label 0 should be any allowed class with equal probability (random order, not "
                       f"sorted); counts {first.astype(int)}, p = {p0:.2g}")


def test_stage1_examples_are_uniform_within_class(nb):
    y = np.array([0] * 8 + [1] * 8, dtype=np.int64)
    g = torch.Generator().manual_seed(3)
    in_support, in_query = np.zeros(16), np.zeros(16)
    for _ in range(3000):
        ep = _sample(nb, y, (0, 1), 2, 1, 2, generator=g)
        in_support[ep["support"].numpy()] += 1
        in_query[ep["query"].numpy()] += 1
    for name, counts in (("support", in_support), ("query", in_query)):
        p = stats.chisquare(counts).pvalue
        assert p > 1e-4, (f"every example of a class should be equally likely to be a {name} example; counts "
                          f"{counts.astype(int)}, p = {p:.2g}. shuffle each class's indices before splitting them")


def test_stage1_errors_inputs_and_generator(nb):
    y = _labels([5, 5, 2])
    with pytest.raises(ValueError):
        nb.sample_episode(y, (0, 1), 3, 1, 1)
    with pytest.raises(ValueError):
        # class 2 has only 2 examples but needs 3; with n_way=3 it is always drawn
        nb.sample_episode(y, (0, 1, 2), 3, 2, 1, generator=torch.Generator().manual_seed(0))
    e1 = _sample(nb, torch.as_tensor(y), (0, 1), 2, 2, 2, generator=torch.Generator().manual_seed(4))
    torch.manual_seed(99)
    e2 = _sample(nb, y, [0, 1], 2, 2, 2, generator=torch.Generator().manual_seed(4))
    assert all(torch.equal(e1[k], e2[k]) for k in e1), (
        "the same generator seed must give the same episode (y may be a tensor or an array, classes any sequence)")


# ---------------------------------------------------------------- stage 2: prototypes and logits


def test_stage2_prototypes_shuffled_and_unequal(nb):
    g = torch.Generator().manual_seed(5)
    z = torch.randn(7, 3, generator=g, requires_grad=True)
    labels = torch.tensor([2, 0, 1, 2, 0, 2, 1])
    got = nb.prototypes(z, labels, 3)
    want = torch.stack([z[labels == j].mean(0) for j in range(3)])
    assert tuple(got.shape) == (3, 3), f"prototypes: (n_way, D) = (3, 3), got {tuple(got.shape)}"
    assert torch.allclose(got, want, atol=1e-6), (
        "prototype j is the mean of the embeddings whose label is j. labels can come in any order and classes can "
        "have different counts: don't reshape into (N, K, D)")
    assert got.requires_grad, "prototypes must stay differentiable (no .detach(), no numpy)"
    with pytest.raises(ValueError):
        nb.prototypes(z, torch.tensor([0, 0, 1, 1, 0, 1, 0]), 3)


def test_stage2_squared_euclidean_logits(nb):
    zq = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, -1.0]])
    protos = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    got = nb.proto_logits(zq, protos)
    want = -torch.tensor([[1.0, 4.0], [4.0, 1.0], [5.0, 18.0]])
    assert tuple(got.shape) == (3, 2), f"logits are (Q, N) = (3, 2), got {tuple(got.shape)}"
    assert torch.allclose(got, want), (
        f"logits are minus the SQUARED Euclidean distances: expected {want.tolist()}, got {got.tolist()}")
    assert torch.allclose(nb.proto_logits(zq, protos, scale=0.5), 0.5 * want), "scale multiplies the logits"


def test_stage2_squared_distance_is_a_linear_classifier(nb):
    g = torch.Generator().manual_seed(6)
    zq, protos = torch.randn(5, 4, generator=g), torch.randn(3, 4, generator=g)
    got = nb.proto_logits(zq, protos)
    linear = 2 * zq @ protos.T - (protos ** 2).sum(1)  # Snell et al.: w_j = 2 c_j, b_j = -|c_j|^2
    shift = got - linear
    assert torch.allclose(shift, shift[:, :1].expand_as(shift), atol=1e-5), (
        "-|z - c_j|^2 = 2 c_j.z - |c_j|^2 - |z|^2: up to a per-query constant, the logits are a linear classifier")
    assert torch.allclose(shift[:, 0], -(zq ** 2).sum(1), atol=1e-5), "and that constant is -|z|^2"


def test_stage2_cosine_logits_and_errors(nb):
    zq = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    protos = torch.tensor([[1.0, 0.0], [0.0, -3.0]])
    s = 1 / math.sqrt(2)
    want = torch.tensor([[1.0, 0.0], [s, -s]])
    got = nb.proto_logits(zq, protos, metric="cosine")
    assert torch.allclose(got, want, atol=1e-6), f"cosine logits: expected {want.tolist()}, got {got.tolist()}"
    assert torch.allclose(nb.proto_logits(zq, protos, metric="cosine", scale=10.0), 10 * want, atol=1e-5), (
        "scale multiplies cosine logits too (that's the whole point of it)")
    with pytest.raises(ValueError):
        nb.proto_logits(zq, protos, metric="manhattan")


# ---------------------------------------------------------------- stage 3: the episode loss


class _Probe(nn.Module):
    """linear embedding that records every call: inputs, train mode, grad mode."""

    def __init__(self, d_in=4, d_out=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(d_in, d_out)
        self.calls = []

    def forward(self, x):
        self.calls.append({"x": x.detach().clone(), "training": self.training, "grad": torch.is_grad_enabled()})
        return self.lin(x)


def _toy(n_per=6, n_cls=5, d=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n_cls).repeat_interleave(n_per)
    X = torch.randn(n_per * n_cls, d, generator=g) + 2 * torch.randn(n_cls, d, generator=g)[y]
    return X, y


def _ref_episode_loss(embed_fn, X, ep, metric="sqeuclidean", scale=1.0):
    zs, zq = embed_fn(X[ep["support"]]), embed_fn(X[ep["query"]])
    n = len(ep["classes"])
    c = torch.stack([zs[ep["support_labels"] == j].mean(0) for j in range(n)])
    if metric == "sqeuclidean":
        logits = -scale * ((zq[:, None] - c[None]) ** 2).sum(-1)
    else:
        logits = scale * F.normalize(zq, dim=-1) @ F.normalize(c, dim=-1).T
    loss = -logits.log_softmax(-1)[torch.arange(len(zq)), ep["query_labels"]].mean()
    return loss, (logits.argmax(1) == ep["query_labels"]).float().mean().item()


def _ep():
    return {"support": torch.tensor([0, 1, 6, 7, 13, 12]), "query": torch.tensor([2, 3, 8, 9, 14, 15]),
            "support_labels": torch.tensor([0, 0, 1, 1, 2, 2]), "query_labels": torch.tensor([0, 0, 1, 1, 2, 2]),
            "classes": torch.tensor([0, 1, 2])}


def test_stage3_worked_example(nb):
    X = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 4.0], [1.0, 1.0], [1.0, 3.0]])
    ep = {"support": torch.tensor([0, 1, 2, 3]), "query": torch.tensor([4, 5]),
          "support_labels": torch.tensor([0, 0, 1, 1]), "query_labels": torch.tensor([0, 1]),
          "classes": torch.tensor([7, 3])}
    loss, acc = nb.episode_loss(nn.Identity(), X, ep)
    want = (np.log1p(np.exp(-4.0)) + np.log1p(np.exp(-8.0))) / 2
    assert abs(loss.item() - want) < 1e-6, (
        f"the lesson's worked example: prototypes (1, 0) and (0, 3), loss (log(1 + e^-4) + log(1 + e^-8)) / 2 = "
        f"{want:.6f}; got {loss.item():.6f}")
    assert acc == 1.0, f"both queries are nearest to their own prototype: accuracy 1.0, got {acc}"


def test_stage3_episode_loss_value_and_gradient(nb):
    X, _ = _toy()
    probe = _Probe()
    ep = _ep()
    loss, acc = nb.episode_loss(probe, X, ep)
    want_loss, want_acc = _ref_episode_loss(probe.lin, X, ep)
    assert torch.is_tensor(loss) and loss.dim() == 0, "the loss is a scalar tensor"
    assert abs(loss.item() - want_loss.item()) < 1e-5, (
        f"loss should be {want_loss.item():.6f}: prototypes from the SUPPORT embeddings, logits for the QUERY "
        f"embeddings, cross-entropy against query_labels; got {loss.item():.6f}")
    assert isinstance(acc, float) and abs(acc - want_acc) < 1e-9, f"accuracy should be the float {want_acc}, got {acc!r}"
    loss.backward()
    assert probe.lin.weight.grad is not None and probe.lin.weight.grad.abs().sum() > 0, (
        "the loss must be differentiable with respect to the embedding's parameters")


def test_stage3_one_embedding_call_support_then_query(nb):
    X, _ = _toy()
    probe = _Probe()
    ep = _ep()
    nb.episode_loss(probe, X, ep)
    assert len(probe.calls) == 1, (
        f"embed support and query in ONE call (so batch-norm statistics are shared); embed was called "
        f"{len(probe.calls)} times")
    want = X[torch.cat([ep["support"], ep["query"]])]
    assert torch.equal(probe.calls[0]["x"], want), "the one call embeds X[support] followed by X[query]"


def test_stage3_metric_and_scale_pass_through(nb):
    X, _ = _toy()
    probe = _Probe(seed=1)
    ep = _ep()
    for kw in ({"metric": "cosine", "scale": 5.0}, {"scale": 0.1}):
        loss, _ = nb.episode_loss(probe, X, ep, **kw)
        want, _ = _ref_episode_loss(probe.lin, X, ep, **kw)
        assert abs(loss.item() - want.item()) < 1e-5, f"episode_loss ignores {kw}: expected {want.item():.5f}, got {loss.item():.5f}"


# ---------------------------------------------------------------- stage 4: training and evaluation


def test_stage4_training_replays_seeded_episodes_with_adam(nb):
    X, y = _toy(n_per=8, n_cls=6, seed=8)
    probe = _Probe(seed=2)
    ref = _Probe(seed=2)
    probe.eval()
    hist = _train(nb, probe, X, y, (0, 1, 2, 4, 5), n_way=3, k_shot=2, n_query=3, n_episodes=4, lr=0.05,
                             metric="cosine", scale=4.0, seed=11)
    assert isinstance(hist, list) and len(hist) == 4, f"train_protonet returns one loss per episode (4), got {hist!r}"
    assert all(c["training"] for c in probe.calls), "train_protonet must put the embedding in train mode (embed.train())"
    g = torch.Generator().manual_seed(11)
    opt = torch.optim.Adam(ref.lin.parameters(), lr=0.05)
    for i in range(4):
        ep = nb.sample_episode(y, (0, 1, 2, 4, 5), 3, 2, 3, generator=g)
        loss, _ = _ref_episode_loss(ref.lin, X, ep, metric="cosine", scale=4.0)
        assert abs(hist[i] - loss.item()) < 1e-4, (
            f"episode {i}: loss {hist[i]:.5f}, expected {loss.item():.5f}. draw every episode with ONE torch.Generator "
            "seeded with `seed`, and pass metric and scale through")
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.allclose(probe.lin.weight, ref.lin.weight, atol=1e-5), (
        "after 4 episodes the weights differ from Adam(embed.parameters(), lr=lr) on the episode losses")
    seen = torch.cat([c["x"] for c in probe.calls])
    rows = {tuple(r.tolist()) for r in X[torch.isin(y, torch.tensor([0, 1, 2, 4, 5]))]}
    assert all(tuple(r.tolist()) in rows for r in seen), "training embedded an example of a class outside `classes`"


def test_stage4_evaluation_mean_interval_and_modes(nb):
    X, y = _toy(n_per=10, n_cls=6, seed=9)
    probe = _Probe(seed=3)
    probe.train()
    mean, half = _eval(nb, probe, X, y, (1, 3, 5), n_way=2, k_shot=1, n_query=4, n_episodes=50,
                                      metric="cosine", scale=3.0, seed=12)
    assert all(not c["training"] for c in probe.calls), "evaluate_protonet must put the embedding in eval mode"
    assert all(not c["grad"] for c in probe.calls), "evaluate under torch.no_grad()"
    g = torch.Generator().manual_seed(12)
    accs = np.array([_ref_episode_loss(probe.lin, X, nb.sample_episode(y, (1, 3, 5), 2, 1, 4, generator=g),
                                       metric="cosine", scale=3.0)[1] for _ in range(50)])
    want_half = 1.96 * accs.std(ddof=1) / math.sqrt(50)
    assert abs(mean - accs.mean()) < 1e-9, (
        f"mean accuracy over the 50 seeded episodes should be {accs.mean():.4f}, got {mean:.4f} (seed, metric and "
        "scale must all be used)")
    assert abs(half - want_half) < 1e-9, (
        f"the 95% half-width is 1.96 * sd (ddof=1) / sqrt(n_episodes) = {want_half:.5f}, got {half:.5f}")
    seen = torch.cat([c["x"] for c in probe.calls])
    allowed = {tuple(r.tolist()) for r in X[torch.isin(y, torch.tensor([1, 3, 5]))]}
    assert all(tuple(r.tolist()) in allowed for r in seen), "evaluation embedded an example of a class outside `classes`"


def test_stage4_evaluation_defaults(nb):
    X, y = _toy(n_per=17, n_cls=3, seed=10)
    probe = _Probe(seed=4)
    _eval(nb, probe, X, y, (0, 1, 2), n_way=3, k_shot=1)
    assert len(probe.calls) == 600, f"n_episodes defaults to 600 (the few-shot convention), embed ran {len(probe.calls)} times"
    assert probe.calls[0]["x"].shape[0] == 3 * 1 + 3 * 15, "n_query defaults to 15 per class"


def test_stage4_episodic_training_learns_to_ignore_noise(nb):
    # 24 classes that differ in 2 of 12 dimensions; the other 10 are loud noise. a linear embedding trained on
    # 16 classes must find the 2 useful directions, which transfer to the 8 held-out classes.
    def data(seed):
        g = torch.Generator().manual_seed(seed)
        y = torch.arange(24).repeat_interleave(20)
        means = torch.zeros(24, 12)
        means[:, :2] = 3 * torch.randn(24, 2, generator=g)
        X = means[y] + torch.cat([0.5 * torch.randn(480, 2, generator=g), 3 * torch.randn(480, 10, generator=g)], 1)
        return X, y

    gains = []
    for s in range(3):
        X, y = data(s)
        train, test = tuple(range(16)), tuple(range(16, 24))
        torch.manual_seed(s)
        emb = nn.Linear(12, 12)
        with torch.no_grad():
            emb.weight.copy_(torch.eye(12))
            emb.bias.zero_()
        before, _ = _eval(nb, emb, X, y, test, n_way=5, k_shot=1, n_query=5, n_episodes=200, seed=100 + s)
        _train(nb, emb, X, y, train, n_way=5, k_shot=1, n_query=5, n_episodes=300, lr=0.01, seed=s, budget=800_000)
        after, _ = _eval(nb, emb, X, y, test, n_way=5, k_shot=1, n_query=5, n_episodes=200, seed=100 + s)
        gains.append(after - before)
    assert np.median(gains) > 0.15, (
        f"episodic training should lift 5-way 1-shot accuracy on held-out classes by well over 15 points here; median "
        f"gain {np.median(gains):+.3f} over 3 seeds (per seed {np.round(gains, 3).tolist()})")
