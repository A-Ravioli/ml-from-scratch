"""Checks for exercise.py (federated averaging).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Aggregation and
one-step client updates are checked exactly against hand computations; FedAvg is checked
against its one exact special case (one full-batch step on every client is centralized
gradient descent), with stand-in client updates and compressors that make the server's
bookkeeping visible; the quantizer is checked for unbiasedness statistically.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mlfs.testing import bounded

torch.set_num_threads(1)  # tiny models: one thread is fastest, and avoids oversubscribing a busy machine


def _net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))


def _blobs(n, seed, shift=0.0):
    g = np.random.default_rng(seed)
    y = g.integers(0, 2, n)
    X = g.normal(size=(n, 3)) + np.where(y[:, None] == 1, 2.0, -2.0) + shift
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)


def _flat_grad(model, X, y):
    model.zero_grad()
    F.cross_entropy(model(X), y).backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).clone()


# ---------------------------------------------------------------- stage 1: vectors and the average


def test_stage1_get_weights_order_and_copy(nb):
    m = _net()
    w = nb.get_weights(m)
    want = torch.cat([p.detach().reshape(-1) for p in m.parameters()])
    assert w.shape == (26,), f"Linear(3,4) + Linear(4,2) has 12 + 4 + 8 + 2 = 26 parameters; got shape {tuple(w.shape)}"
    assert torch.equal(w, want), "flatten each parameter (reshape(-1)) and concatenate them in model.parameters() order"
    assert not w.requires_grad, "get_weights returns a detached vector (no autograd history)"
    w[0] += 100.0
    assert not torch.equal(nb.get_weights(m), w), (
        "changing the returned vector changed the model: get_weights must return a COPY, or the server's "
        "snapshot of the global model would move while a client trains")


def test_stage1_set_weights_in_place(nb):
    m = _net()
    params_before = [id(p) for p in m.parameters()]
    target = torch.arange(26, dtype=torch.float32)
    out = nb.set_weights(m, target)
    assert out is m or out is None, "set_weights works in place (return the model, or nothing)"
    assert [id(p) for p in m.parameters()] == params_before, (
        "set_weights must copy INTO the existing Parameters (p.copy_ under torch.no_grad), not replace them: "
        "an optimizer holding the old Parameter objects would stop seeing the model")
    assert torch.equal(nb.get_weights(m), target), "after set_weights(m, w), get_weights(m) should return w"
    assert torch.equal(m[0].bias.detach(), torch.arange(12, 16, dtype=torch.float32)), (
        "the first layer's bias should hold entries 12..15 (after the 12 weights of Linear(3, 4))")
    try:
        nb.set_weights(m, torch.zeros(25))
    except ValueError:
        pass
    else:
        raise AssertionError("set_weights with 25 entries for a 26-parameter model should raise ValueError")


def test_stage1_aggregate_hand_example(nb):
    got = nb.fedavg_aggregate([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 6.0])], [1, 3])
    assert torch.allclose(got, torch.tensor([2.5, 5.0])), (
        f"(1·(1,2) + 3·(3,6)) / 4 = (2.5, 5); got {got.tolist()}. "
        "if you got (2, 4) you took the plain mean: weight each client by its number of samples")


def test_stage1_aggregate_matches_numpy(nb):
    g = np.random.default_rng(3)
    W = g.normal(size=(5, 7))
    n = np.array([10, 1, 0, 30, 7])
    got = nb.fedavg_aggregate([torch.tensor(w) for w in W], n)
    want = (n[:, None] * W).sum(0) / n.sum()
    assert got.shape == (7,) and np.allclose(got.numpy(), want), "weighted average sum_k n_k w_k / sum_k n_k"
    try:
        nb.fedavg_aggregate([], [])
    except ValueError:
        pass
    else:
        raise AssertionError("aggregating no clients should raise ValueError")
    try:
        nb.fedavg_aggregate([torch.ones(2), torch.ones(2)], [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("num_samples summing to zero should raise ValueError (the weights are undefined)")


# ---------------------------------------------------------------- stage 2: local SGD


def test_stage2_full_batch_is_one_gradient_step(nb):
    X, y = _blobs(20, 0)
    m = _net(1)
    w0 = nb.get_weights(m)
    g = _flat_grad(m, X, y)
    w, steps = nb.client_update(m, X, y, epochs=1, batch_size=None, lr=0.3, seed=0)
    assert steps == 1, f"batch_size=None means one full-batch step per epoch; got n_steps={steps}"
    assert torch.allclose(w, w0 - 0.3 * g, atol=1e-6), (
        "one epoch with batch_size=None should be exactly w - lr * grad(mean cross-entropy over all of X)")
    assert torch.equal(nb.get_weights(m), w), "client_update trains the model in place and returns its weights"


def test_stage2_step_count_and_coverage(nb):
    n, seen = 25, []

    def loss_fn(out, yb):
        seen.append(yb.tolist())
        return out.pow(2).mean()

    m = nn.Linear(3, 1)
    X = torch.randn(n, 3, generator=torch.Generator().manual_seed(0))
    _, steps = bounded(nb.client_update, m, X, torch.arange(n), epochs=3, batch_size=10, lr=0.01, loss_fn=loss_fn,
                       seed=0, what="client_update")
    assert steps == 9, f"3 epochs of ceil(25 / 10) = 3 minibatches each is 9 steps; got {steps}"
    assert [len(b) for b in seen] == [10, 10, 5] * 3, (
        f"minibatch sizes should be 10, 10, 5 in each epoch (the last one smaller); got {[len(b) for b in seen]}")
    epochs = [sum(seen[i:i + 3], []) for i in (0, 3, 6)]
    for e, idx in enumerate(epochs):
        assert sorted(idx) == list(range(n)), f"epoch {e}: every example should be used exactly once per epoch"
    assert epochs[0] != list(range(n)) and epochs[0] != epochs[1], (
        "reshuffle the data every epoch (np.random.default_rng(seed).permutation): SGD on a fixed order is biased")


def test_stage2_seeded_and_learns(nb):
    X, y = _blobs(60, 1)
    a, _ = nb.client_update(_net(2), X, y, epochs=2, batch_size=8, lr=0.1, seed=5)
    b, _ = nb.client_update(_net(2), X, y, epochs=2, batch_size=8, lr=0.1, seed=5)
    c, _ = nb.client_update(_net(2), X, y, epochs=2, batch_size=8, lr=0.1, seed=6)
    assert torch.equal(a, b), "the same seed must give the same result (shuffle with np.random.default_rng(seed))"
    assert not torch.equal(a, c), "different seeds should shuffle differently"
    m = _net(2)
    before = F.cross_entropy(m(X), y).item()
    nb.client_update(m, X, y, epochs=5, batch_size=10, lr=0.1, seed=0)
    after = F.cross_entropy(m(X), y).item()
    assert after < 0.5 * before, f"5 epochs of SGD on separable blobs should halve the loss: {before:.3f} -> {after:.3f}"


# ---------------------------------------------------------------- stage 3: partitions


def test_stage3_iid_partition(nb):
    parts = nb.partition_iid(103, 10, seed=0)
    assert len(parts) == 10, f"expected 10 clients, got {len(parts)}"
    allidx = np.concatenate(parts)
    assert np.issubdtype(allidx.dtype, np.integer), "partitions are integer index arrays"
    assert sorted(allidx.tolist()) == list(range(103)), "every index appears in exactly one client"
    sizes = sorted(len(p) for p in parts)
    assert sizes[-1] - sizes[0] <= 1, f"IID client sizes should differ by at most 1; got {sizes}"
    assert not np.array_equal(parts[0], np.arange(len(parts[0]))), "shuffle before splitting"
    again = nb.partition_iid(103, 10, seed=0)
    assert all(np.array_equal(p, q) for p, q in zip(parts, again)), "the same seed gives the same split"


def test_stage3_shards_partition(nb):
    y = np.repeat(np.arange(10), 120)
    np.random.default_rng(1).shuffle(y)
    parts = nb.partition_shards(y, 20, shards_per_client=3, seed=0)
    assert len(parts) == 20, f"expected 20 clients, got {len(parts)}"
    assert sorted(np.concatenate(parts).tolist()) == list(range(1200)), "every index appears in exactly one client"
    assert {len(p) for p in parts} == {60}, "60 shards of 20 examples, 3 per client: every client gets 60 examples"
    n_labels = [len(np.unique(y[p])) for p in parts]
    assert max(n_labels) <= 3, (
        f"each shard is a run of the label-SORTED indices, so a client with 3 shards sees at most 3 labels; "
        f"got label counts {n_labels}. did you sort by y (np.argsort) before cutting?")
    assert np.mean(n_labels) > 1.5, "shards should be dealt at random, not in sorted order (most clients get mixed labels)"
    two = nb.partition_shards(np.repeat(np.arange(10), 60), 100, seed=1)
    assert len(two) == 100 and max(len(np.unique(np.repeat(np.arange(10), 60)[p])) for p in two) <= 2, (
        "the default is 2 shards per client (McMahan et al.)")


# ---------------------------------------------------------------- stage 4: the simulator


def _clients(sizes, seed=0):
    return [_blobs(s, seed + i, shift=0.3 * i) for i, s in enumerate(sizes)]


def test_stage4_fedsgd_is_centralized_gd(nb):
    data = _clients([5, 12, 30])
    m = _net(3)
    ref = _net(3)
    Xall = torch.cat([X for X, _ in data])
    yall = torch.cat([y for _, y in data])
    w0 = nb.get_weights(ref)
    g = _flat_grad(ref, Xall, yall)
    hist = bounded(nb.fedavg, m, data, rounds=1, client_fraction=1.0, local_epochs=1, batch_size=None, lr=0.5,
                   seed=0, what="fedavg")
    assert torch.allclose(nb.get_weights(m), w0 - 0.5 * g, atol=1e-6), (
        "with every client, one epoch and full batches, FedAvg must equal one step of gradient descent on the "
        "union of the data (FedSGD). if this fails only with unequal client sizes, weight the average by n_k")
    assert hist["round"] == [1] and set(map(int, hist["clients"][0])) == {0, 1, 2}, (
        "client_fraction=1.0 selects all 3 clients")


def test_stage4_bookkeeping(nb):
    data = _clients([8] * 10)
    m = _net(4)
    P = 26
    calls = []

    def fake_update(model, X, y, *, epochs, batch_size, lr, seed):
        w = nb.get_weights(model)
        calls.append(w.clone())
        nb.set_weights(model, w + 1.0)  # a client that moves every weight by +1
        return w + 1.0, 4

    w0 = nb.get_weights(m)
    hist = bounded(nb.fedavg, m, data, rounds=4, client_fraction=0.3, local_epochs=2, batch_size=5, lr=0.1,
                   local_update=fake_update, seed=0, what="fedavg")
    assert len(calls) == 12, f"client_fraction 0.3 of 10 clients is 3 per round; 4 rounds = 12 updates, got {len(calls)}"
    for r in range(4):
        for c in calls[3 * r:3 * r + 3]:
            assert torch.allclose(c, w0 + r), (
                "each client must start from the CURRENT global model: call set_weights(model, w_global) before "
                "every client, or client 2 starts where client 1 stopped")
    assert torch.allclose(nb.get_weights(m), w0 + 4), "4 rounds of clients that each add 1 should move the model by 4"
    for chosen in hist["clients"]:
        assert len(set(map(int, chosen))) == 3, "the 3 clients in a round must be distinct (sample without replacement)"
    assert len({tuple(sorted(map(int, c))) for c in hist["clients"]}) > 1, "sample a new set of clients every round"
    assert hist["bytes_up"] == [3 * P * 4 * r for r in range(1, 5)], (
        f"each client uploads P = {P} float32 numbers = {4 * P} bytes; bytes_up is CUMULATIVE: expected "
        f"{[3 * P * 4 * r for r in range(1, 5)]}, got {hist['bytes_up']}")
    assert hist["bytes_down"] == hist["bytes_up"], "each selected client also downloads the full model"
    assert hist["client_steps"] == [12, 24, 36, 48], (
        f"client_steps is the cumulative number of local steps (4 per client here); got {hist['client_steps']}")


def test_stage4_compress_and_eval_hooks(nb):
    data = _clients([8] * 4)
    m = _net(5)
    w0 = nb.get_weights(m)
    seen = []

    def compress(delta, rng):
        seen.append(delta.clone())
        return torch.zeros_like(delta), 7

    evals = []
    hist = bounded(nb.fedavg, m, data, rounds=5, client_fraction=0.5, local_epochs=1, batch_size=4, lr=0.1,
                   compress=compress, eval_fn=lambda model: evals.append(1) or len(evals), eval_every=2, seed=1,
                   what="fedavg")
    assert len(seen) == 10 and seen[0].abs().sum() > 0, (
        "compress is called once per client on its UPDATE delta = w_k - w_global (nonzero after training)")
    assert torch.equal(nb.get_weights(m), w0), "the server applies the compressed deltas: all-zero deltas leave the model unchanged"
    assert hist["bytes_up"] == [14, 28, 42, 56, 70], (
        f"with a compressor, bytes_up counts the bytes it reports (7 per client here); got {hist['bytes_up']}")
    assert hist["eval_round"] == [0, 2, 4, 5] and hist["eval"] == [1, 2, 3, 4], (
        f"eval_fn runs before training, every eval_every rounds, and after the last round: eval_round should be "
        f"[0, 2, 4, 5]; got {hist['eval_round']}")


def test_stage4_learns(nb):
    data = _clients([20] * 8, seed=10)
    Xt, yt = _blobs(400, 99)
    accs = []
    for s in range(3):
        m = _net(s)
        bounded(nb.fedavg, m, data, rounds=5, client_fraction=0.5, local_epochs=2, batch_size=5, lr=0.1, seed=s,
                what="fedavg")
        with torch.no_grad():
            accs.append((m(Xt).argmax(1) == yt).float().mean().item())
    assert np.median(accs) > 0.9, f"5 rounds of FedAvg on separable blobs should reach >90% test accuracy; got {accs}"


# ---------------------------------------------------------------- stage 5: quantization


def test_stage5_grid_bytes_and_endpoints(nb):
    v = torch.tensor([0.0, 0.1, 0.5, 0.77, 1.0, -0.3])
    q, nbytes = nb.stochastic_quantize(v, 2, rng=np.random.default_rng(0))
    lo, hi = -0.3, 1.0
    grid = lo + torch.arange(4) * (hi - lo) / 3
    assert q.shape == v.shape and q.dtype == v.dtype, "return a tensor with v's shape and dtype"
    assert all(torch.isclose(grid, x, atol=1e-6).any() for x in q), (
        f"2 bits = 4 evenly spaced levels between min and max {grid.tolist()}; got {q.tolist()}")
    assert np.isclose(q[5].item(), lo) and np.isclose(q[4].item(), hi), "the min and max are levels: they come back exactly"
    assert nbytes == math.ceil(6 * 2 / 8) + 8, f"n_bytes = ceil(d * bits / 8) + 8 (two float32 for min and max); got {nbytes}"
    assert nb.stochastic_quantize(torch.randn(1000), 4, rng=np.random.default_rng(0))[1] == 508
    one, _ = nb.stochastic_quantize(v, 1, rng=np.random.default_rng(0))
    assert all(np.isclose(x, -0.3) or np.isclose(x, 1.0) for x in one.tolist()), "1 bit: every entry becomes the min or the max"
    c = torch.full((5,), 2.5)
    cq, _ = nb.stochastic_quantize(c, 3, rng=np.random.default_rng(0))
    assert torch.equal(cq, c), "a constant vector (max = min) comes back unchanged (don't divide by zero)"
    for bad in (0, 17):
        try:
            nb.stochastic_quantize(v, bad, rng=np.random.default_rng(0))
        except ValueError:
            pass
        else:
            raise AssertionError(f"bits={bad} should raise ValueError (allowed: 1..16)")


def test_stage5_unbiased_with_bounded_variance(nb):
    v = torch.tensor([-1.0, -0.62, -0.1, 0.0, 0.33, 0.5, 0.91, 1.0], dtype=torch.float64)
    rng = np.random.default_rng(1)
    draws = torch.stack([nb.stochastic_quantize(v, 2, rng=rng)[0] for _ in range(3000)])
    step = 2.0 / 3
    se = step / 2 / math.sqrt(3000)
    err = (draws.mean(0) - v).abs().max().item()
    assert err < 5 * se, (
        f"stochastic rounding must be unbiased: averaged over 3000 draws it should match v to ~{se:.4f}, but the "
        f"largest error is {err:.4f}. round UP with probability equal to the fractional position between levels "
        "(rounding to the NEAREST level is biased)")
    var = draws.var(0).max().item()
    assert var <= step ** 2 / 4 * 1.1, f"per-entry variance is at most (level spacing)^2 / 4 = {step ** 2 / 4:.3f}; got {var:.3f}"
    assert draws.var(0)[2] > 0.01, "entries between levels must be rounded randomly, not deterministically"
    a = nb.stochastic_quantize(v, 3, rng=np.random.default_rng(7))[0]
    b = nb.stochastic_quantize(v, 3, rng=np.random.default_rng(7))[0]
    assert torch.equal(a, b), "draw the random numbers from `rng`, so the same rng state gives the same result"


# ---------------------------------------------------------------- one test per spec clause


def test_stage1_set_weights_rejects_long_vectors(nb):
    try:
        nb.set_weights(_net(), torch.zeros(27))
    except ValueError:
        pass
    else:
        raise AssertionError("set_weights with 27 entries for a 26-parameter model should raise ValueError (too LONG "
                             "is as wrong as too short)")


def test_stage5_independent_rounding_per_entry(nb):
    v = torch.cat([torch.tensor([0.0, 1.0]), torch.full((1000,), 1 / 6)], ).double()
    rng = np.random.default_rng(0)
    fracs = []
    for _ in range(20):
        q, _ = nb.stochastic_quantize(v, 2, rng=rng)
        fracs.append((q[2:] > 0.2).double().mean().item())  # 1/6 lies halfway between levels 0 and 1/3
    assert all(0.4 < f < 0.6 for f in fracs), (
        f"1,000 entries sit halfway between two levels; in every draw about half should round up, independently. "
        f"fractions rounded up per draw: {np.round(fracs, 2).tolist()}. draw one uniform PER ENTRY (rng.random(v.shape)), "
        "not one for the whole vector")
    for bad in (2.5, "4"):
        try:
            nb.stochastic_quantize(v, bad, rng=np.random.default_rng(0))
        except ValueError:
            pass
        else:
            raise AssertionError(f"bits={bad!r} should raise ValueError: bits must be an integer in 1..16")


def test_stage4_rounding_seeds_and_shared_rng(nb):
    data = _clients([6] * 10)
    seeds, rngs = [], []

    def fake_update(model, X, y, *, epochs, batch_size, lr, seed):
        seeds.append(seed)
        return nb.get_weights(model) + 0.1, 1

    def compress(delta, rng):
        rngs.append((rng, rng.bit_generator.state["state"]["state"]))
        return delta, 5

    hist = bounded(nb.fedavg, _net(6), data, rounds=3, client_fraction=0.29, local_epochs=1, batch_size=3, lr=0.1,
                   local_update=fake_update, compress=compress, seed=2, what="fedavg")
    assert [len(c) for c in hist["clients"]] == [3, 3, 3], (
        f"m = max(1, round(client_fraction * N)) = round(2.9) = 3 clients per round, not int(2.9) = 2; got "
        f"{[len(c) for c in hist['clients']]}")
    assert len(set(seeds)) == len(seeds) == 9, (
        f"every client call gets its own seed, drawn from the round's rng (int(rng.integers(2**31))); got {seeds}")
    assert all(r is rngs[0][0] for r, _ in rngs), (
        "pass fedavg's own rng to compress: one generator shared by all calls, so clients' rounding noise is "
        "independent (a fresh np.random.default_rng(0) per call gives every client the SAME noise)")
    assert len({st for _, st in rngs}) >= 4, "the rng's state must keep advancing across compress calls (don't reset it)"
    P = 26
    assert hist["bytes_down"] == [3 * 4 * P * r for r in (1, 2, 3)], (
        f"downloads are full models (4 * P bytes each), whatever the upload compression; got {hist['bytes_down']}")
    assert hist["bytes_up"] == [15, 30, 45], "uploads count the compressor's reported bytes"
