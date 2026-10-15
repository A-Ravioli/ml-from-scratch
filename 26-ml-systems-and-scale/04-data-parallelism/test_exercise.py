"""Checks for 26 · 04 data parallelism.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the ring all-reduce is
checked for the sum, for independence of its outputs, and for actually being a ring: its floating-point
rounding must match the ring's summation order chunk by chunk, which a direct sum doesn't. the cost
models against hand computations; gradient accumulation against the full batch with unequal micro-
batches; sharding for disjointness, coverage, padding and reshuffling. stage 5 runs two real processes
over torch.distributed (gloo, CPU) with a hard timeout: averaged gradients must equal the full-batch
gradient through exactly one all_reduce, and DDP training must give identical weights on both ranks,
equal to single-process SGD on the combined batches.
"""
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs import notebooks  # noqa: E402
from mlfs.testing import bounded  # noqa: E402


# ---------------------------------------------------------------- stage 1: ring all-reduce


def _ring_order_sum(arrays):
    """the sum each chunk gets in a ring: chunk c accumulates x_c + x_{c+1} + ... + x_{c-1}, left to right."""
    p = len(arrays)
    chunks = [np.array_split(np.asarray(a).ravel(), p) for a in arrays]
    out = []
    for c in range(p):
        acc = chunks[c][c].copy()
        for j in range(1, p):
            acc = acc + chunks[(c + j) % p][c]
        out.append(acc)
    return np.concatenate(out).reshape(np.shape(arrays[0]))


def test_stage1_sum(nb):
    rng = np.random.default_rng(0)
    arrays = [rng.normal(size=(5, 7)) for _ in range(4)]
    copies = [a.copy() for a in arrays]
    out = bounded(nb.ring_allreduce, arrays, what="ring_allreduce")
    assert len(out) == 4, f"one result per worker: 4; got {len(out)}"
    for i, o in enumerate(out):
        assert np.shape(o) == (5, 7), f"worker {i}'s result must keep the input shape (5, 7); got {np.shape(o)}"
        assert np.allclose(o, sum(arrays)), f"worker {i}'s result isn't the elementwise sum of all 4 arrays"
    assert all(np.array_equal(a, c) for a, c in zip(arrays, copies)), "the inputs must not be modified"
    out[0][0, 0] = 1e9
    assert out[1][0, 0] != 1e9, "each worker gets its own array (in a real cluster they're on different machines)"


def test_stage1_is_a_ring(nb):
    rng = np.random.default_rng(1)
    for p, n in ((4, 1000), (3, 10), (5, 13)):  # n not divisible by p: np.array_split chunks
        arrays = [(rng.normal(size=n) * 10.0 ** rng.integers(-3, 4)).astype(np.float32) for _ in range(p)]
        out = bounded(nb.ring_allreduce, arrays, what="ring_allreduce")
        want = _ring_order_sum(arrays)
        assert all(np.array_equal(o, want) for o in out), (
            f"p={p}, n={n}: the results must match a ring's reduce-scatter bit for bit: chunk c (np.array_split into p "
            "chunks) starts at worker c and is passed to c+1, c+2, ..., each adding its own copy, so it is summed in the "
            "order x_c + x_(c+1) + ... + x_(c-1). a direct sum (np.sum, or adding all arrays in order 0, 1, 2, ...) "
            "rounds differently in float32: that's not a ring all-reduce")
        if n == 1000:
            assert not np.array_equal(np.sum(np.stack(arrays), axis=0), want), "sanity: the ring differs from a direct sum"


def test_stage1_edge_cases(nb):
    a = [np.arange(6.0)]
    assert np.array_equal(bounded(nb.ring_allreduce, a)[0], np.arange(6.0)), "one worker: the result is its own array"
    two = bounded(nb.ring_allreduce, [np.ones(3), 2 * np.ones(3)])
    assert all(np.array_equal(t, 3 * np.ones(3)) for t in two)
    three = bounded(nb.ring_allreduce, [np.array([1, 2, 3]), np.array([10, 20, 30]), np.array([100, 200, 300])])
    assert all(np.array_equal(t, [111, 222, 333]) for t in three), "the lesson's worked example: [111, 222, 333] everywhere"


# ---------------------------------------------------------------- stage 2: cost model


def test_stage2_bytes_and_time(nb):
    assert np.isclose(nb.allreduce_bytes(1000, 4), 1500), "2 (p-1)/p n: 2 * 3/4 * 1000 = 1500 bytes per worker"
    assert np.isclose(nb.allreduce_bytes(1000, 1), 0), "one worker sends nothing"
    assert np.isclose(nb.allreduce_bytes(1e9, 1000), 2e9 * 999 / 1000), "approaches 2n for large p, independent of p"
    t = nb.allreduce_time(4e6, 4, bandwidth=1e9, latency=1e-5)
    assert np.isclose(t, 6 * (1e-5 + 1e-3)), (
        f"2 (p-1) = 6 messages of n/p = 1 MB, each 10 us latency + 1 ms transfer: 6.06 ms; got {t * 1e3:.3f} ms")
    assert np.isclose(nb.allreduce_time(4e6, 4, bandwidth=1e9), 6e-3), "latency defaults to 0"
    assert np.isclose(nb.allreduce_time(8e6, 4, bandwidth=1e10, latency=1e-5), 1.26e-3), "the lesson's worked example"
    assert np.isclose(nb.allreduce_bytes(8e6, 4), 12e6)


def test_stage2_efficiency(nb):
    assert np.isclose(nb.ddp_efficiency(1.0, 0.25), 0.8), "no overlap: 1 / (1 + 0.25) = 0.8"
    assert np.isclose(nb.ddp_efficiency(1.0, 0.25, overlap=0.5), 1.0), "0.25 s of comm hides under 0.5 s of compute"
    assert np.isclose(nb.ddp_efficiency(1.0, 0.9, overlap=0.5), 1 / 1.4), "only 0.5 of the 0.9 s can be hidden"
    assert np.isclose(nb.ddp_efficiency(2.0, 0.0), 1.0)


# ---------------------------------------------------------------- stage 3: gradient accumulation


def _mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 3))


def test_stage3_matches_full_batch(nb):
    g = torch.Generator().manual_seed(0)
    X, Y = torch.randn(23, 5, generator=g), torch.randint(0, 3, (23,), generator=g)
    m = _mlp()
    full = nn.functional.cross_entropy(m(X), Y)
    full.backward()
    want = [p.grad.clone() for p in m.parameters()]
    for p in m.parameters():
        p.grad = torch.full_like(p, 7.0)  # stale gradients from an earlier step
    micro = [(X[:10], Y[:10]), (X[10:13], Y[10:13]), (X[13:], Y[13:])]  # unequal sizes: 10, 3, 10
    loss = bounded(nb.accumulate_gradients, m, nn.functional.cross_entropy, micro)
    assert np.isclose(loss, full.item(), atol=1e-6), (
        f"the returned loss is the full batch's mean loss {full.item():.6f}; got {loss}")
    for p, w in zip(m.parameters(), want):
        assert torch.allclose(p.grad, w, atol=1e-6), (
            "the accumulated gradient must equal the full batch's. with unequal micro-batches, weight each micro-batch's "
            "mean loss by len(x) / N (dividing by the number of micro-batches is only right for equal sizes); and start "
            "from zeroed gradients")


# ---------------------------------------------------------------- stage 4: sharding


def test_stage4_disjoint_and_complete(nb):
    shards = [np.asarray(nb.shard_indices(12, r, 3, epoch=0, seed=5)) for r in range(3)]
    assert all(len(s) == 4 for s in shards), f"12 examples over 3 ranks: 4 each; got {[len(s) for s in shards]}"
    assert sorted(np.concatenate(shards).tolist()) == list(range(12)), "together the shards cover every index exactly once"
    perm = np.random.default_rng(5).permutation(12)
    assert np.array_equal(shards[1], perm[1::3]), (
        "rank r takes every world_size-th element of np.random.default_rng(seed + epoch).permutation(n), starting at r. "
        "every rank must compute the same permutation (same seed) for the shards to be disjoint")


def test_stage4_padding_and_epochs(nb):
    shards = [np.asarray(nb.shard_indices(10, r, 4, seed=1)) for r in range(4)]
    assert [len(s) for s in shards] == [3, 3, 3, 3], (
        "10 examples over 4 ranks: pad to 12 by wrapping around the permutation, so every rank does the same number "
        "of steps (a rank with fewer batches would leave the others waiting in all_reduce forever)")
    allidx = np.concatenate(shards)
    perm = np.random.default_rng(1).permutation(10)
    assert sorted(allidx.tolist()) == sorted(perm.tolist() + perm[:2].tolist()), "the 2 padding indices are the first 2"
    tiny = [np.asarray(nb.shard_indices(1, r, 3, seed=1)).tolist() for r in range(3)]
    assert tiny == [[0], [0], [0]], (
        f"1 example over 3 ranks: tile the permutation cyclically so every rank gets one index; got {tiny}")
    two = [np.asarray(nb.shard_indices(2, r, 5, seed=2)).tolist() for r in range(5)]
    p2 = np.random.default_rng(2).permutation(2)
    assert two == [[int(p2[i % 2])] for i in range(5)], f"2 examples over 5 ranks: cycle a, b, a, b, a; got {two}"
    e0 = np.asarray(nb.shard_indices(100, 0, 2, epoch=0, seed=3))
    e1 = np.asarray(nb.shard_indices(100, 0, 2, epoch=1, seed=3))
    assert not np.array_equal(e0, e1), "each epoch reshuffles (seed + epoch)"
    assert np.array_equal(e0, np.asarray(nb.shard_indices(100, 0, 2, seed=3))), "epoch defaults to 0"
    assert not np.array_equal(e0, np.asarray(nb.shard_indices(100, 0, 2, seed=4))), "the seed changes the permutation"


# ---------------------------------------------------------------- stage 5: real processes

WORKER = textwrap.dedent(r"""
    import os, pickle, sys
    import torch, torch.distributed as dist
    from torch import nn
    torch.set_num_threads(1)
    nb_path, rank, world, store, out, task = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    from mlfs import notebooks
    nb = notebooks.load(nb_path)
    dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=world)
    try:
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        X, Y = torch.randn(24, 5, generator=g), torch.randint(0, 3, (24,), generator=g)
        torch.manual_seed(100 + rank)  # different initial weights on each rank: ddp_train must broadcast rank 0's
        m = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 3))
        res = {}
        if task == "avg":
            torch.manual_seed(0)
            m = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 3))
            lo, hi = rank * 12, (rank + 1) * 12
            nn.functional.cross_entropy(m(X[lo:hi]), Y[lo:hi]).backward()
            calls = []
            real = dist.all_reduce
            def counting(*a, **k):
                calls.append(1)
                return real(*a, **k)
            dist.all_reduce = counting
            nb.average_gradients(m, world)
            dist.all_reduce = real
            res = {"grads": [p.grad.clone() for p in m.parameters()], "calls": len(calls)}
        else:
            losses = nb.ddp_train(m, X, Y, nn.functional.cross_entropy, rank=rank, world_size=world, epochs=2,
                                  batch_size=4, lr=0.5, seed=3)
            res = {"params": [p.detach().clone() for p in m.parameters()], "losses": losses}
        with open(out, "wb") as f:
            pickle.dump(res, f)
    finally:
        dist.destroy_process_group()
""")


def _nb_path(nb):
    path = getattr(nb, "__file__", None)
    if path is None:
        pytest.skip("stage 5 starts real processes that load your notebook from disk: run `pytest -k stage5` in a terminal")
    return path


def _run_ranks(nb, task, world=2, timeout=90):
    path = _nb_path(nb)
    with tempfile.TemporaryDirectory() as d:
        outs = [os.path.join(d, f"r{r}.pkl") for r in range(world)]
        env = {**os.environ, "MLFS_ROOT": str(notebooks.repo_root())}
        procs = [subprocess.Popen([sys.executable, "-c", WORKER, path, str(r), str(world), os.path.join(d, "store"),
                                   outs[r], task], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                 for r in range(world)]
        errs = []
        try:
            for p in procs:
                _, err = p.communicate(timeout=timeout)
                errs.append(err)
        except subprocess.TimeoutExpired:
            for p in procs:
                p.kill()
            for p in procs:
                p.communicate()
            raise AssertionError(
                f"the {world} processes didn't finish within {timeout} s: probably a collective that only some ranks "
                "call (every rank must call all_reduce/broadcast the same number of times, in the same order)") from None
        for r, (p, err) in enumerate(zip(procs, errs)):
            if p.returncode != 0:
                tail = "\n".join(err.strip().splitlines()[-12:])
                raise AssertionError(f"rank {r} failed:\n{tail}")
        results = []
        for o in outs:
            with open(o, "rb") as f:
                results.append(pickle.load(f))
        return results


@pytest.mark.slow
def test_stage5_average_gradients(nb):
    res = _run_ranks(nb, "avg")
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    X, Y = torch.randn(24, 5, generator=g), torch.randint(0, 3, (24,), generator=g)
    m = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 3))
    nn.functional.cross_entropy(m(X), Y).backward()
    for r, out in enumerate(res):
        for got, p in zip(out["grads"], m.parameters()):
            assert torch.allclose(got, p.grad, atol=1e-6), (
                f"rank {r}: after average_gradients each rank's grads must equal the full-batch gradient (the mean of the "
                "two ranks' gradients). summing without dividing by world_size gives twice that")
        assert out["calls"] == 1, (
            f"rank {r} called dist.all_reduce {out['calls']} times: flatten every gradient into one buffer and reduce it "
            "with a single call (one call per parameter pays the latency once per tensor)")


@pytest.mark.slow
def test_stage5_ddp_matches_single_process(nb):
    res = _run_ranks(nb, "ddp")
    p0, p1 = res[0]["params"], res[1]["params"]
    assert all(torch.equal(a, b) for a, b in zip(p0, p1)), (
        "after DDP training both ranks must hold identical weights. if they differ, rank 0's initial weights weren't "
        "broadcast, or the gradients weren't averaged before every step")
    g = torch.Generator().manual_seed(0)
    X, Y = torch.randn(24, 5, generator=g), torch.randint(0, 3, (24,), generator=g)
    torch.manual_seed(100)  # rank 0's initial weights
    m = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 3))
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    for epoch in range(2):
        shards = [np.asarray(nb.shard_indices(24, r, 2, epoch=epoch, seed=3)) for r in range(2)]
        for s in range(0, len(shards[0]), 4):
            opt.zero_grad()
            loss = sum(nn.functional.cross_entropy(m(X[sh[s:s + 4]]), Y[sh[s:s + 4]]) for sh in shards) / 2
            loss.backward()
            opt.step()
    for a, b in zip(p0, m.parameters()):
        assert torch.allclose(a, b, atol=1e-5), (
            "DDP must be equivalent to single-process SGD whose batch is the union of the ranks' batches (equal-size "
            "local batches, mean loss): check the shards (shard_indices with each epoch and the seed), the batch "
            "slicing, and that averaging happens after backward and before step")
    assert len(res[0]["losses"]) == 6, f"2 epochs of 12 examples per rank in batches of 4: 6 losses; got {len(res[0]['losses'])}"
