"""Checks for 26 · 05 model parallelism & memory.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. memory accounting is
checked against the ZeRO paper's 16 bytes per parameter and every option (precision, optimizer, stage,
world size, uneven shards); saved-tensor measurement against autograd's documented saves; checkpointing
for exact gradients, for saving only its inputs, for recomputing exactly once, and for replaying dropout
without disturbing the global RNG; the tensor-parallel MLP for equality with the full MLP and for which
hidden units each rank owns; the pipeline schedule for completeness, dependencies, one task per stage
per tick and its bubble; ZeroAdamW against torch.optim.AdamW, in one process and in two gloo processes
with a hard timeout.
"""
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs import notebooks  # noqa: E402
from mlfs.testing import bounded  # noqa: E402


# ---------------------------------------------------------------- stage 1: memory accounting


def test_stage1_zero_paper(nb):
    N = 7_500_000_000
    m = nb.training_memory(N)
    assert (m["params"], m["grads"], m["optimizer"]) == (2 * N, 2 * N, 12 * N), (
        f"mixed-precision AdamW (Rajbhandari et al.'s 2 + 2 + 12 bytes per parameter): want {(2 * N, 2 * N, 12 * N)}; "
        f"got {(m['params'], m['grads'], m['optimizer'])}. the 12 is a float32 master copy plus two float32 moments")
    assert m["total"] == 16 * N, "total = params + grads + optimizer: 16 bytes per parameter, 120 GB for 7.5B"
    z1 = nb.training_memory(N, zero_stage=1, world_size=64)
    assert z1["optimizer"] == 12 * N // 64 and z1["params"] == 2 * N and z1["grads"] == 2 * N, (
        "ZeRO stage 1 shards only the optimizer state over the ranks")
    z2 = nb.training_memory(N, zero_stage=2, world_size=64)
    assert z2["grads"] == 2 * N // 64 and z2["params"] == 2 * N, "stage 2 also shards the gradients"
    z3 = nb.training_memory(N, zero_stage=3, world_size=64)
    assert z3["params"] == 2 * N // 64 and np.isclose(z3["total"], 16 * N / 64), "stage 3 shards everything"
    assert nb.training_memory(N, world_size=64) == m, "zero_stage=0 replicates everything whatever the world size"


def test_stage1_options(nb):
    f32 = nb.training_memory(1000, bytes_per_param=4)
    assert (f32["params"], f32["grads"], f32["optimizer"]) == (4000, 4000, 8000), (
        "pure float32 AdamW: no separate master copy (the weights are already float32), two float32 moments: 4 + 4 + 8")
    sgd = nb.training_memory(1000, bytes_per_param=4, optimizer="sgd")
    assert sgd["optimizer"] == 0, "plain SGD keeps no state"
    mom = nb.training_memory(1000, optimizer="momentum")
    assert mom["optimizer"] == 1000 * (4 + 4), "16-bit SGD with momentum: float32 master + one float32 buffer"
    assert nb.training_memory(1000, optimizer="sgd")["optimizer"] == 4000, "16-bit plain SGD still needs the master copy"
    odd = nb.training_memory(10, zero_stage=3, world_size=4)
    assert (odd["params"], odd["grads"], odd["optimizer"]) == (3 * 2, 3 * 2, 3 * 12), (
        "10 parameters over 4 ranks: each rank holds a shard of ceil(10 / 4) = 3 (the last is padded)")
    with pytest.raises(ValueError):
        nb.training_memory(10, optimizer="adagrad")
    with pytest.raises(ValueError):
        nb.training_memory(10, zero_stage=4)


# ---------------------------------------------------------------- stage 2: activations and checkpointing


def test_stage2_saved_bytes(nb):
    x = torch.randn(100, requires_grad=True)
    y = torch.randn(100, requires_grad=True)
    out, b = nb.saved_tensor_bytes(torch.exp, x)
    assert torch.equal(out, torch.exp(x)) and b == 400, f"exp saves its 100-float output for backward: 400 bytes; got {b}"
    assert nb.saved_tensor_bytes(lambda a, c: a * c, x, y)[1] == 800, "a * b saves both inputs"
    assert nb.saved_tensor_bytes(lambda a: a * a, x)[1] == 400, (
        "x * x saves x twice, but it's one storage: count distinct storages, not saves")
    assert nb.saved_tensor_bytes(lambda a: a.sum(), x)[1] == 0, "a sum saves nothing (its gradient needs only a shape)"
    big = torch.randn(1000, requires_grad=True)
    c = torch.randn(10, requires_grad=True)
    b = nb.saved_tensor_bytes(lambda a, w: a[:10] * w, big, c)[1]
    assert b == 4000 + 40, (
        f"a saved view keeps its whole storage alive: big[:10] * c saves a view of a 1000-float storage (4000 bytes) and "
        f"c (40): want 4040, got {b}. size each storage with untyped_storage().nbytes(), not the view's numel")
    lin = nn.Linear(10, 5)
    xi = torch.randn(3, 10, requires_grad=True)
    _, b_all = nb.saved_tensor_bytes(lin, xi)
    _, b_act = nb.saved_tensor_bytes(lin, xi, exclude=list(lin.parameters()))
    assert (b_all, b_act) == (120 + 200, 120), (
        f"a linear layer saves its input (120 bytes) and its weight (200, a parameter); exclude= drops storages shared "
        f"with the given tensors: want (320, 120), got {(b_all, b_act)}")


def _block(seed=0, p=0.0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Dropout(p), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))


def test_stage2_checkpoint_gradients_and_memory(nb):
    blk = _block()
    x = torch.randn(16, 32, requires_grad=True)
    calls = []

    def fn(a):
        calls.append(1)
        return blk(a)

    out = nb.checkpoint(fn, x)
    assert torch.allclose(out, blk(x)), "checkpoint(fn, x) returns fn(x)"
    assert len(calls) == 1, "the forward calls fn once"
    out.pow(2).sum().backward()
    assert len(calls) == 2, f"backward recomputes fn exactly once (fn ran {len(calls)} times in total)"
    g_x, g_p = x.grad.clone(), [p.grad.clone() for p in blk.parameters()]
    x.grad = None
    blk.zero_grad()
    blk(x).pow(2).sum().backward()
    assert torch.allclose(g_x, x.grad, atol=1e-6), "the input's gradient must equal the plain backward's"
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(g_p, blk.parameters())), (
        "the block's parameter gradients must equal the plain backward's: recompute with grad enabled and backprop "
        "through the recomputed graph")
    seen = {}

    def pack(t):
        seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    params = {p.untyped_storage().data_ptr() for p in blk.parameters()}
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        nb.checkpoint(blk, x)
    act = sum(v for k, v in seen.items() if k not in params)
    assert act == x.numel() * 4, (
        f"a checkpointed call keeps only its input ({x.numel() * 4} bytes) for backward; it saved {act} bytes of "
        "activations. run fn under torch.no_grad() in the forward")


def test_stage2_checkpoint_dropout_rng(nb):
    blk = _block(p=0.5).train()
    x = torch.randn(16, 32, requires_grad=True)
    torch.manual_seed(7)
    out = nb.checkpoint(blk, x)
    between = torch.rand(2)  # something else draws random numbers between forward and backward
    out.sum().backward()
    after = torch.rand(3)
    g_ck = [p.grad.clone() for p in blk.parameters()]
    blk.zero_grad()
    torch.manual_seed(7)
    ref = blk(x)
    assert torch.equal(between, torch.rand(2))
    ref.sum().backward()
    assert torch.allclose(out, ref), "the checkpointed forward uses the same dropout masks as a plain forward"
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(g_ck, blk.parameters())), (
        "with dropout, the recomputation must replay the forward's random numbers: save torch.get_rng_state() in the "
        "forward and restore it before recomputing, or the masks differ and the gradients are wrong")
    assert torch.equal(after, torch.rand(3)), (
        "the global random stream must end where a plain forward + backward leaves it: restore the RNG state only "
        "inside torch.random.fork_rng(), so the recomputation doesn't rewind it")


# ---------------------------------------------------------------- stage 3: tensor parallelism


def test_stage3_tensor_parallel_mlp(nb):
    torch.manual_seed(0)
    fc1, fc2 = nn.Linear(12, 32), nn.Linear(32, 12)
    x = torch.randn(5, 12)
    want = fc2(F.gelu(fc1(x)))
    for p in (1, 2, 4):
        shards = nb.split_mlp(fc1, fc2, p)
        assert len(shards) == p
        w1, b1, w2 = shards[-1]
        assert tuple(w1.shape) == (32 // p, 12) and tuple(b1.shape) == (32 // p,) and tuple(w2.shape) == (12, 32 // p), (
            f"world_size {p}: each rank holds W1 rows (h/p, d), b1 (h/p,) and W2 columns (d, h/p); got "
            f"{tuple(w1.shape)}, {tuple(b1.shape)}, {tuple(w2.shape)}")
        out, partials = nb.tp_mlp_forward(x, shards, fc2.bias)
        assert torch.allclose(out, want, atol=1e-5), f"world_size {p}: the sum of partials + b2 must equal the full MLP"
        k = 32 // p
        for r, part in enumerate(partials):
            mine = F.gelu(x @ fc1.weight[r * k:(r + 1) * k].T + fc1.bias[r * k:(r + 1) * k]) @ fc2.weight[:, r * k:(r + 1) * k].T
            assert torch.allclose(part, mine, atol=1e-5), (
                f"rank {r} of {p} must own hidden units {r * k}..{(r + 1) * k - 1} (contiguous blocks), and its partial "
                "output must NOT include b2 (added once, after the all-reduce)")
    with pytest.raises(ValueError):
        nb.split_mlp(fc1, fc2, 5)


def test_stage3_shards_are_copies(nb):
    torch.manual_seed(1)
    fc1, fc2 = nn.Linear(4, 8), nn.Linear(8, 4)
    shards = nb.split_mlp(fc1, fc2, 2)
    shards[0][0].zero_()
    assert fc1.weight.abs().sum() > 0 and fc1.weight[:4].abs().sum() > 0, (
        "each rank's shard is its own tensor (on a real cluster it lives on another device): clone the slices")


# ---------------------------------------------------------------- stage 4: pipeline parallelism


def test_stage4_schedule(nb):
    for p, m in ((4, 4), (3, 8), (1, 3), (5, 1)):
        g = bounded(nb.gpipe_schedule, p, m)
        assert len(g) == 2 * (m + p - 1) and all(len(row) == p for row in g), (
            f"{p} stages, {m} micro-batches: {2 * (m + p - 1)} ticks of {p} slots; got {len(g)} ticks")
        when = {}
        for t, row in enumerate(g):
            for s, ev in enumerate(row):
                if ev is not None:
                    key = (s, ev[0], ev[1])
                    assert key not in when, f"stage {s} runs {ev} twice"
                    when[key] = t
        assert len(when) == 2 * p * m, f"every stage runs every micro-batch's forward and backward exactly once"
        for s in range(p):
            for k in range(m):
                if s > 0:
                    assert when[(s, "F", k)] > when[(s - 1, "F", k)], "F(s, m) needs F(s-1, m)'s output"
                if s < p - 1:
                    assert when[(s, "B", k)] > when[(s + 1, "B", k)], "B(s, m) needs B(s+1, m)'s gradient"
                assert when[(s, "B", k)] > max(when[(q, "F", j)] for q in range(p) for j in range(m)), (
                    "GPipe runs every forward before any backward (that's what makes its memory grow with m)")
        assert when[(0, "F", 0)] == 0 and when[(0, "B", m - 1)] == len(g) - 1, "no wasted ticks at either end"


def test_stage4_bubble(nb):
    for p, m in ((4, 4), (4, 16), (8, 1), (2, 7)):
        g = nb.gpipe_schedule(p, m)
        idle = sum(ev is None for row in g for ev in row) / (len(g) * p)
        assert np.isclose(nb.bubble_fraction(p, m), (p - 1) / (m + p - 1)), "bubble = (p - 1) / (m + p - 1)"
        assert np.isclose(idle, nb.bubble_fraction(p, m)), "the schedule's idle fraction equals the bubble formula"


# ---------------------------------------------------------------- stage 5: ZeRO-1


def _net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))  # 35 + 7 + 21 + 3 = 66 parameters


def test_stage5_single_rank_matches_adamw(nb):
    m, ref = _net(), _net()
    kw = dict(lr=0.05, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.1)
    opt = nb.ZeroAdamW(m.parameters(), rank=0, world_size=1, **kw)
    ropt = torch.optim.AdamW(ref.parameters(), **kw)
    g = torch.Generator().manual_seed(0)
    for step in range(6):
        x, y = torch.randn(8, 5, generator=g), torch.randn(8, 3, generator=g)
        for net, o in ((m, opt), (ref, ropt)):
            o.zero_grad()
            F.mse_loss(net(x), y).backward()
            o.step()
    for a, b in zip(m.parameters(), ref.parameters()):
        assert torch.allclose(a, b, atol=2e-5), (
            "with world_size=1, ZeroAdamW must match torch.optim.AdamW step for step (bias correction, decoupled weight "
            "decay applied to every parameter, betas, eps)")
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    opt.zero_grad()
    assert all(p.grad is None for p in m.parameters()), "zero_grad sets every .grad to None (not to zeros)"


def test_stage5_params_without_grads(nb):
    m, ref = _net(2), _net(2)
    opt = nb.ZeroAdamW(m.parameters(), rank=0, world_size=1, lr=0.05, weight_decay=0.1)
    ropt = torch.optim.AdamW(ref.parameters(), lr=0.05, weight_decay=0.1)
    g = torch.Generator().manual_seed(1)
    for step in range(4):
        x, y = torch.randn(8, 5, generator=g), torch.randn(8, 3, generator=g)
        for net, o in ((m, opt), (ref, ropt)):
            o.zero_grad()
            F.mse_loss(net(x), y).backward()
            if step in (0, 2):
                net[2].bias.grad = None  # this parameter gets no gradient in steps 0 and 2
            o.step()
    for a, b in zip(m.parameters(), ref.parameters()):
        assert torch.allclose(a, b, atol=2e-5), (
            "a parameter whose .grad is None must be skipped entirely, as torch.optim.AdamW does: no weight decay, no "
            "moment update, and its step count (for bias correction) doesn't advance")


def test_stage5_defaults_and_sharding(nb):
    m, ref = _net(1), _net(1)
    opt = nb.ZeroAdamW(m.parameters(), rank=0, world_size=1)
    ropt = torch.optim.AdamW(ref.parameters())
    x, y = torch.randn(8, 5), torch.randn(8, 3)
    for net, o in ((m, opt), (ref, ropt)):
        F.mse_loss(net(x), y).backward()
        o.step()
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(m.parameters(), ref.parameters())), (
        "the defaults (lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight_decay 0.01) are AdamW's")
    o3 = nb.ZeroAdamW(_net().parameters(), rank=2, world_size=3)
    assert o3.exp_avg.numel() == 22 and tuple(o3.shard) == (44, 66), "66 parameters over 3 ranks: shards of exactly 22"
    for rank in range(4):
        o = nb.ZeroAdamW(_net().parameters(), rank=rank, world_size=4)
        assert o.exp_avg.numel() == 17 and o.exp_avg_sq.numel() == 17, (
            f"66 parameters over 4 ranks: each rank keeps moments for a shard of ceil(66 / 4) = 17 values; rank {rank} "
            f"holds {o.exp_avg.numel()}")
        assert tuple(o.shard) == (17 * rank, min(17 * rank + 17, 66)), (
            f"rank {rank}'s shard of the flattened parameters is [17 r, 17 r + 17) clipped to 66; got {o.shard}")


WORKER = textwrap.dedent(r"""
    import pickle, sys
    import torch, torch.distributed as dist
    from torch import nn
    import torch.nn.functional as F
    torch.set_num_threads(1)
    nb_path, rank, world, store, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    from mlfs import notebooks
    nb = notebooks.load(nb_path)
    dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=world)
    try:
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))
        opt = nb.ZeroAdamW(m.parameters(), rank=rank, world_size=world, lr=0.05, weight_decay=0.1)
        g = torch.Generator().manual_seed(0)
        m1 = None
        for step in range(5):
            x, y = torch.randn(16, 5, generator=g), torch.randn(16, 3, generator=g)
            lo, hi = rank * 8, (rank + 1) * 8
            opt.zero_grad()
            F.mse_loss(m(x[lo:hi]), y[lo:hi]).backward()
            opt.step()
            if step == 0:
                m1 = opt.exp_avg.clone()
        with open(out, "wb") as f:
            pickle.dump({"params": [p.detach().clone() for p in m.parameters()], "state": opt.exp_avg.numel(),
                         "m1": m1}, f)
    finally:
        dist.destroy_process_group()
""")


@pytest.mark.slow
def test_stage5_two_ranks(nb):
    path = getattr(nb, "__file__", None)
    if path is None:
        pytest.skip("this check starts real processes that load your notebook from disk: run `pytest -k stage5`")
    with tempfile.TemporaryDirectory() as d:
        outs = [os.path.join(d, f"r{r}.pkl") for r in range(2)]
        env = {**os.environ, "MLFS_ROOT": str(notebooks.repo_root())}
        procs = [subprocess.Popen([sys.executable, "-c", WORKER, path, str(r), "2", os.path.join(d, "store"), outs[r]],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) for r in range(2)]
        try:
            errs = [p.communicate(timeout=90)[1] for p in procs]
        except subprocess.TimeoutExpired:
            for p in procs:
                p.kill()
            for p in procs:
                p.communicate()
            raise AssertionError("the 2 ranks didn't finish in 90 s: a collective that only one rank calls?") from None
        for r, (p, e) in enumerate(zip(procs, errs)):
            assert p.returncode == 0, f"rank {r} failed:\n" + "\n".join(e.strip().splitlines()[-12:])
        res = []
        for o in outs:
            with open(o, "rb") as f:
                res.append(pickle.load(f))
    assert res[0]["state"] == res[1]["state"] == 33, "66 parameters over 2 ranks: 33 moments each"
    assert all(torch.equal(a, b) for a, b in zip(res[0]["params"], res[1]["params"])), (
        "after the all-gather both ranks must hold identical parameters")
    torch.manual_seed(0)
    ref = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))
    g0 = torch.Generator().manual_seed(0)
    x0, y0 = torch.randn(16, 5, generator=g0), torch.randn(16, 3, generator=g0)
    ((F.mse_loss(ref(x0[:8]), y0[:8]) + F.mse_loss(ref(x0[8:]), y0[8:])) / 2).backward()
    flat = torch.cat([p.grad.flatten() for p in ref.parameters()])
    ref.zero_grad()
    for r in range(2):
        assert torch.allclose(res[r]["m1"], 0.1 * flat[33 * r:33 * r + 33], atol=1e-6), (
            f"rank {r}'s first moment after one step must be (1 - beta1) times its slice of the AVERAGED gradient: "
            "divide the all-reduced sum by world_size (Adam's update hides a factor of 2, its moments don't)")
    ropt = torch.optim.AdamW(ref.parameters(), lr=0.05, weight_decay=0.1)
    g = torch.Generator().manual_seed(0)
    for step in range(5):
        x, y = torch.randn(16, 5, generator=g), torch.randn(16, 3, generator=g)
        ropt.zero_grad()
        ((F.mse_loss(ref(x[:8]), y[:8]) + F.mse_loss(ref(x[8:]), y[8:])) / 2).backward()
        ropt.step()
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(res[0]["params"], ref.parameters())), (
        "two ranks with ZeroAdamW must equal one AdamW on the averaged gradient: all-reduce the flat gradient and "
        "divide by world_size, update only your shard, then all-gather every rank's shard")
