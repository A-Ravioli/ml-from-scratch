"""Checks for 26 · 01 the performance model.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the FLOP and byte
counts are checked against hand computations with three different dimensions (so a swapped m, n, k
shows up); the roofline in both regimes, at the ridge, and on arrays; the benchmark by the exact
order in which it calls the function and the sync hook (no timing thresholds); the GPT-block calculator
against the lesson's table and against the closed forms 24 B T d^2 and 4 B T^2 d; the profiler wrapper
by the number of calls it reports for each op, which is deterministic.
"""
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------- stage 1: flops, bytes, intensity


def test_stage1_matmul_flops(nb):
    got = nb.matmul_flops(2, 3, 5)
    assert got == 60, (
        f"(2, 5) @ (5, 3) does 2*3 = 6 dot products of length 5, each 5 multiplies and 5 adds: 60 FLOPs; got {got}. "
        "if you got 30, you counted multiply-adds instead of FLOPs (one of each per term)")
    assert nb.matmul_flops(7, 1, 1) == 14 and nb.matmul_flops(1, 1, 9) == 18, "2 m n k for any m, n, k"
    assert nb.matmul_flops(4096, 4096, 4096) == 2 * 4096 ** 3, "exact integer arithmetic for big sizes"


def test_stage1_matmul_bytes(nb):
    got = nb.matmul_bytes(2, 3, 5)
    assert got == (2 * 5 + 5 * 3 + 2 * 3) * 4, (
        f"(2, 5) @ (5, 3) -> (2, 3) in float32 reads 10 + 15 elements and writes 6: 124 bytes; got {got}. "
        "count A (m*k), B (k*n) and the output C (m*n) once each")
    assert nb.matmul_bytes(2, 3, 5, bytes_per_element=2) == 62, "bytes_per_element scales the count (2 for bf16/fp16)"
    assert nb.matmul_bytes(1, 1000, 1000) == (1000 + 1000 * 1000 + 1000) * 4, (
        "a matrix-vector product (m = 1) moves the whole matrix for very few FLOPs")


def test_stage1_intensity(nb):
    assert np.isclose(nb.arithmetic_intensity(60, 124), 60 / 124), "intensity is FLOPs divided by bytes"
    n = 1024
    got = nb.arithmetic_intensity(nb.matmul_flops(n, n, n), nb.matmul_bytes(n, n, n))
    assert np.isclose(got, n / 6), (
        f"a square float32 matmul has intensity 2n^3 / (12 n^2) = n/6 = {n / 6:.1f} FLOP/byte; got {got}")
    gemv = nb.arithmetic_intensity(nb.matmul_flops(1, n, n), nb.matmul_bytes(1, n, n))
    assert gemv < 0.6, f"a matrix-vector product does about 2 FLOPs per 4-byte weight: intensity ~0.5; got {gemv}"


# ---------------------------------------------------------------- stage 2: the roofline


def test_stage2_roofline_regimes(nb):
    peak, bw = 10e12, 100e9  # ridge at 100 FLOP/byte
    assert np.isclose(nb.roofline(1.0, peak, bw), 100e9), (
        "below the ridge you're memory-bound: attainable = intensity * bandwidth = 1 * 100e9")
    assert np.isclose(nb.roofline(1000.0, peak, bw), peak), "above the ridge you hit the compute roof: peak_flops"
    assert np.isclose(nb.roofline(100.0, peak, bw), peak), "at the ridge both limits agree"
    xs = np.array([0.25, 10.0, 100.0, 1e4])
    got = np.asarray(nb.roofline(xs, peak, bw))
    assert got.shape == (4,) and np.allclose(got, [25e9, 1e12, 10e12, 10e12]), (
        f"roofline works elementwise on an array of intensities; got {got}. use np.minimum, not min()")


def test_stage2_ridge_and_bound(nb):
    peak, bw = 3.6e12, 90e9
    assert np.isclose(nb.ridge_point(peak, bw), 40.0), "ridge = peak_flops / bandwidth (FLOP/byte)"
    assert nb.bound(5.0, peak, bw) == "memory", "below the ridge an op is memory-bound"
    assert nb.bound(400.0, peak, bw) == "compute", "above the ridge an op is compute-bound"
    assert nb.bound(40.0, peak, bw) == "compute", "exactly at the ridge counts as 'compute' (memory only when below)"
    assert nb.bound(39.9, 3.6e12, 90e9) == "memory"


# ---------------------------------------------------------------- stage 3: benchmarking


def test_stage3_call_order(nb):
    events = []
    out = nb.benchmark(lambda: events.append("fn"), warmup=2, repeats=3, sync=lambda: events.append("sync"))
    want = ["fn", "fn", "sync"] + ["fn", "sync"] * 3
    assert events == want, (
        f"expected the calls {want}, got {events}. warm up first, sync once to drain the warmup's queued work, then "
        "for each repeat: start the clock, call fn, sync, stop the clock. without the sync after fn, a GPU timing "
        "measures only how long it took to *queue* the work")
    assert len(out["times"]) == 3


def test_stage3_statistics(nb):
    calls = []
    out = nb.benchmark(lambda: calls.append(1), warmup=0, repeats=7)
    assert len(calls) == 7, f"warmup=0, repeats=7 calls fn exactly 7 times; got {len(calls)}"
    assert set(out) >= {"best", "median", "times"}, f"returns a dict with 'best', 'median', 'times'; got {set(out)}"
    t = out["times"]
    assert len(t) == 7 and all(x >= 0 for x in t)
    assert out["best"] == min(t), "best is the minimum time: the run least disturbed by other processes"
    assert np.isclose(out["median"], np.median(t)), "median is np.median of the times"


def test_stage3_each_repeat_timed_separately(nb):
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] == 2:
            time.sleep(0.08)  # only the first timed call is slow

    t = nb.benchmark(fn, warmup=1, repeats=3)["times"]
    assert t[0] >= 0.07 and t[2] < 0.05, (
        f"each repeat's time covers only its own call (restart the clock every repeat); got {t}")


def test_stage3_clock_encloses_sync_but_not_warmup(nb):
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(0.25)  # a slow first call (compilation, caches): must not be timed

    out = nb.benchmark(fn, warmup=1, repeats=2, sync=lambda: time.sleep(0.02))
    assert min(out["times"]) >= 0.018, (
        f"each timed interval must include the sync that follows fn (it waits for the device); times {out['times']}")
    assert max(out["times"]) < 0.2, (
        f"the warmup call must not be timed; times {out['times']}. call fn warmup times before the timed loop")


# ---------------------------------------------------------------- stage 4: a GPT block, op by op


def _expected_ops(B, T, d, H, dff, b):
    BT, BH, dh = B * T, B * H, d // H

    def lin(din, dout):
        return 2 * BT * din * dout, (BT * din + din * dout + BT * dout) * b

    return {
        "ln1": (7 * BT * d, 2 * BT * d * b), "q_proj": lin(d, d), "k_proj": lin(d, d), "v_proj": lin(d, d),
        "scores": (BH * 2 * T * T * dh, BH * (T * dh + dh * T + T * T) * b),
        "softmax": (5 * BH * T * T, 2 * BH * T * T * b),
        "attn_v": (BH * 2 * T * dh * T, BH * (T * T + T * dh + T * dh) * b),
        "out_proj": lin(d, d), "residual1": (BT * d, 3 * BT * d * b), "ln2": (7 * BT * d, 2 * BT * d * b),
        "fc1": lin(d, dff), "gelu": (8 * BT * dff, 2 * BT * dff * b), "fc2": lin(dff, d), "residual2": (BT * d, 3 * BT * d * b),
    }


NAMES = ["ln1", "q_proj", "k_proj", "v_proj", "scores", "softmax", "attn_v", "out_proj", "residual1", "ln2", "fc1",
         "gelu", "fc2", "residual2"]


def test_stage4_train_flops(nb):
    assert nb.transformer_train_flops(10**6, 10**9) == 6 * 10**15, "6 N D: 2 N D forward plus 4 N D backward"
    assert nb.transformer_train_flops(3, 5) == 90


def test_stage4_block_table(nb):
    ops = nb.gpt_block_ops(2, 8, 16, 4)
    names = [o["name"] for o in ops]
    assert names == NAMES, f"ops in execution order with the lesson's names:\n  want {NAMES}\n  got  {names}"
    want = _expected_ops(2, 8, 16, 4, 64, 4)
    for o in ops:
        f, by = want[o["name"]]
        assert o["flops"] == f, f"{o['name']}: flops should be {f}, got {o['flops']} (see the spec table)"
        assert o["bytes"] == by, f"{o['name']}: bytes should be {by}, got {o['bytes']} (see the spec table)"


def test_stage4_worked_example(nb):
    ops = {o["name"]: o for o in nb.gpt_block_ops(64, 128, 192, 6)}
    assert (ops["fc1"]["flops"], ops["fc1"]["bytes"]) == (2_415_919_104, 32_047_104), (
        f"the lesson's worked example: fc1 of model M does 2,415,919,104 FLOPs and moves 32,047,104 bytes; got "
        f"{ops['fc1']['flops']:,} and {ops['fc1']['bytes']:,}")
    assert (ops["scores"]["flops"], ops["scores"]["bytes"]) == (402_653_184, 37_748_736), (
        f"scores: 384 products (128, 32) @ (32, 128): 402,653,184 FLOPs, 37,748,736 bytes; got "
        f"{ops['scores']['flops']:,}, {ops['scores']['bytes']:,}")
    assert (ops["softmax"]["flops"], ops["softmax"]["bytes"]) == (31_457_280, 50_331_648), (
        f"softmax over 6,291,456 scores: 31,457,280 FLOPs, 50,331,648 bytes; got "
        f"{ops['softmax']['flops']:,}, {ops['softmax']['bytes']:,}")
    peak, bw = 3e12, 1e11
    kinds = {k: nb.bound(nb.arithmetic_intensity(o["flops"], o["bytes"]), peak, bw) for k, o in ops.items()}
    assert kinds["fc1"] == "compute" and kinds["scores"] == "memory" and kinds["softmax"] == "memory"


def test_stage4_options_and_closed_forms(nb):
    B, T, d, H = 3, 10, 24, 6
    ops = {o["name"]: o for o in nb.gpt_block_ops(B, T, d, H, d_ff=40, bytes_per_element=2)}
    want = _expected_ops(B, T, d, H, 40, 2)
    for k, (f, by) in want.items():
        assert (ops[k]["flops"], ops[k]["bytes"]) == (f, by), (
            f"{k} with d_ff=40, bytes_per_element=2: want flops {f}, bytes {by}; got {ops[k]['flops']}, {ops[k]['bytes']}. "
            "d_ff sets fc1/gelu/fc2's width; bytes_per_element scales every byte count")
    ops = {o["name"]: o for o in nb.gpt_block_ops(B, T, d, H)}
    lin = sum(ops[k]["flops"] for k in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"))
    assert lin == 24 * B * T * d * d, "the six projections do 2 * (12 d^2) FLOPs per token: 24 B T d^2 in total"
    att = ops["scores"]["flops"] + ops["attn_v"]["flops"]
    assert att == 4 * B * T * T * d, "the two attention matmuls do 4 B T^2 d FLOPs, independent of the head count"
    ops2 = {o["name"]: o for o in nb.gpt_block_ops(B, T, d, 2)}
    assert ops2["softmax"]["bytes"] * 3 == ops["softmax"]["bytes"], (
        "the softmax moves B * n_heads * T^2 scores: three times as many heads, three times the bytes")


def test_stage4_mfu(nb):
    assert np.isclose(nb.mfu(1e12, 0.5, 4e12), 0.5), "1e12 FLOPs in 0.5 s is 2e12 FLOP/s: half of a 4e12 peak"
    assert np.isclose(nb.mfu(6e9, 2.0, 1e10), 0.3)


# ---------------------------------------------------------------- stage 5: profiling


def test_stage5_profile_counts(nb):
    torch.manual_seed(0)
    a, b = torch.randn(64, 64), torch.randn(64, 64)
    calls = []

    def step():
        calls.append(1)
        x = a @ b
        x = torch.relu(x @ b)
        torch.relu(x @ a)

    rows = nb.profile_ops(step, n_steps=4, warmup=2)
    assert len(calls) == 6, f"2 warmup calls plus 4 profiled ones: step_fn runs 6 times; got {len(calls)}"
    by_name = {r[0]: r for r in rows}
    assert "aten::mm" in by_name and "aten::relu" in by_name, (
        f"expected rows for 'aten::mm' and 'aten::relu' (torch.profiler's key_averages() keys); got {list(by_name)[:8]}")
    assert by_name["aten::mm"][2] == 12, (
        f"3 matmuls per step x 4 profiled steps = 12 calls of aten::mm; got {by_name['aten::mm'][2]}. "
        "are the warmup calls being profiled too?")
    assert by_name["aten::relu"][2] == 8, f"2 relus x 4 steps = 8 calls; got {by_name['aten::relu'][2]}"
    times = [r[1] for r in rows]
    assert times == sorted(times, reverse=True), "rows are sorted by self time, slowest first"
    assert all(isinstance(r[2], int) and r[1] >= 0 for r in rows)


def test_stage5_defaults(nb):
    calls = []

    def step():
        calls.append(1)
        torch.ones(8) @ torch.ones(8)

    rows = nb.profile_ops(step)
    assert len(calls) == 4, f"defaults: 1 warmup + 3 profiled calls; got {len(calls)}"
    assert {r[0]: r[2] for r in rows}.get("aten::dot") == 3, "a vector dot product shows up as aten::dot, 3 calls"


def test_stage5_self_time_not_total(nb):
    torch.manual_seed(1)
    a = torch.randn(384, 384)
    b = torch.randn(384, 384)

    def step():
        for _ in range(4):
            torch.nn.functional.linear(a, b)  # aten::linear -> aten::matmul -> aten::mm: nested ops

    by = {r[0]: r[1] for r in nb.profile_ops(step, n_steps=3, warmup=1)}
    assert by["aten::mm"] > by["aten::linear"] and by["aten::mm"] > by["aten::matmul"], (
        f"aten::linear and aten::matmul only call aten::mm, which does the work; with self times mm is the biggest row, "
        f"with total times (which include the ops called) linear >= matmul >= mm. got linear {by['aten::linear']:.0f}, "
        f"matmul {by['aten::matmul']:.0f}, mm {by['aten::mm']:.0f} us: report self_cpu_time_total")
