"""Checks for capstone.py (make your GPT 5x faster to train).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. everything runs on the CPU
with tiny GPTs on synthetic token streams; there are no timing thresholds. fused attention is checked
against your chapter 18 attention (outputs, gradients, cached generation, dropout) and for actually
calling the fused kernel; the timer by its call pattern; the training step and trainer by watching the
ops they dispatch (float16/bfloat16 matmuls, the fused AdamW kernel, the fused attention kernel), by a
recording compiler backend, and against train_lm itself: with every switch off, train_fast must take the
same steps on the same batches with the same schedule. the speedup interval against its definition.
"""
import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from mlfs.systems import GradScaler  # noqa: E402  (reference, from topic 03)
from mlfs.transformer import GPT, estimate_loss, generate_cached, train_lm  # noqa: E402  (reference, ch 18)

V = 11


def _gpt(seed=0, dropout=0.0, T=16):
    torch.manual_seed(seed)
    return GPT(V, block_size=T, n_layers=2, n_heads=2, d_model=16, dropout=dropout)


def _stream(n=3000, seed=0):
    return np.random.default_rng(seed).integers(0, V, n)


class _SeeFunctions(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.names = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.names.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))


class _SeeOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        dts = [a.dtype for a in args if isinstance(a, torch.Tensor)]
        self.ops.append((str(func.overloadpacket.__name__), dts))
        self.scalars = getattr(self, "scalars", []) + [a for a in args if isinstance(a, (int, float))]
        return func(*args, **(kwargs or {}))


# ---------------------------------------------------------------- stage 1: fused attention


def test_stage1_same_model(nb):
    ref, m = _gpt(), _gpt()
    out = nb.fused_attention(m)
    assert out is m, "fused_attention patches the model in place and returns it"
    idx = torch.randint(0, V, (3, 16), generator=torch.Generator().manual_seed(0))
    y = torch.randint(0, V, (3, 16), generator=torch.Generator().manual_seed(1))
    with _SeeFunctions() as seen:
        lf, loss_f = m(idx, y)
    assert seen.names.count("scaled_dot_product_attention") == 2, (
        f"each of the 2 blocks must call F.scaled_dot_product_attention once per forward; saw "
        f"{seen.names.count('scaled_dot_product_attention')} calls")
    lr_, loss_r = ref(idx, y)
    assert torch.allclose(lf, lr_, atol=1e-5), "the fused model must compute the same logits as your chapter 18 GPT"
    loss_f.backward()
    loss_r.backward()
    for (n, a), b in zip(m.named_parameters(), ref.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-5), f"gradient of {n} differs from the unfused model's"
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in ref.parameters()), "no parameters added"
    nb.fused_attention(m)
    assert torch.allclose(m(idx)[0], lr_, atol=1e-5), "patching twice is harmless"


def test_stage1_cache_and_dropout(nb):
    ref, m = _gpt(T=32), nb.fused_attention(_gpt(T=32))
    prompt = torch.randint(0, V, (2, 5), generator=torch.Generator().manual_seed(2))
    assert torch.equal(generate_cached(m, prompt, 20, temperature=0.0), generate_cached(ref, prompt, 20, temperature=0.0)), (
        "generation with a KV cache must still work and give the same tokens: when a cache (or a mask, or no causal "
        "flag) is passed, fall back to the original attention")
    d = nb.fused_attention(_gpt(dropout=0.5))
    idx = torch.randint(0, V, (4, 16), generator=torch.Generator().manual_seed(3))
    d.train()
    torch.manual_seed(0)
    a = d(idx)[0]
    b = d(idx)[0]
    assert not torch.allclose(a, b), "in training mode the attention dropout must still be applied"
    for blk in d.blocks:
        blk.mlp.dropout.p = 0.0
        blk.drop.p = 0.0
    d.drop.p = 0.0
    torch.manual_seed(0)
    a = d(idx)[0]
    b = d(idx)[0]
    assert not torch.allclose(a, b), "with only attention dropout left, training-mode outputs must still vary"
    d.eval()
    ref_d = _gpt(dropout=0.5).eval()
    assert torch.allclose(d(idx)[0], ref_d(idx)[0], atol=1e-5), "in eval mode there's no dropout (dropout_p = 0)"


def test_stage1_fallbacks(nb):
    ref, m = _gpt(), nb.fused_attention(_gpt())
    x = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(4))
    a_ref, a = ref.blocks[0].attn, m.blocks[0].attn
    assert torch.allclose(a(x, is_causal=False), a_ref(x, is_causal=False), atol=1e-5), (
        "called with is_causal=False (bidirectional attention), the patched attention must not apply a causal mask: "
        "fall back to the original forward")
    a.save_weights = True
    out = a(x, is_causal=True)
    assert a.last_weights is not None and tuple(a.last_weights.shape) == (2, 2, 9, 9), (
        "with save_weights=True the attention must still store last_weights (the fused kernel never forms them: fall back)")
    assert torch.allclose(out, a_ref(x, is_causal=True), atol=1e-5)


# ---------------------------------------------------------------- stage 2: timing


def test_stage2_call_pattern(nb):
    events = []
    out = nb.time_steps(lambda: events.append("step"), 4, warmup=2, repeats=3, sync=lambda: events.append("sync"))
    want = ["step"] * 2 + (["sync"] + ["step"] * 4 + ["sync"]) * 3
    assert events == want, (
        f"expected {want}, got {events}: warm up, then for each repeat sync, start the clock, run n_steps steps, sync, "
        "stop the clock")
    out = np.asarray(out)
    assert out.shape == (3,) and (out >= 0).all(), "one mean seconds-per-step value per repeat"
    calls = []
    nb.time_steps(lambda: calls.append(1), 5)
    assert len(calls) == 3 + 3 * 5, "defaults: warmup 3, repeats 3"


def test_stage2_is_a_mean_per_step(nb):
    import time as _t
    out = np.asarray(nb.time_steps(lambda: _t.sleep(0.03), 10, warmup=0, repeats=2))
    assert (out >= 0.029).all() and (out < 0.2).all(), (
        f"each value is the window's time divided by n_steps (about 30 ms per step here), not the window's total "
        f"(at least 0.3 s); got {out}")


# ---------------------------------------------------------------- stage 3: the training step


def _xy(B=4, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (B, T), generator=g), torch.randint(0, V, (B, T), generator=g)


def test_stage3_plain_step(nb):
    m, ref = _gpt(), _gpt()
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    ropt = torch.optim.SGD(ref.parameters(), lr=0.5)
    step = nb.make_train_step(m, opt)
    x, y = _xy()
    loss = step(x, y)
    assert isinstance(loss, torch.Tensor) and not loss.requires_grad and loss.dim() == 0, (
        "the step returns the loss as a detached 0-d tensor (not loss.item(): that would sync the device every step)")
    _, rl = ref(x, y)
    rl.backward()
    torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
    ropt.step()
    assert torch.allclose(loss, rl.detach()) and all(
        torch.allclose(a, b, atol=1e-6) for a, b in zip(m.parameters(), ref.parameters())), (
        "one step = forward, backward, clip the gradient norm to grad_clip (default 1.0), optimizer.step()")
    step(*_xy(seed=1))
    ropt.zero_grad()
    _, rl = ref(*_xy(seed=1))
    rl.backward()
    torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
    ropt.step()
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(m.parameters(), ref.parameters())), (
        "gradients must be zeroed every step")
    m2 = _gpt()
    w0 = [p.detach().clone() for p in m2.parameters()]
    nb.make_train_step(m2, torch.optim.SGD(m2.parameters(), lr=1.0), grad_clip=0.01)(x, y)
    moved = math.sqrt(sum((p.detach() - q).pow(2).sum().item() for p, q in zip(m2.parameters(), w0)))
    assert moved <= 0.01 + 1e-5, f"grad_clip=0.01 with lr 1 moves the weights by at most 0.01; moved {moved:.4f}"


def test_stage3_amp_scaler_compile(nb):
    m = _gpt()
    step = nb.make_train_step(m, torch.optim.SGD(m.parameters(), lr=0.1), amp_dtype=torch.bfloat16)
    with _SeeOps() as seen:
        step(*_xy())
    mm = [d for op, d in seen.ops if op in ("mm", "addmm", "bmm")]
    assert mm and any(torch.bfloat16 in d for d in mm), "with amp_dtype=torch.bfloat16 the matmuls run in bfloat16 (autocast)"
    calls = []

    class Spy(GradScaler):
        def scale(self, loss):
            calls.append("scale")
            return super().scale(loss)

        def unscale_(self, opt):
            calls.append("unscale_")
            return super().unscale_(opt)

        def step(self, opt):
            calls.append("step")
            return super().step(opt)

        def update(self):
            calls.append("update")
            return super().update()

    m = _gpt()
    step = nb.make_train_step(m, torch.optim.SGD(m.parameters(), lr=0.1), amp_dtype=torch.float16,
                              scaler=Spy(init_scale=2.0**8))
    step(*_xy())
    assert calls == ["scale", "unscale_", "step", "update"], (
        f"with a scaler: scale the loss, backward, unscale_, clip, scaler.step, scaler.update; calls were {calls}")
    backends = []

    def backend(gm, example_inputs):
        backends.append(len(gm.graph.nodes))
        return gm.forward

    m = _gpt()
    step = nb.make_train_step(m, torch.optim.SGD(m.parameters(), lr=0.1), compile_backend=backend)
    l1 = step(*_xy())
    assert backends, "compile_backend is passed to torch.compile(model, backend=...) and used for the forward"
    assert torch.isfinite(l1)
    torch._dynamo.reset()


class _Big(torch.nn.Module):
    """a causal-LM-protocol stand-in with a large loss: forward(x, y) -> (out, 100 * mse)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(6, 4)

    def forward(self, x, y):
        out = self.lin(x)
        return out, F.mse_loss(out.float(), y) * 100


def test_stage3_clipping_and_skips_with_scaler(nb):
    g = torch.Generator().manual_seed(5)
    x, y = torch.randn(16, 6, generator=g), torch.randn(16, 4, generator=g)
    m = _Big()
    p0 = [p.detach().clone() for p in m.parameters()]
    step = nb.make_train_step(m, torch.optim.SGD(m.parameters(), lr=1.0), amp_dtype=torch.float16,
                              scaler=GradScaler(init_scale=256.0), grad_clip=0.5)
    step(x, y)
    moved = math.sqrt(sum((p.detach() - q).pow(2).sum().item() for p, q in zip(m.parameters(), p0)))
    assert 0.4 < moved <= 0.5 + 1e-3, (
        f"with a scaler and grad_clip=0.5 (lr 1) the update's norm must be ~0.5; got {moved:.4f}. unscale_ BEFORE "
        "clipping (clipping the scaled gradients clips at 0.5 / 256), and don't skip the clip when a scaler is used")
    m = _Big()
    step = nb.make_train_step(m, torch.optim.SGD(m.parameters(), lr=1.0), amp_dtype=torch.float16,
                              scaler=GradScaler(init_scale=2.0**40))
    step(x, y)
    assert step.skipped == 1, f"a step the scaler skips (a 2^40 scale overflows float16) is counted: skipped {step.skipped}"
    step(x, y)
    assert step.skipped == 2, "and every later skip too (the scale backs off one factor of 2 per skip)"


# ---------------------------------------------------------------- stage 4: train_fast


def test_stage4_faithful_to_train_lm(nb):
    data = _stream()
    ref, m = _gpt(), _gpt()
    h_ref = train_lm(ref, data, 16, batch_size=4, max_steps=25, lr=3e-3, warmup_steps=5, eval_interval=5,
                     eval_batches=2, seed=7)
    h = nb.train_fast(m, data, 16, batch_size=4, max_steps=25, lr=3e-3, warmup_steps=5, seed=7)
    for (n, a), b in zip(m.named_parameters(), ref.parameters()):
        assert torch.allclose(a, b, atol=1e-5), (
            f"with every switch off, train_fast must reproduce train_lm exactly (same seed -> same batches, same "
            f"AdamW groups and betas, same warmup + cosine schedule, same clipping); {n} differs")
    assert not m.training, "train_fast leaves the model in eval mode, like train_lm"
    lr = np.asarray(h["lr"])
    assert lr.shape == (25,) and np.allclose(lr[np.array(h_ref["step"]) - 1], h_ref["lr"]), (
        "history['lr'] lists the learning rate used at every step, matching train_lm's schedule")
    loss = np.asarray(h["loss"])
    assert loss.shape == (25,) and np.isfinite(loss).all(), "history['loss'] has every step's training loss"
    assert math.isnan(h["val_loss"]) and h["skipped"] == 0 and h["wall"] > 0


def test_stage4_seed_val_and_options(nb):
    data, val = _stream(), _stream(800, seed=1)
    a = nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=6, warmup_steps=2, seed=1)["loss"]
    b = nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=6, warmup_steps=2, seed=1)["loss"]
    c = nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=6, warmup_steps=2, seed=2)["loss"]
    assert np.allclose(a, b) and not np.allclose(a, c), "the seed drives batch sampling: same seed, same losses"
    m = _gpt()
    h = nb.train_fast(m, data, 16, val_data=val, batch_size=4, max_steps=4, warmup_steps=1, eval_batches=3)
    assert np.isclose(h["val_loss"], estimate_loss(m, val, 16, batch_size=4, n_batches=3)), (
        "val_loss is estimate_loss(model, val_data, block_size, batch_size=batch_size, n_batches=eval_batches) at the end")
    w = _gpt()
    w0 = {n: p.detach().clone() for n, p in w.named_parameters()}
    # step 1 has lr 0 (warmup starts at 0), so step 2 sees the same gradients in both runs
    nb.train_fast(w, data, 16, batch_size=4, max_steps=2, warmup_steps=1, lr=1e-2, weight_decay=0.0, grad_clip=None)
    w2 = _gpt()
    nb.train_fast(w2, data, 16, batch_size=4, max_steps=2, warmup_steps=1, lr=1e-2, weight_decay=0.5, grad_clip=None)
    diff = {n: (p - q).abs().max().item() for (n, p), q in zip(w.named_parameters(), w2.parameters())}
    assert diff["blocks.0.attn.q_proj.weight"] > 0, "weight_decay applies to matrices"
    assert diff["blocks.0.attn.q_proj.bias"] < 1e-7 and diff["ln_f.weight"] < 1e-7, (
        "weight decay must skip 1-D parameters (biases, layer-norm weights), as in train_lm")


def test_stage4_clip_and_short_runs(nb):
    data = _stream()

    def moved(clip):
        m = _gpt()
        w0 = [p.detach().clone() for p in m.parameters()]
        nb.train_fast(m, data, 16, batch_size=4, max_steps=3, warmup_steps=1, lr=1e-2, weight_decay=0.0, grad_clip=clip)
        return math.sqrt(sum((p.detach() - q).pow(2).sum().item() for p, q in zip(m.parameters(), w0)))

    assert moved(1e-12) < 0.01 * moved(None), (
        "grad_clip must be passed through: clipped to 1e-12, the gradients are far below AdamW's eps and the weights "
        "barely move")
    ref = _gpt()
    h_ref = train_lm(ref, data, 16, batch_size=4, max_steps=3, lr=3e-3, warmup_steps=10, eval_interval=1, eval_batches=1)
    m = _gpt()
    h = nb.train_fast(m, data, 16, batch_size=4, max_steps=3, lr=3e-3, warmup_steps=10)
    assert np.allclose(h["lr"], h_ref["lr"]), (
        f"warmup_steps >= max_steps is clamped to max_steps - 1, as in train_lm: lrs {h['lr']} vs train_lm's {h_ref['lr']}")
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(m.parameters(), ref.parameters()))


def test_stage4_switches(nb):
    data = _stream()
    with _SeeFunctions() as f, _SeeOps() as d:
        nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=2, warmup_steps=1, fused_attn=True, fused_optimizer=True)
    assert "scaled_dot_product_attention" in f.names, "fused_attn=True applies your fused_attention"
    assert any(op.startswith("_fused_adamw") for op, _ in d.ops), (
        "fused_optimizer=True creates torch.optim.AdamW(..., fused=True), whose step is one fused kernel")
    with _SeeFunctions() as f, _SeeOps() as d:
        nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=2, warmup_steps=1)
    assert "scaled_dot_product_attention" not in f.names and not any(op.startswith("_fused_adamw") for op, _ in d.ops), (
        "with the switches off, neither is used")
    with _SeeOps() as d:
        h = nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=2, warmup_steps=1, amp_dtype=torch.bfloat16)
    assert any(torch.bfloat16 in dt for op, dt in d.ops if op in ("mm", "addmm", "bmm")), "amp_dtype runs matmuls in it"
    with _SeeOps() as d:
        h = nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=3, warmup_steps=1, amp_dtype=torch.float16)
    assert any(torch.float16 in dt for op, dt in d.ops if op in ("mm", "addmm", "bmm")), "float16 autocast"
    assert any(v in (65536.0, 2.0**-16) for v in getattr(d, "scalars", [])), (
        "amp_dtype=torch.float16 must use a GradScaler (yours, from mlfs.systems, default scale 2^16): no op multiplied "
        "or divided by 65536 (scaling the loss, unscaling the gradients)")
    assert np.isfinite(h["loss"]).all() and isinstance(h["skipped"], int) and h["skipped"] >= 0, (
        "history['skipped'] counts the steps the scaler skipped (an int)")
    used = []

    class Spy(GradScaler):
        def update(self):
            used.append(1)
            return super().update()

    nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=3, warmup_steps=1, amp_dtype=torch.float16, scaler=Spy())
    assert len(used) == 3, "a scaler passed to train_fast is the one used (update() once per step)"
    used.clear()
    nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=2, warmup_steps=1, scaler=Spy())
    assert not used, "without amp_dtype=float16 there's no loss scaling, even if a scaler is passed"
    seen = []

    def backend(gm, example_inputs):
        seen.append(1)
        return gm.forward

    nb.train_fast(_gpt(), data, 16, batch_size=4, max_steps=2, warmup_steps=1, compile_backend=backend)
    assert seen, "compile_backend is passed through to torch.compile"
    torch._dynamo.reset()


# ---------------------------------------------------------------- stage 5: speedups with uncertainty


def test_stage5_speedup_ci(nb):
    base = [0.30, 0.31, 0.29, 0.30, 0.32]
    new = [0.10, 0.11, 0.10, 0.09, 0.10]
    s, lo, hi = nb.speedup_ci(base, new, rng=0)
    assert np.isclose(s, 0.30 / 0.10), f"the point estimate is median(base) / median(new) = 3.0; got {s}"
    assert lo <= s <= hi and 2.5 < lo and hi < 3.6, f"a 95% interval around 3 for these tight samples; got [{lo}, {hi}]"
    assert nb.speedup_ci(base, new, rng=0) == (s, lo, hi), "the same rng seed gives the same interval"
    rng = np.random.default_rng(0)
    b = 1.0 + 0.1 * rng.standard_normal(400)
    n = 0.5 + 0.1 * rng.standard_normal(400)
    _, l1, h1 = nb.speedup_ci(b[:5], n[:5], rng=1)
    _, l2, h2 = nb.speedup_ci(b, n, rng=1)
    assert h2 - l2 < h1 - l1, "more timing windows give a narrower interval"
    _, l3, h3 = nb.speedup_ci(b, n, alpha=0.5, rng=1)
    assert h3 - l3 < h2 - l2, "alpha sets the coverage: a 50% interval is narrower than a 95% one"
    base = 0.2 + 0.05 * np.random.default_rng(3).random(30)
    _, lo_c, hi_c = nb.speedup_ci(base, base / 2, rng=4)
    assert hi_c - lo_c > 0.1, (
        f"the two samples are resampled independently (they're timings of different runs, not pairs): even when new = "
        f"base / 2 exactly, the interval has width from both samples' spread; got [{lo_c:.3f}, {hi_c:.3f}]. resampling the "
        "same indices for both makes it collapse to 2.0")
    _, l4, h4 = nb.speedup_ci(b, n, n_boot=50, rng=1)
    assert (l4, h4) != (l2, h2), "n_boot is the number of bootstrap resamples"
