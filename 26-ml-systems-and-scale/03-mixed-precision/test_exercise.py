"""Checks for 26 · 03 mixed precision.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the format simulator is
checked bit for bit against PyTorch's own float16, bfloat16 and float8_e5m2 casts on values spread over
every regime (subnormal, normal, overflow, exact halfway ties); low-precision SGD against hand
computations of stagnation; the GradScaler against torch.amp.GradScaler's documented behaviour, one
clause at a time (skip on inf/nan without touching parameters or optimizer state, backoff, growth after
exactly growth_interval clean steps, counter reset, no double unscaling); the training step for the
autocast dtype, for gradient underflow with and without a scaler, and for clipping the unscaled grads.
"""
import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

FORMATS = {"float16": (5, 10, torch.float16), "bfloat16": (8, 7, torch.bfloat16), "float8_e5m2": (5, 2, torch.float8_e5m2)}


def _values(rng, fmt):
    e, m, _ = FORMATS[fmt]
    bias = 2 ** (e - 1) - 1
    logs = rng.uniform(-bias - m - 3, bias + 1.5, size=4000)  # from below the smallest subnormal to past the max
    x = np.exp2(logs) * rng.choice([-1, 1], size=logs.size)
    k = np.arange(0, 2 ** m - 1)
    normal_ties = np.concatenate([(1 + (k + 0.5) * 2.0 ** -m) * 2.0 ** e for e in (-3, 0, 2)])  # halfway, normal range
    sub_ties = (np.arange(0, 2 ** min(m, 6)) + 0.5) * 2.0 ** (1 - bias - m)  # halfway between subnormals
    ties = np.concatenate([normal_ties, sub_ties])
    out = np.concatenate([x, ties, -ties, [0.0, -0.0, np.inf, -np.inf]])
    return out.astype(np.float32).astype(np.float64)  # float32-exact values: torch's cast then rounds only once


# ---------------------------------------------------------------- stage 1: float formats


def test_stage1_format_constants(nb):
    f16 = nb.float_format(5, 10)
    want = {"max": 65504.0, "min_normal": 2.0**-14, "min_subnormal": 2.0**-24, "eps": 2.0**-10}
    for k, v in want.items():
        assert math.isclose(f16[k], v), f"float16 {k} should be {v!r}, got {f16[k]!r}"
    bf = nb.float_format(8, 7)
    assert math.isclose(bf["max"], float(torch.finfo(torch.bfloat16).max)), "bfloat16 max ~3.39e38, like float32's range"
    assert math.isclose(bf["eps"], 2.0**-7) and math.isclose(bf["min_normal"], 2.0**-126)
    e5 = nb.float_format(5, 2)
    assert math.isclose(e5["max"], 57344.0) and math.isclose(e5["min_subnormal"], 2.0**-16), "float8 e5m2: max 57344"


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_stage1_matches_torch(nb, fmt):
    e, m, dt = FORMATS[fmt]
    x = _values(np.random.default_rng(0), fmt)
    got = np.asarray(nb.round_to_format(x, e, m), dtype=np.float64)
    want = torch.tensor(x, dtype=torch.float64).to(torch.float32).to(dt).double().numpy()
    bad = ~((got == want) | (np.isnan(got) & np.isnan(want)))
    i = np.flatnonzero(bad)[:1]
    assert not bad.any(), (
        f"{fmt} (e={e}, m={m}): {bad.sum()} of {bad.size} values round differently from torch; e.g. x = {x[i]} -> "
        f"yours {got[i]}, torch {want[i]}. tiny values: below the normal range the spacing stays 2^(1 - bias - m) "
        "(subnormals); halfway ties round to even (np.round does); values that round above the max become inf")
    assert np.signbit(np.asarray(nb.round_to_format(np.array([-0.0, -1e-30]), e, m))).all(), "the sign survives, even for 0"


def test_stage1_shapes(nb):
    x = np.linspace(-3, 3, 12).reshape(3, 4)
    got = np.asarray(nb.round_to_format(x, 8, 7))
    assert got.shape == (3, 4) and got.dtype == np.float64, "same shape as x, float64 values"
    assert np.isnan(nb.round_to_format(np.array([np.nan]), 5, 10)).all(), "nan stays nan"
    assert float(nb.round_to_format(np.array(0.1), 5, 10)) == 0.0999755859375, "the worked example: 0.1 in float16"
    assert float(nb.round_to_format(np.array(0.1), 8, 7)) == 0.10009765625, "the worked example: 0.1 in bfloat16"
    assert float(nb.round_to_format(np.array(1e-8 * 65536), 5, 10)) != 0.0 == float(nb.round_to_format(np.array(1e-8), 5, 10))


# ---------------------------------------------------------------- stage 2: master weights


def test_stage2_stagnation(nb):
    grads = [np.array([1.0, -1.0])] * 100
    got = np.asarray(nb.low_precision_sgd(np.array([1.0, 1.0]), grads, 1e-3, 8, 7))
    assert np.array_equal(got, [1.0, 1.0]), (
        f"in bfloat16 the neighbours of 1.0 are 1 - 2^-8 and 1 + 2^-7: an update of 0.001 is less than half a step, so "
        f"every update rounds back to 1.0 and 100 of them change nothing. expected [1, 1], got {got}")
    got = np.asarray(nb.low_precision_sgd(np.array([1.0, 1.0]), grads, 1e-3, 8, 7, master=True))
    want = nb.round_to_format(np.array([0.9, 1.1]), 8, 7)
    assert np.allclose(got, want), (
        f"with a float64 master copy the 100 updates accumulate to -/+0.1, then round once: {want}; got {got}")


def test_stage2_grads_are_rounded(nb):
    g = np.array([3e-8, 1e-8])  # float16 rounds the first up to 2^-24, the second to 0
    got = np.asarray(nb.low_precision_sgd(np.array([0.0, 0.0]), [g], 1.0, 5, 10, master=True))
    assert np.array_equal(got, [-(2.0**-24), 0.0]), (
        f"gradients arrive in the low-precision format too: round each g before the update. got {got}")
    w0 = np.array([1.0039])  # rounds UP to 1 + 2^-7 in bfloat16
    got0 = np.asarray(nb.low_precision_sgd(w0, [np.array([1.0])], 0.0025, 8, 7))
    want0 = nb.round_to_format(nb.round_to_format(w0, 8, 7) - 0.0025, 8, 7)
    assert np.array_equal(got0, want0), (
        f"without a master copy the weights live in the format from the start: round w0 first. want {want0}, got {got0}")
    many = np.asarray(nb.low_precision_sgd(np.array([0.0]), [np.array([1e-8])] * 100, 1.0, 5, 10, master=True))
    assert np.array_equal(many, [0.0]), (
        "100 gradients of 1e-8 each round to 0 in float16 before they reach the master copy, so nothing accumulates; "
        f"got {many}")
    got = np.asarray(nb.low_precision_sgd(np.array([2.0]), [np.array([1.0])], 0.25, 5, 10))
    assert np.array_equal(got, [1.75]), "a representable update is applied exactly"


# ---------------------------------------------------------------- stage 3: the GradScaler


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(3, 2)


def test_stage3_scale_and_unscale(nb):
    s = nb.GradScaler()
    assert s.get_scale() == 65536.0, "the default initial scale is 2^16"
    loss = torch.tensor(0.5, requires_grad=True)
    out = s.scale(loss)
    assert out.dtype == torch.float32 and math.isclose(out.item(), 32768.0), "scale(loss) = loss * scale"
    half = s.scale(torch.tensor(2.0, dtype=torch.float16))
    assert half.dtype == torch.float32 and math.isfinite(half.item()), (
        "scale() converts the loss to float32 first: 2.0 * 65536 overflows float16 (max 65504)")
    m = _model()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    (m(torch.ones(4, 3)).sum() * 8.0).backward()
    want = [p.grad.clone() / 8.0 for p in m.parameters()]
    s2 = nb.GradScaler(init_scale=8.0)
    s2.unscale_(opt)
    assert all(torch.allclose(p.grad, w) for p, w in zip(m.parameters(), want)), "unscale_ divides every grad by the scale"
    before = [p.detach().clone() for p in m.parameters()]
    assert s2.step(opt) is True, "a finite step returns True"
    assert all(torch.allclose(p, b - 0.1 * w) for p, b, w in zip(m.parameters(), before, want)), (
        "step after an explicit unscale_ must not divide the grads a second time")
    with pytest.raises(RuntimeError):
        s3 = nb.GradScaler()
        s3.unscale_(opt)
        s3.unscale_(opt)


def test_stage3_all_param_groups(nb):
    torch.manual_seed(3)
    a, b = nn.Linear(3, 2), nn.Linear(2, 2)
    opt = torch.optim.SGD([{"params": a.parameters()}, {"params": b.parameters(), "lr": 0.01}], lr=0.1)
    for p in [*a.parameters(), *b.parameters()]:
        p.grad = torch.full_like(p, 8.0)
    s = nb.GradScaler(init_scale=8.0)
    s.unscale_(opt)
    assert all(torch.allclose(p.grad, torch.ones_like(p)) for p in [*a.parameters(), *b.parameters()]), (
        "unscale_ must divide the grads of EVERY param group (the capstone's AdamW has two: with and without decay)")
    s.update()
    b.bias.grad = torch.tensor([float("nan"), 1.0])
    assert s.step(opt) is False, "an inf/nan in the second param group must also skip the step"


def test_stage3_skip_on_overflow(nb):
    for bad in (float("inf"), float("nan")):
        m = _model()
        opt = torch.optim.Adam(m.parameters(), lr=0.1)
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        m.bias.grad[0] = bad
        s = nb.GradScaler(init_scale=4.0)
        before = [p.detach().clone() for p in m.parameters()]
        assert s.step(opt) is False, f"a grad containing {bad} must make step() skip and return False"
        assert all(torch.equal(p, b) for p, b in zip(m.parameters(), before)), "a skipped step changes no parameter"
        assert len(opt.state) == 0, "optimizer.step() must not even be called (Adam's moments would absorb the inf)"
        s.update()
        assert s.get_scale() == 2.0, f"after an overflow the scale backs off: 4 * 0.5 = 2; got {s.get_scale()}"


def test_stage3_growth_and_reset(nb):
    m = _model()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    s = nb.GradScaler(init_scale=16.0, growth_factor=3.0, backoff_factor=0.25, growth_interval=3)

    def step(ok):
        for p in m.parameters():
            p.grad = torch.ones_like(p) * (1.0 if ok else float("inf"))
        r = s.step(opt)
        s.update()
        return r

    for i in range(2):
        assert step(True) and s.get_scale() == 16.0, "no growth before growth_interval clean steps"
    step(True)
    assert s.get_scale() == 48.0, f"after exactly 3 clean steps the scale grows by growth_factor 3: 48; got {s.get_scale()}"
    step(True)
    step(True)
    assert not step(False) and s.get_scale() == 12.0, "an overflow multiplies by backoff_factor 0.25 and resets the count"
    step(True)
    step(True)
    assert s.get_scale() == 12.0, "the clean-step count restarted after the overflow: 2 clean steps aren't enough"
    step(True)
    assert s.get_scale() == 36.0, "3 clean steps after the overflow: grow"
    step(True)
    step(True)
    assert s.get_scale() == 36.0, "the count restarts after every growth: 2 more clean steps aren't enough"
    step(True)
    assert s.get_scale() == 108.0, (
        f"growth repeats: 3 more clean steps after a growth grow the scale again (36 * 3 = 108); got {s.get_scale()}. "
        "reset the clean-step count to 0 whenever the scale grows")
    for p in m.parameters():
        p.grad = None
    assert s.step(opt) is True, "parameters without grads (.grad None) are skipped, not treated as overflow"


def test_stage3_default_growth_interval(nb):
    m = _model()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    s = nb.GradScaler(init_scale=1.0)
    for i in range(2000):
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        s.step(opt)
        s.update()
        if i == 1998:
            assert s.get_scale() == 1.0, "the default growth_interval is 2000"
    assert s.get_scale() == 2.0, "after 2000 clean steps the default growth_factor 2 doubles the scale"


# ---------------------------------------------------------------- stage 4: a mixed-precision step


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, 6, generator=g), torch.randn(32, 4, generator=g)


def test_stage4_autocast_dtype(nb):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
    seen = []
    m[0].register_forward_hook(lambda mod, i, o: seen.append(o.dtype))
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x, y = _data()
    for dt in (torch.float16, torch.bfloat16):
        loss, stepped = nb.amp_step(m, nn.functional.mse_loss, opt, x, y, dtype=dt)
        assert seen[-1] == dt, f"with dtype={dt}, the linear layers must run in {dt} under torch.autocast; saw {seen[-1]}"
        assert isinstance(loss, float) and stepped is True
    assert all(p.dtype == torch.float32 for p in m.parameters()), "the parameters themselves stay float32 (master copy)"
    nb.amp_step(m, nn.functional.mse_loss, opt, x, y)
    assert seen[-1] == torch.float16, "the default dtype is float16"


def test_stage4_underflow_and_scaling(nb):
    x, y = _data(1)
    tiny = lambda out, t: nn.functional.mse_loss(out, t) * 1e-7  # gradients far below float16's smallest subnormal

    def run(dtype, scaler):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 4))
        opt = torch.optim.SGD(m.parameters(), lr=1e4)
        w0 = m[0].weight.detach().clone()
        nb.amp_step(m, tiny, opt, x, y, dtype=dtype, scaler=scaler)
        return (m[0].weight.detach() - w0).abs().max().item()

    assert run(torch.float16, None) == 0.0, (
        "without loss scaling, the float16 activation gradients underflow to zero and the first layer doesn't move "
        "(if it moved, the backward pass isn't running under float16 autocast)")
    assert run(torch.float16, nb.GradScaler()) > 0, (
        "with a GradScaler the scaled gradients survive float16 and the first layer is updated")
    assert run(torch.bfloat16, None) > 0, "bfloat16 has float32's range: no underflow even without a scaler"


def test_stage4_overflow_skips_and_backs_off(nb):
    torch.manual_seed(0)
    m = nn.Linear(6, 4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    w0 = m.weight.detach().clone()
    x, y = _data(2)
    s = nb.GradScaler(init_scale=2.0**40)
    loss, stepped = nb.amp_step(m, nn.functional.mse_loss, opt, x, y, scaler=s)
    assert stepped is False and torch.equal(m.weight, w0), "a scale of 2^40 overflows the float16 grads: skip the step"
    assert s.get_scale() == 2.0**39, "amp_step calls scaler.update(), which backs off after the overflow"
    with torch.autocast("cpu", dtype=torch.float16):
        want = torch.nn.functional.mse_loss(m(x), y).item()
    assert math.isfinite(loss) and math.isclose(loss, want, rel_tol=1e-2), (
        f"the returned loss is the unscaled loss {want:.4f}; got {loss}")


def test_stage4_fresh_grads_and_clipping(nb):
    torch.manual_seed(0)
    m = nn.Linear(6, 4)
    ref = copy.deepcopy(m)
    x, y = _data(3)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    for _ in range(2):
        nb.amp_step(m, nn.functional.mse_loss, opt, x, y, dtype=torch.bfloat16, scaler=nb.GradScaler(init_scale=1024.0))
    ropt = torch.optim.SGD(ref.parameters(), lr=1.0)
    for _ in range(2):
        ropt.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            l = nn.functional.mse_loss(ref(x), y)
        l.backward()
        ropt.step()
    assert torch.allclose(m.weight, ref.weight, atol=2e-2), (
        "two steps must equal two plain steps on fresh gradients: zero the grads first, and unscale before stepping")
    torch.manual_seed(0)
    m, ref = nn.Linear(6, 4), None
    ref = copy.deepcopy(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    ropt = torch.optim.SGD(ref.parameters(), lr=0.5)
    for _ in range(2):
        nb.amp_step(m, nn.functional.mse_loss, opt, x, y, dtype=torch.bfloat16)
        ropt.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            l = nn.functional.mse_loss(ref(x), y)
        l.backward()
        ropt.step()
    assert torch.allclose(m.weight, ref.weight, atol=1e-2), (
        "without a scaler too, every step must start from zeroed gradients (optimizer.zero_grad())")
    torch.manual_seed(0)
    m = nn.Linear(6, 4)
    p0 = [p.detach().clone() for p in m.parameters()]
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    nb.amp_step(m, lambda o, t: nn.functional.mse_loss(o, t) * 100, opt, x, y, scaler=nb.GradScaler(init_scale=256.0),
                grad_clip=0.5)
    norm = math.sqrt(sum((p.detach() - q).pow(2).sum().item() for p, q in zip(m.parameters(), p0)))
    assert norm <= 0.5 + 1e-3, (
        f"with grad_clip=0.5 and lr=1 the update's norm is at most 0.5; got {norm:.3f}. clip the *unscaled* gradients "
        "(scaler.unscale_ first): clipping scaled ones clips at 0.5 / 65536")
    assert norm > 0.4, f"the clipped update should have norm ~0.5 (its gradient was large); got {norm:.3f}"
    torch.manual_seed(0)
    m = nn.Linear(6, 4)
    p0 = [p.detach().clone() for p in m.parameters()]
    nb.amp_step(m, lambda o, t: nn.functional.mse_loss(o, t) * 100, torch.optim.SGD(m.parameters(), lr=1.0), x, y,
                dtype=torch.bfloat16, grad_clip=0.5)
    norm = math.sqrt(sum((p.detach() - q).pow(2).sum().item() for p, q in zip(m.parameters(), p0)))
    assert 0.4 < norm <= 0.5 + 1e-3, f"without a scaler, grad_clip still clips the update to norm 0.5; got {norm:.3f}"
