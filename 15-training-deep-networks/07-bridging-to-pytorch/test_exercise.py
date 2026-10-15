"""Checks for exercise.py (bridging to PyTorch).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. `to_torch` and
`gradient_parity` are checked against the reference mlfs.nn / mlfs.train modules (forward outputs,
gradients and buffers to ~1e-12 in float64); the torch modules and helpers against their specs;
`fit` and `evaluate` against hand-computed losses, a counting scheduler, a clipping bound and a small
learning task aggregated over seeds; `ema_update` against the formula. Training runs under
`mlfs.testing.bounded`.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlfs import nn as mnn  # noqa: E402
from mlfs import train as mtrain  # noqa: E402
from mlfs.tensor import Tensor  # noqa: E402
from mlfs.testing import bounded  # noqa: E402


def _mlfs_net(seed=0, norm=None, p=0.0):
    rng = np.random.default_rng(seed)
    layers = [mnn.Linear(5, 7, rng=rng)]
    if norm == "batch":
        layers.append(mtrain.BatchNorm1d(7))
    elif norm == "layer":
        layers.append(mtrain.LayerNorm(7))
    layers.append(mnn.Tanh())
    if p:
        layers.append(mtrain.Dropout(p, rng=rng))
    layers += [mnn.Linear(7, 6, rng=rng), mnn.ReLU(), mnn.Linear(6, 3, rng=rng), mnn.Sigmoid()]
    return mnn.Sequential(*layers)


def _xy(seed=0, n=16, d=5, k=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.integers(0, k, n)


# ---------------------------------------------------------------- stage 1: to_torch


def test_stage1_structure_and_weights(nb):
    net = _mlfs_net()
    t = nb.to_torch(net)
    assert isinstance(t, torch.nn.Sequential) and len(t) == len(net), "a Sequential becomes a torch.nn.Sequential"
    kinds = [type(m).__name__ for m in t]
    assert kinds == ["Linear", "Tanh", "Linear", "ReLU", "Linear", "Sigmoid"], f"layer by layer conversion; got {kinds}"
    assert t[0].weight.dtype == torch.float64, "weights are copied in float64 (call .float() yourself later)"
    assert np.array_equal(t[0].weight.detach().numpy(), net[0].weight.data), (
        "torch.nn.Linear uses the same (out, in) layout as your Linear: copy the weight as is, no transpose")
    assert np.array_equal(t[4].bias.detach().numpy(), net[4].bias.data)
    names = [n for n, _ in t.named_parameters()]
    assert names == [n for n, _ in net.named_parameters()], (
        f"the parameter names match your Sequential's (0.weight, 0.bias, ...); got {names}")
    X, _ = _xy()
    assert np.allclose(t(torch.tensor(X)).detach().numpy(), net(X).data, atol=1e-12), "same outputs, to ~1e-12"


def test_stage1_norms_dropout_and_modes(nb):
    rng = np.random.default_rng(1)
    net = _mlfs_net(norm="batch", p=0.3)
    for _ in range(3):
        net(rng.normal(size=(8, 5)))  # moves the running statistics away from 0 / 1
    net.eval()
    t = nb.to_torch(net)
    bn = t[1]
    assert isinstance(bn, torch.nn.BatchNorm1d) and bn.momentum == 0.1 and bn.eps == 1e-5
    assert np.allclose(bn.running_mean.numpy(), net[1].running_mean) and np.allclose(bn.running_var.numpy(),
                                                                                    net[1].running_var), (
        "BatchNorm1d's running_mean and running_var buffers are copied too")
    assert isinstance(t[3], torch.nn.Dropout) and t[3].p == 0.3
    assert not t.training and not t[1].training, "an mlfs model in eval mode becomes a torch module in eval mode"
    X = rng.normal(size=(4, 5))
    assert np.allclose(t(torch.tensor(X)).detach().numpy(), net(X).data, atol=1e-12), (
        "in eval mode (running statistics, no dropout) the outputs agree")
    ln = nb.to_torch(_mlfs_net(norm="layer"))
    assert isinstance(ln[1], torch.nn.LayerNorm) and ln[1].eps == 1e-5 and ln.training
    odd = mnn.Sequential(mnn.Linear(5, 4, rng=0), mtrain.BatchNorm1d(4, eps=1e-3, momentum=0.3),
                         mtrain.LayerNorm(4, eps=1e-2))
    t_odd = nb.to_torch(odd)
    assert t_odd[1].momentum == 0.3 and t_odd[1].eps == 1e-3 and t_odd[2].eps == 1e-2, (
        "copy each layer's own eps and momentum, not torch's defaults")


def test_stage1_independent_copy_and_errors(nb):
    net = _mlfs_net()
    t = nb.to_torch(net)
    with torch.no_grad():
        t[0].weight.add_(1.0)
    assert not np.allclose(t[0].weight.detach().numpy(), net[0].weight.data), (
        "to_torch copies the weights: changing the torch module mustn't change the mlfs one")

    class Mystery(mnn.Module):
        def forward(self, x):
            return x

    with pytest.raises(TypeError):
        nb.to_torch(mnn.Sequential(Mystery()))


# ---------------------------------------------------------------- stage 2: gradient_parity


def test_stage2_parity_classification_and_regression(nb):
    X, y = _xy(2)
    res = nb.gradient_parity(_mlfs_net(3, norm="layer"), X, y)
    assert set(res) >= {"loss", "output", "grad"}, f"gradient_parity returns a dict with loss, output, grad; got {res}"
    assert all(v < 1e-10 for v in res.values()), (
        f"your engine and torch should agree to ~1e-12 in float64 (integer y: cross-entropy); got {res}")
    yr = np.random.default_rng(4).normal(size=(16, 3))
    res = nb.gradient_parity(_mlfs_net(5), X, yr)
    assert all(v < 1e-10 for v in res.values()), f"float targets use the MSE loss; got {res}"


def test_stage2_batchnorm_in_training_mode(nb):
    X, y = _xy(6)
    res = nb.gradient_parity(_mlfs_net(7, norm="batch"), X, y)
    assert all(v < 1e-10 for v in res.values()), (
        f"in training mode both frameworks normalize with the batch statistics, and your hand-written backward must "
        f"match torch's: got {res}")


def test_stage2_no_side_effects(nb):
    net = _mlfs_net(8, norm="batch")
    before = {k: v.copy() for k, v in net.state_dict().items()}
    rm = net[1].running_mean.copy()
    X, y = _xy(9)
    nb.gradient_parity(net, X, y)
    assert all(np.array_equal(before[k], v) for k, v in net.state_dict().items()), "parameters untouched"
    assert np.array_equal(net[1].running_mean, rm), (
        "the caller's running statistics are untouched: run the mlfs side on a copy (copy.deepcopy)")
    assert all(p.grad is None for p in net.parameters()), "and no gradients are left on the caller's model"


def test_stage2_detects_a_wrong_gradient(nb):
    class Linear(mnn.Linear):  # same name, same forward, but a backward that doubles the gradient
        def forward(self, x):
            out = super().forward(x)
            return Tensor.from_op(out.data, (out,), lambda g: (2.0 * g,))

    rng = np.random.default_rng(12)
    net = mnn.Sequential(Linear(5, 4, rng=rng), mnn.Tanh(), mnn.Linear(4, 3, rng=rng))
    X, y = _xy(13)
    res = nb.gradient_parity(net, X, y)
    assert res["output"] < 1e-12 and res["loss"] < 1e-12, "the forward pass is correct, so outputs and losses agree"
    assert res["grad"] > 1e-3, (
        f"this network's first layer has a wrong backward pass (gradients doubled): gradient_parity must report a "
        f"large 'grad' difference, the max |mlfs grad - torch grad| over all parameters; got {res['grad']:.3g}")


def test_stage2_detects_a_tiny_error_in_the_last_bias(nb):
    delta = np.zeros(3)
    delta[1] = 1e-6

    class Linear(mnn.Linear):  # correct forward; the bias gradient is off by 1e-6 in one entry
        def forward(self, x):
            y = (x if isinstance(x, Tensor) else Tensor(x)) @ self.weight.T
            b = Tensor.from_op(self.bias.data.copy(), (self.bias,), lambda g: (g + delta,))
            return y + b

    rng = np.random.default_rng(14)
    net = mnn.Sequential(mnn.Linear(5, 4, rng=rng), mnn.Tanh(), Linear(4, 3, rng=rng))
    X, y = _xy(15)
    res = nb.gradient_parity(net, X, y)
    assert res["output"] < 1e-12 and res["loss"] < 1e-12
    assert 0.9e-6 < res["grad"] < 1.1e-6, (
        f"one entry of the LAST layer's BIAS gradient is wrong by 1e-6: 'grad' must be the max |mlfs - torch| over "
        f"every entry of every parameter (not the first parameter, not a mean); got {res['grad']:.3g}")


def test_stage2_parity_with_rmsnorm_in_float64(nb):
    rng = np.random.default_rng(16)
    net = mnn.Sequential(mnn.Linear(5, 6, rng=rng), mtrain.RMSNorm(6), mnn.Tanh(), mnn.Linear(6, 3, rng=rng))
    net[1].weight.data[...] = rng.normal(size=6)
    X, y = _xy(17)
    res = nb.gradient_parity(net, X, y)
    assert all(v < 1e-10 for v in res.values()), (
        f"with an RMSNorm inside, float64 parity still holds to ~1e-15 only if your torch RMSNorm computes float64 "
        f"inputs in float64 (upcast only float16/bfloat16); got {res}")


# ---------------------------------------------------------------- stage 3: modules and helpers


def test_stage3_mlp_structure(nb):
    m = nb.MLP(4, [8, 6], 3, activation="gelu", dropout=0.1, norm="batch")
    assert isinstance(m, torch.nn.Module)
    kinds = [type(x).__name__ for x in m.modules() if not isinstance(x, (torch.nn.Sequential, nb.MLP))]
    assert kinds == ["Linear", "BatchNorm1d", "GELU", "Dropout", "Linear", "BatchNorm1d", "GELU", "Dropout", "Linear"], (
        f"per hidden layer: Linear -> norm -> activation -> dropout; then a plain Linear output; got {kinds}")
    assert nb.count_parameters(m) == (4 * 8 + 8) + 16 + (8 * 6 + 6) + 12 + (6 * 3 + 3)
    plain = nb.MLP(4, [8], 2)
    assert [type(x).__name__ for x in plain.modules()][-3:] == ["Linear", "ReLU", "Linear"], "defaults: relu, no norm, no dropout"
    assert plain(torch.zeros(5, 4)).shape == (5, 2)
    for act, cls in [("tanh", torch.nn.Tanh), ("silu", torch.nn.SiLU)]:
        assert any(isinstance(x, cls) for x in nb.MLP(2, [3], 1, activation=act).modules())
    assert any(isinstance(x, torch.nn.LayerNorm) for x in nb.MLP(2, [3], 1, norm="layer").modules())
    with pytest.raises(ValueError):
        nb.MLP(2, [3], 1, activation="swish")
    with pytest.raises(ValueError):
        nb.MLP(2, [3], 1, norm="group")


def test_stage3_rmsnorm(nb):
    r = nb.RMSNorm(4)
    assert [n for n, _ in r.named_parameters()] == ["weight"] and torch.equal(r.weight.data, torch.ones(4))
    assert r.eps == 1e-6
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float64)
    out = r(x)
    want = x / math.sqrt(25 / 4 + 1e-6)
    assert out.dtype == torch.float64 and torch.allclose(out, want, atol=1e-6), (
        "x / sqrt(mean(x^2) + eps) * weight; the result has the input's dtype")
    with torch.no_grad():
        r.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(r(x), want * torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64), atol=1e-6)
    half = torch.full((2, 8), 300.0, dtype=torch.float16)
    out_half = nb.RMSNorm(8)(half)
    assert torch.allclose(out_half.float(), torch.ones(2, 8), atol=1e-2), (
        "in float16, 300^2 = 90,000 overflows to inf (float16 tops out at 65,504), so squaring in the input's dtype "
        f"gives zeros; compute the normalization in float32 (x.float()) and cast back. got {out_half[0, :3].tolist()}")
    mine = mtrain.RMSNorm(4)
    X = np.random.default_rng(0).normal(size=(3, 4))
    assert np.allclose(nb.RMSNorm(4)(torch.tensor(X)).detach().numpy(), mine(X).data, atol=1e-12), (
        "on float64 input your torch RMSNorm agrees with your mlfs.train.RMSNorm to float64 precision")
    tiny = torch.tensor(np.random.default_rng(1).normal(size=(3, 4)) * 1e-4)
    if hasattr(torch.nn, "RMSNorm"):
        ref = torch.nn.RMSNorm(4, eps=1e-6).double()
        assert torch.allclose(nb.RMSNorm(4)(tiny).detach(), ref(tiny).detach(), rtol=1e-9, atol=0), (
            "for inputs of size ~1e-4, mean(x^2) ~ 1e-8 is comparable to eps: eps goes INSIDE the square root, "
            "x / sqrt(mean(x^2) + eps), like torch.nn.RMSNorm")
    f32 = nb.RMSNorm(4)(torch.ones(2, 4))
    assert f32.dtype == torch.float32
    h = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)
    normed = h.float() / torch.sqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    got_h = nb.RMSNorm(4)(h)
    assert torch.equal(got_h.float(), normed.half().float()), (
        "a float16 input is normalized in float32 and cast BACK to float16 before the weight is applied (so the "
        "values are float16-rounded); got values that weren't rounded to float16")


def test_stage3_count_parameters_and_device(nb):
    m = nb.MLP(3, [4], 2)
    total = (3 * 4 + 4) + (4 * 2 + 2)
    m.net[0].weight.requires_grad_(False)
    assert nb.count_parameters(m) == total - 12, "trainable_only=True skips parameters with requires_grad=False"
    assert nb.count_parameters(m, trainable_only=False) == total
    d = nb.get_device()
    assert isinstance(d, torch.device) and d.type in ("cuda", "mps", "cpu")
    if torch.cuda.is_available():
        assert d.type == "cuda", "cuda first when it's available"
    assert nb.get_device("cpu") == torch.device("cpu"), "prefer='cpu' forces the CPU"


# ---------------------------------------------------------------- stage 4: loaders, evaluate, fit


def test_stage4_make_loader(nb):
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10)
    xb, yb = next(iter(nb.make_loader(X, y, batch_size=4, shuffle=False)))
    assert xb.dtype == torch.float32 and yb.dtype == torch.int64, "X -> float32, integer y -> int64 (cross-entropy's dtype)"
    assert torch.equal(yb, torch.arange(4)), "shuffle=False keeps the order"
    batches = [b[1].tolist() for b in nb.make_loader(X, y, batch_size=4, shuffle=False)]
    assert [len(b) for b in batches] == [4, 4, 2], "the last batch is smaller (no drop_last)"
    a = [b[1].tolist() for b in nb.make_loader(X, y, batch_size=10, seed=3)]
    b = [b[1].tolist() for b in nb.make_loader(X, y, batch_size=10, seed=3)]
    c = [b[1].tolist() for b in nb.make_loader(X, y, batch_size=10, seed=4)]
    assert a == b and a != c, "seed fixes the shuffling: same seed, same order; another seed, another order"
    _, yf = next(iter(nb.make_loader(X, y.astype(float), batch_size=3)))
    assert yf.dtype == torch.float32, "float targets (regression) become float32"
    only = next(iter(nb.make_loader(X, batch_size=3)))
    assert len(only) == 1 and only[0].shape == (3, 2), "without y, batches are 1-tuples (x,)"


class _Probe(torch.nn.Module):
    """identity layer that records the grad mode and the training flag it sees."""

    def __init__(self):
        super().__init__()
        self.grad_modes, self.train_flags = [], []

    def forward(self, x):
        self.grad_modes.append(torch.is_grad_enabled())
        self.train_flags.append(self.training)
        return x


def test_stage4_evaluate_runs_without_a_graph(nb):
    probe = _Probe()
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), probe)
    X, y = np.zeros((6, 2)), np.zeros(6, dtype=int)
    nb.evaluate(model, nb.make_loader(X, y, batch_size=3, shuffle=False), torch.nn.functional.cross_entropy)
    assert probe.grad_modes and not any(probe.grad_modes), "evaluate runs the model under torch.no_grad()"
    assert not any(probe.train_flags), "...and in eval mode"


def test_stage4_evaluate(nb):
    torch.manual_seed(0)
    m = nb.MLP(2, [4], 3, dropout=0.5)
    X, y = np.random.default_rng(0).normal(size=(10, 2)), np.random.default_rng(1).integers(0, 3, 10)
    loader = nb.make_loader(X, y, batch_size=4, shuffle=False)
    m.train()
    res = nb.evaluate(m, loader, torch.nn.functional.cross_entropy)
    m.eval()
    with torch.no_grad():
        out = m(torch.tensor(X, dtype=torch.float32))
    want_loss = torch.nn.functional.cross_entropy(out, torch.tensor(y)).item()
    assert math.isclose(res["loss"], want_loss, rel_tol=1e-5), (
        f"'loss' is the mean over all examples in eval mode (dropout off; weight batch means by batch size): "
        f"want {want_loss:.6f}, got {res['loss']:.6f}")
    assert math.isclose(res["accuracy"], (out.argmax(1).numpy() == y).mean()), "'accuracy' for (B, C) outputs"
    m.train()
    nb.evaluate(m, loader)
    assert m.training, "evaluate restores the model's previous mode"
    assert "loss" not in nb.evaluate(m, loader), "no loss_fn, no 'loss'"
    reg = nb.evaluate(nb.MLP(2, [3], 1), nb.make_loader(X, y.astype(float)[:, None], shuffle=False),
                      torch.nn.functional.mse_loss)
    assert "accuracy" not in reg and "loss" in reg, "regression targets (float) get no accuracy"


def _fit(nb, *args, **kwargs):
    return bounded(nb.fit, *args, budget=2_000_000, what="fit", hint="loop over range(epochs) and over the loader once.",
                   **kwargs)


def test_stage4_fit_bookkeeping(nb):
    torch.manual_seed(1)
    X, y = np.random.default_rng(2).normal(size=(30, 2)), np.random.default_rng(3).integers(0, 2, 30)
    m = nb.MLP(2, [8], 2)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    steps = []

    class Counting:
        def step(self):
            steps.append(1)

    h = _fit(nb, m, torch.nn.functional.cross_entropy, opt, nb.make_loader(X, y, batch_size=8, seed=0), epochs=3,
             val_loader=nb.make_loader(X, y, shuffle=False), scheduler=Counting())
    assert set(h) >= {"train_loss", "val_loss", "lr"} and all(len(h[k]) == 3 for k in ("train_loss", "val_loss", "lr")), (
        f"history has train_loss, val_loss and lr, one entry per epoch; got {({k: len(v) for k, v in h.items()})}")
    assert len(steps) == 3 * 4, "scheduler.step() is called once per OPTIMIZER step (4 batches x 3 epochs), not per epoch"
    assert h["lr"] == [0.1] * 3
    stop = _fit(nb, m, torch.nn.functional.cross_entropy, opt, nb.make_loader(X, y, seed=0), epochs=10,
                callback=lambda e, hist: e == 1)
    assert len(stop["train_loss"]) == 2, "callback(epoch, history) returning True stops after that epoch"
    probe = _Probe()
    frozen = torch.nn.Sequential(torch.nn.Linear(2, 2), probe)
    frozen.eval()  # e.g. left in eval mode by an earlier evaluation
    h0 = _fit(nb, frozen, torch.nn.functional.cross_entropy, torch.optim.SGD(frozen.parameters(), lr=0.0),
              nb.make_loader(X, y, batch_size=8, shuffle=False))
    assert probe.train_flags and all(probe.train_flags), "fit puts the model in train mode at the start of every epoch"
    with torch.no_grad():
        full = torch.nn.functional.cross_entropy(frozen(torch.tensor(X, dtype=torch.float32)), torch.tensor(y)).item()
    assert math.isclose(h0["train_loss"][0], full, rel_tol=1e-5), (
        f"with lr 0 the epoch's train_loss is the mean loss over ALL {len(X)} examples ({full:.5f}), weighting each "
        f"batch by its size; got {h0['train_loss'][0]:.5f}")


def test_stage4_fit_clips_and_drives_mlfs_schedulers(nb):
    torch.manual_seed(2)
    m = nb.MLP(1, [], 1)  # a single Linear layer
    X, y = np.array([[100.0]]), np.array([[1e4]])
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    before = torch.cat([p.detach().flatten().clone() for p in m.parameters()])
    _fit(nb, m, torch.nn.functional.mse_loss, opt, nb.make_loader(X, y, shuffle=False), grad_clip=1.0)
    after = torch.cat([p.detach().flatten() for p in m.parameters()])
    assert (after - before).norm().item() <= 0.1 + 1e-6, (
        "with grad_clip=1.0 the gradient's global norm is clipped (torch.nn.utils.clip_grad_norm_) BEFORE "
        "optimizer.step(), so one SGD step moves the parameters by at most lr * 1.0")
    m2 = nb.MLP(1, [2], 1)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.5)
    sched = mtrain.CosineSchedule(opt2, total_steps=4, warmup_steps=1)
    h = _fit(nb, m2, torch.nn.functional.mse_loss, opt2, nb.make_loader(np.ones((4, 1)), np.ones((4, 1)), batch_size=2),
             epochs=2, scheduler=sched)
    assert h["lr"] == pytest.approx([sched.lr_at(2), sched.lr_at(4)]), (
        "your mlfs.train schedulers drive torch optimizers through set_lr; fit records the lr after each epoch")


def test_stage4_fit_learns(nb):
    accs = []
    for seed in range(5):
        torch.manual_seed(100 + seed)
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(200, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)  # XOR of the signs: needs the hidden layer
        m = nb.MLP(2, [32], 2)
        _fit(nb, m, torch.nn.functional.cross_entropy, torch.optim.Adam(m.parameters(), lr=0.02),
             nb.make_loader(X, y, batch_size=32, seed=seed), epochs=40)
        accs.append(nb.evaluate(m, nb.make_loader(X, y, shuffle=False))["accuracy"])
    assert np.median(accs) > 0.9, f"an MLP(2, [32], 2) should fit sign-XOR (median accuracy over 5 seeds > 0.9); got {accs}"


# ---------------------------------------------------------------- stage 5: ema_update


def test_stage5_formula(nb):
    torch.manual_seed(3)
    target, source = nb.MLP(3, [4], 2, norm="batch"), nb.MLP(3, [4], 2, norm="batch")
    source(torch.randn(8, 3))  # moves source's BatchNorm running statistics
    t0 = [p.detach().clone() for p in target.parameters()]
    s0 = [p.detach().clone() for p in source.parameters()]
    nb.ema_update(target, source, decay=0.9)
    for t, a, b in zip(target.parameters(), t0, s0):
        assert torch.allclose(t, 0.9 * a + 0.1 * b), "target <- decay * target + (1 - decay) * source, for every parameter"
        assert t.requires_grad and t.grad_fn is None, "update in place under no_grad: target stays a leaf"
    for tb, sb in zip(target.buffers(), source.buffers()):
        assert torch.equal(tb, sb), "buffers (BatchNorm running stats) are copied from the source, not averaged"
    assert all(torch.equal(p, q) for p, q in zip(source.parameters(), s0)), "the source is untouched"


def test_stage5_edge_decays(nb):
    torch.manual_seed(4)
    a, b = nb.MLP(2, [3], 1), nb.MLP(2, [3], 1)
    a0 = [p.detach().clone() for p in a.parameters()]
    nb.ema_update(a, b, decay=1.0)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), a0)), "decay = 1 leaves the target unchanged"
    nb.ema_update(a, b, decay=0.0)
    assert all(torch.allclose(p, q) for p, q in zip(a.parameters(), b.parameters())), "decay = 0 copies the source"
    c = nb.MLP(2, [3], 1)
    for p in c.parameters():
        p.data.zero_()
    for _ in range(10):
        nb.ema_update(c, b)  # default decay 0.999
    want = 1 - 0.999 ** 10
    p_c, p_b = next(iter(c.parameters())), next(iter(b.parameters()))
    assert torch.allclose(p_c, want * p_b, atol=1e-6), "the default decay is 0.999"
