"""Checks for exercise.py (MAML on sine regression).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the functional MLP
is checked against `torch.nn` with copied weights; the task sampler against its formula and with
Kolmogorov-Smirnov tests of every uniform draw; the inner loop against a per-task loop written
here (one `torch.autograd.grad` call per task); the meta-gradient against autograd through that
unrolled loop, against central finite differences in float64, and against the closed form for a
linear model; the outer loop against a hand-rolled SGD step and against `torch.optim.Adam`.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy import stats  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

# the first optimizer step in a process runs ~3M lines of torch's lazy imports; do it here, untraced, so the
# budgets below only count your training loop
_w = torch.zeros(1, requires_grad=True)
_w.grad = torch.zeros(1)
torch.optim.Adam([_w]).step()
torch.optim.SGD([_w], lr=0.1).step()


def _train(nb, *args, budget=200_000, **kwargs):
    """train_maml, stopped with a teaching message if its loop doesn't end (about 1,000 lines per iteration)."""
    return bounded(nb.train_maml, *args, budget=budget, what="train_maml",
                   hint="the loop must run exactly n_iters iterations, appending one loss per iteration.", **kwargs)



# ---------------------------------------------------------------- reference pieces (written here)


def _ref_forward(params, x):
    h = x
    n = len(params) // 2
    for i in range(n):
        h = h @ params[2 * i] + params[2 * i + 1].unsqueeze(-2)
        if i < n - 1:
            h = torch.relu(h)
    return h


def _tanh_forward(params, x):
    """a smooth two-layer net (finite differences need no kinks)."""
    return torch.tanh(x @ params[0] + params[1].unsqueeze(-2)) @ params[2] + params[3].unsqueeze(-2)


def _mse(pred, y):
    return ((pred - y) ** 2).flatten(1).mean(1)


def _l1(pred, y):
    return (pred - y).abs().flatten(1).mean(1)


def _ref_adapt_one(params, x, y, forward, loss_fn, lr, steps, first_order):
    """task by task: x (n, d) -> adapted list of tensors with the task's own shape."""
    fast = list(params)
    for _ in range(steps):
        loss = loss_fn(forward(fast, x[None]), y[None]).sum()
        grads = torch.autograd.grad(loss, fast, create_graph=not first_order)
        fast = [w - lr * g for w, g in zip(fast, grads)]
    return fast


def _ref_maml_loss(params, batch, forward=_ref_forward, loss_fn=_mse, lr=0.01, steps=1, first_order=False):
    losses = []
    for t in range(batch["xs"].shape[0]):
        fast = _ref_adapt_one(params, batch["xs"][t], batch["ys"][t], forward, loss_fn, lr, steps, first_order)
        losses.append(loss_fn(forward(fast, batch["xq"][t][None]), batch["yq"][t][None]).sum())
    return torch.stack(losses).mean()


def _params(sizes, seed, dtype=torch.float32, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        out.append((scale * torch.randn(a, b, generator=g) / math.sqrt(a)).to(dtype).requires_grad_())
        out.append((0.3 * scale * torch.randn(b, generator=g)).to(dtype).requires_grad_())
    return out


def _batch(T=3, ns=5, nq=4, d=1, k=1, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return {"xs": torch.randn(T, ns, d, generator=g).to(dtype), "ys": torch.randn(T, ns, k, generator=g).to(dtype),
            "xq": torch.randn(T, nq, d, generator=g).to(dtype), "yq": torch.randn(T, nq, k, generator=g).to(dtype)}


def _grads(loss, params):
    return [g.detach() for g in torch.autograd.grad(loss, params)]


def _assert_lists_close(got, want, what, tol=1e-5, hint=""):
    assert isinstance(got, (list, tuple)), f"{what}: expected a list of tensors, got {type(got).__name__}"
    assert len(got) == len(want), f"{what}: expected {len(want)} tensors, got {len(got)}"
    for i, (g, w) in enumerate(zip(got, want)):
        assert tuple(g.shape) == tuple(w.shape), f"{what}: tensor {i} should have shape {tuple(w.shape)}, got {tuple(g.shape)}"
        err = float((g.detach() - w.detach()).abs().max())
        scale = float(w.detach().abs().max())
        assert err <= tol * max(1.0, scale), (
            f"{what}: tensor {i} is off by {err:.3g} (largest true entry {scale:.3g}). {hint}")


# ---------------------------------------------------------------- stage 1: functional MLP and sine tasks


def test_stage1_init_mlp_shapes_and_distribution(nb):
    g = torch.Generator().manual_seed(0)
    ps = nb.init_mlp([300, 200, 300], generator=g)
    assert isinstance(ps, list) and len(ps) == 4, "init_mlp([300, 200, 300]) should return [W1, b1, W2, b2]"
    shapes = [tuple(p.shape) for p in ps]
    assert shapes == [(300, 200), (200,), (200, 300), (300,)], (
        f"shapes should be [(300, 200), (200,), (200, 300), (300,)] (W is (fan_in, fan_out), so x @ W works), "
        f"got {shapes}")
    for p in ps:
        assert p.dtype == torch.float32, f"parameters should be float32, got {p.dtype}"
        assert p.is_leaf and p.requires_grad, "parameters must be leaf tensors with requires_grad=True"
    for i, (p, fan_in) in enumerate(zip(ps, [300, 300, 200, 200])):
        bound = 1 / math.sqrt(fan_in)
        v = p.detach().flatten().numpy()
        assert float(abs(v).max()) <= bound + 1e-7, (
            f"entries must lie in [-1/sqrt(fan_in), 1/sqrt(fan_in)] = ±{bound:.4f} (fan_in = {fan_in}, the "
            f"layer's input size, for weights and biases alike); found {float(abs(v).max()):.4f}")
        name = f"{'W' if i % 2 == 0 else 'b'}{i // 2 + 1}"
        sd = bound / math.sqrt(3)  # the sd of U(-bound, bound)
        assert abs(v.std() - sd) < 0.15 * sd, (
            f"{name} should be U(±1/sqrt(fan_in)) like nn.Linear (sd {sd:.4f}), got sd {v.std():.4f}: every layer's "
            "weights AND biases are random, the last layer's too")
        p_ks = stats.kstest(v, "uniform", args=(-bound, 2 * bound)).pvalue
        assert p_ks > 1e-4, f"{name} entries should be uniform on ±1/sqrt({fan_in}); KS test p = {p_ks:.2g}"


def test_stage1_init_mlp_uses_generator(nb):
    a = nb.init_mlp([3, 5, 2], generator=torch.Generator().manual_seed(7))
    torch.manual_seed(123)
    b = nb.init_mlp([3, 5, 2], generator=torch.Generator().manual_seed(7))
    c = nb.init_mlp([3, 5, 2], generator=torch.Generator().manual_seed(8))
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "the same generator seed must give the same parameters"
    assert not torch.equal(a[0], c[0]), "different generator seeds must give different parameters"


def test_stage1_mlp_forward_matches_torch_nn(nb):
    ps = _params([2, 6, 5, 3], seed=1)
    with torch.no_grad():
        ps[5][0] -= 5.0  # a very negative output bias: a ReLU after the last layer would clip it to 0
    net = torch.nn.Sequential(torch.nn.Linear(2, 6), torch.nn.ReLU(), torch.nn.Linear(6, 5), torch.nn.ReLU(),
                              torch.nn.Linear(5, 3))
    with torch.no_grad():
        for layer, (W, b) in zip([net[0], net[2], net[4]], zip(ps[0::2], ps[1::2])):
            layer.weight.copy_(W.T)
            layer.bias.copy_(b)
    x = torch.randn(7, 2, generator=torch.Generator().manual_seed(2))
    got = nb.mlp_forward(ps, x)
    want = net(x)
    assert tuple(got.shape) == (7, 3), f"(n, d_in) -> (n, d_out): expected (7, 3), got {tuple(got.shape)}"
    err = float((got - want).detach().abs().max())
    assert err < 1e-5, (f"mlp_forward differs from Linear-ReLU-Linear-ReLU-Linear by {err:.3g}. "
                        "ReLU after every layer except the last; x @ W + b with W (d_in, d_out)")
    assert (got < 0).any(), "the output layer has no activation, so outputs can be negative"
    single = _params([4, 1], seed=3)
    x4 = torch.randn(5, 4)
    assert torch.allclose(nb.mlp_forward(single, x4), x4 @ single[0] + single[1], atol=1e-6), (
        "a single-layer MLP is the linear model x @ W + b")


def test_stage1_mlp_forward_per_task_weights(nb):
    ps = _params([1, 4, 2], seed=4)
    T = 3
    g = torch.Generator().manual_seed(5)
    batched = [p.detach() + 0.5 * torch.randn(T, *p.shape, generator=g) for p in ps]
    x = torch.randn(T, 6, 1, generator=g)
    got = nb.mlp_forward(batched, x)
    assert tuple(got.shape) == (T, 6, 2), f"per-task weights (T, d_in, d_out): expected output (3, 6, 2), got {tuple(got.shape)}"
    for t in range(T):
        want = _ref_forward([p[t] for p in batched], x[t])
        assert torch.allclose(got[t], want, atol=1e-5), (
            f"task {t}: with per-task weights, output t must use weights [p[t] for p in params]. "
            "biases are (T, d_out): broadcast them over the n points (b.unsqueeze(-2))")
    shared = nb.mlp_forward(ps, x)
    assert tuple(shared.shape) == (T, 6, 2), "shared weights must also work on a (T, n, d_in) batch"
    assert torch.allclose(shared[1], _ref_forward(ps, x[1]), atol=1e-5), "shared weights on a batch: wrong values"


def test_stage1_sine_tasks_formula_and_shapes(nb):
    b = nb.sample_sine_tasks(6, 10, 7, generator=torch.Generator().manual_seed(0))
    want_shapes = {"xs": (6, 10, 1), "ys": (6, 10, 1), "xq": (6, 7, 1), "yq": (6, 7, 1), "amp": (6,), "phase": (6,)}
    for k, s in want_shapes.items():
        assert k in b, f"the batch dict needs key {k!r}"
        assert tuple(b[k].shape) == s, f"{k} should have shape {s}, got {tuple(b[k].shape)}"
        assert b[k].dtype == torch.float32, f"{k} should be float32"
    for xk, yk in (("xs", "ys"), ("xq", "yq")):
        want = b["amp"][:, None, None] * torch.sin(b[xk] - b["phase"][:, None, None])
        err = float((b[yk] - want).abs().max())
        assert err < 1e-5, (f"{yk} should be amp * sin({xk} - phase) (Finn et al.'s convention: minus the phase); "
                            f"off by {err:.3g}")
    assert not torch.equal(b["xs"][0], b["xs"][1]), "each task draws its own inputs"
    assert len(set(b["amp"].tolist())) == 6, "each task draws its own amplitude"


def test_stage1_sine_tasks_distributions(nb):
    b = nb.sample_sine_tasks(4000, 3, 2, generator=torch.Generator().manual_seed(1))
    checks = [("amp", b["amp"], 0.1, 5.0), ("phase", b["phase"], 0.0, math.pi),
              ("xs", b["xs"], -5.0, 5.0), ("xq", b["xq"], -5.0, 5.0)]
    for name, v, lo, hi in checks:
        v = v.flatten().numpy()
        assert v.min() >= lo and v.max() <= hi, f"{name} must lie in [{lo}, {hi:.4g}], got [{v.min():.4g}, {v.max():.4g}]"
        p = stats.kstest(v, "uniform", args=(lo, hi - lo)).pvalue
        assert p > 1e-4, f"{name} should be uniform on [{lo}, {hi:.4g}]; KS test p = {p:.2g}"
    r = np.corrcoef(b["xs"][:, 0, 0].numpy(), b["xq"][:, 0, 0].numpy())[0, 1]
    assert abs(r) < 0.1, f"support and query inputs must be drawn independently; correlation {r:.2f}"
    r2 = np.corrcoef(b["amp"].numpy(), b["phase"].numpy())[0, 1]
    assert abs(r2) < 0.1, f"amplitude and phase must be independent; correlation {r2:.2f}"


def test_stage1_sine_tasks_ranges_and_generator(nb):
    b = nb.sample_sine_tasks(500, 4, 4, amp_range=(2.0, 3.0), phase_range=(1.0, 1.5), x_range=(0.0, 1.0),
                             generator=torch.Generator().manual_seed(2))
    for name, key, lo, hi in [("amp_range", "amp", 2.0, 3.0), ("phase_range", "phase", 1.0, 1.5),
                              ("x_range", "xs", 0.0, 1.0), ("x_range", "xq", 0.0, 1.0)]:
        v = b[key]
        assert float(v.min()) >= lo and float(v.max()) <= hi, f"{key} ignores {name}=({lo}, {hi})"
        assert float(v.max()) - float(v.min()) > 0.8 * (hi - lo), f"{key} should fill {name}=({lo}, {hi})"
    a1 = nb.sample_sine_tasks(3, 2, 2, generator=torch.Generator().manual_seed(9))
    torch.manual_seed(77)
    a2 = nb.sample_sine_tasks(3, 2, 2, generator=torch.Generator().manual_seed(9))
    assert all(torch.equal(a1[k], a2[k]) for k in a1), "the same generator seed must give the same tasks"


# ---------------------------------------------------------------- stage 2: the inner loop


def test_stage2_task_mse(nb):
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [1.0, 1.0]]])
    y = torch.tensor([[[1.0, 0.0], [3.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
    got = nb.task_mse(pred, y)
    want = torch.tensor([(0 + 4 + 0 + 4) / 4, (1 + 1 + 0 + 0) / 4])
    assert tuple(got.shape) == (2,), f"task_mse returns one loss per task, shape (T,); got {tuple(got.shape)}"
    assert torch.allclose(got, want), f"per-task mean over points and outputs: expected {want.tolist()}, got {got.tolist()}"


def test_stage2_worked_example(nb):
    # the lesson's worked example: y = w x + b from (w, b) = (1, 0), support (1, 3), (2, 5), lr 0.1
    params = [torch.tensor([[1.0]], requires_grad=True), torch.tensor([0.0], requires_grad=True)]
    xs = torch.tensor([[[1.0], [2.0]]])
    ys = torch.tensor([[[3.0], [5.0]]])
    fast = nb.adapt(params, xs, ys, inner_lr=0.1, n_steps=1)
    w, b = float(fast[0].detach().reshape(-1)[0]), float(fast[1].detach().reshape(-1)[0])
    assert abs(w - 1.8) < 1e-5 and abs(b - 0.5) < 1e-5, (
        f"one step from (w, b) = (1, 0) with lr 0.1 should give (1.8, 0.5), got ({w:.4f}, {b:.4f}). the loss is the "
        "MEAN squared error over the support points: dL/dw = -8, dL/db = -5")


def test_stage2_adapt_matches_per_task_loop(nb):
    params = _params([1, 8, 8, 1], seed=10)
    batch = _batch(T=4, ns=6, seed=11)
    fast = nb.adapt(params, batch["xs"], batch["ys"], inner_lr=0.05, n_steps=3)
    assert isinstance(fast, (list, tuple)) and len(fast) == len(params), "adapt returns a list like params"
    for p, f in zip(params, fast):
        assert tuple(f.shape) == (4, *p.shape), f"adapted params carry a task dimension: expected {(4, *p.shape)}, got {tuple(f.shape)}"
    for t in range(4):
        want = _ref_adapt_one(params, batch["xs"][t], batch["ys"][t], _ref_forward, _mse, 0.05, 3, False)
        _assert_lists_close([f[t] for f in fast], want, f"task {t} after 3 steps at lr 0.05", tol=1e-4,
                            hint="each task takes its OWN gradient steps (not the batch-average gradient), "
                                 "theta <- theta - inner_lr * grad of that task's support loss")


def test_stage2_defaults_zero_steps_and_inputs_untouched(nb):
    params = _params([1, 5, 1], seed=12)
    before = [p.detach().clone() for p in params]
    batch = _batch(T=2, seed=13)
    fast = nb.adapt(params, batch["xs"], batch["ys"])
    for t in range(2):
        want = _ref_adapt_one(params, batch["xs"][t], batch["ys"][t], _ref_forward, _mse, 0.01, 1, False)
        _assert_lists_close([f[t] for f in fast], want, "adapt with defaults (inner_lr=0.01, n_steps=1)", tol=1e-5)
    zero = nb.adapt(params, batch["xs"], batch["ys"], n_steps=0)
    for p, f in zip(params, zero):
        assert tuple(f.shape) == (2, *p.shape), "n_steps=0 still returns per-task copies (T, *p.shape)"
        assert torch.allclose(f, p.expand(2, *p.shape)), "n_steps=0 returns the shared params, unchanged"
    for p, b in zip(params, before):
        assert torch.equal(p.detach(), b), "adapt must not modify the shared params in place"
    frozen = [p.detach() for p in params]
    fast2 = nb.adapt(frozen, batch["xs"], batch["ys"], inner_lr=0.1, n_steps=2)
    want = _ref_adapt_one(params, batch["xs"][0], batch["ys"][0], _ref_forward, _mse, 0.1, 2, False)
    _assert_lists_close([f[0] for f in fast2], want, "adapt on params with requires_grad=False", tol=1e-5,
                        hint="at evaluation the weights may be detached: the inner loop still needs gradients")


def test_stage2_forward_and_loss_fn_are_used(nb):
    params = _params([1, 4, 1], seed=14)
    batch = _batch(T=3, seed=15)
    fast = nb.adapt(params, batch["xs"], batch["ys"], forward=_tanh_forward, loss_fn=_l1, inner_lr=0.2, n_steps=2)
    for t in range(3):
        want = _ref_adapt_one(params, batch["xs"][t], batch["ys"][t], _tanh_forward, _l1, 0.2, 2, False)
        _assert_lists_close([f[t] for f in fast], want, f"task {t} with a custom forward (tanh) and loss (L1)",
                            tol=1e-5, hint="use the forward and loss_fn you were given, not mlp_forward/task_mse")


def test_stage2_graph_kept_or_cut(nb):
    params = _params([1, 4, 1], seed=16, dtype=torch.float64)
    batch = _batch(T=3, seed=17, dtype=torch.float64)
    for first_order in (False, True):
        fast = nb.adapt(params, batch["xs"], batch["ys"], forward=_tanh_forward, inner_lr=0.3, n_steps=2,
                        first_order=first_order)
        got = _grads(_mse(_tanh_forward(fast, batch["xq"]), batch["yq"]).sum(), params)
        want = _grads(3 * _ref_maml_loss(params, batch, _tanh_forward, _mse, 0.3, 2, first_order), params)
        what = "first-order (inner gradients detached)" if first_order else "second-order (graph through the inner steps)"
        hint = ("first_order=True: compute inner gradients WITHOUT create_graph, so theta' = theta - lr * (a constant)"
                if first_order else
                "first_order=False: torch.autograd.grad(..., create_graph=True) at EVERY inner step")
        _assert_lists_close(got, want, f"d(query loss of adapted params)/d(params), {what}", tol=1e-7, hint=hint)


# ---------------------------------------------------------------- stage 3: the meta-objective and its gradient


def test_stage3_maml_loss_value(nb):
    params = _params([1, 6, 1], seed=20)
    batch = _batch(T=5, ns=4, nq=6, seed=21)
    got = nb.maml_loss(params, batch, inner_lr=0.05, n_steps=2)
    want = _ref_maml_loss(params, batch, lr=0.05, steps=2)
    assert got.dim() == 0, "maml_loss returns a scalar tensor"
    assert abs(got.item() - want.item()) < 1e-5, (
        f"maml_loss should be {want.item():.6f} (MEAN over tasks of the QUERY loss after adapting on the SUPPORT "
        f"set), got {got.item():.6f}")
    zero = nb.maml_loss(params, batch, n_steps=0)
    plain = _mse(_ref_forward(params, batch["xq"]), batch["yq"]).mean()
    assert abs(zero.item() - plain.item()) < 1e-6, "with n_steps=0 there is no adaptation: the plain query loss"
    defaults = nb.maml_loss(params, batch)
    assert abs(defaults.item() - _ref_maml_loss(params, batch).item()) < 1e-6, "defaults: inner_lr=0.01, n_steps=1"


def test_stage3_second_order_gradient_vs_autograd_unrolled(nb):
    params = _params([1, 10, 10, 1], seed=22)
    batch = _batch(T=4, ns=5, nq=5, seed=23)
    got = _grads(nb.maml_loss(params, batch, inner_lr=0.1, n_steps=3), params)
    want = _grads(_ref_maml_loss(params, batch, lr=0.1, steps=3), params)
    _assert_lists_close(got, want, "MAML meta-gradient (3 inner steps)", tol=1e-4,
                        hint="the graph must run through every inner step: create_graph=True")
    fo = _grads(_ref_maml_loss(params, batch, lr=0.1, steps=3, first_order=True), params)
    gap = max(float((a - b).abs().max()) for a, b in zip(want, fo))
    assert gap > 1e-3, "(test sanity) second- and first-order gradients should differ here"


def test_stage3_gradient_vs_finite_differences(nb):
    params = _params([1, 3, 1], seed=24, dtype=torch.float64)
    batch = _batch(T=2, ns=4, nq=3, seed=25, dtype=torch.float64)
    kw = dict(forward=_tanh_forward, inner_lr=0.4, n_steps=2)
    got = _grads(nb.maml_loss(params, batch, **kw), params)
    eps = 1e-6
    for i, p in enumerate(params):
        flat = p.detach().reshape(-1)
        for j in range(flat.numel()):
            plus = [q.detach().clone().requires_grad_() for q in params]
            minus = [q.detach().clone().requires_grad_() for q in params]
            with torch.no_grad():
                plus[i].view(-1)[j] += eps
                minus[i].view(-1)[j] -= eps
            fd = (_ref_maml_loss(plus, batch, _tanh_forward, _mse, 0.4, 2).item()
                  - _ref_maml_loss(minus, batch, _tanh_forward, _mse, 0.4, 2).item()) / (2 * eps)
            g = float(got[i].reshape(-1)[j])
            assert abs(g - fd) < 1e-6 * max(1.0, abs(fd)), (
                f"d maml_loss / d params[{i}][{j}]: autograd gives {g:.8f}, central finite differences "
                f"{fd:.8f}. the meta-gradient must include the second-order term (and use the forward you pass)")


def test_stage3_first_order_gradient(nb):
    params = _params([1, 10, 1], seed=26)
    batch = _batch(T=3, seed=27)
    got = _grads(nb.maml_loss(params, batch, inner_lr=0.2, n_steps=2, first_order=True), params)
    want = _grads(_ref_maml_loss(params, batch, lr=0.2, steps=2, first_order=True), params)
    _assert_lists_close(got, want, "first-order MAML gradient", tol=1e-5,
                        hint="FOMAML = the query-loss gradient at the adapted params, summed back onto the shared "
                             "params (the inner gradients are treated as constants)")
    custom = _grads(nb.maml_loss(params, batch, loss_fn=_l1, inner_lr=0.2, n_steps=1), params)
    want_c = _grads(_ref_maml_loss(params, batch, loss_fn=_l1, lr=0.2, steps=1), params)
    _assert_lists_close(custom, want_c, "maml_loss with loss_fn=L1", tol=1e-5,
                        hint="loss_fn is used for both the inner (support) and outer (query) losses")


def _np_meta_grad(theta, Xs, ys, Xq, yq, a, k, first_order):
    H = 2 * Xs.T @ Xs / len(ys)
    th = theta.copy()
    for _ in range(k):
        th = th - a * 2 * Xs.T @ (Xs @ th - ys) / len(ys)
    g = 2 * Xq.T @ (Xq @ th - yq) / len(yq)
    return g if first_order else np.linalg.matrix_power(np.eye(len(theta)) - a * H, k) @ g


def test_stage3_linear_meta_gradient_worked_example(nb):
    Xs = np.array([[1.0, 1.0], [2.0, 1.0]])
    ys = np.array([3.0, 5.0])
    Xq = np.array([[3.0, 1.0]])
    yq = np.array([7.0])
    got = np.asarray(nb.linear_meta_gradient(np.array([1.0, 0.0]), Xs, ys, Xq, yq, inner_lr=0.1))
    assert got.shape == (2,), f"one gradient entry per parameter: shape (2,), got {got.shape}"
    assert np.allclose(got, [-2.64, 0.22], atol=1e-9), (
        f"the lesson's worked example gives the MAML meta-gradient (-2.64, 0.22), got {np.round(got, 4)}. "
        "(I - a H_s) with H_s = 2/n Xs^T Xs multiplies the query gradient at the adapted theta")
    fo = np.asarray(nb.linear_meta_gradient(np.array([1.0, 0.0]), Xs, ys, Xq, yq, inner_lr=0.1, first_order=True))
    assert np.allclose(fo, [-6.6, -2.2], atol=1e-9), f"first-order: the query gradient (-6.6, -2.2), got {np.round(fo, 4)}"


def test_stage3_linear_meta_gradient_vs_autograd(nb):
    rng = np.random.default_rng(28)
    d = 4
    theta = rng.normal(size=d)
    Xs, ys = rng.normal(size=(6, d)), rng.normal(size=6)
    Xq, yq = rng.normal(size=(5, d)), rng.normal(size=5)
    for k, fo in [(1, False), (3, False), (3, True), (0, False)]:
        got = np.asarray(nb.linear_meta_gradient(theta, Xs, ys, Xq, yq, inner_lr=0.15, n_steps=k, first_order=fo))
        # autograd oracle: the learner-independent unrolled loop, in float64
        th = torch.tensor(theta, requires_grad=True)
        p = th
        Xs_t, ys_t, Xq_t, yq_t = (torch.tensor(a) for a in (Xs, ys, Xq, yq))
        for _ in range(k):
            g = torch.autograd.grad(((Xs_t @ p - ys_t) ** 2).mean(), p, create_graph=not fo)[0]
            p = p - 0.15 * g
        want = torch.autograd.grad(((Xq_t @ p - yq_t) ** 2).mean(), th)[0].numpy()
        assert np.allclose(want, _np_meta_grad(theta, Xs, ys, Xq, yq, 0.15, k, fo), atol=1e-10)  # test sanity
        assert np.allclose(got, want, atol=1e-8), (
            f"n_steps={k}, first_order={fo}: expected {np.round(want, 5)}, got {np.round(got, 5)}. after k steps the "
            "Jacobian d theta_k / d theta is (I - a H_s)^k")


def test_stage3_maml_loss_linear_model_matches_closed_form(nb):
    rng = np.random.default_rng(29)
    d, T = 3, 4
    theta = rng.normal(size=d)
    batch = {k: torch.tensor(rng.normal(size=s)) for k, s in
             [("xs", (T, 5, d)), ("ys", (T, 5, 1)), ("xq", (T, 4, d)), ("yq", (T, 4, 1))]}
    p = [torch.tensor(theta.reshape(d, 1), requires_grad=True)]

    def linear(params, x):
        return x @ params[0]

    got = _grads(nb.maml_loss(p, batch, forward=linear, inner_lr=0.2, n_steps=2), p)[0].numpy().ravel()
    want = np.mean([_np_meta_grad(theta, batch["xs"][t].numpy(), batch["ys"][t, :, 0].numpy(), batch["xq"][t].numpy(),
                                  batch["yq"][t, :, 0].numpy(), 0.2, 2, False) for t in range(T)], axis=0)
    assert np.allclose(got, want, atol=1e-8), (
        f"for a linear model, the gradient of maml_loss must equal the mean over tasks of (I - a H_s)^k grad L_q; "
        f"expected {np.round(want, 5)}, got {np.round(got, 5)}")


# ---------------------------------------------------------------- stage 4: the outer loop


def _fixed_sampler(batches, log=None):
    state = {"i": 0}

    def sample(generator):
        if log is not None:
            log.append(generator)
        b = batches[state["i"] % len(batches)]
        state["i"] += 1
        return b
    return sample


def test_stage4_sgd_steps_are_exact(nb):
    params = _params([1, 6, 1], seed=30)
    batches = [_batch(T=3, seed=31), _batch(T=3, seed=32)]
    for first_order in (False, True):
        new, hist = _train(nb, params, _fixed_sampler(batches), n_iters=2, meta_lr=0.1, inner_lr=0.1, n_steps=2,
                           first_order=first_order, optimizer="sgd")
        theta = [p.detach().clone().requires_grad_() for p in params]
        refs = []
        for b in batches:
            loss = _ref_maml_loss(theta, b, lr=0.1, steps=2, first_order=first_order)
            refs.append(loss.item())
            grads = _grads(loss, theta)
            theta = [(p.detach() - 0.1 * g).requires_grad_() for p, g in zip(theta, grads)]
        _assert_lists_close(new, theta, f"params after two SGD meta-steps (first_order={first_order})", tol=1e-5,
                            hint="theta <- theta - meta_lr * grad maml_loss each iteration (plain SGD: no momentum), "
                                 "with YOUR inner_lr, n_steps and first_order")
        assert len(hist) == 2 and all(abs(h - r) < 1e-5 * max(1.0, r) for h, r in zip(hist, refs)), (
            f"history[i] is the meta-loss (a float) computed at iteration i, before that iteration's update; "
            f"expected {refs}, got {hist}")


def test_stage4_forward_loss_fn_and_inputs_untouched(nb):
    params = _params([1, 4, 1], seed=32)
    before = [p.detach().clone() for p in params]
    batch = _batch(T=2, seed=33)
    new, _ = _train(nb, params, _fixed_sampler([batch]), n_iters=1, forward=_tanh_forward, loss_fn=_l1,
                           meta_lr=0.5, inner_lr=0.2, optimizer="sgd")
    grads = _grads(_ref_maml_loss(params, batch, _tanh_forward, _l1, 0.2, 1), params)
    _assert_lists_close(new, [p.detach() - 0.5 * g for p, g in zip(params, grads)],
                        "one SGD step with a custom forward and loss_fn", tol=1e-5,
                        hint="pass forward and loss_fn through to maml_loss")
    for p, b in zip(params, before):
        assert torch.equal(p.detach(), b), "train_maml trains a COPY: the params you pass in must not change"
    for p in new:
        assert p.is_leaf and p.requires_grad, "returned params are leaf tensors with requires_grad=True"


def test_stage4_adam_matches_torch(nb):
    params = _params([1, 5, 1], seed=34)
    batches = [_batch(T=2, seed=35 + i) for i in range(3)]
    new, hist = _train(nb, params, _fixed_sampler(batches), n_iters=3, meta_lr=0.01, inner_lr=0.1)
    theta = [p.detach().clone().requires_grad_() for p in params]
    opt = torch.optim.Adam(theta, lr=0.01)
    for b in batches:
        loss = _ref_maml_loss(theta, b, lr=0.1, steps=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _assert_lists_close(new, theta, "params after 3 Adam meta-steps (default optimizer)", tol=1e-5,
                        hint="optimizer='adam' is torch.optim.Adam(params, lr=meta_lr) with default betas, "
                             "created once, before the loop")
    assert len(hist) == 3, f"history has one entry per iteration: expected 3, got {len(hist)}"
    with pytest.raises(ValueError):
        _train(nb, params, _fixed_sampler(batches), n_iters=1, optimizer="rmsprop")


def test_stage4_sampling_uses_seeded_generator(nb):
    params = _params([1, 4, 1], seed=36)
    log = []

    def sampler(generator):
        log.append(generator)
        return nb.sample_sine_tasks(2, 3, 3, generator=generator)

    _, h1 = _train(nb, params, sampler, n_iters=4, seed=5)
    assert len(log) == 4, f"sample_batch must be called once per iteration (4), was called {len(log)} times"
    assert all(isinstance(g, torch.Generator) for g in log), "sample_batch receives a torch.Generator"
    assert all(g is log[0] for g in log), "use ONE generator for the whole run (seeded once), not a new one per call"
    assert log[0].initial_seed() == 5, f"the generator must be seeded with `seed` (5), got {log[0].initial_seed()}"
    _, h2 = _train(nb, params, sampler, n_iters=4, seed=5)
    _, h3 = _train(nb, params, sampler, n_iters=4, seed=6)
    assert h1 == h2, "the same seed must reproduce the same run"
    assert h1 != h3, "a different seed must draw different task batches"


def _post_adapt_loss(nb, params, seed, steps=1):
    b = nb.sample_sine_tasks(200, 10, 20, generator=torch.Generator().manual_seed(seed))
    fast = nb.adapt([p.detach() for p in params], b["xs"], b["ys"], n_steps=steps, first_order=True)
    with torch.no_grad():
        return _mse(_ref_forward(fast, b["xq"]), b["yq"]).mean().item()


def test_stage4_meta_training_makes_adaptation_work(nb):
    # MAML vs the same budget of plain multi-task training (n_steps=0), median over 3 seeds
    def sampler(g):
        return nb.sample_sine_tasks(10, 10, 10, generator=g)

    maml_after, pre_after, maml_before = [], [], []
    for s in range(3):
        init = nb.init_mlp([1, 40, 40, 1], generator=torch.Generator().manual_seed(100 + s))
        m, _ = _train(nb, init, sampler, n_iters=400, meta_lr=3e-3, n_steps=1, seed=s, budget=1_500_000)
        p, _ = _train(nb, init, sampler, n_iters=400, meta_lr=3e-3, n_steps=0, seed=s, budget=1_500_000)
        maml_after.append(_post_adapt_loss(nb, m, 1000 + s))
        maml_before.append(_post_adapt_loss(nb, m, 1000 + s, steps=0))
        pre_after.append(_post_adapt_loss(nb, p, 1000 + s))
    ma, pa, mb = np.median(maml_after), np.median(pre_after), np.median(maml_before)
    assert ma < 0.8 * pa, (
        f"after one adaptation step, MAML's init should beat plain multi-task training's by a wide margin; median "
        f"query MSE {ma:.3f} vs {pa:.3f}. is the meta-gradient flowing through the inner step?")
    assert ma < 0.7 * mb, f"one inner step should cut MAML's query loss sharply: {mb:.3f} before, {ma:.3f} after"
