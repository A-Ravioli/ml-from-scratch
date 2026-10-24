"""Checks for exercise.py (in-context learning as meta-learning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the task sampler is
checked with moment tests; the least-squares baseline against NumPy (`lstsq` for the minimum-norm
solution, the normal equations for ridge) and against the noiseless-recovery property; gradient
descent against a hand-computed step; linear self-attention against its formula and against one
GD step with the constructed weights; the transformer for causality (perturbing a later x or the
same y must not change a prediction), shapes and structure; training against a reference Adam
loop replaying the same tasks; and (slow) a trained transformer whose error falls far below its
one-example error once the context passes the dimension, as least squares' does.
"""
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


def _train(nb, *args, budget=200_000, **kwargs):
    """train_icl, stopped with a teaching message if its loop doesn't end (about 4,000 lines per step)."""
    return bounded(nb.train_icl, *args, budget=budget, what="train_icl",
                   hint="the loop must run exactly n_steps steps.", **kwargs)


def _tasks(B=3, n=7, d=3, seed=0, noise=0.0):
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(B, n, d, generator=g)
    w = torch.randn(B, d, generator=g)
    ys = (xs @ w.unsqueeze(-1)).squeeze(-1) + noise * torch.randn(B, n, generator=g)
    return xs, ys, w


# ---------------------------------------------------------------- stage 1: tasks and least squares


def test_stage1_sample_linear_tasks(nb):
    b = nb.sample_linear_tasks(4000, 6, 3, generator=torch.Generator().manual_seed(1))
    assert tuple(b["xs"].shape) == (4000, 6, 3) and tuple(b["ys"].shape) == (4000, 6) and tuple(b["w"].shape) == (4000, 3), (
        "xs (B, n, d), ys (B, n), w (B, d)")
    assert torch.allclose(b["ys"], (b["xs"] @ b["w"].unsqueeze(-1)).squeeze(-1), atol=1e-5), (
        "noiseless: ys = xs @ w for every task")
    for name, v in (("w", b["w"]), ("xs", b["xs"])):
        v = v.flatten().numpy()
        p = stats.kstest(v, "norm").pvalue
        assert p > 1e-4, f"{name} entries should be N(0, 1); KS test p = {p:.2g} (mean {v.mean():.3f}, sd {v.std():.3f})"
    c = np.corrcoef(b["xs"][:, 0, 0].numpy(), b["xs"][:, 1, 0].numpy())[0, 1]
    assert abs(c) < 0.1, "the points of a task are independent draws"
    assert not torch.allclose(b["w"][0], b["w"][1]), "each task has its own w"


def test_stage1_noise_scale_and_generator(nb):
    b = nb.sample_linear_tasks(3000, 5, 2, noise_std=0.5, x_scale=2.0, generator=torch.Generator().manual_seed(2))
    resid = (b["ys"] - (b["xs"] @ b["w"].unsqueeze(-1)).squeeze(-1)).flatten()
    assert abs(float(resid.std()) - 0.5) < 0.02, f"noise_std=0.5: residuals should have sd 0.5, got {float(resid.std()):.3f}"
    assert abs(float(b["xs"].std()) - 2.0) < 0.05, f"x_scale=2: inputs should have sd 2, got {float(b['xs'].std()):.3f}"
    assert abs(float(b["w"].std()) - 1.0) < 0.05, "x_scale scales the inputs only, not w"
    r = (b["ys"] - (b["xs"] @ b["w"].unsqueeze(-1)).squeeze(-1)).numpy()
    c = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
    assert abs(c) < 0.1, (
        f"label noise must be drawn independently for every point, not once per task: the residuals of points 0 "
        f"and 1 of a task correlate at {c:.2f}")
    within = float(r.std(axis=1, ddof=1).mean())
    assert within > 0.4, f"within a task, the residuals should vary with sd about 0.5; their average sd is {within:.3f}"
    a1 = nb.sample_linear_tasks(2, 3, 2, noise_std=0.5, generator=torch.Generator().manual_seed(3))
    torch.manual_seed(42)
    a2 = nb.sample_linear_tasks(2, 3, 2, noise_std=0.5, generator=torch.Generator().manual_seed(3))
    assert all(torch.equal(a1[k], a2[k]) for k in a1), "the same generator seed must give the same tasks, noise included (draw it from the generator too)"


def test_stage1_least_squares_matches_numpy(nb):
    xs, ys, _ = _tasks(B=3, n=7, d=3, seed=4, noise=0.3)
    got = nb.least_squares_predictions(xs, ys)
    assert tuple(got.shape) == (3, 7), f"one prediction per position: (B, n) = (3, 7), got {tuple(got.shape)}"
    assert got.dtype == xs.dtype, f"return the input's dtype ({xs.dtype}), got {got.dtype}"
    X, Y = xs.double().numpy(), ys.double().numpy()
    for b in range(3):
        assert abs(float(got[b, 0])) < 1e-12, "position 0 has no context: predict 0"
        for i in range(1, 7):
            w = np.linalg.lstsq(X[b, :i], Y[b, :i], rcond=None)[0]  # minimum-norm when i < d
            want = X[b, i] @ w
            assert abs(float(got[b, i]) - want) < 1e-4 * max(1, abs(want)), (
                f"task {b}, position {i}: expected {want:.5f} (least squares on the first {i} pairs; minimum-norm while "
                f"{i} < d = 3), got {float(got[b, i]):.5f}. use only pairs BEFORE position i")


def test_stage1_ridge_and_noiseless_recovery(nb):
    xs, ys, _ = _tasks(B=2, n=6, d=3, seed=5, noise=0.5)
    got = nb.least_squares_predictions(xs, ys, ridge=0.7)
    X, Y = xs.double().numpy(), ys.double().numpy()
    for b in range(2):
        for i in range(1, 6):
            w = np.linalg.solve(X[b, :i].T @ X[b, :i] + 0.7 * np.eye(3), X[b, :i].T @ Y[b, :i])
            assert abs(float(got[b, i]) - X[b, i] @ w) < 1e-4, (
                f"ridge=0.7, task {b}, position {i}: w = (X^T X + 0.7 I)^-1 X^T y with SUMS over the context, not means")
    xs, ys, _ = _tasks(B=50, n=8, d=4, seed=6)
    e = nb.position_mse(nb.least_squares_predictions(xs, ys), ys)
    assert tuple(e.shape) == (8,), f"position_mse returns one error per position, (n,), got {tuple(e.shape)}"
    assert float(e[5:].max()) < 1e-8, f"noiseless, d = 4: from 4 points on, least squares is exact; errors {e.tolist()}"
    assert float(e[3]) > 0.05, "with 3 < d points the problem is underdetermined: the error can't be 0"


def test_stage1_position_mse(nb):
    preds = torch.tensor([[1.0, 2.0, 0.0], [3.0, 2.0, 1.0]])
    ys = torch.tensor([[0.0, 2.0, 2.0], [1.0, 0.0, 1.0]])
    got = nb.position_mse(preds, ys)
    assert torch.allclose(got, torch.tensor([2.5, 2.0, 2.0])), f"mean over tasks per position: expected [2.5, 2, 2], got {got.tolist()}"


# ---------------------------------------------------------------- stage 2: gradient descent as attention


def test_stage2_gd_hand_example(nb):
    # the lesson's worked example: d = 2, context (1, 0) -> 2 and (1, 1) -> 1, query (0, 1), lr 0.5
    xs = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    ys = torch.tensor([[2.0, 1.0, -1.0]])
    one = nb.gd_predictions(xs, ys, lr=0.5)
    assert tuple(one.shape) == (1, 3), f"(B, n) predictions, got {tuple(one.shape)}"
    assert abs(float(one[0, 2]) - 0.5) < 1e-6, (
        f"one step from w = 0: w = 0.5 * (2 (1, 0) + 1 (1, 1)) = (1.5, 0.5), prediction for (0, 1) is 0.5; got "
        f"{float(one[0, 2]):.4f}. the loss is 1/2 * a SUM over the context")
    assert abs(float(one[0, 1]) - 1.0) < 1e-6 and float(one[0, 0]) == 0.0, "positions 0 and 1: 0, then 0.5 * 2 * (1 . 1) = 1"
    two = nb.gd_predictions(xs, ys, lr=0.5, n_steps=2)
    assert abs(float(two[0, 2]) - 0.0) < 1e-6, (
        f"two steps: residuals (1.5 - 2, 2 - 1) = (-0.5, 1), gradient (0.5, 1), w = (1.25, 0), prediction 0; got "
        f"{float(two[0, 2]):.4f}")


def test_stage2_gd_steps_match_unrolled_numpy(nb):
    xs, ys, _ = _tasks(B=4, n=6, d=3, seed=7, noise=0.2)
    xs, ys = xs.double(), ys.double()
    for k, lr in ((7, 0.05), (1, 0.2), (0, 0.1)):
        got = nb.gd_predictions(xs, ys, lr=lr, n_steps=k)
        X, Y = xs.numpy(), ys.numpy()
        for b in range(4):
            for i in range(1, 6):
                w = np.zeros(3)
                for _ in range(k):
                    w = w - lr * X[b, :i].T @ (X[b, :i] @ w - Y[b, :i])
                want = X[b, i] @ w
                assert abs(float(got[b, i]) - want) < 1e-8 * max(1, abs(want)), (
                    f"n_steps={k}, lr={lr}, task {b}, position {i}: expected {want:.6f}, got {float(got[b, i]):.6f}. "
                    "each position restarts from w = 0 and uses only the pairs before it")


def test_stage2_lsa_formula(nb):
    g = torch.Generator().manual_seed(8)
    xs, ys = torch.randn(2, 5, 3, generator=g), torch.randn(2, 5, generator=g)
    xq = torch.randn(2, 3, generator=g)
    Wkq, Wpv = torch.randn(4, 4, generator=g), torch.randn(4, 4, generator=g)
    got = nb.lsa_predict(xs, ys, xq, Wkq, Wpv)
    assert tuple(got.shape) == (2,), f"one prediction per task, (B,), got {tuple(got.shape)}"
    for b in range(2):
        E = torch.cat([xs[b], ys[b, :, None]], 1)
        eq = torch.cat([xq[b], torch.zeros(1)])
        upd = sum(Wpv @ E[j] * (E[j] @ Wkq @ eq) for j in range(5))
        want = (eq + upd)[-1]
        assert abs(float(got[b]) - float(want)) < 1e-4, (
            f"task {b}: expected {float(want):.5f}, got {float(got[b]):.5f}. e_q + W_pv sum_j e_j (e_j^T W_kq e_q), "
            "last coordinate; tokens e_j = (x_j, y_j), the query token (x_q, 0)")


def test_stage2_constructed_weights_do_one_gd_step(nb):
    xs, ys, _ = _tasks(B=5, n=9, d=4, seed=9, noise=0.3)
    for lr in (0.1, 0.03):
        Wkq, Wpv = nb.gd_lsa_weights(4, lr)
        assert tuple(Wkq.shape) == (5, 5) and tuple(Wpv.shape) == (5, 5), "weights are (d+1, d+1)"
        gd = nb.gd_predictions(xs, ys, lr=lr)
        for i in (1, 4, 8):
            got = nb.lsa_predict(xs[:, :i], ys[:, :i], xs[:, i], Wkq, Wpv)
            assert torch.allclose(got, gd[:, i], atol=1e-4), (
                f"lr={lr}, context of {i}: linear self-attention with your weights gives {got.tolist()}, one GD step "
                f"gives {gd[:, i].tolist()}")


# ---------------------------------------------------------------- stage 3: the transformer


def test_stage3_interleave(nb):
    xs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    ys = torch.tensor([[5.0, 6.0]])
    got = nb.interleave(xs, ys)
    want = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 0.0, 5.0], [3.0, 4.0, 0.0], [0.0, 0.0, 6.0]]])
    assert torch.equal(got, want), f"tokens x_1, y_1, x_2, y_2 with x = (x, 0) and y = (0, 0, y): expected {want.tolist()}, got {got.tolist()}"


def _model(nb, seed=0, **kw):
    torch.manual_seed(seed)
    m = nb.ICLTransformer(3, max_points=10, d_model=16, n_layers=2, n_heads=2, **kw)
    m.eval()
    return m


def test_stage3_structure_and_shapes(nb):
    m = _model(nb)
    assert m.dim == 3, "store dim as an attribute (train_icl uses it)"
    assert len(m.blocks) == 2 and all(type(b).__name__ == "TransformerBlock" for b in m.blocks), (
        "blocks: an nn.ModuleList of n_layers TransformerBlocks (from mlfs.transformer)")
    assert m.read_in.in_features == 4 and m.read_in.out_features == 16, "read_in: Linear(d + 1, d_model)"
    assert m.read_out.out_features == 1, "read_out: Linear(d_model, 1)"
    assert tuple(m.pos.shape) == (20, 16), "pos: learned positions, (2 * max_points, d_model)"
    xs, ys, _ = _tasks(B=2, n=6, d=3, seed=10)
    out = m(xs, ys)
    assert tuple(out.shape) == (2, 6), f"forward returns (B, n) predictions, got {tuple(out.shape)}"
    assert tuple(m(*_tasks(B=1, n=10, d=3)[:2]).shape) == (1, 10), "n == max_points is allowed"
    with pytest.raises(ValueError):
        m(*_tasks(B=1, n=11, d=3)[:2])
    with pytest.raises(ValueError):
        m(*_tasks(B=1, n=4, d=2)[:2])


def test_stage3_positions_and_final_norm_are_used(nb):
    m = _model(nb, seed=2)
    xs, ys, _ = _tasks(B=2, n=4, d=3, seed=13)
    # random perturbations: a constant shift of a token would be removed by the layer norms
    delta = torch.randn(16, generator=torch.Generator().manual_seed(14))
    with torch.no_grad():
        base = m(xs, ys)
        m.pos[2] += delta  # the position of the x_1 token (token 2)
        out = m(xs, ys)
        assert torch.allclose(out[:, 0], base[:, 0], atol=1e-5), (
            "changing pos[2] changed the prediction at token 0: positions must line up with tokens (token t gets pos[t])")
        assert not torch.allclose(out[:, 1], base[:, 1], atol=1e-5), (
            "changing pos[2] must change the prediction read at token 2 (y_1's): add pos[:2n] to the read-in tokens")
        m.pos[2] -= delta
        m.pos[8:] += 3.0 * delta  # rows beyond the 2n = 8 tokens of this sequence
        assert torch.allclose(m(xs, ys), base, atol=1e-5), "a sequence of n pairs uses only pos[:2n]"
        m.pos[8:] -= 3.0 * delta
        m.ln_f.weight.mul_(1.0 + delta.abs())
        assert not torch.allclose(m(xs, ys), base, atol=1e-5), "apply ln_f before read_out"


def test_stage3_causality(nb):
    m = _model(nb, seed=1)
    xs, ys, _ = _tasks(B=2, n=6, d=3, seed=11)
    with torch.no_grad():
        base = m(xs, ys)
        for i in range(6):
            y2 = ys.clone()
            y2[:, i] += 5.0
            out = m(xs, y2)
            assert torch.allclose(out[:, :i + 1], base[:, :i + 1], atol=1e-5), (
                f"changing y_{i} changed the prediction FOR y_{i} (or an earlier one): the model sees the answer. read "
                "predictions at the x tokens (positions 0, 2, 4, ...) and attend causally")
            if i < 5:
                assert not torch.allclose(out[:, i + 1:], base[:, i + 1:], atol=1e-5), (
                    f"changing y_{i} must change later predictions: they condition on it")
            x2 = xs.clone()
            x2[:, i] += 1.0
            out = m(x2, ys)
            assert torch.allclose(out[:, :i], base[:, :i], atol=1e-5), f"changing x_{i} changed an earlier prediction"
            assert not torch.allclose(out[:, i], base[:, i], atol=1e-5), (
                f"the prediction for y_{i} must depend on x_{i}: read it at x_{i}'s own token")


# ---------------------------------------------------------------- stage 4: meta-training


class _Recorder(torch.nn.Module):
    """wraps a model and records train mode and the batches it sees."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.dim, self.calls = inner, inner.dim, []

    def forward(self, xs, ys):
        self.calls.append((self.training, xs.detach().clone(), ys.detach().clone()))
        return self.inner(xs, ys)


def test_stage4_training_matches_reference_loop(nb):
    torch.manual_seed(12)
    inner = nb.ICLTransformer(2, max_points=5, d_model=8, n_layers=1, n_heads=2)
    ref = nb.ICLTransformer(2, max_points=5, d_model=8, n_layers=1, n_heads=2)
    ref.load_state_dict(inner.state_dict())
    rec = _Recorder(inner)
    rec.eval()
    hist = _train(nb, rec, n_points=4, n_steps=3, batch_size=5, lr=0.01, noise_std=0.3, seed=7)
    assert isinstance(hist, list) and len(hist) == 3, f"one loss per step (3), got {hist!r}"
    assert all(t for t, _, _ in rec.calls), "train_icl must put the model in train mode"
    assert not rec.training, "train_icl leaves the model in eval mode"
    g = torch.Generator().manual_seed(7)
    opt = torch.optim.Adam(ref.parameters(), lr=0.01)
    for step in range(3):
        b = nb.sample_linear_tasks(5, 4, 2, noise_std=0.3, generator=g)
        assert torch.equal(rec.calls[step][1], b["xs"]) and torch.equal(rec.calls[step][2], b["ys"]), (
            f"step {step}: draw fresh tasks with sample_linear_tasks(batch_size, n_points, model.dim, noise_std=..., "
            "generator=...) from ONE generator seeded with `seed`")
        loss = ((ref(b["xs"], b["ys"]) - b["ys"]) ** 2).mean()
        assert abs(hist[step] - loss.item()) < 1e-5, f"step {step}: the loss is the plain mean squared error over tasks and positions"
        opt.zero_grad()
        loss.backward()
        opt.step()
    for (n1, p1), (_, p2) in zip(inner.named_parameters(), ref.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-5), f"after 3 steps {n1} differs from Adam(model.parameters(), lr=lr)"


@pytest.mark.slow
def test_stage4_learns_in_context_past_the_dimension(nb):
    # d = 2: least squares is exact from 2 examples on; with 1 example nothing can beat an error of about 1.
    # small transformers sometimes stall for a while before they "get" the task, so two seeds, either may pass.
    b = nb.sample_linear_tasks(2000, 6, 2, generator=torch.Generator().manual_seed(103))
    ls = nb.position_mse(nb.least_squares_predictions(b["xs"], b["ys"]), b["ys"])
    assert float(ls[3:].max()) < 1e-8  # test sanity: least squares is exact past d
    curves = []
    for seed in (3, 4):
        torch.manual_seed(seed)
        m = nb.ICLTransformer(2, max_points=6, d_model=32, n_layers=2, n_heads=2)
        _train(nb, m, n_points=6, n_steps=800, batch_size=64, lr=3e-3, seed=seed, budget=6_000_000)
        with torch.no_grad():
            e = nb.position_mse(m(b["xs"], b["ys"]), b["ys"])
        curves.append(e.numpy().round(3))
        if float(e[3:].mean()) < 0.75 * float(e[1]) and float(e[0]) > 1.5 and float(e[5]) < float(e[2]):
            return
    raise AssertionError(
        f"after 800 steps (two seeds), the transformer's error by position was {curves[0]} and {curves[1]}. past the "
        "dimension (positions 3-5) it should fall far below its error with one example (position 1), the way least "
        "squares drops to 0 there, starting from about d = 2 with no context")
