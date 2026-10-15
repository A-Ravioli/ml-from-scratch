"""Checks for exercise.py (stochastic optimization).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The optimizers are
checked three ways: hand-computed single steps, the parameter protocol (in-place updates, None
grads skipped, grads never modified, lr changes honoured, state per parameter index, any object
with .data/.grad accepted), and an independent re-implementation of PyTorch's update rules run
on random sequences of gradients (including steps where some gradients are None).
"""
import inspect

import numpy as np
import pytest


class _Obj:
    """a duck-typed parameter: not a Param, just .data and .grad."""

    def __init__(self, data):
        self.data = np.array(data, dtype=float)
        self.grad = None


def _grad_sequence(shapes, n_steps, seed, p_none=0.25):
    rng = np.random.default_rng(seed)
    seq = []
    for t in range(n_steps):
        seq.append([None if (t > 0 and rng.random() < p_none) else np.asarray(rng.normal(size=s) * rng.uniform(0.1, 3))
                    for s in shapes])
    return seq


def _reference(kind, inits, seq, hp, lr_at=None):
    """independent re-implementation of torch.optim semantics for the given optimizer kind."""
    data = [np.array(d, dtype=float) for d in inits]
    st = [dict() for _ in inits]
    for t, grads in enumerate(seq):
        lr = hp["lr"] if lr_at is None else lr_at(t)
        for i, g in enumerate(grads):
            if g is None:
                continue
            s, p = st[i], data[i]
            if kind == "sgd":
                d = g + hp.get("weight_decay", 0.0) * p
                mu = hp.get("momentum", 0.0)
                if mu:
                    s["b"] = d.copy() if "b" not in s else mu * s["b"] + d
                    d = d + mu * s["b"] if hp.get("nesterov") else s["b"]
                data[i] = p - lr * d
            elif kind == "adagrad":
                s["s"] = s.get("s", 0) + g * g
                data[i] = p - lr * g / (np.sqrt(s["s"]) + hp.get("eps", 1e-10))
            elif kind == "rmsprop":
                a = hp.get("alpha", 0.99)
                s["v"] = a * s.get("v", 0) + (1 - a) * g * g
                data[i] = p - lr * g / (np.sqrt(s["v"]) + hp.get("eps", 1e-8))
            elif kind in ("adam", "adamw"):
                b1, b2 = hp.get("betas", (0.9, 0.999))
                wd = hp.get("weight_decay", 0.0 if kind == "adam" else 0.01)
                if kind == "adamw":
                    p = p * (1 - lr * wd)
                    d = g
                else:
                    d = g + wd * p
                s["t"] = s.get("t", 0) + 1
                s["m"] = b1 * s.get("m", 0) + (1 - b1) * d
                s["v"] = b2 * s.get("v", 0) + (1 - b2) * d * d
                mh, vh = s["m"] / (1 - b1 ** s["t"]), s["v"] / (1 - b2 ** s["t"])
                data[i] = p - lr * mh / (np.sqrt(vh) + hp.get("eps", 1e-8))
    return data


def _run(cls, inits, seq, hp, lr_at=None):
    params = [_Obj(d) for d in inits]  # duck-typed on purpose: the protocol is .data/.grad, not Param
    opt = cls(params, **hp)
    aliases = [p.data for p in params]
    for t, grads in enumerate(seq):
        if lr_at is not None:
            opt.lr = lr_at(t)
        opt.zero_grad()
        for p, g in zip(params, grads):
            p.grad = None if g is None else g.copy()
        opt.step()
    return params, aliases


def _check_against_reference(nb, name, kind, hp, seed):
    shapes = [(3,), (2, 2), ()]
    inits = [np.random.default_rng(seed + 100).normal(size=s) for s in shapes]
    seq = _grad_sequence(shapes, 12, seed)
    lr_at = lambda t: hp["lr"] * (1.0 if t < 6 else 0.3)  # noqa: E731  (a scheduler changes .lr mid-run)
    params, aliases = _run(getattr(nb, name), inits, seq, hp, lr_at=lr_at)
    want = _reference(kind, inits, seq, hp, lr_at=lr_at)
    for i, (p, w, a) in enumerate(zip(params, want, aliases)):
        assert p.data is a, (
            f"{name}: step() must update p.data IN PLACE (p.data -= ..., p.data *= ...), not rebind it: other code "
            "(a model, a later chapter's Tensor) holds references to the same array")
        assert np.allclose(p.data, w, rtol=1e-10, atol=1e-12), (
            f"{name} with {hp}: after 12 steps (some gradients None, lr cut by 0.3 after step 6) parameter {i} is "
            f"{np.round(p.data, 6).tolist()}, the PyTorch rule gives {np.round(w, 6).tolist()} (max |difference| "
            f"{np.max(np.abs(np.asarray(p.data) - w)):.2e}). check the update "
            "formula, that parameters with grad None are skipped (their state doesn't advance either), and that you "
            "read self.lr inside step()")


# ---------------------------------------------------------------- stage 1: the protocol


def test_stage1_param_copies_and_converts(nb):
    src = np.array([1, 2, 3])
    p = nb.Param(src)
    assert isinstance(p.data, np.ndarray) and p.data.dtype == np.float64, (
        f"Param.data must be a float64 ndarray (int input gets converted); got {type(p.data).__name__} "
        f"{getattr(p.data, 'dtype', '')}")
    assert p.grad is None, "a new Param has grad None"
    p.data[0] = 99.0
    assert src[0] == 1, "Param must store a COPY: training it must never change the caller's array"
    fsrc = np.array([0.5, 1.5])
    pf = nb.Param(fsrc)
    pf.data[0] = 7.0
    assert fsrc[0] == 0.5, (
        "Param must copy even a float64 array (np.array(data, dtype=np.float64), not np.asarray, which returns the "
        "same array when no conversion is needed)")
    q = nb.Param(np.array([[1.5, 2.5]]))
    assert q.data.shape == (1, 2) and nb.Param(3.0).data.shape == (), "Param keeps the shape (a scalar becomes 0-d)"


def test_stage1_optimizer_base(nb):
    ps = [nb.Param(np.ones(2)), nb.Param(np.zeros(3))]
    opt = nb.Optimizer((p for p in ps), 0.5)  # a generator: materialize it once
    assert isinstance(opt.params, list) and len(opt.params) == 2 and opt.params[0] is ps[0], (
        "store the parameters as a list self.params (params may be a generator, which can only be read once), "
        "keeping the original objects")
    assert opt.lr == 0.5, "store lr as self.lr"
    assert isinstance(opt.state, dict) and not opt.state, "self.state starts as an empty dict"
    for p in ps:
        p.grad = np.ones_like(p.data)
    opt.zero_grad()
    assert all(p.grad is None for p in ps), (
        "zero_grad() sets every .grad to None (not to zeros): None tells step() to skip the parameter")


def test_stage1_zero_grad_on_any_object(nb):
    objs = [_Obj([1.0]), _Obj([2.0, 3.0])]
    objs[0].grad = np.array([5.0])
    opt = nb.Optimizer(objs, 0.1)
    opt.zero_grad()
    assert objs[0].grad is None and objs[1].grad is None, (
        "optimizers work with ANY object that has .data and .grad (chapter 14's Tensor is one): zero_grad must not "
        "assume Param")


def test_stage1_minibatches_cover_once(nb):
    batches = list(nb.minibatch_indices(10, 3, rng=0))
    sizes = [len(b) for b in batches]
    allidx = np.concatenate(batches)
    assert sizes == [3, 3, 3, 1], f"10 examples in batches of 3: sizes [3, 3, 3, 1]; got {sizes}"
    assert sorted(allidx.tolist()) == list(range(10)), "one epoch covers every index exactly once"
    assert np.issubdtype(allidx.dtype, np.integer), f"indices must be integers, got {allidx.dtype}"
    assert allidx.tolist() != list(range(10)), "shuffle=True should permute the indices"
    ordered = [b.tolist() for b in nb.minibatch_indices(7, 3, shuffle=False)]
    assert ordered == [[0, 1, 2], [3, 4, 5], [6]], f"shuffle=False gives 0..n-1 in order: got {ordered}"


def test_stage1_minibatches_drop_last_and_generator(nb):
    gen = nb.minibatch_indices(10, 3, drop_last=True, rng=1)
    assert inspect.isgenerator(gen) or hasattr(gen, "__next__"), "minibatch_indices should be a generator (use yield)"
    sizes = [len(b) for b in gen]
    assert sizes == [3, 3, 3], f"drop_last=True drops the short final batch: [3, 3, 3]; got {sizes}"
    assert list(nb.minibatch_indices(2, 5, drop_last=True, rng=0)) == [], (
        "with n < batch_size and drop_last=True there are no full batches: yield nothing")
    one = list(nb.minibatch_indices(4, 4, rng=0))
    assert len(one) == 1 and sorted(one[0].tolist()) == [0, 1, 2, 3], "batch_size = n gives one full batch"


def test_stage1_minibatches_use_rng(nb):
    a = np.concatenate(list(nb.minibatch_indices(20, 4, rng=7)))
    np.random.seed(123)  # the global state must not matter
    b = np.concatenate(list(nb.minibatch_indices(20, 4, rng=7)))
    assert a.tolist() == b.tolist(), "the same int seed must give the same order (use as_rng(rng), not np.random)"
    g = np.random.default_rng(3)
    e1 = np.concatenate(list(nb.minibatch_indices(20, 4, rng=g)))
    e2 = np.concatenate(list(nb.minibatch_indices(20, 4, rng=g)))
    assert e1.tolist() != e2.tolist(), (
        "passing the same Generator for two epochs should give two different shuffles (the generator advances)")


# ---------------------------------------------------------------- stage 2: SGD


def test_stage2_one_step_in_place(nb):
    p = nb.Param([1.0, 2.0])
    alias = p.data
    p.grad = np.array([0.5, -1.0])
    nb.SGD([p], lr=0.1).step()
    assert np.allclose(p.data, [0.95, 2.1]), f"x - lr * g = (1 - 0.05, 2 + 0.1) = (0.95, 2.1); got {p.data}"
    assert p.data is alias and np.allclose(alias, [0.95, 2.1]), (
        "step() must change p.data IN PLACE (p.data -= ...), not assign a new array to p.data: a model holding "
        "the old array would never see the update")
    assert p.grad.tolist() == [0.5, -1.0], "step() must not modify p.grad"


def test_stage2_skips_none_and_respects_lr_changes(nb):
    a, b = nb.Param([1.0]), nb.Param([1.0])
    opt = nb.SGD([a, b], lr=0.1)
    a.grad = np.array([1.0])
    opt.step()
    assert b.data.tolist() == [1.0] and np.isclose(a.data[0], 0.9), (
        "a parameter whose .grad is None must be skipped, not treated as zero gradient (and not crash)")
    opt.lr = 0.01
    opt.step()
    assert np.isclose(a.data[0], 0.89), (
        f"after opt.lr = 0.01 the next step must use the new rate (read self.lr inside step()): expected 0.89, got "
        f"{a.data[0]}")


def test_stage2_momentum_and_nesterov_hand_values(nb):
    p = nb.Param([0.0])
    opt = nb.SGD([p], lr=0.1, momentum=0.9)
    xs = []
    for _ in range(3):
        p.grad = np.array([1.0])
        opt.step()
        xs.append(p.data[0])
    assert np.allclose(xs, [-0.1, -0.29, -0.561]), (
        f"constant gradient 1, momentum 0.9: the buffer is 1, 1.9, 2.71 (b = g on the first step, then 0.9 b + g), "
        f"so x = -0.1, -0.29, -0.561; got {np.round(xs, 4).tolist()}")
    q = nb.Param([0.0])
    opt = nb.SGD([q], lr=0.1, momentum=0.9, nesterov=True)
    ys = []
    for _ in range(2):
        q.grad = np.array([1.0])
        opt.step()
        ys.append(q.data[0])
    assert np.allclose(ys, [-0.19, -0.461]), (
        f"nesterov uses g + momentum * b: step 1 b = 1, update 1 + 0.9 = 1.9; step 2 b = 1.9, update 1 + 1.71 = "
        f"2.71. so x = -0.19, -0.461; got {np.round(ys, 4).tolist()}")
    with pytest.raises(ValueError):
        nb.SGD([nb.Param([0.0])], lr=0.1, nesterov=True)


def test_stage2_weight_decay_does_not_touch_grad(nb):
    p = nb.Param([2.0, -4.0])
    g = np.array([0.0, 1.0])
    p.grad = g
    nb.SGD([p], lr=0.1, weight_decay=0.5).step()
    assert np.allclose(p.data, [1.9, -3.9]), (
        f"weight decay adds wd * p to the gradient: (0 + 1, 1 - 2) = (1, -1), so x = (2 - 0.1, -4 + 0.1) = "
        f"(1.9, -3.9); got {p.data}")
    assert g.tolist() == [0.0, 1.0] and p.grad is g, (
        "build the decayed gradient as a NEW array (g = p.grad + wd * p.data); `p.grad += ...` corrupts the "
        "caller's gradient")


def test_stage2_state_per_parameter_and_late_start(nb):
    a, b = nb.Param([0.0]), nb.Param([0.0])
    opt = nb.SGD([a, b], lr=1.0, momentum=0.5)
    a.grad = np.array([1.0])
    opt.step()
    opt.step()  # b has no gradient yet
    b.grad = np.array([1.0])
    a.grad = np.array([1.0])
    opt.step()
    assert np.isclose(b.data[0], -1.0), (
        f"b gets its first gradient at step 3: its momentum buffer starts fresh (b = g = 1), so it moves by 1; got "
        f"{b.data[0]}. keep a separate buffer per parameter (self.state[i]) and create it on the first real step.")
    assert np.isclose(a.data[0], -(1 + 1.5 + 1.75)), (
        f"a's buffer goes 1, 1.5, 1.75 over its three steps: x = -4.25; got {a.data[0]}")


def test_stage2_matches_pytorch_rule(nb):
    for hp in (dict(lr=0.05), dict(lr=0.05, momentum=0.9), dict(lr=0.05, momentum=0.8, nesterov=True),
               dict(lr=0.05, momentum=0.7, weight_decay=0.1)):
        _check_against_reference(nb, "SGD", "sgd", hp, seed=1)


# ---------------------------------------------------------------- stage 3: Adagrad and RMSProp


def test_stage3_adagrad_hand_values(nb):
    p = nb.Param([1.0, 1.0])
    opt = nb.Adagrad([p], lr=0.1)
    p.grad = np.array([100.0, 0.01])
    opt.step()
    assert np.allclose(p.data, [0.9, 0.9], atol=1e-8), (
        f"Adagrad's first step is lr * g / sqrt(g^2) = lr * sign(g) for EVERY coordinate, whatever the gradient's "
        f"size: (0.9, 0.9); got {p.data}")
    p.grad = np.array([100.0, 0.01])
    opt.step()
    assert np.allclose(p.data, 0.9 - 0.1 / np.sqrt(2), atol=1e-8), (
        f"second step with the same gradient: s = 2 g^2, so the step is lr / sqrt(2) = 0.0707; got {p.data}")


def test_stage3_rmsprop_first_step_is_large(nb):
    p = nb.Param([0.0])
    opt = nb.RMSProp([p], lr=0.01)
    p.grad = np.array([3.0])
    opt.step()
    assert np.isclose(p.data[0], -0.1, rtol=1e-6), (
        f"RMSProp's v starts at 0, so after one step v = 0.01 g^2 and the step is lr g / (0.1 |g|) = 10 lr = 0.1 "
        f"(no bias correction); got {-p.data[0]}")


def test_stage3_matches_pytorch_rule(nb):
    for name, kind, hp in (("Adagrad", "adagrad", dict(lr=0.1)), ("RMSProp", "rmsprop", dict(lr=0.01)),
                           ("RMSProp", "rmsprop", dict(lr=0.01, alpha=0.9, eps=1e-3))):
        _check_against_reference(nb, name, kind, hp, seed=2)


def test_stage3_grad_untouched_and_duck_typed(nb):
    for name in ("Adagrad", "RMSProp"):
        o = _Obj([1.0, -1.0])
        g = np.array([0.3, -0.2])
        o.grad = g
        getattr(nb, name)([o], lr=0.1).step()
        assert g.tolist() == [0.3, -0.2], f"{name}.step() must not modify the gradient array"
        assert o.data[0] < 1.0 and o.data[1] > -1.0, f"{name} must work on any object with .data/.grad"


# ---------------------------------------------------------------- stage 4: Adam and AdamW


def test_stage4_adam_first_step_is_lr_sign(nb):
    for scale in (1e-3, 1.0, 1e3):
        p = nb.Param([1.0, 1.0])
        p.grad = np.array([scale, -scale])
        nb.Adam([p], lr=0.1).step()
        assert np.allclose(p.data, [0.9, 1.1], atol=1e-5), (
            f"with bias correction, Adam's first step is lr * g / |g| = lr * sign(g) for any gradient size (here "
            f"{scale}): expected (0.9, 1.1), got {p.data}. without bias correction it would be about "
            f"lr * 0.1 g / (0.0316 |g|) = 3.16 lr.")


def test_stage4_coupled_versus_decoupled_decay(nb):
    p = nb.Param([2.0])
    p.grad = np.array([0.0])
    nb.Adam([p], lr=0.1, weight_decay=0.01).step()
    assert np.isclose(p.data[0], 1.9, atol=1e-5), (
        f"Adam's weight decay is COUPLED: g = 0 + 0.01 * 2 = 0.02 goes through the adaptive scaling, and the first "
        f"step is lr * sign(g) = 0.1, so x = 1.9; got {p.data[0]}")
    q = nb.Param([2.0])
    q.grad = np.array([0.0])
    nb.AdamW([q], lr=0.1, weight_decay=0.01).step()
    assert np.isclose(q.data[0], 2.0 * (1 - 0.1 * 0.01)), (
        f"AdamW's decay is DECOUPLED: x *= 1 - lr * wd = 0.999 (x = 1.998), and with g = 0 the Adam part does "
        f"nothing; got {q.data[0]}")


def test_stage4_step_count_per_parameter(nb):
    a, b = nb.Param([0.0]), nb.Param([0.0])
    opt = nb.Adam([a, b], lr=0.1)
    for _ in range(3):
        a.grad = np.array([1.0])
        opt.step()
    b.grad = np.array([1.0])
    a.grad = None
    opt.step()
    assert np.isclose(b.data[0], -0.1, atol=1e-6), (
        f"b's first gradient arrives at the 4th call of step(): its own step count is 1, so its bias-corrected "
        f"step is lr * sign(g) = 0.1; got {-b.data[0]:.6f}. count steps PER PARAMETER (self.state[i]), only when "
        "that parameter has a gradient.")


def test_stage4_matches_pytorch_rule(nb):
    for name, kind, hp in (("Adam", "adam", dict(lr=0.01)),
                           ("Adam", "adam", dict(lr=0.01, betas=(0.8, 0.99), weight_decay=0.1)),
                           ("AdamW", "adamw", dict(lr=0.01)),
                           ("AdamW", "adamw", dict(lr=0.02, betas=(0.5, 0.9), eps=1e-3, weight_decay=0.5))):
        _check_against_reference(nb, name, kind, hp, seed=3)


def test_stage4_trains_logistic_regression(nb):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X @ np.array([2.0, -1.0, 0.5]) + 0.3 * rng.normal(size=200) > 0).astype(float)

    def loss_grad(w, idx):
        z = X[idx] @ w
        p = 0.5 * (1 + np.tanh(0.5 * z))
        return float(np.mean(np.logaddexp(0, z) - y[idx] * z)), X[idx].T @ (p - y[idx]) / len(idx)

    for name, hp in (("Adam", dict(lr=0.05)), ("AdamW", dict(lr=0.05, weight_decay=0.001))):
        w = nb.Param(np.zeros(3))
        opt = getattr(nb, name)([w], **hp)
        for epoch in range(30):
            for idx in nb.minibatch_indices(200, 20, rng=epoch):
                opt.zero_grad()
                _, w.grad = loss_grad(w.data, idx)
                opt.step()
        final, _ = loss_grad(w.data, np.arange(200))
        assert final < 0.25, f"{name} should fit this logistic regression to a mean loss below 0.25 in 30 epochs; got {final:.3f}"


# ---------------------------------------------------------------- stage 5: gradient noise


def test_stage5_variance_matches_enumeration(nb):
    from itertools import combinations

    rng = np.random.default_rng(4)
    G = rng.normal(size=(6, 2))
    g_full = G.mean(axis=0)
    for B in (1, 2, 3, 6):
        errs = [np.sum((G[list(c)].mean(axis=0) - g_full) ** 2) for c in combinations(range(6), B)]
        want = float(np.mean(errs))
        got = nb.minibatch_gradient_variance(G, B)
        assert np.isclose(got, want), (
            f"averaging ||g_B - g_full||^2 over ALL {len(errs)} batches of size {B} (without replacement) gives "
            f"{want:.6f}; got {got:.6f}. the factor (n - B)/(n - 1) is the finite-population correction: with B = n "
            "there is no noise at all.")


def test_stage5_noise_floor_formula(nb):
    got = nb.sgd_noise_floor(0.1, [1.0, 4.0], [2.0, 0.5])
    want = 0.1 * 2.0 / (2 * (2 - 0.1)) + 0.1 * 0.5 / (2 * (2 - 0.4))
    assert np.isclose(got, want), f"sum_i lr s_i / (2 (2 - lr a_i)) = {want:.6f}; got {got}"


def test_stage5_noise_floor_matches_your_sgd(nb):
    rng = np.random.default_rng(5)
    k = 400
    a = np.full(k, 2.0)
    s = np.full(k, 0.5)
    p = nb.Param(np.zeros(k))
    lr = 0.1
    opt = nb.SGD([p], lr=lr)
    vals = []
    for t in range(1_500):
        opt.zero_grad()
        p.grad = a * p.data + np.sqrt(s) * rng.normal(size=k)
        opt.step()
        if t >= 300:
            vals.append(0.5 * np.sum(a * p.data**2))
    measured = float(np.mean(vals))
    predicted = nb.sgd_noise_floor(lr, a, s)
    assert abs(measured - predicted) < 0.06 * predicted, (
        f"constant-step SGD on a 400-D quadratic with noisy gradients settles at an average excess loss of "
        f"{measured:.4f}; your formula predicts {predicted:.4f}")
