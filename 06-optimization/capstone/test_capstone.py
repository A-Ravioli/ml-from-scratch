"""Checks for capstone.py (an optimizer benchmark suite).

All problems are synthetic and deterministic: small logistic regressions with known structure and
quadratics. The optimizers themselves come from mlfs.optim (the reference solutions under pytest),
so these checks test your harness: the objective, the bookkeeping of calls and costs, the
minibatch loop, and the report functions. Counting checks only use relations that hold for any
correct optimizer (totals, monotonicity), not one implementation's call order.
"""
import numpy as np
import pytest

from mlfs.testing import bounded


def _logistic_data(n=300, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = np.hstack([np.ones((n, 1)), rng.normal(size=(n, d - 1))])
    w = rng.normal(size=d)
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ w))).astype(float)
    return X, y


def _fd(fn, w, h=1e-6):
    out = []
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        out.append((fn(w + e) - fn(w - e)) / (2 * h))
    return np.array(out)


class _Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, x):
        self.calls += 1
        return self.fn(x)


# ---------------------------------------------------------------- stage 1: the objective


def test_stage1_hand_values(nb):
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, 0.0])
    f, g, h = nb.logistic_objective(X, y, 0.0)
    assert np.isclose(f(np.zeros(1)), np.log(2)), f"at w = 0 every prediction is 1/2: the mean loss is log 2; got {f(np.zeros(1))}"
    assert np.allclose(g(np.zeros(1)), [-0.5]), (
        f"grad = X^T (sigmoid(Xw) - y) / n = (1 * (0.5 - 1) + (-1) * (0.5 - 0)) / 2 = -0.5; got {g(np.zeros(1))}")
    assert np.allclose(h(np.zeros(1)), [[0.25]]), "hess = X^T diag(s(1 - s)) X / n = (0.25 + 0.25) / 2 = 0.25"
    f2, g2, h2 = nb.logistic_objective(X, y, 2.0)
    w = np.array([0.7])
    assert np.isclose(f2(w), f(w) + 0.49) and np.allclose(g2(w), g(w) + 1.4) and np.allclose(h2(w), h(w) + 2.0), (
        "the ridge term alpha/2 ||w||^2 adds alpha w to the gradient and alpha I to the Hessian")
    assert isinstance(f(w), float), "f must return a Python float"


def test_stage1_derivatives_match_finite_differences(nb):
    X, y = _logistic_data()
    f, g, h = nb.logistic_objective(X, y, 0.01)
    rng = np.random.default_rng(1)
    for _ in range(3):
        w = rng.normal(size=4)
        assert np.allclose(g(w), _fd(f, w), atol=1e-7), "grad must be the derivative of f (check it against finite differences)"
        H = np.asarray(h(w))
        assert np.allclose(H, _fd(g, w).T, atol=1e-6) and np.allclose(H, H.T), "hess must be the (symmetric) derivative of grad"
        assert np.linalg.eigvalsh(H).min() > 0, "with alpha > 0 the Hessian is positive definite (the loss is strongly convex)"


def test_stage1_no_overflow(nb):
    X, y = _logistic_data()
    f, g, h = nb.logistic_objective(1e3 * X, y, 0.0)
    w = 10 * np.ones(4)
    with np.errstate(over="raise", invalid="raise"):
        vals = (f(w), g(w), h(w))
    assert all(np.all(np.isfinite(v)) for v in vals), (
        "with |z| in the thousands, f, grad and hess must stay finite and raise no overflow: use np.logaddexp(0, z) "
        "for log(1 + e^z) and (1 + tanh(z/2))/2 for the sigmoid")


# ---------------------------------------------------------------- stage 2: deterministic runs


def test_stage2_record_and_counts(nb):
    X, y = _logistic_data()
    f, g, h = nb.logistic_objective(X, y, 0.01)
    fs, gs = _Spy(f), _Spy(g)
    rec = bounded(nb.run_deterministic, "lbfgs", fs, gs, np.zeros(4), max_iters=200, tol=1e-8)
    for key in ("method", "x", "fun", "converged", "n_iters", "history", "costs", "n_f", "n_grad", "n_hess", "seconds"):
        assert key in rec, f"the record must have the key {key!r}"
    assert rec["method"] == "lbfgs" and rec["converged"], f"L-BFGS should converge on this problem; got {rec['converged']}"
    assert rec["n_f"] == fs.calls and rec["n_grad"] == gs.calls and rec["n_hess"] == 0, (
        f"count every call: f was called {fs.calls} times and grad {gs.calls} times by the optimizer; your record says "
        f"{rec['n_f']} and {rec['n_grad']}")
    h_, c_ = rec["history"], rec["costs"]
    assert len(c_) == len(h_) == rec["n_iters"] + 1 and c_[0] == 0.0, (
        "costs has one entry per history entry (n_iters + 1), starting with 0.0 for x0")
    assert all(b >= a for a, b in zip(c_, c_[1:])), "costs can only grow"
    total = rec["n_f"] + rec["n_grad"]
    assert total - 2 <= c_[-1] <= total, (
        f"the last cost should be (nearly) the total work n_f + n_grad = {total}; got {c_[-1]}. record the work inside a "
        "callback passed to minimize.")
    assert isinstance(rec["seconds"], float) and rec["seconds"] >= 0


def test_stage2_hessian_cost_and_options(nb):
    X, y = _logistic_data()
    f, g, h = nb.logistic_objective(X, y, 0.01)
    hs = _Spy(h)
    rec = bounded(nb.run_deterministic, "newton", f, g, np.zeros(4), hess=hs, tol=1e-10)
    d = 4
    assert rec["n_hess"] == hs.calls > 0, "count the Hessian calls too"
    total = rec["n_f"] + rec["n_grad"] + d * rec["n_hess"]
    assert total - d - 2 <= rec["costs"][-1] <= total, (
        f"a Hessian costs d = {d} gradient-equivalents: the total work is n_f + n_grad + d n_hess = {total}; your last "
        f"cost is {rec['costs'][-1]}")
    assert rec["converged"] and rec["n_iters"] <= 10, "Newton should solve this logistic regression in a handful of steps"
    from mlfs.optim import minimize

    direct = minimize(f, g, np.ones(4), method="gd", lr=0.3, max_iters=7, tol=0.0)
    rec = bounded(nb.run_deterministic, "gd", f, g, np.ones(4), lr=0.3, max_iters=7, tol=0.0)
    assert rec["n_iters"] == 7 and np.allclose(rec["x"], direct.x), (
        "pass max_iters, tol and the method's options (lr, momentum, ...) through to minimize unchanged")
    with pytest.raises(ValueError):
        bounded(nb.run_deterministic, "newton", f, g, np.zeros(4))


# ---------------------------------------------------------------- stage 3: minibatch runs


def test_stage3_loop_and_bookkeeping(nb):
    from mlfs.optim import SGD

    X, y = _logistic_data(n=103)
    f, g, h = nb.logistic_objective(X, y, 0.01)
    seen_params, batches = [], []

    def make(params):
        seen_params.append(params)
        return SGD(params, lr=0.5, momentum=0.9)

    def lg(w, idx):
        batches.append(np.array(idx))
        s = 0.5 * (1 + np.tanh(0.5 * (X[idx] @ w)))
        return X[idx].T @ (s - y[idx]) / len(idx) + 0.01 * w

    w0 = np.zeros(4)
    rec = bounded(nb.run_stochastic, make, lg, f, w0, 103, epochs=3, batch_size=20, rng=0)
    assert len(seen_params) == 1 and len(seen_params[0]) == 1 and hasattr(seen_params[0][0], "grad"), (
        "make_optimizer is called once, with a list holding one Param")
    assert rec["method"] == "SGD" and rec["n_steps"] == 18, (
        f"3 epochs of ceil(103 / 20) = 6 batches is 18 steps; got method {rec['method']!r}, n_steps {rec['n_steps']}")
    assert rec["costs"] == [0.0, 1.0, 2.0, 3.0] and len(rec["history"]) == 4, (
        "history: the full loss at w0 and after each epoch; costs: 0, 1, 2, 3 passes over the data")
    assert np.isclose(rec["history"][0], np.log(2)), "the first history entry is the full loss at w0"
    for e in range(3):
        ep = np.concatenate(batches[6 * e:6 * e + 6])
        assert sorted(ep.tolist()) == list(range(103)), "each epoch must visit every example once (use minibatch_indices)"
    assert batches[0].tolist() != batches[6].tolist(), (
        "each epoch should be shuffled differently: create the Generator once (as_rng(rng)) and reuse it, rather "
        "than reseeding every epoch")
    assert w0.tolist() == [0.0] * 4, "w0 must not be modified (Param copies it)"
    assert np.isclose(rec["fun"], rec["history"][-1]) and np.isclose(f(rec["x"]), rec["fun"])


def test_stage3_reproducible_and_converges(nb):
    from mlfs.optim import Adam, minimize

    X, y = _logistic_data(n=400, seed=2)
    f, g, h = nb.logistic_objective(X, y, 0.01)
    f_star = minimize(f, g, np.zeros(4), method="newton", hess=h, tol=1e-10).fun

    def lg(w, idx):
        s = 0.5 * (1 + np.tanh(0.5 * (X[idx] @ w)))
        return X[idx].T @ (s - y[idx]) / len(idx) + 0.01 * w

    a = bounded(nb.run_stochastic, lambda ps: Adam(ps, lr=0.05), lg, f, np.zeros(4), 400, epochs=30, batch_size=32, rng=5)
    b = bounded(nb.run_stochastic, lambda ps: Adam(ps, lr=0.05), lg, f, np.zeros(4), 400, epochs=30, batch_size=32, rng=5)
    assert a["history"] == b["history"], "the same rng must give the same run (pass it to minibatch_indices via as_rng)"
    c = bounded(nb.run_stochastic, lambda ps: Adam(ps, lr=0.05), lg, f, np.zeros(4), 400, epochs=3, batch_size=32, rng=6)
    assert c["history"][1:] != a["history"][1:4], (
        "a different rng must give a different run: shuffle with the rng you were given, as_rng(rng), once, and "
        "pass that Generator to minibatch_indices every epoch")
    assert a["fun"] - f_star < 5e-3, f"Adam should get within 5e-3 of the optimum in 30 epochs; got {a['fun'] - f_star:.2e}"


# ---------------------------------------------------------------- stage 4: the report


def test_stage4_cost_to_reach(nb):
    rec = {"history": [10.0, 5.0, 2.0, 1.1, 1.01], "costs": [0.0, 2.0, 4.0, 6.0, 8.0]}
    assert nb.cost_to_reach(rec, 1.0, 0.5) == 2.0, (
        "f0 - f* = 9; the first iterate with f - f* <= 0.5 * 9 = 4.5 is the second (f - f* = 4), reached at cost 2")
    assert nb.cost_to_reach(rec, 1.0, 0.01) == 8.0, "f - f* <= 0.09 first at the last iterate (0.01), cost 8"
    assert nb.cost_to_reach(rec, 1.0, 1e-3) == np.inf, "never within 1e-3 * 9: return inf"
    assert nb.cost_to_reach(rec, 1.0, 1.0) == 0.0, "rel_tol 1 is met by x0 itself, at cost 0"


def test_stage4_rank(nb):
    recs = [{"method": "a", "history": [4.0, 3.0, 1.0], "costs": [0.0, 5.0, 9.0]},
            {"method": "d", "history": [4.0, 2.0, 1.0], "costs": [0.0, 2.0, 3.0]},
            {"method": "c", "history": [4.0, 3.5], "costs": [0.0, 1.0]},
            {"method": "b", "history": [4.0, 1.0], "costs": [0.0, 3.0]}]
    got = nb.rank_methods(recs, 1.0, 0.1)
    assert [m for m, _ in got] == ["d", "b", "a", "c"] and got[-1][1] == np.inf, (
        f"sorted by cost to reach rel_tol, cheapest first, ties in input order (d came before b), inf last: expected "
        f"[('d', 3), ('b', 3), ('a', 9), ('c', inf)]; got {got}")
