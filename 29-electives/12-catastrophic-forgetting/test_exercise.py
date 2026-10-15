"""Checks for exercise.py (catastrophic forgetting: a continual-learning harness).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. task builders are
checked on hand-made arrays (order, dtypes, a permutation shared between train and test sets);
the metrics on a hand-computed accuracy matrix; evaluation and training against exact one-step
SGD updates computed here with autograd; the sequence runner against an "oracle" training
function whose accuracy matrix is known exactly. everything is tiny and synthetic, so the whole
file runs in a couple of seconds.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

# warm up torch.optim once, so its lazy first-call imports don't count against `bounded` budgets
_w = nn.Linear(1, 1)
_o = torch.optim.SGD(_w.parameters(), lr=0.1)
_w(torch.ones(1, 1)).sum().backward()
_o.step()


def _linear(d, c, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(d, c)


def _task(X, y, classes):
    return {"X": torch.as_tensor(X, dtype=torch.float32), "y": torch.as_tensor(y, dtype=torch.int64),
            "classes": tuple(classes)}


def _blobs(n, classes, d=5, seed=0, sep=3.0):
    rng = np.random.default_rng(seed)
    eye = np.eye(d)
    centers = sep * np.stack([(1 - 2 * (k % 2)) * eye[(k // 2) % d] for k in range(10)])  # +-e1, +-e2, ...
    y = rng.choice(np.asarray(classes), size=n)
    X = centers[y] + rng.normal(size=(n, d))
    return _task(X, y, classes)


def _train(nb, *args, **kwargs):
    return bounded(nb.train_task, *args, budget=5_000_000, what="train_task",
                   hint="loop over epochs and over minibatches of a fixed size; don't loop `while` the loss is high.",
                   **kwargs)


# ---------------------------------------------------------------- stage 1: task sequences


def test_stage1_split_hand_example(nb):
    X = np.arange(24, dtype=np.uint8).reshape(6, 2, 2)
    y = np.array([0, 1, 2, 1, 0, 3])
    tasks = nb.split_tasks(X, y, [(0, 1), (2, 3)])
    assert isinstance(tasks, list) and len(tasks) == 2, f"expected a list of 2 tasks, got {type(tasks).__name__}"
    t0, t1 = tasks
    assert set(t0) >= {"X", "y", "classes"}, f"a task is a dict with keys X, y, classes; got keys {sorted(t0)}"
    assert torch.is_tensor(t0["X"]) and t0["X"].dtype == torch.float32, (
        f"task['X'] should be a float32 torch tensor, got {type(t0['X']).__name__} "
        f"{getattr(t0['X'], 'dtype', '')}")
    assert torch.is_tensor(t0["y"]) and t0["y"].dtype == torch.int64, "task['y'] should be an int64 torch tensor"
    assert tuple(t0["X"].shape) == (4, 4), (
        f"task 0 keeps the 4 examples with labels 0 or 1, flattened to (4, 4); got shape {tuple(t0['X'].shape)}")
    assert t0["y"].tolist() == [0, 1, 1, 0], (
        f"task 0's labels should be [0, 1, 1, 0] (original order, original labels), got {t0['y'].tolist()}")
    assert t0["X"][:, 0].tolist() == [0.0, 4.0, 12.0, 16.0], (
        "task 0's rows should be examples 0, 1, 3, 4 in their original order, with values unchanged "
        f"(no rescaling); got first column {t0['X'][:, 0].tolist()}")
    assert t1["y"].tolist() == [2, 3] and tuple(t1["classes"]) == (2, 3), (
        f"task 1 should hold labels [2, 3] with classes (2, 3); got {t1['y'].tolist()}, {t1['classes']}")
    assert tuple(t0["classes"]) == (0, 1), f"task 0's classes should be (0, 1), got {t0['classes']}"


def test_stage1_permuted_shares_permutations(nb):
    d = 12
    Xa = np.tile(np.arange(d, dtype=float), (5, 1))       # column i holds the value i
    Xb = 2 * np.tile(np.arange(d, dtype=float), (3, 1))   # a "test set" of a different size
    ya, yb = np.array([3, 1, 4, 1, 5]), np.array([9, 2, 6])
    A = nb.permuted_tasks(Xa, ya, 4, seed=7)
    B = nb.permuted_tasks(Xb, yb, 4, seed=7)
    assert len(A) == 4 and len(B) == 4, f"expected 4 tasks, got {len(A)} and {len(B)}"
    assert torch.equal(A[0]["X"], torch.as_tensor(Xa, dtype=torch.float32)), "task 0 should be the data unpermuted"
    perms = []
    for k in range(4):
        pa = A[k]["X"][0].numpy().astype(int)
        assert sorted(pa.tolist()) == list(range(d)), f"task {k} must permute the columns, got {pa.tolist()}"
        assert torch.equal(B[k]["X"], 2 * torch.as_tensor(np.tile(pa, (3, 1)), dtype=torch.float32)), (
            f"task {k}: the same seed must give the same permutation for every dataset (train and test "
            "have to be permuted identically, or the test task is a different task). draw the permutations "
            "from the seed alone, not from the data or a global random state.")
        perms.append(tuple(pa))
    assert len(set(perms)) == 4, "the four tasks should use four different permutations"
    assert A[2]["y"].tolist() == ya.tolist() and A[2]["y"].dtype == torch.int64, "labels are unchanged, int64"
    assert tuple(A[1]["classes"]) == (1, 3, 4, 5), (
        f"classes should be the sorted distinct labels (1, 3, 4, 5), got {A[1]['classes']}")
    D = nb.permuted_tasks(Xa, ya, 6, seed=7)
    assert all(torch.equal(D[k]["X"], A[k]["X"]) for k in range(4)), (
        "permutation k must depend only on (seed, k): asking for 6 tasks instead of 4 must not change the first 4")
    C = nb.permuted_tasks(Xa, ya, 4, seed=8)
    assert not torch.equal(C[1]["X"], A[1]["X"]), "a different seed should give different permutations"


# ---------------------------------------------------------------- stage 2: metrics

R_HAND = np.array([[0.90, 0.20, 0.20],
                   [0.95, 0.80, 0.30],
                   [0.50, 0.70, 0.95]])
B_HAND = np.array([0.10, 0.15, 0.25])


def test_stage2_hand_example(nb):
    cases = [
        ("average_accuracy", nb.average_accuracy(R_HAND), 0.716666667,
         "the mean of the last row: (0.5 + 0.7 + 0.95) / 3. not the mean of the whole matrix."),
        ("backward_transfer", nb.backward_transfer(R_HAND), -0.25,
         "((0.5 - 0.9) + (0.7 - 0.8)) / 2: final accuracy minus the accuracy right after learning each task "
         "(the diagonal), over every task but the last."),
        ("forward_transfer", nb.forward_transfer(R_HAND, B_HAND), 0.05,
         "((R[0,1] - b[1]) + (R[1,2] - b[2])) / 2 = ((0.2 - 0.15) + (0.3 - 0.25)) / 2: accuracy on task j "
         "just *before* training on it (row j-1), minus the untrained baseline."),
        ("forgetting", nb.forgetting(R_HAND), 0.275,
         "((0.95 - 0.5) + (0.8 - 0.7)) / 2: the drop from the *best* earlier accuracy (task 0 peaked at 0.95 "
         "after task 1, not at 0.9) to the final one."),
    ]
    for name, got, want, why in cases:
        assert isinstance(got, float), f"{name} should return a python float, got {type(got).__name__}"
        assert abs(got - want) < 1e-6, f"{name} should be {want}, got {got}. {why}"


def test_stage2_edge_cases(nb):
    R2 = np.array([[0.5, 0.0], [0.7, 0.9]])
    got = nb.forgetting(R2)
    assert abs(got - (-0.2)) < 1e-9, (
        f"forgetting of [[0.5, 0], [0.7, 0.9]] is 0.5 - 0.7 = -0.2 (the max runs over rows before the "
        f"last; a task that improved has negative forgetting), got {got}")
    assert abs(nb.backward_transfer(R2.tolist()) - 0.2) < 1e-9, "backward_transfer should accept nested lists"
    for name, args in (("backward_transfer", ()), ("forgetting", ()), ("forward_transfer", ([0.1],))):
        with pytest.raises(ValueError):
            getattr(nb, name)(np.array([[0.9]]), *args)
    assert abs(nb.average_accuracy(np.array([[0.9]])) - 0.9) < 1e-12, "with one task, ACC is that task's accuracy"


# ---------------------------------------------------------------- stage 3: scenarios and evaluation


def test_stage3_scenario_logits(nb):
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0], [4.0, 0.0, -1.0, 2.0]], requires_grad=True)
    same = nb.scenario_logits(logits, (2, 3), "class")
    assert torch.equal(same, logits), "scenario 'class' returns the logits unchanged"
    masked = nb.scenario_logits(logits, (2, 3), "task")
    assert tuple(masked.shape) == (2, 4), f"masking keeps the shape (2, 4), got {tuple(masked.shape)}"
    assert torch.isinf(masked[:, :2]).all() and (masked[:, :2] < 0).all(), (
        "scenario 'task': columns outside `classes` must be -inf, so softmax gives them probability 0")
    assert torch.equal(masked[:, 2:], logits[:, 2:]), "columns in `classes` keep their values"
    assert masked.argmax(1).tolist() == [3, 3], "after masking, argmax picks among classes 2 and 3 only"
    F.cross_entropy(masked, torch.tensor([2, 3])).backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad[:, :2] == 0).all(), (
        "the masked loss must be differentiable, with zero gradient for masked columns. use masked_fill "
        "(not in-place assignment on the input)")
    assert logits[0, 0].item() == 1.0, "don't modify the input logits in place"
    with pytest.raises(ValueError):
        nb.scenario_logits(logits, (2, 3), "domain")


def test_stage3_evaluate(nb):
    model = nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.0], [0.0, 0.9]]))
    # true classes are 2 and 3, but classes 0 and 1 always score slightly higher
    X = [[1, 0], [0, 1], [2, 0], [0, 2], [1, 0.2], [0.3, 1], [1, 0], [0, 1], [3, 0], [0, 1]]
    y = [2, 3, 2, 3, 2, 3, 2, 3, 3, 3]
    task = _task(X, y, (2, 3))
    model.train()
    acc_c = nb.evaluate(model, task, scenario="class")
    assert isinstance(acc_c, float), f"evaluate should return a python float, got {type(acc_c).__name__}"
    assert acc_c == 0.0, f"over all 4 classes the model always picks class 0 or 1: accuracy 0.0, got {acc_c}"
    acc_t = nb.evaluate(model, task, scenario="task")
    assert abs(acc_t - 0.9) < 1e-12, (
        f"restricted to classes (2, 3) the model gets 9 of 10 right (the [3, 0] example labelled 3 is wrong), "
        f"got {acc_t}")
    assert model.training, "evaluate must restore train mode afterwards (it switches to eval mode inside)"
    model.eval()
    acc_b = nb.evaluate(model, task, scenario="task", batch_size=3)
    assert abs(acc_b - 0.9) < 1e-12, (
        f"with batch_size=3 the accuracy should still be 0.9, got {acc_b}: don't drop the last partial batch, "
        "and divide by the number of examples, not batches")
    assert not model.training, "evaluate must leave a model in eval mode in eval mode"
    assert model.weight.grad is None, "evaluate must not compute gradients"


# ---------------------------------------------------------------- stage 4: training on one task


def _expected_step(model, x, y, lr, classes=None, extra=None):
    m = _linear(model.in_features, model.out_features)
    m.load_state_dict(model.state_dict())
    out = m(x)
    if classes is not None:
        mask = torch.ones(out.shape[1], dtype=torch.bool)
        mask[list(classes)] = False
        out = out.masked_fill(mask, float("-inf"))
    loss = F.cross_entropy(out, y)
    if extra is not None:
        loss = loss + extra(m)
    loss.backward()
    with torch.no_grad():
        return m.weight - lr * m.weight.grad, m.bias - lr * m.bias.grad


def test_stage4_one_exact_sgd_step(nb):
    task = _blobs(5, (0, 1, 2, 3), seed=1)
    for scenario, classes in (("class", None), ("task", (0, 1, 2, 3))):
        task2 = dict(task, classes=(0, 1, 2, 3))
        model = _linear(5, 6, seed=2)
        w, b = _expected_step(model, task2["X"], task2["y"], 0.3, classes=classes)
        losses = _train(nb, model, task2, scenario=scenario, epochs=1, lr=0.3, batch_size=64, seed=0)
        assert torch.allclose(model.weight, w, atol=1e-6) and torch.allclose(model.bias, b, atol=1e-6), (
            f"scenario {scenario!r}: 5 examples with batch_size 64 is one full-batch step, so the weights should "
            "be W - lr * grad of the mean cross-entropy. use plain SGD (no momentum, no weight decay) and "
            "the mean (not the sum) over the batch")
        assert isinstance(losses, list) and len(losses) == 1, f"one step should return a list of one loss, got {losses}"
    # two full-batch steps (epochs=2): plain SGD, so the second step uses only the new gradient
    model = _linear(5, 6, seed=2)
    ref = _linear(5, 6, seed=2)
    for _ in range(2):
        w, b = _expected_step(ref, task["X"], task["y"], 0.3)
        with torch.no_grad():
            ref.weight.copy_(w)
            ref.bias.copy_(b)
    _train(nb, model, task, epochs=2, lr=0.3)
    assert torch.allclose(model.weight, ref.weight, atol=1e-6), (
        "two epochs of one full batch should be two plain SGD steps; the second step differs, so check for "
        "momentum (use torch.optim.SGD with its defaults) or an optimizer that persists in some other way")
    # task scenario: masking changes the update
    task3 = dict(task, classes=(0, 1, 2, 3))
    model = _linear(5, 6, seed=2)
    w, _ = _expected_step(model, task3["X"], task3["y"], 0.3, classes=(0, 1, 2, 3))
    model_u = _linear(5, 6, seed=2)
    w_u, _ = _expected_step(model_u, task3["X"], task3["y"], 0.3)
    _train(nb, model, task3, scenario="task", lr=0.3)
    assert not torch.allclose(w, w_u) and torch.allclose(model.weight, w, atol=1e-6), (
        "scenario 'task': train on scenario_logits (classes 4 and 5 masked), so their rows get no gradient")
    assert torch.allclose(model.weight[4:], _linear(5, 6, seed=2).weight[4:]), (
        "with scenario 'task', the output rows of classes outside the task must not change")


def test_stage4_steps_seeds_and_learning(nb):
    task = _blobs(10, (0, 1), seed=3)
    model = _linear(5, 2)
    losses = _train(nb, model, task, epochs=2, batch_size=4, lr=0.1)
    assert len(losses) == 6, (
        f"10 examples, batch_size 4, 2 epochs: 3 steps per epoch (4 + 4 + 2) = 6 losses; got {len(losses)}. "
        "include the last partial batch")
    assert all(isinstance(v, float) for v in losses), "losses should be python floats (use loss.item())"
    big = _blobs(40, (0, 1), seed=4)
    m1, m2, m3 = _linear(5, 2), _linear(5, 2), _linear(5, 2)
    _train(nb, m1, big, batch_size=4, seed=11)
    _train(nb, m2, big, batch_size=4, seed=11)
    _train(nb, m3, big, batch_size=4, seed=12)
    assert torch.equal(m1.weight, m2.weight), "the same seed must give the same shuffling and the same result"
    assert not torch.equal(m1.weight, m3.weight), "a different seed should shuffle differently"
    train, test = _blobs(400, (0, 1, 2), seed=5), _blobs(200, (0, 1, 2), seed=6)
    m = _linear(5, 3)
    before = nb.evaluate(m, test)  # noqa: F841 (shown in the message)
    hist = _train(nb, m, train, epochs=3, lr=0.1)
    after = nb.evaluate(m, test)
    assert np.mean(hist[-7:]) < np.mean(hist[:7]), "the training loss should go down over 3 epochs"
    assert after > 0.8, (
        f"3 epochs on well-separated blobs should reach > 80% test accuracy: {before:.2f} -> {after:.2f}. "
        "is the optimizer stepping, and the loss computed on the model's output?")


def test_stage4_hooks(nb):
    task = _blobs(5, (0, 1, 2), seed=7)
    model = _linear(5, 3)
    pen = lambda m: 0.5 * (m.weight ** 2).sum()  # noqa: E731
    w, _ = _expected_step(model, task["X"], task["y"], 0.2, extra=pen)
    _train(nb, model, task, lr=0.2, penalty=pen)
    assert torch.allclose(model.weight, w, atol=1e-6), "penalty(model) must be added to the loss before backward"

    seen = []
    big = _blobs(10, (0, 1, 2), seed=8)
    model = _linear(5, 3)
    _train(nb, model, big, batch_size=4, after_step=lambda m: seen.append(m.weight.detach().clone()))
    assert len(seen) == 3, f"after_step should be called once per optimizer step (3 here), got {len(seen)} calls"
    assert torch.equal(seen[-1], model.weight), "after_step must run *after* optimizer.step(), seeing the new weights"

    xr, yr = torch.randn(3, 5, generator=torch.Generator().manual_seed(9)), torch.tensor([2, 0, 1])
    model = _linear(5, 3)
    w, _ = _expected_step(model, torch.cat([task["X"], xr]), torch.cat([task["y"], yr]), 0.2)
    _train(nb, model, task, lr=0.2, replay=lambda: (xr, yr))
    assert torch.allclose(model.weight, w, atol=1e-6), (
        "replay() returns (x, y) to append to the batch: the loss is the mean cross-entropy over the "
        "combined batch")
    model, ref = _linear(5, 3), _linear(5, 3)
    _train(nb, model, task, lr=0.2, replay=lambda: None)
    _train(nb, ref, task, lr=0.2)
    assert torch.equal(model.weight, ref.weight), "replay() returning None means no replay for that step"
    with pytest.raises(ValueError):
        _train(nb, _linear(5, 3), task, scenario="task", replay=lambda: (xr, yr))


def test_stage4_adam_option(nb):
    task = _blobs(12, (0, 1, 2), seed=12)
    model, ref = _linear(5, 3, seed=4), _linear(5, 3, seed=4)
    _train(nb, model, task, lr=0.01, batch_size=6, optimizer="adam", seed=3)
    opt = torch.optim.Adam(ref.parameters(), lr=0.01)
    order = torch.randperm(12, generator=torch.Generator().manual_seed(3))
    for i in (0, 6):
        idx = order[i:i + 6]
        loss = F.cross_entropy(ref(task["X"][idx]), task["y"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    # the batch order is yours, so compare only what doesn't depend on it: Adam's first steps move every
    # weight by about lr (its update is normalised), SGD's by lr * |gradient|
    moved = (model.weight - _linear(5, 3, seed=4).weight).abs()
    assert torch.allclose(moved.mean(), (ref.weight - _linear(5, 3, seed=4).weight).abs().mean(), rtol=0.3), (
        f"optimizer='adam' should use torch.optim.Adam(model.parameters(), lr=lr): after two steps with lr = 0.01 "
        f"the weights should have moved by about 0.02 each, got a mean move of {moved.mean():.4f}")
    with pytest.raises(ValueError):
        _train(nb, _linear(5, 3), task, optimizer="rmsprop")


# ---------------------------------------------------------------- stage 5: the sequence runner


def test_stage5_oracle_accuracy_matrix(nb):
    tasks = [_blobs(60, g, seed=10 + i) for i, g in enumerate([(0, 1), (2, 3), (4, 5)])]
    tests = [_blobs(50, g, seed=20 + i) for i, g in enumerate([(0, 1), (2, 3), (4, 5)])]
    rng = np.random.default_rng(0)
    weights = [torch.tensor(rng.normal(size=(6, 5)), dtype=torch.float32) for _ in range(3)]
    calls = []

    def oracle(model, task, t):
        calls.append((t, task is tasks[t]))
        with torch.no_grad():
            model.weight.copy_(weights[t])
            model.bias.zero_()

    model = _linear(5, 6, seed=3)
    init = _linear(5, 6, seed=3)
    for scenario in ("class", "task"):
        calls.clear()
        model.load_state_dict(init.state_dict())
        R, base = bounded(nb.run_sequence, model, tasks, tests, train_fn=oracle, scenario=scenario,
                          what="run_sequence")
        assert calls == [(0, True), (1, True), (2, True)], (
            f"train_fn should be called as train_fn(model, train_tasks[t], t) for t = 0, 1, 2; got {calls}")
        R = np.asarray(R)
        assert R.shape == (3, 3), f"R should be (3, 3), got {R.shape}"
        want = np.zeros((3, 3))
        m = _linear(5, 6)
        for i in range(3):
            with torch.no_grad():
                m.weight.copy_(weights[i])
                m.bias.zero_()
            want[i] = [nb.evaluate(m, tt, scenario=scenario) for tt in tests]
        assert np.allclose(R, want), (
            f"scenario {scenario!r}: R[i, j] must be the accuracy on test task j after training task i, for *every* "
            f"j (including future tasks, which forward transfer needs). expected\n{want.round(3)}\ngot\n{R.round(3)}")
        want_b = [nb.evaluate(init, tt, scenario=scenario) for tt in tests]
        assert np.allclose(np.asarray(base), want_b), (
            f"baseline should be each test task's accuracy before any training: {np.round(want_b, 3)}, "
            f"got {np.round(base, 3)}")
    with pytest.raises(ValueError):
        nb.run_sequence(model, tasks, tests[:2], train_fn=oracle)


def test_stage5_default_training(nb):
    tasks = [_blobs(40, g, seed=30 + i) for i, g in enumerate([(0, 1), (2, 3)])]
    tests = [_blobs(30, g, seed=40 + i) for i, g in enumerate([(0, 1), (2, 3)])]
    a, b = _linear(5, 4, seed=5), _linear(5, 4, seed=5)
    R, _ = bounded(nb.run_sequence, a, tasks, tests, scenario="task", lr=0.2, batch_size=8, what="run_sequence")
    want = np.zeros((2, 2))
    for t, task in enumerate(tasks):
        nb.train_task(b, task, scenario="task", lr=0.2, batch_size=8, seed=t)
        want[t] = [nb.evaluate(b, tt, scenario="task") for tt in tests]
    assert np.allclose(np.asarray(R), want) and torch.equal(a.weight, b.weight), (
        "without train_fn, run_sequence trains task t with train_task(model, task, scenario=scenario, seed=t, "
        "**train_kwargs): extra keyword arguments (lr, batch_size, ...) go to train_task")


class ModeRecorder(nn.Module):
    """a linear model with dropout that records the train/eval mode of every forward pass."""

    def __init__(self, d, c, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(d, c)
        self.drop = nn.Dropout(0.5)
        self.modes = []

    def forward(self, x):
        self.modes.append(self.training)
        return self.lin(self.drop(x))


def test_stage3_evaluate_uses_eval_mode(nb):
    task = _blobs(200, (0, 1, 2), seed=13)
    model = ModeRecorder(5, 3)
    model.train()
    got = nb.evaluate(model, task)
    with torch.no_grad():
        want = float((model.lin(task["X"]).argmax(1) == task["y"]).float().mean())
    assert model.modes and not any(model.modes), (
        "evaluate must call model.eval() before the forward passes (dropout must be off at test time)")
    assert abs(got - want) < 1e-6, f"with dropout off the accuracy is {want:.3f}, got {got:.3f}"
    assert model.training, "and restore train mode afterwards"


def test_stage4_train_mode_and_shuffling(nb):
    task = _blobs(12, (0, 1, 2), seed=14)
    model = ModeRecorder(5, 3)
    model.eval()
    _train(nb, model, task, lr=0.1, batch_size=4)
    assert model.modes and all(model.modes), "train_task must call model.train() before training (dropout on)"
    # exact order: one torch.Generator per call, torch.randperm(n, generator=g) at the start of every epoch
    m, ref = _linear(5, 3, seed=6), _linear(5, 3, seed=6)
    _train(nb, m, task, epochs=2, lr=0.2, batch_size=4, seed=21)
    opt = torch.optim.SGD(ref.parameters(), lr=0.2)
    g = torch.Generator().manual_seed(21)
    for _ in range(2):
        order = torch.randperm(12, generator=g)
        for i in range(0, 12, 4):
            idx = order[i:i + 4]
            loss = F.cross_entropy(ref(task["X"][idx]), task["y"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert torch.allclose(m.weight, ref.weight, atol=1e-6), (
        "2 epochs with batch_size 4 should follow the spec's order exactly: create g = torch.Generator()."
        "manual_seed(seed) once per call, and draw order = torch.randperm(n, generator=g) at the start of each "
        "epoch (re-creating the generator every epoch repeats the same order)")


def test_stage4_fresh_optimizer_each_call(nb):
    task = _blobs(10, (0, 1, 2), seed=15)
    model, ref = _linear(5, 3, seed=7), _linear(5, 3, seed=7)
    for _ in range(2):
        _train(nb, model, task, lr=0.05, batch_size=64, optimizer="adam")
        opt = torch.optim.Adam(ref.parameters(), lr=0.05)  # a new optimizer for every task
        loss = F.cross_entropy(ref(task["X"]), task["y"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.allclose(model.weight, ref.weight, atol=1e-6), (
        "two calls of train_task (one full-batch Adam step each) should equal two steps of two *fresh* Adam "
        "optimizers. build the optimizer inside train_task on every call; don't keep Adam's moment estimates "
        "from the previous task")
