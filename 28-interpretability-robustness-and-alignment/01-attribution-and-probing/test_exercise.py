"""Checks for 28 · 01 attribution & probing.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. tiny torch models written here
stand in for a classifier. saliency against the analytic gradient of linear and tanh models (per-row targets,
absolute value, eval mode, parameter gradients untouched); integrated gradients against an explicit trapezoid
rule (n_steps and the baseline both matter), exactness on linear models and completeness on nonlinear ones;
control labels for consistency per type, uniformity and the rng; probe selectivity on features that memorise
types versus features that encode the property; the sanity-check tools against scipy and by what they change.
"""
import numpy as np
import pytest
from scipy import stats as sps

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _mlp(d=6, h=16, k=4, seed=0, dropout=0.0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Dropout(dropout), nn.Linear(h, k))


def _x(B=5, d=6, seed=1):
    return torch.randn(B, d, generator=torch.Generator().manual_seed(seed))


def _grad_of(model, x, targets):
    """reference gradient of model(x)[b, targets[b]] w.r.t. x[b], one row at a time."""
    rows = []
    was = model.training
    model.eval()
    for b in range(x.shape[0]):
        xb = x[b:b + 1].clone().requires_grad_(True)
        model(xb)[0, int(targets[b])].backward()
        rows.append(xb.grad[0])
    model.zero_grad(set_to_none=True)
    model.train(was)
    return torch.stack(rows)


class _Cubic(nn.Module):
    """f_c(x) = sum_i a_c,i x_i^3 + w_c . x : the gradient along a straight path is quadratic in alpha."""

    def __init__(self, d=4, k=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = nn.Parameter(torch.randn(k, d, generator=g))
        self.w = nn.Parameter(torch.randn(k, d, generator=g))

    def forward(self, x):
        return (x ** 3) @ self.a.T + x @ self.w.T

    def grad(self, x, c):
        return 3 * self.a[c] * x ** 2 + self.w[c]


class _Worked(nn.Module):
    """the lesson's worked example: one logit f(x) = x1^2 + x1 x2."""

    def forward(self, x):
        return (x[:, 0] ** 2 + x[:, 0] * x[:, 1])[:, None]


# ---------------------------------------------------------------- stage 1: saliency


def test_stage1_worked_example(nb):
    got = nb.saliency(_Worked(), torch.tensor([[2.0, 1.0]]), 0)
    assert torch.allclose(got, torch.tensor([[5.0, 2.0]])), (
        f"f = x1^2 + x1 x2 at (2, 1): gradient (2 x1 + x2, x1) = (5, 2); got {got.tolist()}")


def test_stage1_linear_model(nb):
    torch.manual_seed(0)
    lin = nn.Linear(6, 4)
    x = _x()
    got = nb.saliency(lin, x, 2)
    assert got.shape == x.shape, f"saliency has the shape of x {tuple(x.shape)}; got {tuple(got.shape)}"
    want = lin.weight[2].detach().abs().expand_as(x)
    assert torch.allclose(got, want, atol=1e-6), (
        "for a linear model f(x) = W x + b the gradient of logit c is W[c], whatever x is: saliency = |W[c]|. "
        "if you got signed values, take the absolute value; if every row is the same sum, you differentiated the "
        "sum of all logits")
    tg = torch.tensor([0, 3, 1, 1, 2])
    got = nb.saliency(lin, x, tg)
    assert torch.allclose(got, lin.weight[tg].detach().abs(), atol=1e-6), (
        "a (B,) target picks a different logit for each row: row b differentiates model(x)[b, target[b]] only "
        "(out[:, target] would mix every row's target into every row)")


def test_stage1_nonlinear_and_batch_independent(nb):
    m = _mlp()
    x = _x(B=6)
    tg = torch.tensor([0, 1, 2, 3, 0, 1])
    got = nb.saliency(m, x, tg)
    want = _grad_of(m, x, tg).abs()
    assert torch.allclose(got, want, atol=1e-5), "saliency of a tanh MLP should match autograd row by row"
    assert torch.allclose(nb.saliency(m, x[:2], tg[:2]), want[:2], atol=1e-5), "rows are independent of each other"
    assert all(p.grad is None for p in m.parameters()), (
        "saliency must not accumulate gradients into the model's parameters (use torch.autograd.grad(out, x), "
        "or zero them): a later optimizer step would use them")
    assert x.grad is None and not got.requires_grad, "x itself is not modified; the result is a plain tensor"
    with torch.no_grad():
        inside = nb.saliency(m, x, tg)
    assert torch.allclose(inside, want, atol=1e-5), (
        "saliency also works when called inside torch.no_grad() (wrap the gradient part in torch.enable_grad())")


def test_stage1_eval_mode_restored(nb):
    m = _mlp(dropout=0.5)
    m.train()
    x = _x()
    a, b = nb.saliency(m, x, 1), nb.saliency(m, x, 1)
    assert torch.allclose(a, b), "with dropout, two calls disagree: run the model in eval mode"
    assert torch.allclose(a, _grad_of(m, x, [1] * 5).abs(), atol=1e-5), "eval mode: dropout off"
    assert m.training, "put the model back in the mode you found it in (it was in train mode)"
    m.eval()
    nb.saliency(m, x, 1)
    assert not m.training, "a model that came in eval mode leaves in eval mode"


# ---------------------------------------------------------------- stage 2: integrated gradients


def _ig(nb, *args, **kw):
    return bounded(nb.integrated_gradients, *args, budget=3_000_000, what="integrated_gradients",
                   hint="the path has exactly n_steps + 1 points.", **kw)


def _trapezoid_ig(model, x, base, c, n):
    total = torch.zeros_like(x)
    for k in range(n + 1):
        w = (0.5 if k in (0, n) else 1.0) / n
        total += w * model.grad(base + k / n * (x - base), c)
    return (x - base) * total


def test_stage2_worked_example(nb):
    x = torch.tensor([[2.0, 1.0]])
    for n in (1, 2, 7):
        got = _ig(nb, _Worked(), x, 0, n_steps=n)
        assert torch.allclose(got, torch.tensor([[5.0, 1.0]]), atol=1e-5), (
            f"the worked example: IG = (5, 1) (x1 gets all of x1^2 = 4 and half of x1 x2 = 2), exact for any n_steps "
            f"because the gradient is linear along the path; got {got.tolist()} at n_steps={n}. a left Riemann sum "
            "gives (2.5, 0.5) at n_steps=2")


def test_stage2_trapezoid_rule(nb):
    m = _Cubic()
    x = torch.tensor([[1.0, -2.0, 0.5, 1.5], [0.3, 0.8, -1.2, 2.0]])
    with torch.no_grad():
        want = {n: torch.stack([_trapezoid_ig(m, x[i], torch.zeros(4), c, n) for i, c in enumerate([0, 2])])
                for n in (1, 2, 5)}
    for n in (1, 2, 5):
        got = _ig(nb, m, x, torch.tensor([0, 2]), n_steps=n)
        assert got.shape == x.shape and torch.allclose(got, want[n], atol=1e-5), (
            f"n_steps={n}: expected the trapezoid rule over alpha = 0, 1/{n}, ..., 1 (end points weighted 1/2): "
            f"{want[n][0].tolist()}, got {got[0].tolist()}. a left or right Riemann sum, or the midpoint rule, "
            "gives different numbers; so does ignoring n_steps")


def test_stage2_baseline(nb):
    m = _Cubic(seed=3)
    x = torch.tensor([[1.0, -2.0, 0.5, 1.5]])
    base = torch.tensor([0.5, 0.5, -0.5, 1.0])
    with torch.no_grad():
        want = _trapezoid_ig(m, x[0], base, 1, 4)[None]
    got = _ig(nb, m, x, 1, baseline=base, n_steps=4)
    assert torch.allclose(got, want, atol=1e-5), (
        "with a baseline b the path runs from b to x and the attribution is (x - b) * (average gradient): "
        f"expected {want[0].tolist()}, got {got[0].tolist()}")
    got_b = _ig(nb, m, x, 1, baseline=base[None], n_steps=4)
    assert torch.allclose(got_b, want, atol=1e-5), "a (B, ...) baseline works the same as a broadcastable one"
    zero = _ig(nb, m, x, 1, n_steps=4)
    assert torch.allclose(zero, _ig(nb, m, x, 1, baseline=torch.zeros(4), n_steps=4), atol=1e-6), (
        "the default baseline is zeros")
    assert torch.allclose(_ig(nb, m, x, 1, baseline=x, n_steps=4), torch.zeros_like(x)), (
        "baseline = x: the path has length zero, every attribution is 0")


def test_stage2_linear_exact_and_completeness(nb):
    torch.manual_seed(0)
    lin = nn.Linear(6, 4)
    x = _x()
    got = _ig(nb, lin, x, 3, n_steps=1)
    assert torch.allclose(got, x * lin.weight[3].detach(), atol=1e-6), (
        "for a linear model IG with a zero baseline is x * W[c] exactly, for any n_steps (signed, not |.|)")
    m = _mlp(seed=2)
    x, base = _x(B=4, seed=5), _x(B=1, seed=6)
    tg = torch.tensor([0, 1, 2, 3])
    attr = _ig(nb, m, x, tg, baseline=base, n_steps=200)
    with torch.no_grad():
        gap = m(x).gather(1, tg[:, None])[:, 0] - m(base.expand_as(x)).gather(1, tg[:, None])[:, 0]
    assert torch.allclose(attr.sum(1), gap, atol=2e-3 * (1 + gap.abs().max())), (
        f"completeness: each row's attributions sum to f(x) - f(baseline) = {gap.tolist()}; got "
        f"{attr.sum(1).tolist()}")
    assert all(p.grad is None for p in m.parameters()), "IG doesn't accumulate parameter gradients"
    with torch.no_grad():
        inside = _ig(nb, m, x, tg, baseline=base, n_steps=5)
    assert torch.allclose(inside, _ig(nb, m, x, tg, baseline=base, n_steps=5), atol=1e-6), (
        "IG also works inside torch.no_grad() (torch.enable_grad() around the gradient computation)")


def test_stage2_eval_mode_and_errors(nb):
    m = _mlp(dropout=0.5)
    m.train()
    x = _x(B=2)
    a, b = _ig(nb, m, x, 0, n_steps=3), _ig(nb, m, x, 0, n_steps=3)
    assert torch.allclose(a, b), "IG must run the model in eval mode (dropout off)"
    assert m.training, "and restore train mode afterwards"
    with pytest.raises(ValueError):
        nb.integrated_gradients(m, x, 0, n_steps=0)


# ---------------------------------------------------------------- stage 3: probes and control tasks


def test_stage3_control_labels(nb):
    types = np.array([3, 7, 3, 3, 9, 7, 1, 9])
    lab = nb.control_labels(types, 4, rng=0)
    assert lab.shape == (8,) and lab.dtype.kind == "i", "one integer label per token"
    for t in np.unique(types):
        assert len(set(lab[types == t])) == 1, (
            f"every occurrence of type {t} gets the same control label (it's a random function of the type, "
            "not a random label per token)")
    assert np.array_equal(lab, nb.control_labels(types, 4, rng=0)), "the same rng seed gives the same labels"
    many = np.arange(4000)
    draws = [nb.control_labels(many, 4, rng=s) for s in range(3)]
    assert not np.array_equal(draws[0], draws[1]), "the labels come from rng: different seeds, different tables"
    counts = np.bincount(draws[2], minlength=4)
    chi2 = ((counts - 1000) ** 2 / 1000).sum()
    assert counts.size == 4 and chi2 < 25, (
        f"labels should be uniform over range(n_classes) = 0..3; counts over 4000 types {counts.tolist()}. "
        "rng.integers(0, n_classes) excludes n_classes itself already")
    words = np.array(["the", "cat", "the", "sat"])
    wl = nb.control_labels(words, 3, rng=1)
    assert wl[0] == wl[2], "types can be any hashable values, e.g. words"


def _type_data(seed, n_types=30, per_type=12, mode="memorise"):
    rng = np.random.default_rng(seed)
    types = np.repeat(np.arange(n_types), per_type)
    prop = rng.integers(0, 2, size=n_types)  # a binary property of each type
    if mode == "memorise":  # features identify the type, nothing else
        feats = np.eye(n_types)[types] + 0.05 * rng.normal(size=(len(types), n_types))
    else:  # features encode only the property, plus noise
        feats = np.outer(2 * prop[types] - 1, rng.normal(size=8)) + 0.3 * rng.normal(size=(len(types), 8))
    labels = prop[types]
    test = rng.random(len(types)) < 0.3
    return feats[~test], labels[~test], feats[test], labels[test], types[~test], types[test]


def test_stage3_selectivity(nb):
    mem = nb.probe_selectivity(*_type_data(0, mode="memorise"), rng=0)
    assert set(mem) >= {"task", "control", "selectivity"}, "return a dict with task, control and selectivity"
    assert np.isclose(mem["selectivity"], mem["task"] - mem["control"]), "selectivity = task - control"
    assert mem["task"] > 0.9 and mem["control"] > 0.85, (
        f"features that just identify the type can be probed for ANY function of the type, the property and the "
        f"control task alike: got task {mem['task']:.2f}, control {mem['control']:.2f}. if control is near chance, "
        "you drew separate control tables for the train and test splits: one table, shared by both")
    assert mem["selectivity"] < 0.15, f"so their selectivity is near 0; got {mem['selectivity']:.2f}"
    enc = nb.probe_selectivity(*_type_data(1, n_types=120, per_type=5, mode="encode"), rng=0)
    assert enc["task"] > 0.9 and enc["control"] < 0.75 and enc["selectivity"] > 0.25, (
        f"features that encode the property (and not the type) give a good task probe and a poor control probe: "
        f"got {enc}")


def test_stage3_options(nb):
    data = _type_data(2, mode="memorise")
    a = nb.probe_selectivity(*data, rng=5)
    assert a == nb.probe_selectivity(*data, rng=5), "the same rng gives the same result"
    rng = np.random.default_rng(7)
    enc = _type_data(2, mode="encode")
    controls = {nb.probe_selectivity(*enc, rng=rng)["control"] for _ in range(5)}
    assert len(controls) > 1, (
        "a Generator passed as rng keeps advancing: repeated calls draw new control tables (use mlfs.utils.as_rng)")


# ---------------------------------------------------------------- stage 4: sanity checks


def test_stage4_rank_correlation(nb):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, size=(6, 7)).astype(float)  # many ties
    b = a + rng.normal(size=a.shape)
    got = nb.rank_correlation(a, b)
    want = sps.spearmanr(a.ravel(), b.ravel()).statistic
    assert np.isclose(got, want, atol=1e-10), f"Spearman with average ranks for ties: {want:.6f}, got {got:.6f}"
    assert np.isclose(nb.rank_correlation(a, np.exp(a)), 1.0), "invariant to monotone transformations"
    assert np.isclose(nb.rank_correlation(b, -b), -1.0)
    assert isinstance(got, float)
    with pytest.raises(ValueError):
        nb.rank_correlation(np.zeros(3), np.zeros(4))


def test_stage4_randomize_layers(nb):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.Linear(8, 3))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    r1 = nb.randomize_layers(m, 1, generator=torch.Generator().manual_seed(0))
    assert r1 is not m and all(torch.equal(v, m.state_dict()[k]) for k, v in before.items()), (
        "return a deep copy and leave the original model untouched")
    sd = r1.state_dict()
    assert not torch.equal(sd["3.weight"], before["3.weight"]), "n=1 re-initialises the LAST layer (the output)"
    assert torch.equal(sd["0.weight"], before["0.weight"]) and torch.equal(sd["2.weight"], before["2.weight"]), (
        "n=1 leaves every other layer alone")
    r2 = nb.randomize_layers(m, 2, generator=torch.Generator().manual_seed(0)).state_dict()
    assert not torch.equal(r2["2.weight"], before["2.weight"]) and torch.equal(r2["0.weight"], before["0.weight"]), (
        "n=2: the last two parameterised layers (ReLU has no parameters and doesn't count)")
    again = nb.randomize_layers(m, 2, generator=torch.Generator().manual_seed(0)).state_dict()
    other = nb.randomize_layers(m, 2, generator=torch.Generator().manual_seed(1)).state_dict()
    assert all(torch.equal(again[k], r2[k]) for k in r2), "the same generator seed gives the same weights"
    assert not torch.equal(other["3.weight"], r2["3.weight"]), "a different seed gives different weights"
    r0 = nb.randomize_layers(m, 0)
    assert r0 is not m and all(torch.equal(r0.state_dict()[k], v) for k, v in before.items()), "n=0: an exact copy"
    with pytest.raises(ValueError):
        nb.randomize_layers(m, 4)


def test_stage4_cascading(nb):
    m = _mlp(seed=4)  # Linear, Tanh, Dropout, Linear: two parameterised layers
    x = _x(B=3)
    calls = []

    def ignores_model(model, x, target):
        calls.append(model)
        return x

    out = bounded(nb.cascading_randomization, m, ignores_model, x, 0, budget=2_000_000,
                  what="cascading_randomization", generator=torch.Generator().manual_seed(0))
    assert isinstance(out, np.ndarray) and out.shape == (2,), "one correlation per parameterised layer (L = 2)"
    assert np.allclose(out, 1.0), "an attribution that ignores the model is perfectly 'stable' (all 1.0)"
    assert len(calls) == 3 and calls[0] is m and all(c is not m for c in calls[1:]), (
        "attribute is called once on the model, then once per randomized copy (top 1, then top 2 layers)")
    sal = bounded(nb.cascading_randomization, m, nb.saliency, x, 0, budget=2_000_000, what="cascading_randomization",
                  generator=torch.Generator().manual_seed(0))
    assert np.all(np.abs(sal) <= 1) and sal[-1] < 0.9, (
        f"saliency of a fully re-initialised MLP should not rank the inputs like the original does; got {sal}")
    assert sal[0] < 0.99, (
        f"entry 0 is the model with its top layer re-initialised (k = 1), which changes the saliency map; got "
        f"{sal[0]:.3f}. a value of 1.0 means you compared the model with an unrandomized copy (k = 0)")
    m3 = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))
    original = {k: v.clone() for k, v in m3.state_dict().items()}
    seen = []

    def signed(model, xx, t):  # a signed attribution that depends on every layer
        changed = sorted({k.split(".")[0] for k, v in model.state_dict().items() if not torch.equal(v, original[k])})
        out = (xx * (model[0].weight.sum(0) + model[2].weight.sum() - model[4].weight.sum())).detach()
        seen.append((changed, out.numpy()))
        return out

    got = bounded(nb.cascading_randomization, m3, signed, _x(B=4), 0, budget=2_000_000,
                  what="cascading_randomization", generator=torch.Generator().manual_seed(1))
    assert [c for c, _ in seen] == [[], ["4"], ["2", "4"], ["0", "2", "4"]], (
        f"the k-th randomized copy re-initialises the LAST k parameterised layers, in order k = 1, 2, 3 (layers "
        f"changed in each call: {[c for c, _ in seen]})")
    want = [nb.rank_correlation(np.abs(seen[0][1]), np.abs(o)) for _, o in seen[1:]]
    assert np.allclose(got, want), (
        f"entry k-1 is the Spearman correlation of |attribution| for the model and for the k-th copy: "
        f"{np.round(want, 4)}; got {np.round(got, 4)}. correlate absolute values: |attribution| is what saliency "
        "maps show")
    state = torch.random.get_rng_state()
    nb.randomize_layers(m3, 3, generator=torch.Generator().manual_seed(2))
    assert torch.equal(state, torch.random.get_rng_state()), "randomize_layers leaves the global RNG as it was"
