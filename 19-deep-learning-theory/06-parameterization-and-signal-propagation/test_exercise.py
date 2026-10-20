"""Checks for 19 · 06 parameterization & signal propagation.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the Gaussian expectation
against moments known in closed form (and a too-small quadrature); the variance map, fixed point and
chi_1 against ReLU's closed forms and Monte Carlo for tanh; the simulated network against the mean-field
maps, in the ordered and the chaotic phase; the μP model for its initialization, forward pass, learning
rate groups (every parameter exactly once) and the width scaling of its coordinate check against SP.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

def _run(fn, *args, **kwargs):
    """every call that loops runs under a line budget, so a learner's endless loop fails instead of hanging."""
    return bounded(fn, *args, budget=20_000_000, what=getattr(fn, "__name__", None), **kwargs)



# ---------------------------------------------------------------- stage 1: the variance map


def test_stage1_gaussian_expectation(nb):
    for q in (0.5, 2.0):
        assert math.isclose(nb.gaussian_expectation(lambda u: u ** 2, q), q, rel_tol=1e-10), "E[(sqrt(q) z)^2] = q"
        assert math.isclose(nb.gaussian_expectation(lambda u: u ** 4, q), 3 * q * q, rel_tol=1e-10), (
            "E[(sqrt(q) z)^4] = 3 q^2: the expectation is over z ~ N(0, 1) (divide hermegauss's weights by sqrt(2 pi))")
        assert math.isclose(nb.gaussian_expectation(np.cos, q), math.exp(-q / 2), rel_tol=1e-8), "E[cos(sqrt(q) z)] = exp(-q/2)"
    rough = nb.gaussian_expectation(lambda u: u ** 4, 1.0, n_points=2)
    assert not math.isclose(rough, 3.0, rel_tol=1e-3), "with n_points=2 the quadrature can't be exact for u^4: use n_points"


def test_stage1_variance_map(nb):
    assert math.isclose(nb.variance_map(1.5, 1.3, 0.2, "relu"), 1.3 ** 2 * 1.5 / 2 + 0.04, rel_tol=1e-9), (
        "for ReLU, E[relu(u)^2] = q/2, so the map is sigma_w^2 q / 2 + sigma_b^2")
    rng = np.random.default_rng(0)
    z = rng.normal(size=2_000_000)
    mc = 1.7 ** 2 * np.mean(np.tanh(np.sqrt(0.8) * z) ** 2) + 0.1 ** 2
    assert math.isclose(nb.variance_map(0.8, 1.7, 0.1, "tanh"), mc, rel_tol=2e-3), (
        f"tanh variance map at q = 0.8, sigma_w = 1.7, sigma_b = 0.1 should be about {mc:.4f}")


# ---------------------------------------------------------------- stage 2: fixed point and chi_1


def test_stage2_fixed_point_and_chi1(nb):
    q = bounded(nb.q_fixed_point, 1.0, 0.5, "relu", what="q_fixed_point", budget=20_000_000)
    assert math.isclose(q, 0.5, rel_tol=1e-6), (
        f"ReLU with sigma_w = 1, sigma_b = 0.5: q* = sigma_b^2 / (1 - sigma_w^2 / 2) = 0.5; got {q}")
    q = _run(nb.q_fixed_point, 1.8, 0.3, "tanh")
    assert math.isclose(nb.variance_map(q, 1.8, 0.3, "tanh"), q, rel_tol=1e-8), "q* must satisfy q* = variance_map(q*)"
    assert math.isclose(nb.q_fixed_point(1.0, 0.5, "relu", q0=7.0, n_iter=1), 3.75, rel_tol=1e-9), (
        "q0 is the starting point and n_iter the number of applications of the map (one step from 7: 7/2 + 0.25)")
    assert math.isclose(nb.chi1(0.7, math.sqrt(2.0), "relu"), 1.0, rel_tol=1e-6), (
        "for ReLU, E[relu'(u)^2] = 1/2, so chi_1 = sigma_w^2 / 2: exactly 1 at He initialization, sigma_w^2 = 2")
    rng = np.random.default_rng(1)
    z = rng.normal(size=2_000_000)
    mc = 1.5 ** 2 * np.mean((1 - np.tanh(np.sqrt(0.4) * z) ** 2) ** 2)
    assert math.isclose(nb.chi1(0.4, 1.5, "tanh"), mc, rel_tol=2e-3), f"tanh chi_1 at q = 0.4, sigma_w = 1.5 is about {mc:.4f}"


# ---------------------------------------------------------------- stage 3: a real random network


def _inputs():
    """8 inputs of mean square 1; inputs 0 and 1 have correlation 0.6."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 100))
    X[1] = 0.6 * X[0] + 0.8 * X[1]
    return X / np.sqrt(np.mean(X ** 2, axis=1, keepdims=True))


def test_stage3_matches_mean_field(nb):
    x = _inputs()
    out = bounded(nb.propagate, x, 12, 2000, sigma_w=1.6, sigma_b=0.2, activation="tanh",
                  generator=torch.Generator().manual_seed(0), what="propagate")
    q, corr = np.asarray(out["q"]), np.asarray(out["corr"])
    assert q.shape == (12, 8) and corr.shape == (12,), f"q is (depth, n) = (12, 8), corr (depth,) = (12,); got {q.shape}, {corr.shape}"
    q_theory = [1.6 ** 2 + 0.04]
    for _ in range(11):
        q_theory.append(nb.variance_map(q_theory[-1], 1.6, 0.2, "tanh"))
    assert np.allclose(q.mean(axis=1), q_theory, rtol=0.06), (
        f"at width 2,000 the measured mean squared pre-activation should follow the variance map:\n"
        f"theory {np.round(q_theory, 3).tolist()}\nyours (mean of 8 inputs) {np.round(q.mean(axis=1), 3).tolist()}\n"
        "weights ~ N(0, sigma_w^2 / fan_in), biases ~ N(0, sigma_b^2), the first layer acts on x itself")
    qa = 1.6 ** 2 * np.mean(x[0] ** 2) + 0.04
    qb = 1.6 ** 2 * np.mean(x[1] ** 2) + 0.04
    c1 = (1.6 ** 2 * np.mean(x[0] * x[1]) + 0.04) / np.sqrt(qa * qb)
    assert abs(corr[0] - c1) < 0.05, (
        f"layer 1's correlation should be about {c1:.3f}, got {corr[0]:.3f}. "
        "corr is the cosine similarity of the two inputs' pre-activation vectors in each layer")


def test_stage3_ordered_and_chaotic(nb):
    x = _inputs()
    g = torch.Generator().manual_seed(1)
    ordered = _run(nb.propagate, x, 40, 500, sigma_w=0.8, sigma_b=0.3, activation="tanh", generator=g)
    chaotic = _run(nb.propagate, x, 40, 500, sigma_w=3.0, sigma_b=0.3, activation="tanh", generator=g)
    q_o, q_c = _run(nb.q_fixed_point, 0.8, 0.3, "tanh"), _run(nb.q_fixed_point, 3.0, 0.3, "tanh")
    assert nb.chi1(q_o, 0.8, "tanh") < 1 < nb.chi1(q_c, 3.0, "tanh"), "sigma_w = 0.8 is ordered (chi_1 < 1) and 3.0 chaotic"
    assert np.asarray(ordered["corr"])[-1] > 0.99, (
        f"in the ordered phase two inputs become indistinguishable: correlation after 40 layers should be ~1, got "
        f"{np.asarray(ordered['corr'])[-1]:.3f}")
    assert np.asarray(chaotic["corr"])[-1] < 0.6, (
        f"in the chaotic phase nearby inputs decorrelate: correlation after 40 layers should fall well below its start "
        f"(0.6), got {np.asarray(chaotic['corr'])[-1]:.3f}")
    r = _run(nb.propagate, x, 3, 50, sigma_w=1.0, sigma_b=0.0, activation="relu", generator=torch.Generator().manual_seed(2))
    r2 = _run(nb.propagate, x, 3, 50, sigma_w=1.0, sigma_b=0.0, activation="relu", generator=torch.Generator().manual_seed(2))
    r3 = _run(nb.propagate, x, 3, 50, sigma_w=1.0, sigma_b=0.0, activation="relu", generator=torch.Generator().manual_seed(3))
    assert np.allclose(r["q"], r2["q"]) and not np.allclose(r["q"], r3["q"]), "draw every weight with generator"
    he = np.asarray(nb.propagate(x, 4, 2000, sigma_w=math.sqrt(2.0), sigma_b=0.0, activation="relu",
                                 generator=torch.Generator().manual_seed(4))["q"]).mean(axis=1)
    assert 0.75 < he[-1] / he[0] < 1.3, (
        f"ReLU at He initialization (sigma_w^2 = 2, no bias) preserves the signal's size layer after layer; got "
        f"{np.round(he, 3).tolist()} (is activation='relu' really using relu?)")


def test_stage3_bias_and_activation_check(nb):
    x = _inputs()
    out = _run(nb.propagate, x, 3, 4000, sigma_w=0.0, sigma_b=0.5, activation="tanh",
               generator=torch.Generator().manual_seed(0))
    q, corr = np.asarray(out["q"]), np.asarray(out["corr"])
    assert np.allclose(q, 0.25, rtol=0.08), (
        f"with sigma_w = 0 the pre-activations are the biases alone, N(0, sigma_b^2): mean square 0.25 for sigma_b = "
        f"0.5; got {np.round(q[:, 0], 3).tolist()}. the bias std is sigma_b, not sigma_b^2")
    assert np.allclose(corr, 1.0), "with sigma_w = 0 every input gets the same bias vector: correlation 1"
    with pytest.raises(ValueError):
        nb.propagate(x, 2, 10, sigma_w=1.0, sigma_b=0.0, activation="sigmoid")
        raise AssertionError("propagate must raise ValueError for an unknown activation")


# ---------------------------------------------------------------- stage 4: μP


def test_stage4_mumlp_structure(nb):
    torch.manual_seed(0)
    for param in ("sp", "mup"):
        m = nb.MuMLP(30, 512, 7, depth=3, parameterization=param, base_width=64)
        assert len(m.hidden) == 3 and all(isinstance(l, nn.Linear) for l in m.hidden) and isinstance(m.readout, nn.Linear)
        assert (m.hidden[0].in_features, m.hidden[0].out_features, m.readout.in_features, m.readout.out_features) == (30, 512, 512, 7)
        assert m.width_mult == 8.0 and m.parameterization == param, "width_mult = width / base_width"
        for l in m.hidden:
            assert abs(l.weight.std().item() * math.sqrt(l.in_features) - 1) < 0.05 and torch.all(l.bias == 0), (
                "hidden weights ~ N(0, 1/fan_in), biases zero")
        if param == "mup":
            assert torch.all(m.readout.weight == 0), "μP zero-initializes the readout weights"
        else:
            assert abs(m.readout.weight.std().item() * math.sqrt(512) - 1) < 0.1, "SP readout ~ N(0, 1/fan_in)"
    with pytest.raises(ValueError):
        nb.MuMLP(3, 8, 2, parameterization="ntk")


def test_stage4_forward_and_activations(nb):
    torch.manual_seed(1)
    x = torch.randn(5, 6)
    for param, scale in (("sp", 1.0), ("mup", 64 / 256)):
        m = nb.MuMLP(6, 256, 3, depth=2, parameterization=param)
        with torch.no_grad():
            m.readout.weight.normal_()
            m.readout.bias.fill_(0.5)
            h1 = torch.relu(m.hidden[0](x))
            h2 = torch.relu(m.hidden[1](h1))
            out = (h2 * scale) @ m.readout.weight.T + m.readout.bias
            acts = m.activations(x)
        assert len(acts) == 3 and torch.allclose(acts[0], h1) and torch.allclose(acts[1], h2), (
            "activations returns the hidden ReLU outputs, then the output")
        assert torch.allclose(acts[2], out, atol=1e-5) and torch.allclose(m(x), out, atol=1e-5), (
            f"{param}: output = readout(h * {scale}): μP scales the readout's INPUT by base_width / width (so its bias "
            "isn't scaled); SP doesn't scale")


def test_stage4_param_groups(nb):
    m = nb.MuMLP(10, 256, 4, depth=3, parameterization="mup", base_width=64)
    groups = nb.param_groups(m, 0.01)
    lr_of = {}
    for g in groups:
        for p in g["params"]:
            assert id(p) not in lr_of, "every parameter appears in exactly one group"
            lr_of[id(p)] = g["lr"]
    assert set(lr_of) == {id(p) for p in m.parameters()}, "every parameter of the model must be in some group"
    for i, l in enumerate(m.hidden):
        want = 0.01 if i == 0 else 0.01 / 4
        assert math.isclose(lr_of[id(l.weight)], want), (
            f"hidden[{i}].weight gets lr {want} (input layer: lr; width-by-width layers: lr / width_mult = lr / 4)")
        assert math.isclose(lr_of[id(l.bias)], 0.01), "biases are vector-like: lr"
    assert math.isclose(lr_of[id(m.readout.weight)], 0.01) and math.isclose(lr_of[id(m.readout.bias)], 0.01), (
        "the readout keeps lr (its width scaling lives in the forward multiplier)")
    sp = nb.MuMLP(10, 256, 4, depth=3, parameterization="sp")
    g = nb.param_groups(sp, 0.03)
    assert len(g) == 1 and g[0]["lr"] == 0.03 and len(list(g[0]["params"])) == len(list(sp.parameters())), (
        "standard parameterization: one group, every parameter at lr")


# ---------------------------------------------------------------- stage 5: the coordinate check


def _deltas(nb, param, width, seed, lr=0.01):
    torch.manual_seed(seed)
    x = torch.randn(64, 20)
    y = torch.randint(0, 5, (64,))
    m = nb.MuMLP(20, width, 5, depth=2, parameterization=param)
    return _run(nb.activation_deltas, m, x, y, lr=lr, steps=2)


def test_stage5_deltas_definition(nb):
    torch.manual_seed(0)
    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    m = nb.MuMLP(4, 32, 3, depth=2, parameterization="sp")
    m.eval()
    ref = nb.MuMLP(4, 32, 3, depth=2, parameterization="sp")
    ref.load_state_dict(m.state_dict())
    with torch.no_grad():
        before = [a.clone() for a in ref.activations(x)]
    opt = torch.optim.Adam(ref.parameters(), lr=0.05)
    for _ in range(3):
        loss = F.cross_entropy(ref(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        want = [float(((a - b) ** 2).mean().sqrt()) for a, b in zip(ref.activations(x), before)]
    got = _run(nb.activation_deltas, m, x, y, lr=0.05, steps=3)
    assert len(got) == 3 and np.allclose(got, want, rtol=1e-4, atol=1e-7), (
        f"3 full-batch Adam steps (param_groups, cross-entropy) then the RMS change of each activation: expected "
        f"{np.round(want, 5).tolist()}, got {np.round(got, 5).tolist()}")
    assert torch.allclose(m(x), ref(x), atol=1e-5), "the model is trained in place"
    zero = _run(nb.activation_deltas, nb.MuMLP(4, 32, 3), x, y, lr=0.0, steps=2)
    assert max(zero) == 0.0, "lr = 0 changes nothing"


def test_stage5_sp_blows_up_mup_does_not(nb):
    ratios = {}
    for param in ("sp", "mup"):
        small = np.median([_deltas(nb, param, 64, s) for s in range(3)], axis=0)
        large = np.median([_deltas(nb, param, 1024, s) for s in range(3)], axis=0)
        ratios[param] = large / small
    assert ratios["sp"][1] > 4 and ratios["sp"][2] > 4, (
        f"under SP with Adam, a step moves a width-1024 network's second hidden layer and output much more than a "
        f"width-64 one's (ratios {np.round(ratios['sp'], 2).tolist()})")
    assert np.all(ratios["mup"] > 0.4) and np.all(ratios["mup"] < 2.5), (
        f"under μP each layer's activations move by about the same amount at width 64 and 1024 (ratios "
        f"{np.round(ratios['mup'], 2).tolist()}); check the hidden learning rate lr / width_mult and the readout multiplier")
