"""Checks for 19 · 05 scaling laws.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the power law and local
exponents against hand values (chapter 18's capstone numbers); the fit against exact power laws, against
a brute-force log-space optimum on noisy data, and at the c >= 0 boundary; the compute rule against
Chinchilla's 70B model; the allocation against numerical minimization; the parameter count against
a model with chapter 18's GPT layout (tied, untied, no learned positions).
"""
import math

import numpy as np
import pytest
import scipy.optimize

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

def _run(fn, *args, **kwargs):
    """every call that loops runs under a line budget, so a learner's endless loop fails instead of hanging."""
    return bounded(fn, *args, budget=20_000_000, what=getattr(fn, "__name__", None), **kwargs)



def _log_sse(x, y, a, b, c):
    pred = a * np.asarray(x, float) ** (-b) + c
    return float(np.sum((np.log(y) - np.log(pred)) ** 2))


def _brute_fit(x, y):
    """profile over c on a fine grid, polish with Nelder-Mead: a reference optimum for the tests."""
    lx, ly = np.log(x), np.log(y)
    best = None
    for c in np.min(y) * (1 - np.geomspace(1, 1e-7, 2000)):
        s, i = np.polyfit(lx, np.log(y - c), 1)
        cand = (math.exp(i), -s, c)
        if best is None or _log_sse(x, y, *cand) < _log_sse(x, y, *best):
            best = cand
    return best


# ---------------------------------------------------------------- stage 1: the law and its local slope


def test_stage1_power_law_values(nb):
    assert math.isclose(float(nb.power_law(4.0, 2.0, 0.5)), 1.0), "power_law(4, a=2, b=0.5) = 2 * 4^-0.5 = 1 (c defaults to 0)"
    got = np.asarray(nb.power_law(np.array([1.0, 10.0, 100.0]), 3.0, 1.0, 0.5))
    assert got.shape == (3,) and np.allclose(got, [3.5, 0.8, 0.53]), (
        f"power_law([1, 10, 100], 3, 1, 0.5) = 3 x^-1 + 0.5 = [3.5, 0.8, 0.53]; got {got}. the exponent is -b")


def test_stage1_local_exponents_ch18(nb):
    n = np.array([354_240, 1_816_896, 7_460_160])
    bits = np.array([2.567, 2.190, 2.062])
    got = np.asarray(nb.local_exponents(n, bits))
    assert got.shape == (2,), f"one slope per pair of neighbours: shape (2,), got {got.shape}"
    assert np.allclose(got, [0.0972, 0.0426], atol=5e-4), (
        f"chapter 18's capstone: -log(2.190/2.567)/log(1816896/354240) = 0.097 and 0.043 for the next pair; got "
        f"{np.round(got, 4)}. slopes are positive when the loss falls")
    x = np.geomspace(10, 1e5, 5)
    got = np.asarray(nb.local_exponents(x, 3 * x ** -0.4 + 1.5, c=1.5))
    assert np.allclose(got, 0.4), f"with the right offset c, an exact power law has constant local exponent 0.4; got {got}"


# ---------------------------------------------------------------- stage 2: fitting


def test_stage2_no_offset_is_loglinear_least_squares(nb):
    x = np.geomspace(1e3, 1e7, 7)
    a, b, c = bounded(nb.fit_power_law, x, 40 * x ** -0.25, offset=False, what="fit_power_law")
    assert c == 0.0 and math.isclose(a, 40, rel_tol=1e-8) and math.isclose(b, 0.25, rel_tol=1e-8), (
        f"offset=False on y = 40 x^-0.25 must return (40, 0.25, 0.0); got {(a, b, c)}")
    rng = np.random.default_rng(0)
    y = 40 * x ** -0.25 * np.exp(rng.normal(0, 0.2, x.size)) + 2.0
    s, i = np.polyfit(np.log(x), np.log(y), 1)
    a, b, _ = _run(nb.fit_power_law, x, y, offset=False)
    assert math.isclose(a, math.exp(i), rel_tol=1e-8) and math.isclose(b, -s, rel_tol=1e-8), (
        "offset=False is ordinary least squares of log y on log x (a = exp(intercept), b = -slope); fitting in "
        f"linear space weights the large-y points more. expected {(math.exp(i), -s)}, got {(a, b)}")


@pytest.mark.parametrize("a0,b0,c0", [(5.0, 0.3, 1.2), (200.0, 0.5, 0.05), (2.0, 0.08, 3.0)])
def test_stage2_offset_recovers_exact_laws(nb, a0, b0, c0):
    x = np.geomspace(1e3, 1e7, 8)
    a, b, c = bounded(nb.fit_power_law, x, a0 * x ** -b0 + c0, what="fit_power_law")
    assert math.isclose(b, b0, rel_tol=1e-3) and math.isclose(c, c0, rel_tol=1e-3, abs_tol=1e-6) and \
        math.isclose(a, a0, rel_tol=2e-3), (
            f"y = {a0} x^-{b0} + {c0} exactly: the fit should recover (a, b, c); got ({a:.4g}, {b:.4g}, {c:.4g}). a plain "
            "optimizer from a fixed start can stall here; profile over c (for fixed c, the fit is linear in log "
            "space), then polish")


def test_stage2_offset_minimizes_log_space_error(nb):
    rng = np.random.default_rng(1)
    x = np.geomspace(10, 1e6, 10)  # y spans two orders of magnitude, with multiplicative noise
    y = (300 * x ** -0.5 + 1.0) * np.exp(rng.normal(0, 0.1, x.size))
    a, b, c = _run(nb.fit_power_law, x, y)
    assert 0 <= c < y.min(), f"the offset is an irreducible loss: 0 <= c < min(y) = {y.min():.4f}; got {c}"
    ref = _brute_fit(x, y)
    got, want = _log_sse(x, y, a, b, c), _log_sse(x, y, *ref)
    assert got <= want * (1 + 1e-3) + 1e-12, (
        f"the fit must minimize sum (log y - log(a x^-b + c))^2; yours reaches {got:.6g}, a brute-force search "
        f"{want:.6g} at (a, b, c) = {tuple(round(v, 4) for v in ref)}. residuals in linear space (y - prediction) "
        "give a different, worse answer by this criterion")


def test_stage2_offset_respects_bounds(nb):
    x = np.geomspace(1e2, 1e5, 8)
    y = 10 * x ** -0.2 - 0.3  # the unconstrained best offset would be negative
    a, b, c = _run(nb.fit_power_law, x, y)
    assert c >= 0 and c < y.min(), f"c must stay in [0, min(y)); got {c}"
    assert _log_sse(x, y, a, b, c) <= _log_sse(x, y, *_run(nb.fit_power_law, x, y, offset=False)) + 1e-12, (
        "with c pinned at 0, the offset fit must still be at least as good as the no-offset fit")


# ---------------------------------------------------------------- stage 3: the Chinchilla rule


def test_stage3_compute_optimal(nb):
    C = 6 * 70e9 * 1.4e12  # Chinchilla: 70B parameters, 1.4T tokens
    n, d = nb.compute_optimal(C)
    assert math.isclose(float(n), 70e9, rel_tol=1e-9) and math.isclose(float(d), 1.4e12, rel_tol=1e-9), (
        f"C = 6 N D with D = 20 N gives N = sqrt(C / 120): for Chinchilla's compute, (70e9, 1.4e12); got ({float(n):.4g}, {float(d):.4g})")
    n, d = nb.compute_optimal(np.array([6e18, 6e20]), tokens_per_param=100.0)
    assert np.allclose(n, np.sqrt(np.array([6e18, 6e20]) / 600)) and np.allclose(d, 100 * n), (
        "tokens_per_param sets D / N; arrays of budgets work elementwise")


# ---------------------------------------------------------------- stage 4: the parametric loss


HOFF = dict(A=406.4, B=410.7, alpha=0.34, beta=0.28)


def test_stage4_chinchilla_loss(nb):
    got = float(nb.chinchilla_loss(70e9, 1.4e12, E=1.69, **HOFF))
    want = 1.69 + 406.4 / 70e9 ** 0.34 + 410.7 / 1.4e12 ** 0.28
    assert math.isclose(got, want, rel_tol=1e-12), f"E + A/N^alpha + B/D^beta = {want:.5f}, got {got:.5f}"


def test_stage4_allocation_matches_numerical_optimum(nb):
    for C in (1e18, 1e21, 5.76e23):
        n, d = nb.optimal_allocation(C, **HOFF)
        n, d = float(n), float(d)
        assert math.isclose(6 * n * d, C, rel_tol=1e-9), f"the allocation must spend the budget: 6 N D = {6 * n * d:.4g}, C = {C:.4g}"

        def f(logn):
            N = math.exp(logn)
            return HOFF["A"] / N ** HOFF["alpha"] + HOFF["B"] / (C / (6 * N)) ** HOFF["beta"]

        res = scipy.optimize.minimize_scalar(f, bounds=(math.log(1e3), math.log(C)), method="bounded",
                                             options={"xatol": 1e-10})
        assert math.isclose(n, math.exp(res.x), rel_tol=1e-4), (
            f"at C = {C:.3g} the loss is minimized at N = {math.exp(res.x):.4g}; got {n:.4g}. N_opt grows like "
            "C^(beta/(alpha+beta)), with G = (alpha A / (beta B))^(1/(alpha+beta))")
    n1, _ = nb.optimal_allocation(1e20, A=1.0, B=1.0, alpha=0.3, beta=0.3)
    assert math.isclose(float(n1), math.sqrt(1e20 / 6), rel_tol=1e-9), "with A = B and alpha = beta, N = D = sqrt(C/6)"
    n2, _ = nb.optimal_allocation(np.array([1e20, 1e21]), A=1.0, B=2.0, alpha=0.2, beta=0.6)
    assert math.isclose(float(n2[1] / n2[0]), 10 ** 0.75, rel_tol=1e-9), (
        "with alpha = 0.2 and beta = 0.6, ten times the compute multiplies N_opt by 10^(beta/(alpha+beta)) = 10^0.75")


# ---------------------------------------------------------------- stage 5: non-embedding parameters


class TinyLM(nn.Module):
    """the attribute layout of chapter 18's GPT: tok_emb, pos_emb (or None), blocks, ln_f, lm_head."""

    def __init__(self, vocab=65, d=16, block=32, tie=True, learned_pos=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(block, d) if learned_pos else None
        if not learned_pos:
            self.register_buffer("pe", torch.zeros(block, d), persistent=False)
        self.blocks = nn.ModuleList([nn.Linear(d, d), nn.Linear(d, d)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.tok_emb.weight


def test_stage5_non_embedding_params(nb):
    body = 2 * (16 * 16 + 16) + 2 * 16  # two linear blocks and the final layer norm
    for kw, want in ((dict(), body), (dict(tie=False), body + 65 * 16), (dict(learned_pos=False), body)):
        m = TinyLM(**kw)
        got = nb.non_embedding_params(m)
        assert got == want, (
            f"TinyLM({kw}): expected {want} non-embedding parameters, got {got}. skip tok_emb.weight and pos_emb.weight "
            "(pos_emb may be None); a tied lm_head IS the token embedding (count it once, and here not at all); an "
            "untied head counts; buffers never count")
    mlp = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    assert nb.non_embedding_params(mlp) == 26, "a model without tok_emb/pos_emb: every parameter counts (26)"
