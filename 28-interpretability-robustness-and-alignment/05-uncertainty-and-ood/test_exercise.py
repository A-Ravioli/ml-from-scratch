"""Checks for 28 · 05 uncertainty & OOD.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. all NumPy, synthetic data.
temperature scaling against scipy's 1-D minimiser and a planted temperature (over- and underconfident),
the NLL it reaches, max_iters and the argmax; the conformal quantile against order statistics by hand; split
conformal sets and intervals for their exact expected coverage ceil((n+1)(1-alpha))/(n+1), checked over many
random splits with enough power to see an off-by-one rank; the ensemble decomposition and the OOD AUROC
against hand computations.
"""
import math

import numpy as np
import pytest
from scipy import optimize
from scipy.special import logsumexp, softmax

from mlfs.testing import bounded


def _nll(z, y, T):
    return float(np.mean(logsumexp(z / T, axis=1) - z[np.arange(len(y)), y] / T))


def _planted(T0, seed, n=4000, k=5, scale=3.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, scale, (n, k))
    p = softmax(z / T0, axis=1)
    u = rng.random(n)[:, None]
    y = (p.cumsum(1) < u).sum(1).clip(0, k - 1)
    return z, y


def _fit(nb, z, y, **kw):
    return bounded(nb.fit_temperature, z, y, budget=2_000_000, what="fit_temperature",
                   hint="cap the iterations at max_iters.", **kw)


# ---------------------------------------------------------------- stage 1: temperature scaling


def test_stage1_matches_minimiser(nb):
    for T0, seed in ((2.5, 0), (0.5, 1)):
        z, y = _planted(T0, seed)
        T = _fit(nb, z, y)
        want = math.exp(optimize.minimize_scalar(lambda u: _nll(z, y, math.exp(u)), bounds=(-5, 5), method="bounded",
                                                 options={"xatol": 1e-10}).x)
        assert isinstance(T, float) and T > 0, "return a positive Python float"
        assert abs(T - want) < 1e-4 * want, (
            f"labels drawn from softmax(z / {T0}): the NLL-minimising temperature is {want:.5f}; got {T:.5f}")
        assert _nll(z, y, T) <= _nll(z, y, 1.0) + 1e-12, "the fitted temperature never has a worse NLL than T = 1"
        assert _nll(z, y, T) <= min(_nll(z, y, 1.02 * T), _nll(z, y, 0.98 * T)), "it's a minimum of the NLL"
    z, y = _planted(2.5, 0)
    assert _fit(nb, z, y) > 1.5, "overconfident logits (true T 2.5) need T > 1"
    z, y = _planted(0.5, 1)
    assert _fit(nb, z, y) < 0.8, "underconfident logits (true T 0.5) need T < 1: don't restrict T to >= 1"


def test_stage1_properties(nb):
    z, y = _planted(2.0, 2, n=2000)
    T = _fit(nb, z, y)
    shifted = _fit(nb, z + np.random.default_rng(3).normal(size=(2000, 1)) * 5, y)
    assert abs(T - shifted) < 1e-6 * T, "adding a constant to each row of logits changes nothing (softmax invariance)"
    assert np.array_equal((z / T).argmax(1), z.argmax(1)), "a positive temperature never changes the argmax"
    scaled = _fit(nb, 3.0 * z, y)
    assert abs(scaled - 3.0 * T) < 1e-4 * T, "logits 3x larger need a temperature 3x larger"
    one = _fit(nb, z, y, max_iters=1)
    assert abs(one - T) > 1e-3 * T, (
        f"max_iters caps the optimiser: one iteration from T = 1 shouldn't already land on the optimum {T:.4f} "
        f"(got {one:.4f})")
    zl = z.tolist()
    assert abs(_fit(nb, zl, list(y)) - T) < 1e-6 * T, "accept lists as well as arrays"


# ---------------------------------------------------------------- stage 2: conformal classification


def test_stage2_quantile_by_hand(nb):
    s = np.array([0.7, 0.1, 0.4, 0.9, 0.2, 0.6, 1.0, 0.3, 0.8, 0.5])
    assert nb.conformal_quantile(s, 0.2) == 0.9, (
        "n = 10, alpha = 0.2: rank ceil(11 * 0.8) = 9, so the 9th smallest score, 0.9 (not an interpolated quantile)")
    assert nb.conformal_quantile(s, 0.5) == 0.6, "alpha = 0.5: rank ceil(5.5) = 6 -> 0.6"
    assert nb.conformal_quantile(s, 0.05) == math.inf, "alpha = 0.05: rank ceil(10.45) = 11 > 10 -> inf"
    assert np.array_equal(s, [0.7, 0.1, 0.4, 0.9, 0.2, 0.6, 1.0, 0.3, 0.8, 0.5]), "don't sort the input in place"
    assert nb.conformal_quantile(np.arange(99) / 100, 0.1) == 0.89, "n = 99, alpha = 0.1: rank exactly 90"
    nine = np.arange(1, 10) / 10
    assert nb.conformal_quantile(nine, 0.7) == 0.3, (
        "n = 9, alpha = 0.7: the rank is (n+1)(1-alpha) = 3 exactly, so the 3rd smallest score, 0.3. in floating point "
        "10 * (1 - 0.7) = 3.0000000000000004, and math.ceil makes that 4: subtract a tiny tolerance before ceil")
    with pytest.raises(ValueError):
        nb.conformal_quantile(s, 1.0)


def test_stage2_sets_by_hand(nb):
    cal = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.5, 0.4, 0.1]])
    labels = np.array([0, 1, 2, 1])
    # scores 1 - p_y = 0.3, 0.2, 0.6, 0.6; n = 4, alpha = 0.4: rank ceil(5 * 0.6) = 3 -> q = 0.6
    test = np.array([[0.4, 0.45, 0.15], [0.9, 0.05, 0.05]])
    got = nb.conformal_prediction_sets(cal, labels, test, alpha=0.4)
    want = np.array([[True, True, False], [True, False, False]])
    assert got.dtype == bool and got.shape == (2, 3) and np.array_equal(got, want), (
        f"q = 0.6 (third smallest of 0.2, 0.3, 0.6, 0.6); class c is in the set when 1 - p_c <= q, i.e. p_c >= 0.4 "
        f"(inclusive: p = 0.4 is in): {want.tolist()}; got {got.tolist()}")


def _classification_split(rng, n_cal=99, n_test=500, k=6):
    """a miscalibrated model: probabilities from random logits, labels from a different temperature."""
    z = rng.normal(0, 2.5, (n_cal + n_test, k))
    y = (softmax(z / 2.0, axis=1).cumsum(1) < rng.random(n_cal + n_test)[:, None]).sum(1).clip(0, k - 1)
    p = softmax(z, axis=1)
    return p[:n_cal], y[:n_cal], p[n_cal:], y[n_cal:]


def test_stage2_coverage_is_exact(nb):
    rng = np.random.default_rng(0)
    covs = {0.1: [], 0.25: []}
    for _ in range(300):
        pc, yc, pt, yt = _classification_split(rng)
        for alpha in covs:
            sets = nb.conformal_prediction_sets(pc, yc, pt, alpha=alpha)
            covs[alpha].append(sets[np.arange(len(yt)), yt].mean())
    for alpha, c in covs.items():
        k = math.ceil(100 * (1 - alpha))
        want, se = k / 100, np.std(c) / np.sqrt(len(c))
        assert abs(np.mean(c) - want) < 4 * se + 1e-3, (
            f"alpha = {alpha}, n_cal = 99: split conformal covers with probability exactly ceil((n+1)(1-alpha))/(n+1) "
            f"= {want:.2f} on average over random splits; got {np.mean(c):.4f} (+- {se:.4f}). off by about 0.01 means "
            "an off-by-one rank (or an interpolated quantile)")


# ---------------------------------------------------------------- stage 3: conformal regression


def test_stage3_intervals_by_hand(nb):
    lo, hi = nb.conformal_intervals([1.0, 2.0, 3.0, 4.0], [1.5, 1.0, 3.2, 6.0], [10.0, 20.0], alpha=0.4)
    # scores 0.5, 1.0, 0.2, 2.0 -> sorted 0.2, 0.5, 1.0, 2.0; rank ceil(5 * 0.6) = 3 -> q = 1.0
    assert np.allclose(lo, [9.0, 19.0]) and np.allclose(hi, [11.0, 21.0]), (
        f"q = third smallest |y - pred| = 1.0, intervals pred -/+ q; got lo={lo}, hi={hi}")
    lo, hi = nb.conformal_intervals([0.0, 0.0], [1.0, 2.0], [5.0], alpha=0.1)
    assert np.isinf(lo).all() and np.isinf(hi).all(), "too few calibration points for 90%: infinite intervals"


def test_stage3_coverage_is_exact(nb):
    rng = np.random.default_rng(1)
    covs, widths = [], []
    for _ in range(400):
        x = rng.uniform(-2, 2, 99 + 400)
        y = np.sin(2 * x) + rng.standard_t(3, size=len(x)) * 0.3  # heavy-tailed noise
        pred = np.sin(2 * x) * 0.8  # a biased model
        lo, hi = nb.conformal_intervals(pred[:99], y[:99], pred[99:], alpha=0.2)
        covs.append(np.mean((y[99:] >= lo) & (y[99:] <= hi)))
        widths.append(np.mean(hi - lo))
    se = np.std(covs) / np.sqrt(len(covs))
    assert abs(np.mean(covs) - 0.8) < 4 * se + 1e-3, (
        f"alpha = 0.2, n = 99: expected coverage exactly ceil(100 * 0.8) / 100 = 0.80 for any model and noise; got "
        f"{np.mean(covs):.4f} (+- {se:.4f})")
    assert np.std(widths) > 0, "the width depends on the calibration split"


# ---------------------------------------------------------------- stage 4: ensembles and OOD


def test_stage4_decomposition(nb):
    agree = np.array([[[0.5, 0.5]], [[0.5, 0.5]]])
    u = nb.ensemble_uncertainty(agree)
    assert np.allclose(u["total"], np.log(2)) and np.allclose(u["aleatoric"], np.log(2)) and np.allclose(u["epistemic"], 0), (
        "members agree on 50/50: all uncertainty is aleatoric (log 2 nats), none epistemic")
    disagree = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    u = nb.ensemble_uncertainty(disagree)
    assert np.allclose(u["aleatoric"], 0) and np.allclose(u["epistemic"], np.log(2)) and not np.isnan(u["total"]).any(), (
        "confident members that disagree: no aleatoric uncertainty, log 2 epistemic (0 log 0 = 0, no NaN)")
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=(5, 7))
    u = nb.ensemble_uncertainty(p)
    H = lambda q: -(q * np.log(q)).sum(-1)  # noqa: E731
    assert u["mean"].shape == (7, 4) and np.allclose(u["mean"], p.mean(0)), "mean is the ensemble's average prediction"
    assert np.allclose(u["total"], H(p.mean(0))) and np.allclose(u["aleatoric"], H(p).mean(0)), (
        "total = entropy of the mean prediction; aleatoric = mean of the members' entropies")
    assert np.allclose(u["epistemic"], u["total"] - u["aleatoric"]) and (u["epistemic"] >= -1e-12).all(), (
        "epistemic = total - aleatoric, the mutual information between the prediction and the member: never negative")
    with pytest.raises(ValueError):
        nb.ensemble_uncertainty(p[0])


def test_stage4_ood_auroc(nb):
    assert np.isclose(nb.ood_auroc([0.1, 0.2, 0.3], [0.4, 0.5]), 1.0), "every OOD score above every in-distribution one"
    assert np.isclose(nb.ood_auroc([0.4, 0.5], [0.1, 0.2, 0.3]), 0.0), "higher score = more OOD (orientation)"
    assert np.isclose(nb.ood_auroc([0.1, 0.3, 0.5], [0.2, 0.4]), 0.5), "P(out > in) = 3/6 here"
    assert np.isclose(nb.ood_auroc([1.0, 1.0], [1.0, 1.0]), 0.5), "ties count 1/2"
    assert isinstance(nb.ood_auroc([0.0], [1.0]), float)
