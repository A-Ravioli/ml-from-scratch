"""Checks for exercise.py (density estimation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scikit-learn's
KernelDensity and SciPy are used only as oracles.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

# ---------------------------------------------------------------- stage 1: histograms


def test_stage1_worked_example(nb):
    x = np.array([0.0, 1.0, 3.0])
    got = np.asarray(nb.histogram_density(x, np.array([1.0, 3.5, 5.0, -1.0]), 2.0))
    assert got.shape == (4,), f"one density value per evaluation point, got shape {got.shape}"
    assert np.allclose(got, [1 / 3, 1 / 6, 0.0, 0.0]), (
        f"bins [0, 2) and [2, 4) hold 2 and 1 of the 3 points, so the density is 2/(3*2) = 1/3 and 1/(3*2) = 1/6; "
        f"empty bins give 0. got {got.tolist()}")


def test_stage1_integrates_to_one_and_depends_on_origin(nb):
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    h = 0.4
    for origin in [0.0, 0.17]:
        centers = origin + h * (np.arange(-30, 30) + 0.5)
        total = np.sum(nb.histogram_density(x, centers, h, origin)) * h
        assert np.isclose(total, 1.0), (
            f"origin {origin}: summing density * bin width over all bins should give 1, got {total:.4f}. divide each "
            "count by n * bin_width")
    pts = np.linspace(-2, 2, 21) + 0.05
    a = nb.histogram_density(x, pts, 0.4, 0.0)
    b = nb.histogram_density(x, pts, 0.4, 0.2)
    assert not np.allclose(a, b), "moving the origin moves the bin edges, which changes the estimate"
    assert np.allclose(nb.histogram_density(x, [0.39999], 0.4, 0.0), nb.histogram_density(x, [0.0], 0.4, 0.0)), (
        "bins are half-open, [origin + j h, origin + (j+1) h): 0 and 0.39999 share a bin")


def test_stage1_vectorized(nb):
    rng = np.random.default_rng(1)
    bounded(nb.histogram_density, rng.normal(size=20_000), rng.normal(size=20_000), 0.1, budget=20_000,
            what="histogram_density on 20,000 points",
            hint="compute every bin index with np.floor at once, then count with np.searchsorted on the sorted training "
                 "bins (or np.unique): no Python loop over points.")


# ---------------------------------------------------------------- stage 2: kernels


def test_stage2_gaussian_matches_normal_pdf(nb):
    from scipy.stats import norm, multivariate_normal
    r = np.array([0.0, 0.3, 1.0, 2.5])
    got = np.asarray(nb.log_kernel(r**2, 0.7, 1, "gaussian"))
    assert np.allclose(got, norm.logpdf(r, scale=0.7)), (
        f"in 1-D the gaussian kernel with bandwidth h is the N(0, h^2) density; log values {norm.logpdf(r, scale=0.7)}, "
        f"got {got}")
    pts = np.array([[0.2, -0.4, 1.0]])
    want = multivariate_normal(np.zeros(3), 0.5**2 * np.eye(3)).logpdf(pts)
    assert np.isclose(nb.log_kernel(np.sum(pts**2), 0.5, 3, "gaussian"), want), (
        "in d dimensions it's the N(0, h^2 I) density: the normalizer is (2 pi h^2)^(-d/2)")


def test_stage2_hand_values(nb):
    got = float(nb.log_kernel((2 / 3 * 1.5) ** 2, 1.5, 1, "epanechnikov"))
    assert np.isclose(np.exp(got), 0.75 * (1 - 4 / 9) / 1.5), (
        f"1-D epanechnikov: K_h(r) = 0.75 (1 - (r/h)^2) / h; at r/h = 2/3, h = 1.5 that's 0.2778. got exp = {np.exp(got)}")
    assert np.isclose(np.exp(float(nb.log_kernel(0.0, 1.0, 3, "epanechnikov"))), 15 / (8 * np.pi)), (
        "3-D epanechnikov at 0: (d + 2) / (2 V_d) with V_3 = 4 pi / 3, i.e. 15 / (8 pi)")
    assert np.isclose(np.exp(float(nb.log_kernel(0.25, 1.0, 3, "tophat"))), 3 / (4 * np.pi)), (
        "3-D top-hat inside the unit ball: 1 / V_3 = 3 / (4 pi)")
    out = np.asarray(nb.log_kernel(np.array([1.0, 1.21, 4.0]), 1.0, 2, "tophat"))
    assert np.all(np.isneginf(out)), f"outside the support (distance >= h, boundary included) the log kernel is -inf, got {out}"
    with pytest.raises(ValueError):
        nb.log_kernel(1.0, 1.0, 1, "cosine")
        pytest.fail("an unknown kernel name should raise ValueError")


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov", "tophat"])
@pytest.mark.parametrize("d", [1, 2])
def test_stage2_integrates_to_one(nb, kernel, d):
    h = 0.8
    g = np.linspace(-4, 4, 1601 if d == 1 else 401)
    step = g[1] - g[0]
    if d == 1:
        r2 = g**2
    else:
        A, B = np.meshgrid(g, g)
        r2 = A**2 + B**2
    total = np.sum(np.exp(nb.log_kernel(r2, h, d, kernel))) * step**d
    assert abs(total - 1) < 0.01, (
        f"{kernel} kernel in {d}-D with h = {h} should integrate to 1, got {total:.4f}. check the normalizing "
        "constant: it depends on d (through the volume of the unit ball) and on h^d")


# ---------------------------------------------------------------- stage 3: the KDE


def test_stage3_worked_example(nb):
    kde = nb.KernelDensity(1.0).fit(np.array([[0.0], [1.0], [3.0]]))
    got = float(kde.score_samples(np.array([[1.0]]))[0])
    assert np.isclose(np.exp(got), 0.23163, atol=1e-5), (
        f"p(1) = (phi(1) + phi(0) + phi(2)) / 3 = 0.23163 with phi the standard normal pdf; got exp = {np.exp(got):.5f}")
    epa = nb.KernelDensity(1.5, kernel="epanechnikov").fit(np.array([[0.0], [1.0], [3.0]]))
    got = float(np.exp(epa.score_samples(np.array([[1.0]]))[0]))
    assert np.isclose(got, 0.25926, atol=1e-5), f"the lesson's epanechnikov value is 0.25926, got {got:.5f}"


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov", "tophat"])
def test_stage3_matches_sklearn(nb, kernel):
    from sklearn.neighbors import KernelDensity
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2))
    Q = rng.normal(size=(25, 2)) * 1.2
    ours = nb.KernelDensity(0.6, kernel=kernel)
    assert ours.fit(X) is ours, "fit returns self"
    ref = KernelDensity(bandwidth=0.6, kernel=kernel).fit(X)
    got, want = np.asarray(ours.score_samples(Q)), ref.score_samples(Q)
    finite = np.isfinite(want)
    assert np.array_equal(np.isfinite(got), finite), (
        f"{kernel}: points outside every kernel's support get log density -inf (and only those)")
    assert np.allclose(got[finite], want[finite]), f"{kernel}: score_samples should be the log of the KDE"
    assert np.isclose(ours.score(Q[finite]), want[finite].sum()), "score is the total log-likelihood"
    assert ours.get_params() == {"bandwidth": 0.6, "kernel": kernel}


def test_stage3_stable_in_log_space(nb):
    kde = nb.KernelDensity(0.1).fit(np.array([[0.0], [0.5]]))
    got = float(kde.score_samples(np.array([[100.0]]))[0])
    want = -0.5 * (99.5 / 0.1) ** 2 - math.log(2) - 0.5 * math.log(2 * math.pi * 0.01) + math.log(
        1 + math.exp(-0.5 * (100**2 - 99.5**2) / 0.01))
    assert np.isfinite(got) and np.isclose(got, want, rtol=1e-9), (
        f"far from the data the density underflows to 0 in floating point, but its log is a perfectly good number "
        f"(about {want:.0f}). got {got}. combine the log kernel values with logsumexp instead of exponentiating")


def test_stage3_vectorized(nb):
    rng = np.random.default_rng(3)
    kde = nb.KernelDensity(0.5).fit(rng.normal(size=(2000, 3)))
    bounded(kde.score_samples, rng.normal(size=(2000, 3)), budget=20_000, what="score_samples on 2,000 points",
            hint="one pairwise_distances call, one log_kernel call on the whole matrix, one logsumexp along axis 1.")


# ---------------------------------------------------------------- stage 4: sampling


def test_stage4_sample_moments(nb):
    rng = np.random.default_rng(4)
    X = rng.normal(loc=[3.0, -1.0], scale=[1.0, 0.5], size=(400, 2))
    kde = nb.KernelDensity(0.3).fit(X)
    S = np.asarray(kde.sample(20_000, rng=0))
    assert S.shape == (20_000, 2), f"sample(n) returns (n, d), got {S.shape}"
    assert np.allclose(S.mean(axis=0), X.mean(axis=0), atol=0.03), (
        "a KDE sample is a random training row plus zero-mean noise, so its mean matches the data's mean")
    want_var = X.var(axis=0) + 0.3**2
    assert np.allclose(S.var(axis=0), want_var, rtol=0.05), (
        f"its variance is the data's variance plus h^2 = 0.09 per feature: about {want_var.round(3)}, got "
        f"{S.var(axis=0).round(3)}. did you scale the noise by h (not h^2)?")


def test_stage4_sample_rng_and_kernels(nb):
    kde = nb.KernelDensity(0.5).fit(np.arange(10.0).reshape(-1, 1))
    assert np.array_equal(kde.sample(5, rng=1), kde.sample(5, rng=1)), "the same seed gives the same samples"
    assert np.asarray(kde.sample(rng=2)).shape == (1, 1), "sample() draws one point by default, shape (1, d)"
    small = nb.KernelDensity(0.01).fit(np.array([[0.0], [10.0], [20.0]]))
    nearest = np.rint(np.asarray(small.sample(600, rng=3))[:, 0] / 10).astype(int)
    assert np.array_equal(np.bincount(nearest, minlength=3) > 0, [True, True, True]), (
        "every training row must be drawable: pick indices with integers(n), whose upper end is exclusive")
    with pytest.raises(NotImplementedError):
        nb.KernelDensity(0.5, kernel="tophat").fit(np.zeros((3, 1))).sample(2, rng=0)
        pytest.fail("sampling from a non-gaussian kernel isn't required: raise NotImplementedError")


# ---------------------------------------------------------------- stage 5: bandwidth


def test_stage5_silverman(nb):
    rng = np.random.default_rng(5)
    x = rng.normal(scale=2.0, size=(200, 1))
    got = nb.silverman_bandwidth(x)
    want = (4 / 3) ** 0.2 * 200 ** -0.2 * x.std(ddof=1)
    assert np.isclose(got, want), f"in 1-D the rule is 1.06 sigma n^(-1/5) = {want:.4f}, got {got}"
    X = rng.normal(scale=[1.0, 3.0], size=(100, 2))
    want2 = (4 / 4) ** (1 / 6) * 100 ** (-1 / 6) * X.std(axis=0, ddof=1).mean()
    assert np.isclose(nb.silverman_bandwidth(X), want2), "in d dimensions: (4/(d+2))^(1/(d+4)) n^(-1/(d+4)) sigma"


def test_stage5_select_bandwidth(nb):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 1))
    hs = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.7, 1.0, 2.0, 4.0])
    best, scores = bounded(nb.select_bandwidth, X, hs, cv=5, rng=0, budget=1_000_000, what="select_bandwidth")
    scores = np.asarray(scores)
    assert scores.shape == (10,), f"one mean score per bandwidth, got shape {scores.shape}"
    assert best == hs[np.argmax(scores)], "best is the bandwidth with the highest mean held-out log-likelihood"
    assert 0.1 <= best <= 0.7, f"for 300 standard-normal points the best bandwidth is around 0.2-0.5, got {best}"
    assert scores[0] < scores.max() - 0.1 and scores[-1] < scores.max() - 0.2, (
        "both a tiny and a huge bandwidth should score clearly worse: undersmoothing spikes, oversmoothing blurs")
    assert -1.75 < scores.max() < -1.2, (
        f"scores are mean held-out log densities PER POINT (a standard normal's is about -1.42), got max {scores.max():.3f}. "
        "average score_samples over each test fold, then over folds")
    again = nb.select_bandwidth(X, hs, cv=5, rng=0)
    assert np.allclose(again[1], scores), "the same rng gives the same folds and the same scores"
    from sklearn.neighbors import KernelDensity
    from mlfs.model_selection import KFold
    gen = np.random.default_rng(1)
    splits = list(KFold(4, shuffle=True, rng=np.random.default_rng(1)).split(X))
    want = [np.mean([KernelDensity(bandwidth=h).fit(X[tr]).score_samples(X[te]).mean() for tr, te in splits])
            for h in hs[2:6]]
    got = np.asarray(nb.select_bandwidth(X, hs[2:6], cv=4, rng=gen)[1])
    assert np.allclose(got, want), (
        "the folds are drawn ONCE, from KFold(cv, shuffle=True, rng=as_rng(rng)).split(X), and shared by every bandwidth "
        "(with a Generator as rng, drawing new folds per bandwidth gives each candidate different folds)")
    best_sorted, _ = nb.select_bandwidth(np.sort(X, axis=0), hs, cv=5, rng=0)
    assert 0.1 <= best_sorted <= 0.7, (
        f"on the same points sorted in increasing order the answer must not change much, got {best_sorted}. "
        "contiguous (unshuffled) folds hold out a whole tail of the data: shuffle the folds")
