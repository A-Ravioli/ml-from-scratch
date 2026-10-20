"""Checks for exercise.py (estimation).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.

Simulated quantities are compared with exact answers using tolerances of about 5 Monte Carlo
standard errors: a correct harness passes whatever random numbers it draws, and the bugs these
tests look for miss by far more.
"""
import math

import numpy as np


def _normal(size, rng):
    return rng.normal(0.0, 1.0, size)


def _uniform(theta):
    return lambda size, rng: theta * rng.random(size)


def _row_mean(x):
    return np.mean(x, axis=-1)


# ---------------------------------------------------------------- stage 1: bias, variance, MSE


def test_stage1_hand_example(nb):
    got = nb.bias_variance_mse(np.array([1.0, 2.0, 3.0, 6.0]), 2.0)
    assert set(got) >= {"bias", "variance", "mse"}, f"return a dict with keys 'bias', 'variance', 'mse'; got {sorted(got)}"
    assert abs(got["bias"] - 1.0) < 1e-12, (
        f"estimates (1, 2, 3, 6) of theta = 2 average 3, so the bias is 3 - 2 = 1; got {got['bias']}")
    assert abs(got["variance"] - 3.5) < 1e-12, (
        f"the estimates' spread around their own mean 3 is (4 + 1 + 0 + 9) / 4 = 3.5; got {got['variance']}. "
        + ("you divided by n - 1 = 3: use ddof=0 here, so that mse = bias^2 + variance exactly."
           if abs(got["variance"] - 14 / 3) < 1e-9 else
           "measure the spread around mean(estimates), not around theta." if abs(got["variance"] - 4.5) < 1e-9 else ""))
    assert abs(got["mse"] - 4.5) < 1e-12, (
        f"mse = mean((estimates - theta)^2) = (1 + 0 + 1 + 16) / 4 = 4.5; got {got['mse']}")


def test_stage1_decomposition_and_types(nb):
    rng = np.random.default_rng(0)
    for theta in (-3.0, 0.0, 10.0):
        est = rng.normal(theta + 0.7, 2.0, size=1001)
        got = nb.bias_variance_mse(est, theta)
        assert all(isinstance(got[k], float) for k in ("bias", "variance", "mse")), "return Python floats"
        assert abs(got["mse"] - (got["bias"] ** 2 + got["variance"])) < 1e-9 * max(1.0, got["mse"]), (
            "with ddof=0, mse must equal bias^2 + variance exactly: that's the bias-variance decomposition. "
            f"got mse = {got['mse']:.6f} but bias^2 + variance = {got['bias'] ** 2 + got['variance']:.6f}")
    unbiased = nb.bias_variance_mse([4.0, 6.0], 5.0)
    assert unbiased["bias"] == 0.0 and abs(unbiased["mse"] - 1.0) < 1e-12, (
        "estimates (4, 6) of 5: bias 0, mse 1. bias is mean(estimates) - theta, not mean(|estimates - theta|).")


# ---------------------------------------------------------------- stage 2: the harness


def test_stage2_simulate_shape_and_rows(nb):
    est = np.asarray(nb.simulate(lambda x: x[:, 0] * 0 + x.shape[1], _normal, 7, 13, rng=0))
    assert est.shape == (13,), (
        f"simulate(..., n=7, n_reps=13) gives one estimate per dataset: shape (13,), got {est.shape}. "
        "draw sampler((n_reps, n), rng): one dataset per ROW.")
    assert np.all(est == 7), "each dataset (row) should have n = 7 draws: draw sampler((n_reps, n), rng), not (n, n_reps)"


def test_stage2_simulate_uses_the_rng(nb):
    a = np.asarray(nb.simulate(_row_mean, _normal, 5, 10, rng=42))
    b = np.asarray(nb.simulate(_row_mean, _normal, 5, 10, rng=42))
    c = np.asarray(nb.simulate(_row_mean, _normal, 5, 10, rng=43))
    assert np.array_equal(a, b), "the same seed should give the same estimates: normalize rng with as_rng(rng)"
    assert not np.array_equal(a, c), "seeds 42 and 43 gave identical estimates: pass the rng you were given to the sampler"
    g = np.random.default_rng(7)
    d = np.asarray(nb.simulate(_row_mean, _normal, 5, 10, rng=g))
    e = np.asarray(nb.simulate(_row_mean, _normal, 5, 10, rng=g))
    assert not np.array_equal(d, e), "two calls with the same Generator should continue its stream, not restart it"


def test_stage2_evaluate_sample_mean(nb):
    reps = 40_000
    got = nb.evaluate(_row_mean, _normal, 0.0, 25, reps, rng=1)
    assert set(got) >= {"bias", "variance", "mse", "mse_se"}, f"evaluate returns bias, variance, mse and mse_se; got {sorted(got)}"
    # the mean of 25 N(0,1) draws has variance 1/25; its squared error has sd sqrt(2)/25
    se = math.sqrt(2) / 25 / math.sqrt(reps)
    assert abs(got["mse"] - 0.04) < 5 * se, f"the mean of 25 standard normals has MSE 1/25 = 0.04, you measured {got['mse']:.5f}"
    assert abs(got["bias"]) < 5 * math.sqrt(0.04 / reps), f"the sample mean is unbiased, you measured bias {got['bias']:.5f}"
    assert abs(got["mse_se"] - se) < 0.1 * se, (
        f"mse_se should be std((est - theta)^2, ddof=1) / sqrt(n_reps) ≈ {se:.2e}, got {got['mse_se']:.2e}. "
        + ("did you forget to divide by sqrt(n_reps)?" if got["mse_se"] > 10 * se else ""))


def test_stage2_evaluate_biased_estimator(nb):
    got = nb.evaluate(lambda x: np.mean(x, axis=-1) + 0.5, _normal, 0.0, 4, 20_000, rng=2)
    assert abs(got["bias"] - 0.5) < 0.03, f"x̄ + 0.5 has bias 0.5, you measured {got['bias']:.4f}"
    assert abs(got["variance"] - 0.25) < 0.02, f"x̄ of 4 standard normals has variance 1/4, you measured {got['variance']:.4f}"
    assert abs(got["mse"] - 0.5) < 0.03, f"MSE = 0.5^2 + 1/4 = 0.5, you measured {got['mse']:.4f}"


# ---------------------------------------------------------------- stage 3: estimators of a uniform's end


def test_stage3_hand_example(nb):
    x = np.array([1.0, 4.0, 2.0, 5.0])
    for name, want in (("theta_mom", 6.0), ("theta_max", 5.0), ("theta_unbiased_max", 6.25)):
        got = getattr(nb, name)(x)
        assert abs(float(got) - want) < 1e-12, (
            f"{name}([1, 4, 2, 5]) should be {want}, got {got}"
            + (". the correction is (n + 1) / n with n = 4, so 5/4 * 5" if name == "theta_unbiased_max" else ""))


def test_stage3_rowwise(nb):
    x = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 2.0]])
    for name, want in (("theta_mom", [4.0, 2.0]), ("theta_max", [3.0, 2.0]), ("theta_unbiased_max", [4.0, 8 / 3])):
        got = np.asarray(getattr(nb, name)(x))
        assert got.shape == (2,) and np.allclose(got, want), (
            f"{name} on a (2, 3) array should work row by row (axis=-1) and give {want}, got {got.tolist()}. "
            "with axis=0 you'd get one value per COLUMN.")


def test_stage3_exact_bias_and_mse(nb):
    theta, n, reps = 3.0, 5, 40_000
    rng = np.random.default_rng(3)
    x = theta * rng.random((reps, n))
    exact = {  # (bias, mse) for Uniform(0, theta)
        "theta_mom": (0.0, theta**2 / (3 * n)),
        "theta_max": (-theta / (n + 1), 2 * theta**2 / ((n + 1) * (n + 2))),
        "theta_unbiased_max": (0.0, theta**2 / (n * (n + 2))),
    }
    for name, (bias, mse) in exact.items():
        est = np.asarray(getattr(nb, name)(x))
        sq = (est - theta) ** 2
        tol = 5 * sq.std() / math.sqrt(reps)
        assert abs(sq.mean() - mse) < tol, f"{name}: for n = 5 the exact MSE is {mse:.4f}, simulated {sq.mean():.4f}"
        assert abs(est.mean() - theta - bias) < 5 * est.std() / math.sqrt(reps), (
            f"{name}: the exact bias is {bias:.4f}, simulated {est.mean() - theta:.4f}")


# ---------------------------------------------------------------- stage 4: n - 1, n, or n + 1?


def test_stage4_scaled_variance(nb):
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])  # sum of squared deviations from 5 is 32
    for c, want in ((7, 32 / 7), (8, 4.0), (9, 32 / 9)):
        got = float(nb.scaled_variance(x, c))
        assert abs(got - want) < 1e-12, (
            f"the squared deviations of (2, 4, 4, 4, 5, 5, 7, 9) from their mean 5 sum to 32, so c = {c} "
            f"gives {want:.4f}; got {got:.4f}")
    rows = np.asarray(nb.scaled_variance(np.array([[1.0, 3.0], [0.0, 0.0], [10.0, 14.0]]), 1))
    assert rows.shape == (3,) and np.allclose(rows, [2.0, 0.0, 8.0]), (
        f"row-wise: each row's deviations are from THAT row's mean (keepdims=True); expected [2, 0, 8], got {rows}")


def test_stage4_exact_mse(nb):
    n, s2 = 10, 2.0
    for c, want in ((9, 2 * s2**2 / 9), (10, (2 * n - 1) * s2**2 / n**2), (11, 2 * s2**2 / 11)):
        got = nb.scaled_variance_mse(n, c, s2)
        assert abs(got - want) < 1e-12, (
            f"n = 10, sigma^2 = 2, c = {c}: the exact MSE is {want:.6f}, got {got:.6f}. bias = sigma^2 ((n-1)/c - 1), "
            "variance = 2 sigma^4 (n - 1) / c^2.")
    cs = np.arange(5, 30)
    m = np.asarray(nb.scaled_variance_mse(n, cs))
    assert m.shape == cs.shape, "scaled_variance_mse should accept an array of c"
    assert cs[np.argmin(m)] == n + 1, f"the MSE-optimal divisor for Gaussian data is n + 1 = 11, your formula says {cs[np.argmin(m)]}"


def test_stage4_formula_matches_simulation(nb):
    n, reps = 6, 50_000
    rng = np.random.default_rng(4)
    x = rng.normal(1.0, math.sqrt(3.0), (reps, n))
    for c in (n - 1, n, n + 1):
        sq = (np.asarray(nb.scaled_variance(x, c)) - 3.0) ** 2
        exact = nb.scaled_variance_mse(n, c, 3.0)
        assert abs(sq.mean() - exact) < 5 * sq.std() / math.sqrt(reps), (
            f"c = {c}: your exact MSE {exact:.4f} disagrees with your simulation {sq.mean():.4f}")


# ---------------------------------------------------------------- stage 5: consistency


def test_stage5_convergence_rate(nb):
    ns = np.array([10, 20, 40, 80])
    assert abs(nb.convergence_rate(ns, 3.0 / ns) + 1) < 1e-12, "values 3/n have rate -1"
    assert abs(nb.convergence_rate(ns, 0.5 * ns**-2.0) + 2) < 1e-12, "values 0.5/n^2 have rate -2"
    noisy = 2.0 / ns * np.array([1.1, 0.9, 1.05, 0.95])
    want = float(np.polyfit(np.log(ns), np.log(noisy), 1)[0])
    got = nb.convergence_rate(ns, noisy)
    assert abs(got - want) < 1e-12, (
        f"fit a least-squares line through ALL the points (log n, log value): expected slope {want:.4f}, got {got:.4f}. "
        "using only the first and last points is not a least-squares fit.")


def test_stage5_rates_of_three_estimators(nb):
    ns = [8, 16, 32, 64, 128]
    mom = np.asarray(nb.mse_curve(lambda x: 2 * np.mean(x, axis=-1), _uniform(1.0), 1.0, ns, 4000, rng=5))
    assert mom.shape == (5,), f"one MSE per n: shape (5,), got {mom.shape}"
    exact = 1 / (3 * np.array(ns))
    assert np.all(np.abs(mom / exact - 1) < 0.12), (
        f"2 x̄ for Uniform(0, 1) has MSE 1/(3n) = {np.round(exact, 5).tolist()}, you got {np.round(mom, 5).tolist()}")
    mx = np.asarray(nb.mse_curve(lambda x: (x.shape[-1] + 1) / x.shape[-1] * np.max(x, axis=-1),
                                 _uniform(1.0), 1.0, ns, 4000, rng=6))
    r_mom, r_max = nb.convergence_rate(ns, mom), nb.convergence_rate(ns, mx)
    assert abs(r_mom + 1) < 0.1, f"2 x̄ is consistent with MSE ~ 1/n: rate -1, you measured {r_mom:.3f}"
    assert abs(r_max + 2) < 0.15, f"the corrected max has MSE 1/(n(n+2)) ~ 1/n^2: rate -2, you measured {r_max:.3f}"
    first = np.asarray(nb.mse_curve(lambda x: 2 * x[:, 0], _uniform(1.0), 1.0, ns, 4000, rng=7))
    assert abs(nb.convergence_rate(ns, first)) < 0.1, (
        "2 x_1 (twice the FIRST draw) is unbiased but ignores the rest of the data: its MSE stays 1/3 for every n "
        f"(rate 0). you measured {nb.convergence_rate(ns, first):.3f}")


def test_stage5_uses_the_rng(nb):
    a = np.asarray(nb.mse_curve(_row_mean, _normal, 0.0, [4, 8], 500, rng=1))
    b = np.asarray(nb.mse_curve(_row_mean, _normal, 0.0, [4, 8], 500, rng=1))
    c = np.asarray(nb.mse_curve(_row_mean, _normal, 0.0, [4, 8], 500, rng=2))
    assert np.array_equal(a, b), "the same seed should give the same curve"
    assert not np.array_equal(a, c), "seeds 1 and 2 gave identical curves: pass the given rng (normalized once) down to evaluate"
    d = np.asarray(nb.mse_curve(_row_mean, _normal, 0.0, [5, 5], 500, rng=3))
    assert d[0] != d[1], (
        "two entries with the same n gave the identical MSE: you re-seeded for every n. normalize rng ONCE "
        "with as_rng(rng) before the loop and pass that same generator to every evaluate call.")
