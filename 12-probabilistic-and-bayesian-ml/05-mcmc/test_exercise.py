"""Checks for exercise.py (markov chain monte carlo).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.

Samplers are checked against EXACT posteriors (a correlated Gaussian, and a Gamma posterior
sampled in log space) with tolerances set by the effective sample size: a posterior mean is
accepted within 4.5 Monte Carlo standard errors, sd / sqrt(ESS), where the ESS comes from an
independent reference implementation in this file (not from your stage 4). Bookkeeping (burn-in,
thinning, call counts, sweep order) is checked exactly with deterministic or counting targets.
Diagnostics are checked against direct formulas. Every sampler call runs under mlfs.testing.bounded.
"""
import time

import numpy as np

from mlfs.testing import bounded


def _acf(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / ac[0]


def _ess(x):
    """reference: Geyer initial positive sequence, tau floored at 1 / log10(n)."""
    x = np.asarray(x, float)
    n = len(x)
    rho = _acf(x)
    tau = -1.0
    for m in range(n // 2):
        pair = rho[2 * m] + rho[2 * m + 1] if 2 * m + 1 < n else -1
        if pair <= 0:
            break
        tau += 2 * pair
    return n / max(tau, 1 / np.log10(n))


def _check_mean(samples, mean, sd, what, k=4.5):
    for j in range(samples.shape[1]):
        ess = _ess(samples[:, j])
        err = abs(samples[:, j].mean() - mean[j])
        se = sd[j] / np.sqrt(ess)
        assert err < k * se, (
            f"{what}: coordinate {j} has sample mean {samples[:, j].mean():.4f}, exact posterior mean {mean[j]:.4f}. "
            f"with an effective sample size of {ess:.0f} the Monte Carlo standard error is {se:.4f}, so an error of "
            f"{err:.4f} ({err / se:.1f} standard errors) means the chain isn't targeting the right distribution. check "
            "the acceptance rule: accept if log(u) < log_prob(proposal) - log_prob(current), and KEEP the current "
            "state (recording it again) when you reject")


def _check_var(samples, var, what, k=5.0):
    for j in range(samples.shape[1]):
        sq = (samples[:, j] - samples[:, j].mean()) ** 2
        ess = _ess(sq)
        rel = abs(sq.mean() / var[j] - 1)
        tol = k * np.std(sq) / var[j] / np.sqrt(ess)
        assert rel < tol, (
            f"{what}: coordinate {j} has sample variance {sq.mean():.4f}, exact {var[j]:.4f} (relative error "
            f"{rel:.2f}, tolerance {tol:.2f} from the ESS of the squared deviations)")


def _gauss(mean, cov):
    P = np.linalg.inv(cov)
    return lambda x: -0.5 * (x - mean) @ P @ (x - mean)


# ---------------------------------------------------------------- stage 1: Metropolis-Hastings

MEAN = np.array([1.0, -2.0])
COV = np.array([[1.0, 0.6], [0.6, 2.0]])


def test_stage1_correlated_gaussian(nb):
    s, acc = bounded(nb.metropolis_hastings, _gauss(MEAN, COV), np.zeros(2), 6000, proposal_scale=1.5,
                     burn_in=500, rng=0, what="metropolis_hastings", budget=1_000_000)
    s = np.asarray(s)
    assert s.shape == (6000, 2), f"samples should have shape (n_samples, d) = (6000, 2); got {s.shape}"
    assert 0.15 < acc < 0.7, f"acceptance rate for this target and scale should be roughly 0.3-0.5; got {acc:.3f}"
    _check_mean(s, MEAN, np.sqrt(np.diag(COV)), "MH on N([1, -2], [[1, .6], [.6, 2]])")
    _check_var(s, np.diag(COV), "MH on the correlated Gaussian")
    c = np.cov(s.T)[0, 1]
    assert abs(c - 0.6) < 0.25, f"the sample covariance of the two coordinates should be near 0.6; got {c:.3f}"


def test_stage1_gamma_posterior_in_log_space(nb):
    # Poisson counts with a Gamma(2, 1) prior on the rate: the posterior is Gamma(a, b) exactly.
    y = np.array([3, 5, 2, 4, 6, 3, 4])
    a, b = 2 + y.sum(), 1 + len(y)

    def log_post(u):  # u = log(rate); the + u is the Jacobian of rate = exp(u)
        lam = np.exp(u[0])
        return (a - 1) * u[0] - b * lam + u[0]

    s, acc = bounded(nb.metropolis_hastings, log_post, np.array([0.0]), 6000, proposal_scale=0.5, burn_in=300, rng=1,
                     what="metropolis_hastings", budget=1_000_000)
    lam = np.exp(np.asarray(s))
    _check_mean(lam, [a / b], [np.sqrt(a) / b], f"MH on a Gamma({a}, {b}) posterior (sampled in log space)")


def test_stage1_burn_in_thinning_and_calls(nb):
    calls = [0]

    def lp(x):
        calls[0] += 1
        return -0.5 * float(x @ x)

    out, _ = bounded(nb.metropolis_hastings, lp, np.zeros(3), 40, proposal_scale=0.8, burn_in=25, thin=3, rng=7,
                     what="metropolis_hastings")
    assert calls[0] == 1 + 25 + 40 * 3, (
        f"log_prob should be evaluated once at x0 and once per proposal: 1 + burn_in + n_samples * thin = 146 calls "
        f"for burn_in=25, n_samples=40, thin=3; you made {calls[0]}. (re-evaluating the current state wastes work)")
    full, acc_full = nb.metropolis_hastings(lp, np.zeros(3), 25 + 120, proposal_scale=0.8, rng=7)
    want = np.asarray(full)[25 + 2::3]
    moves = np.sum(np.any(np.diff(np.vstack([np.zeros(3), full]), axis=0) != 0, axis=1))
    _, acc = nb.metropolis_hastings(lp, np.zeros(3), 40, proposal_scale=0.8, burn_in=25, thin=3, rng=7)
    assert np.isclose(acc, moves / 145) and np.isclose(acc_full, moves / 145), (
        f"the acceptance rate counts every proposal, burn-in and thinned-away steps included: the chain moved "
        f"{moves} times in 145 steps ({moves / 145:.3f}); got {acc:.3f}")
    assert np.shape(out) == (40, 3) and np.allclose(out, want), (
        "with the same seed, a burn_in=25, thin=3 run must equal an unthinned run of 145 steps with the first 25 "
        "discarded and every 3rd kept after that (steps 28, 31, ..., 145, 1-based)")


def test_stage1_acceptance_and_support(nb):
    lp = lambda x: -0.5 * float(x @ x)
    _, hi = bounded(nb.metropolis_hastings, lp, np.zeros(2), 500, proposal_scale=1e-3, rng=2, what="metropolis_hastings")
    _, lo = bounded(nb.metropolis_hastings, lp, np.zeros(2), 500, proposal_scale=50.0, rng=3, what="metropolis_hastings")
    assert hi > 0.97 and lo < 0.08, (
        f"tiny steps are almost always accepted (got {hi:.3f}), huge steps almost never (got {lo:.3f}). acceptance "
        "rate = accepted proposals / all proposals (including burn-in)")
    box = lambda x: 0.0 if np.all((x >= 0) & (x <= 1)) else -np.inf
    s, _ = bounded(nb.metropolis_hastings, box, np.array([0.5]), 3000, proposal_scale=0.4, rng=4, what="metropolis_hastings")
    s = np.asarray(s)
    assert np.all((s >= 0) & (s <= 1)), "a proposal with log_prob = -inf must never be accepted"
    _check_mean(s, [0.5], [np.sqrt(1 / 12)], "MH on Uniform(0, 1)")


# ---------------------------------------------------------------- stage 2: Gibbs


def test_stage2_sweep_order_and_bookkeeping(nb):
    conds = [lambda x, rng: x[1] + 1.0, lambda x, rng: x[0] * 2.0]
    out = np.asarray(bounded(nb.gibbs_sampler, conds, np.zeros(2), 3, rng=0, what="gibbs_sampler"))
    want = np.array([[1, 2], [3, 6], [7, 14]], dtype=float)
    assert np.allclose(out, want), (
        f"deterministic 'conditionals' x0 <- x1 + 1, x1 <- 2 x0 from (0, 0): each sweep updates x0 FIRST and then x1 "
        f"using the NEW x0, giving (1, 2), (3, 6), (7, 14); got {out.tolist()}")
    out2 = np.asarray(bounded(nb.gibbs_sampler, conds, np.zeros(2), 3, burn_in=1, thin=2, rng=0, what="gibbs_sampler"))
    s, states = np.zeros(2), []
    for _ in range(7):
        s = np.array([s[1] + 1, 0.0])
        s[1] = 2 * s[0]
        states.append(s.copy())
    assert np.allclose(out2, np.array(states)[[2, 4, 6]]), (
        "burn_in=1, thin=2, n_samples=3: run 1 + 3 * 2 = 7 sweeps and keep the states after sweeps 3, 5 and 7")
    x0 = np.array([5.0, 5.0])
    nb.gibbs_sampler(conds, x0, 2, rng=0)
    assert np.array_equal(x0, [5.0, 5.0]), "gibbs_sampler must not modify x0"


def test_stage2_bivariate_normal(nb):
    rho = 0.8
    sd = np.sqrt(1 - rho**2)
    conds = [lambda x, rng: rho * x[1] + sd * rng.standard_normal(),
             lambda x, rng: rho * x[0] + sd * rng.standard_normal()]
    s = np.asarray(bounded(nb.gibbs_sampler, conds, np.array([3.0, -3.0]), 6000, burn_in=200, rng=5,
                           what="gibbs_sampler", budget=1_000_000))
    assert s.shape == (6000, 2)
    _check_mean(s, [0.0, 0.0], [1.0, 1.0], "Gibbs on a standard bivariate normal with rho = 0.8")
    _check_var(s, [1.0, 1.0], "Gibbs on the bivariate normal")
    r = np.corrcoef(s.T)[0, 1]
    assert abs(r - rho) < 0.06, f"the sample correlation should be near {rho}; got {r:.3f}"


def test_stage2_discrete_joint(nb):
    P = np.array([[0.1, 0.3], [0.4, 0.2]])  # P(a, b)

    def cond_a(x, rng):
        p = P[:, int(x[1])] / P[:, int(x[1])].sum()
        return float(rng.random() < p[1])

    def cond_b(x, rng):
        p = P[int(x[0])] / P[int(x[0])].sum()
        return float(rng.random() < p[1])

    s = np.asarray(bounded(nb.gibbs_sampler, [cond_a, cond_b], np.zeros(2), 6000, rng=6, what="gibbs_sampler",
                           budget=1_000_000)).astype(int)
    for a in (0, 1):
        for b in (0, 1):
            ind = ((s[:, 0] == a) & (s[:, 1] == b)).astype(float)
            se = np.sqrt(P[a, b] * (1 - P[a, b]) / _ess(ind))
            assert abs(ind.mean() - P[a, b]) < 4.5 * se, (
                f"Gibbs on a 2x2 table: the frequency of (a={a}, b={b}) is {ind.mean():.3f}, the exact probability "
                f"{P[a, b]} (tolerance {4.5 * se:.3f})")


# ---------------------------------------------------------------- stage 3: autocorrelation


def test_stage3_matches_direct_formula(nb):
    x = np.random.default_rng(10).normal(size=300).cumsum()
    got = np.asarray(nb.autocorrelation(x, max_lag=40))
    y = x - x.mean()
    want = np.array([np.sum(y[: len(y) - k] * y[k:]) for k in range(41)]) / np.sum(y * y)
    assert got.shape == (41,), f"lags 0..max_lag: shape (max_lag + 1,) = (41,); got {got.shape}"
    assert np.allclose(got, want, atol=1e-10), (
        "rho_k = sum_t (x_t - mean)(x_t+k - mean) / sum_t (x_t - mean)^2: divide by the lag-0 sum (not by n - k), "
        "and zero-pad the FFT to at least 2n - 1 so the end of the chain doesn't wrap around onto the start")
    full = np.asarray(nb.autocorrelation(x[:50]))
    assert full.shape == (50,) and np.isclose(full[0], 1.0), "max_lag=None returns every lag 0..n-1, with rho_0 = 1"
    flat = np.asarray(nb.autocorrelation(np.full(20, 3.0), max_lag=5))
    assert np.allclose(flat, 1.0), (
        "a chain that never moves (a sampler that rejects everything) has zero variance: return all ones (perfectly "
        f"correlated) instead of dividing by zero; got {flat.tolist()}")


def test_stage3_ar1_and_speed(nb):
    rng = np.random.default_rng(11)
    phi, n = 0.7, 50_000
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    t0 = time.perf_counter()
    got = np.asarray(bounded(nb.autocorrelation, x, what="autocorrelation"))
    t_learner = time.perf_counter() - t0
    assert got.shape == (n,) and np.allclose(got[:11], phi ** np.arange(11), atol=0.03), (
        f"an AR(1) chain x_t = 0.7 x_t-1 + noise has autocorrelation 0.7^k; got {np.round(got[:5], 3).tolist()}")
    best = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        f = np.fft.rfft(x - x.mean(), 2 * n)
        np.fft.irfft(f * np.conj(f))
        best = min(best, time.perf_counter() - t0)
    assert t_learner < 30 * best + 0.05, (
        f"all 50,000 lags of a 50,000-step chain took {t_learner:.2f} s, against {best * 1000:.1f} ms for one FFT: "
        "use np.fft (O(n log n)), not a loop over lags (O(n^2))")


# ---------------------------------------------------------------- stage 4: effective sample size


def _ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_stage4_exact_definition(nb):
    for phi, seed in [(0.5, 20), (0.95, 21), (0.0, 22), (-0.3, 24), (-0.6, 23)]:
        x = _ar1(phi, 2000, seed)
        got = nb.effective_sample_size(x)
        want = _ess(x)
        assert np.isclose(got, want, rtol=1e-8), (
            f"AR(1) with phi = {phi}: ESS = n / tau with tau = -1 + 2 * (sum of the pair sums rho_2m + rho_2m+1, "
            f"stopping BEFORE the first pair sum <= 0), and tau floored at 1 / log10(n). expected {want:.2f}, got {got}")


def test_stage4_statistical(nb):
    n = 20_000
    for phi in [0.0, 0.9]:
        want = n * (1 - phi) / (1 + phi)
        got = float(np.median([nb.effective_sample_size(_ar1(phi, n, 30 + k)) for k in range(3)]))
        assert abs(got / want - 1) < 0.3, (
            f"AR(1) with phi = {phi}: the true ESS is n (1 - phi) / (1 + phi) = {want:.0f}; the median over three "
            f"chains of your estimate is {got:.0f}. (each estimate is noisy, about 9% for phi = 0.9, so this only "
            "catches gross errors)")
    X = np.stack([_ar1(0.5, 3000, 31), _ar1(0.9, 3000, 32)], axis=1)
    got = np.asarray(nb.effective_sample_size(X))
    assert got.shape == (2,) and got[0] > 2 * got[1], "a (n, d) array gives one ESS per column; the stickier chain has the smaller one"


# ---------------------------------------------------------------- stage 5: R-hat


def _rhat(ch):
    half = ch.shape[1] // 2
    s = np.concatenate([ch[:, :half], ch[:, -half:]], axis=0)
    W = np.mean(np.var(s, axis=1, ddof=1))
    B = half * np.var(s.mean(axis=1), ddof=1)
    return np.sqrt(((half - 1) / half * W + B / half) / W)


def test_stage5_worked_example(nb):
    got = nb.r_hat(np.array([[0.0, 2.0, 1.0, 3.0], [4.0, 6.0, 5.0, 7.0]]))
    assert np.isclose(got, np.sqrt(10 / 3)), (
        f"lesson worked example: halves (0, 2), (1, 3), (4, 6), (5, 7) give W = 2, B = 2 * 17/3, var+ = 20/3, "
        f"R-hat = sqrt(10/3) = 1.826; got {got}")


def test_stage5_formula(nb):
    rng = np.random.default_rng(40)
    for n in (100, 101):
        ch = rng.normal(size=(4, n)) + np.array([[0.0], [0.1], [0.3], [-0.2]])
        got = nb.r_hat(ch)
        assert np.isclose(got, _rhat(ch), rtol=1e-10), (
            f"n = {n}: split each chain into its first and last floor(n/2) draws (dropping the middle draw when n is "
            f"odd), then R-hat = sqrt(((n'-1)/n' W + B/n') / W) with n' = floor(n/2). expected {_rhat(ch):.6f}, got {got}")
    ch3 = rng.normal(size=(3, 200, 2))
    got = np.asarray(nb.r_hat(ch3))
    assert got.shape == (2,) and np.allclose(got, [_rhat(ch3[:, :, 0]), _rhat(ch3[:, :, 1])]), "(m, n, d) gives one R-hat per coordinate"


def test_stage5_detects_problems(nb):
    rng = np.random.default_rng(41)
    good = rng.normal(size=(4, 2000))
    assert nb.r_hat(good) < 1.01, f"four i.i.d. chains from the same distribution: R-hat ~ 1; got {nb.r_hat(good):.4f}"
    stuck = good + np.array([[0.0], [0.0], [0.0], [2.0]])
    assert nb.r_hat(stuck) > 1.2, f"one chain shifted by two sds: R-hat should be about 1.35; got {nb.r_hat(stuck):.3f}"
    trend = good + np.linspace(-1.5, 1.5, 2000)  # every chain drifts the same way
    h = 1000
    whole = np.sqrt(((h * 2 - 1) / (h * 2) * np.mean(np.var(trend, axis=1, ddof=1))
                     + np.var(trend.mean(1), ddof=1)) / np.mean(np.var(trend, axis=1, ddof=1)))
    assert whole < 1.01 < nb.r_hat(trend) and nb.r_hat(trend) > 1.2, (
        f"four chains that all drift together agree with each other (unsplit R-hat {whole:.3f}), but each one isn't "
        f"stationary: SPLIT R-hat compares first halves with second halves and must flag it (> 1.2); got "
        f"{nb.r_hat(trend):.3f}")
