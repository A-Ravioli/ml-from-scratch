"""Checks for exercise.py (hamiltonian monte carlo).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The integrator is
checked by hand on the harmonic oscillator and for the three properties HMC needs: time
reversibility, volume preservation (a numerical Jacobian determinant of 1) and an energy error
that shrinks like step_size^2. The sampler is checked against EXACT posteriors (a correlated
Gaussian, a badly scaled Gaussian with a mass matrix, a Gamma posterior in log space) with
tolerances of 4.5 Monte Carlo standard errors from a reference ESS in this file; bookkeeping is
checked exactly with seeded runs. Every sampler call runs under mlfs.testing.bounded.
"""
import numpy as np

from mlfs.testing import bounded


def _acf(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / ac[0]


def _ess(x):
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


def _check_moments(s, mean, var, what):
    for j in range(s.shape[1]):
        ess = _ess(s[:, j])
        se = np.sqrt(var[j] / ess)
        err = abs(s[:, j].mean() - mean[j])
        assert err < 4.5 * se, (
            f"{what}: coordinate {j} has sample mean {s[:, j].mean():.4f}, exact {mean[j]:.4f}; that's {err / se:.1f} "
            f"Monte Carlo standard errors (ESS {ess:.0f}). check the accept rule (accept if log u < H_old - H_new) and "
            "that a rejected step keeps the OLD position")
        sq = (s[:, j] - s[:, j].mean()) ** 2
        rel = abs(sq.mean() / var[j] - 1)
        tol = 5 * np.std(sq) / var[j] / np.sqrt(_ess(sq))
        assert rel < tol, (
            f"{what}: coordinate {j} has sample variance {sq.mean():.4f}, exact {var[j]:.4f} (relative error {rel:.2f}, "
            f"tolerance {tol:.2f}). if you use inv_mass: draw p ~ N(0, M) with M = 1 / inv_mass, and use the same "
            "inv_mass in the kinetic energy and in the position update")


def _leap(nb, *args, **kwargs):
    return bounded(nb.leapfrog, *args, what="leapfrog", budget=300_000,
                   hint="loop exactly n_steps times (for i in range(n_steps)).", **kwargs)


def _lf(q, p, g, eps, n, m=1.0):
    q, p = np.array(q, float), np.array(p, float)
    for _ in range(n):
        p = p + 0.5 * eps * g(q)
        q = q + eps * m * p
        p = p + 0.5 * eps * g(q)
    return q, p


def _quartic(q):
    return -0.25 * np.sum(q**4) - 0.5 * q[0] * q[1]


def _quartic_grad(q):
    g = -q**3
    g = g - 0.5 * q[::-1]
    return g


# ---------------------------------------------------------------- stage 1: energy and leapfrog


def test_stage1_hamiltonian(nb):
    lp = lambda q: -0.5 * float(q @ q)
    H = nb.hamiltonian(np.array([1.0, 2.0]), np.array([3.0, 4.0]), lp)
    assert np.isclose(H, 2.5 + 12.5), f"H = -log_prob(q) + |p|^2 / 2 = 2.5 + 12.5 = 15; got {H}"
    H2 = nb.hamiltonian(np.array([1.0, 2.0]), np.array([3.0, 4.0]), lp, inv_mass=np.array([2.0, 0.5]))
    assert np.isclose(H2, 2.5 + 13.0), (
        f"with inv_mass (2, 0.5) the kinetic energy is 1/2 (2 * 9 + 0.5 * 16) = 13, so H = 15.5; got {H2}")


def test_stage1_leapfrog_by_hand(nb):
    g = lambda q: -q  # standard normal: log_prob = -q^2 / 2
    q, p = _leap(nb, np.array([1.0]), np.array([0.0]), g, 0.5, 1)
    assert np.allclose(q, [0.875]) and np.allclose(p, [-0.46875]), (
        f"lesson worked example, one step of size 0.5 from (q, p) = (1, 0): p_half = -0.25, q = 0.875, "
        f"p = -0.25 - 0.25 * 0.875 = -0.46875; got q = {np.asarray(q).tolist()}, p = {np.asarray(p).tolist()}")
    q0, p0 = np.array([0.3, -1.2]), np.array([0.7, 0.4])
    q, p = _leap(nb, q0, p0, _quartic_grad, 0.05, 37)
    qw, pw = _lf(q0, p0, _quartic_grad, 0.05, 37)
    assert np.allclose(q, qw, atol=1e-12) and np.allclose(p, pw, atol=1e-12), (
        "37 steps on a non-Gaussian target disagree with half-step / full-step / half-step leapfrog")
    assert np.array_equal(q0, [0.3, -1.2]) and np.array_equal(p0, [0.7, 0.4]), "leapfrog must not modify its inputs"
    q, p = _leap(nb, q0, p0, _quartic_grad, 0.05, 0)
    assert np.allclose(q, q0) and np.allclose(p, p0), "zero steps changes nothing"
    m = np.array([4.0, 0.25])
    q, p = _leap(nb, q0, p0, _quartic_grad, 0.05, 20, inv_mass=m)
    qw, pw = _lf(q0, p0, _quartic_grad, 0.05, 20, m)
    assert np.allclose(q, qw, atol=1e-12) and np.allclose(p, pw, atol=1e-12), (
        "with inv_mass the position update is q += step * inv_mass * p (the velocity is M^-1 p)")


def test_stage1_gradient_calls(nb):
    calls = [0]

    def g(q):
        calls[0] += 1
        return _quartic_grad(q)

    _leap(nb, np.array([0.1, 0.2]), np.array([1.0, -1.0]), g, 0.1, 25)
    assert calls[0] <= 26, (
        f"25 leapfrog steps need only 26 gradient evaluations (merge each step's second half-kick with the next "
        f"step's first); you made {calls[0]}")


def test_stage1_reversible_and_volume_preserving(nb):
    q0, p0 = np.array([0.8, -0.5]), np.array([-0.3, 1.1])
    q1, p1 = _leap(nb, q0, p0, _quartic_grad, 0.1, 30)
    q2, p2 = _leap(nb, q1, -np.asarray(p1), _quartic_grad, 0.1, 30)
    assert np.allclose(q2, q0, atol=1e-10) and np.allclose(-np.asarray(p2), p0, atol=1e-10), (
        "time reversibility: flipping the momentum and integrating again must return exactly to the start")
    z0 = np.concatenate([q0, p0])

    def flow(z):
        q, p = nb.leapfrog(z[:2], z[2:], _quartic_grad, 0.1, 30)  # already shown to terminate above
        return np.concatenate([q, p])

    h = 1e-6
    J = np.stack([(flow(z0 + h * e) - flow(z0 - h * e)) / (2 * h) for e in np.eye(4)], axis=1)
    det = np.linalg.det(J)
    assert abs(det - 1) < 1e-5, (
        f"leapfrog preserves volume in (q, p) space: the Jacobian determinant of 30 steps should be 1; got {det:.8f}")


def test_stage1_energy_error_is_second_order(nb):
    lp = lambda q: -0.5 * (q[0] ** 2 + 4 * q[1] ** 2)
    g = lambda q: -np.array([q[0], 4 * q[1]])
    q0, p0 = np.array([1.0, 0.5]), np.array([0.3, -0.7])
    H0 = nb.hamiltonian(q0, p0, lp)
    errs = []
    for eps in (0.1, 0.05, 0.025):
        n = int(round(2.0 / eps))
        worst = 0.0
        q, p = q0, p0
        for _ in range(n):
            q, p = _leap(nb, q, p, g, eps, 1)
            worst = max(worst, abs(nb.hamiltonian(q, p, lp) - H0))
        errs.append(worst)
    r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
    assert 3.0 < r1 < 5.0 and 3.0 < r2 < 5.0, (
        f"over a fixed time, halving the step size should cut the energy error by about 4 (it's O(step^2)); got "
        f"errors {[f'{e:.2e}' for e in errs]}, ratios {r1:.2f}, {r2:.2f}. a ratio near 2 means a first-order "
        "integrator (full momentum steps instead of half steps at the ends)")


# ---------------------------------------------------------------- stage 2: the sampler

COV = np.array([[1.0, 0.9], [0.9, 1.0]])
PREC = np.linalg.inv(COV)


def test_stage2_correlated_gaussian(nb):
    lp = lambda x: -0.5 * x @ PREC @ x
    g = lambda x: -PREC @ x
    s, acc = bounded(nb.hmc, lp, np.array([2.0, -2.0]), 2000, grad_log_prob=g, step_size=0.15, n_leapfrog=12,
                     burn_in=100, rng=0, what="hmc", budget=2_000_000)
    s = np.asarray(s)
    assert s.shape == (2000, 2), f"samples have shape (n_samples, d) = (2000, 2); got {s.shape}"
    assert 0.6 < acc <= 1.0, f"step 0.15 on this target should accept most proposals; got {acc:.3f}"
    _check_moments(s, [0, 0], [1, 1], "HMC on a Gaussian with correlation 0.9")
    r = np.corrcoef(s.T)[0, 1]
    assert abs(r - 0.9) < 0.06, f"the sample correlation should be near 0.9; got {r:.3f}"


def test_stage2_gamma_posterior(nb):
    y = np.array([3, 5, 2, 4, 6, 3, 4])
    a, b = 2 + y.sum(), 1 + len(y)
    lp = lambda u: a * u[0] - b * np.exp(u[0])  # log-rate parameterization, Jacobian included
    g = lambda u: np.array([a - b * np.exp(u[0])])
    s, _ = bounded(nb.hmc, lp, np.array([0.0]), 2000, grad_log_prob=g, step_size=0.08, n_leapfrog=10, burn_in=100,
                   rng=1, what="hmc", budget=2_000_000)
    lam = np.exp(np.asarray(s))
    ess = _ess(lam[:, 0])
    se = np.sqrt(a) / b / np.sqrt(ess)
    assert abs(lam.mean() - a / b) < 4.5 * se, (
        f"HMC on a Gamma({a}, {b}) posterior (in log space): E[rate] = {a / b:.4f}, sample mean {lam.mean():.4f}, "
        f"standard error {se:.4f}")


def test_stage2_mass_matrix(nb):
    sd = np.array([10.0, 0.1])
    lp = lambda x: -0.5 * np.sum((x / sd) ** 2)
    g = lambda x: -x / sd**2
    s, acc = bounded(nb.hmc, lp, np.zeros(2), 1500, grad_log_prob=g, step_size=0.3, n_leapfrog=8, burn_in=100,
                     inv_mass=sd**2, rng=2, what="hmc", budget=2_000_000)
    assert acc > 0.8, (
        f"with inv_mass equal to the target's variances the problem looks isotropic, so step 0.3 is easy; acceptance "
        f"should be high, got {acc:.3f}")
    _check_moments(np.asarray(s), [0, 0], sd**2, "HMC with a diagonal mass matrix on N(0, diag(100, 0.01))")
    _, acc2 = bounded(nb.hmc, lp, np.zeros(2), 200, grad_log_prob=g, step_size=0.3, n_leapfrog=8, rng=3, what="hmc")
    assert acc2 < 0.1, (
        f"without the mass matrix, step 0.3 is far beyond the stability limit 2 * 0.1 of the narrow direction: "
        f"nearly everything should be rejected; got acceptance {acc2:.3f}")


def test_stage2_bookkeeping_and_divergence(nb):
    lp = lambda x: -0.5 * float(x @ x)
    g = lambda x: -x
    a, _ = bounded(nb.hmc, lp, np.array([1.0, 1.0]), 30, grad_log_prob=g, step_size=0.3, n_leapfrog=5, burn_in=10,
                   rng=4, what="hmc")
    b, _ = bounded(nb.hmc, lp, np.array([1.0, 1.0]), 40, grad_log_prob=g, step_size=0.3, n_leapfrog=5, rng=4,
                   what="hmc")
    assert np.allclose(a, np.asarray(b)[10:]), (
        "with the same seed, burn_in=10 and n_samples=30 must equal the last 30 of an n_samples=40 run without burn-in")
    s, acc = bounded(nb.hmc, lp, np.array([0.5]), 100, grad_log_prob=g, step_size=3.0, n_leapfrog=10, rng=5,
                     what="hmc")
    s = np.asarray(s)
    assert np.all(np.abs(s) < 5) and acc < 0.1, (
        f"step size 3 on a unit Gaussian is beyond leapfrog's stability limit (2): trajectories explode, the energy "
        f"error is huge, and (almost) every proposal must be rejected, so the chain stays near where it started. got "
        f"acceptance {acc:.2f}, largest |sample| {np.max(np.abs(s)):.3g}. a rejected trajectory must leave q unchanged")


def test_stage2_moderate_step_size(nb):
    # large steps on a unit Gaussian: acceptance is well below 1, so the accept/reject rule really matters
    lp = lambda x: -0.5 * float(x @ x)
    g = lambda x: -x
    for eps in (1.2, 1.6):
        s, acc = bounded(nb.hmc, lp, np.array([0.3]), 3000, grad_log_prob=g, step_size=eps, n_leapfrog=3,
                         burn_in=100, rng=11, what="hmc", budget=2_000_000)
        assert 0.2 < acc < 0.95, f"step {eps}, 3 leapfrog steps on N(0, 1): acceptance should be moderate; got {acc:.2f}"
        _check_moments(np.asarray(s), [0.0], [1.0],
                       f"HMC on N(0, 1) with step {eps} (acceptance {acc:.2f}): the accept test is log u < H_old - H_new "
                       "exactly (no scaling), and a rejected step records the OLD position")


# ---------------------------------------------------------------- stage 3: gradients for free


def _banana(x):
    return -0.5 * (x[0] * x[0] + 4.0 * (x[1] - 0.5 * x[0] * x[0]) ** 2)


def _banana_grad(x):
    r = x[1] - 0.5 * x[0] ** 2
    return np.array([-x[0] + 4.0 * r * x[0], -4.0 * r])


def test_stage3_autodiff_default(nb):
    kw = dict(step_size=0.2, n_leapfrog=6, burn_in=5, rng=6)
    a, acc_a = bounded(nb.hmc, _banana, np.array([0.5, 0.2]), 25, grad_log_prob=_banana_grad, what="hmc", **kw)
    b, acc_b = bounded(nb.hmc, _banana, np.array([0.5, 0.2]), 25, what="hmc", budget=3_000_000, **kw)
    assert np.allclose(a, b, atol=1e-9) and acc_a == acc_b, (
        "with grad_log_prob=None, hmc must use mlfs.autodiff.grad(log_prob, x): the same seed must give the same "
        "samples as passing the exact gradient (forward-mode autodiff is exact up to rounding)")


# ---------------------------------------------------------------- stage 4: U-turns


def test_stage4_no_u_turn(nb):
    assert nb.no_u_turn([0.0], [1.0], [1.0], [0.5]), "both ends still moving apart: keep going"
    assert not nb.no_u_turn([0.0], [1.0], [1.0], [-0.5]), "the front end turned back towards the start: stop"
    assert not nb.no_u_turn([0.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [1.0, 0.0]), (
        "the back end is moving towards the front ((q+ - q-) . p- < 0): stop")
    assert nb.no_u_turn([0.0, 0.0], [1.0, 1.0], [1.0, -0.5], [0.0, 2.0])


def test_stage4_u_turn_steps(nb):
    g = lambda q: -q
    s = bounded(nb.u_turn_steps, np.array([1.0]), np.array([0.0]), g, 0.1, what="u_turn_steps")
    q, p, want = np.array([1.0]), np.array([0.0]), None
    for k in range(1, 1001):
        q, p = _lf(q, p, g, 0.1, 1)
        if (q - 1.0) @ p < 0:
            want = k
            break
    assert s == want and 30 <= s <= 33, (
        f"from q = 1 at rest on a unit Gaussian, the particle swings to q = -1 and turns around after half a period "
        f"(time pi): about pi / 0.1 = 31 steps. expected {want}, got {s}. count steps until (q - q0) . p < 0")
    g2 = lambda q: -q / np.array([1.0, 25.0])
    s2 = nb.u_turn_steps(np.array([0.0, 0.0]), np.array([0.0, 1.0]), g2, 0.1)
    assert 75 <= s2 <= 82, (
        f"starting AT THE MODE and moving along a direction with sd 5 (period 10 pi), the particle reaches the far "
        f"point of its swing, and the U-turn test fires, after a quarter period, 5 pi / 2 = 7.85 time units, i.e. "
        f"about 79 steps of 0.1 (the trajectory length adapts to the scale); got {s2}")
    flat = lambda q: np.zeros_like(q)
    s3 = bounded(nb.u_turn_steps, np.array([0.0]), np.array([1.0]), flat, 0.1, 50, what="u_turn_steps")
    assert s3 == 50, f"on a flat target the particle never turns: return max_steps (50); got {s3}"
