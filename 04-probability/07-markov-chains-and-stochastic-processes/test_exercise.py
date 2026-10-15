"""Checks for exercise.py (markov chains & stochastic processes).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
import math

import numpy as np
import pytest

# the lesson's weather chain: 0 = sunny, 1 = rainy
WEATHER = np.array([[0.9, 0.1], [0.5, 0.5]])
# a lopsided 4-state chain (not symmetric, not doubly stochastic)
P4 = np.array([[0.1, 0.6, 0.3, 0.0],
               [0.0, 0.2, 0.5, 0.3],
               [0.4, 0.0, 0.1, 0.5],
               [0.7, 0.1, 0.0, 0.2]])


def _finishes(fn, *args, budget=300_000):
    """call fn(*args) under a line-count budget, in this thread; fail (instead of hanging the check)
    if it runs too long. raising inside the traced code really stops a runaway loop."""
    import sys

    class _Runaway(Exception):
        pass

    count = [0]

    def tracer(frame, event, arg):
        if event == "line":
            count[0] += 1
            if count[0] > budget:
                raise _Runaway
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        return fn(*args)
    except _Runaway:
        raise AssertionError(
            f"{fn.__name__} didn't finish after {budget:,} steps on this chain. if you iterate p @ P until it "
            "stops changing, that never happens for a periodic chain: solve the linear system instead.") from None
    finally:
        sys.settrace(old)


def _pi_true(P):
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, np.argmin(np.abs(w - 1))])
    return v / v.sum()


def _cycle(n, lazy=0.5):
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] += lazy
        P[i, (i + 1) % n] += (1 - lazy) / 2
        P[i, (i - 1) % n] += (1 - lazy) / 2
    return P


# ---------------------------------------------------------------- stage 1: evolving a distribution


def test_stage1_worked_example(nb):
    out = np.asarray(nb.evolve([1.0, 0.0], WEATHER, 2))
    assert out.shape == (3, 2), f"evolve(p0, P, 2) returns p_0, p_1, p_2: shape (3, 2), got {out.shape}"
    assert np.allclose(out, [[1, 0], [0.9, 0.1], [0.86, 0.14]]), (
        f"starting sunny: p1 = (0.9, 0.1), p2 = (0.9*0.9 + 0.1*0.5, 0.9*0.1 + 0.1*0.5) = (0.86, 0.14). got "
        f"{np.round(out, 4).tolist()}. multiply the ROW vector on the left: p @ P, not P @ p.")


def test_stage1_rows_stay_distributions(nb):
    p0 = np.array([0.25, 0.25, 0.25, 0.25])
    out = np.asarray(nb.evolve(p0, P4, 30))
    assert np.allclose(out.sum(axis=1), 1) and np.all(out >= -1e-15), "every p_t must be a distribution (non-negative, sums to 1)"
    assert np.allclose(out[0], p0), "row 0 is the starting distribution itself"
    assert np.allclose(out[5], p0 @ np.linalg.matrix_power(P4, 5)), "p_5 should equal p_0 @ P^5"


# ---------------------------------------------------------------- stage 2: simulation


def test_stage2_path_basics(nb):
    path = np.asarray(nb.simulate_markov_chain(P4, 2, 50, rng=0))
    assert path.shape == (51,), f"n_steps = 50 gives 51 states (including the start), got shape {path.shape}"
    assert np.issubdtype(path.dtype, np.integer), f"states should be ints, got dtype {path.dtype}"
    assert path[0] == 2, f"the path should start at x0 = 2, got {path[0]}"
    for a, b in zip(path[:-1], path[1:]):
        assert P4[a, b] > 0, (
            f"the path jumps {a} -> {b}, but P[{a}, {b}] = 0: that transition is impossible. draw the next "
            "state from row P[current state].")


class _Zeros(np.random.Generator):
    """a Generator whose .random() always returns 0.0, which a real one does once in 2^53 draws."""

    def __init__(self):
        super().__init__(np.random.PCG64(0))

    def random(self, size=None, dtype=np.float64, out=None):
        return 0.0 if size is None else np.zeros(size)


def test_stage2_never_takes_an_impossible_step(nb):
    P = np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # state 0 can never be entered
    path = np.asarray(nb.simulate_markov_chain(P, 1, 6, rng=_Zeros()))
    assert np.all(path[1:] != 0), (
        f"with u = 0.0 (rng.random() can return exactly 0), your path {path.tolist()} entered state 0, whose "
        "probability is 0 from every state. with np.searchsorted(cdf_row, u), use side='right' so that a "
        "zero-probability state (an empty interval of the cumulative sums) is skipped.")


def test_stage2_transition_frequencies(nb):
    n = 100_000
    path = np.asarray(nb.simulate_markov_chain(P4, 0, n, rng=1))
    counts = np.zeros((4, 4))
    np.add.at(counts, (path[:-1], path[1:]), 1)
    freq = counts / counts.sum(axis=1, keepdims=True)
    se = np.sqrt(P4 * (1 - P4) / counts.sum(axis=1, keepdims=True))
    worst = np.max(np.abs(freq - P4) / np.maximum(se, 1e-12))
    assert worst < 6, (
        f"the observed transition frequencies should match P (within 6 standard errors), but the worst entry "
        f"is {worst:.1f} SE off.\nobserved\n{np.round(freq, 3)}\nexpected\n{P4}")


def test_stage2_reproducible_and_absorbing(nb):
    a = np.asarray(nb.simulate_markov_chain(P4, 1, 30, rng=5))
    b = np.asarray(nb.simulate_markov_chain(P4, 1, 30, rng=5))
    assert np.array_equal(a, b), "the same seed must give the same path"
    absorbing = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.5]])
    path = np.asarray(nb.simulate_markov_chain(absorbing, 1, 20, rng=2))
    assert np.all(path == 1), "state 1 is absorbing (P[1, 1] = 1): a path started there never leaves"


def test_stage2_validates(nb):
    with pytest.raises(ValueError):
        nb.simulate_markov_chain(np.array([[0.5, 0.4], [0.5, 0.5]]), 0, 5)
        pytest.fail("row 0 sums to 0.9: that's not a transition matrix, raise ValueError")
    with pytest.raises(ValueError):
        nb.simulate_markov_chain(np.array([[1.2, -0.2], [0.5, 0.5]]), 0, 5)
        pytest.fail("negative entries aren't probabilities: raise ValueError")


# ---------------------------------------------------------------- stage 3: stationary distribution


def test_stage3_worked_example(nb):
    pi = np.asarray(nb.stationary_distribution(WEATHER))
    assert np.allclose(pi, [5 / 6, 1 / 6]), (
        f"the weather chain's stationary distribution is (5/6, 1/6): pi_0 * 0.1 = pi_1 * 0.5 balances the flow. "
        f"got {np.round(pi, 4).tolist()}. "
        + ("(0.5, 0.5) is the RIGHT eigenvector P v = v (the all-ones vector, normalized): you need pi P = pi."
           if np.allclose(pi, 0.5) else ""))


def test_stage3_lopsided_chain(nb):
    pi = np.asarray(nb.stationary_distribution(P4))
    assert pi.shape == (4,) and abs(pi.sum() - 1) < 1e-12 and np.all(pi >= 0), "pi is a distribution over the 4 states"
    assert np.allclose(pi @ P4, pi, atol=1e-12), f"pi @ P should equal pi, got {np.round(pi @ P4, 4)} vs {np.round(pi, 4)}"
    assert np.allclose(pi, _pi_true(P4)), "pi should be the left eigenvector of P for eigenvalue 1, normalized"


def test_stage3_periodic_chain(nb):
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    pi = np.asarray(_finishes(nb.stationary_distribution, flip))
    assert np.allclose(pi, [0.5, 0.5]), (
        f"the chain that always switches state has stationary distribution (1/2, 1/2), got {pi.tolist()}. "
        "repeated multiplication p @ P never converges here (it oscillates), so solve the linear system instead.")
    # period 2 with a NON-uniform stationary distribution: 0 -> 1 -> (0 or 2) -> 1 -> ...
    walk = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    pi = np.asarray(_finishes(nb.stationary_distribution, walk))
    assert np.allclose(pi, [0.25, 0.5, 0.25]), (
        f"the periodic walk 0 <-> 1 <-> 2 spends half its time in the middle: pi = (1/4, 1/2, 1/4), got "
        f"{np.round(pi, 4).tolist()}. iterating p @ P from the uniform start oscillates between (1/6, 2/3, 1/6) "
        "and (1/3, 1/3, 1/3) and never settles: solve pi P = pi, sum(pi) = 1 as a linear system.")


# ---------------------------------------------------------------- stage 4: mixing


def test_stage4_total_variation(nb):
    tv = nb.total_variation([0.5, 0.5, 0.0], [0.2, 0.3, 0.5])
    assert abs(tv - 0.5) < 1e-12, (
        f"TV between (.5, .5, 0) and (.2, .3, .5) is (|.3| + |.2| + |-.5|) / 2 = 0.5, got {tv}. it's HALF the L1 distance.")
    assert nb.total_variation([1.0, 0.0], [0.0, 1.0]) == 1.0, "distributions with disjoint supports have TV 1"


def test_stage4_spectral_gap(nb):
    g = nb.spectral_gap(WEATHER)
    assert abs(g - 0.6) < 1e-12, f"the weather chain's eigenvalues are 1 and 0.4, so the gap is 0.6; got {g}"
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert abs(nb.spectral_gap(flip)) < 1e-12, (
        "the flip chain has eigenvalues 1 and -1: |lambda_2| = 1, gap 0 (it never mixes). use the MODULUS.")
    for P in (P4, _cycle(8)):  # several distinct eigenvalue moduli
        mods = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        got = nb.spectral_gap(P)
        assert abs(got - (1 - mods[1])) < 1e-10, (
            f"eigenvalue moduli (sorted, largest first) {np.round(mods, 4).tolist()}: the gap is 1 - (the SECOND "
            f"largest) = {1 - mods[1]:.4f}, got {got:.4f}")
    rot = np.array([[0.2, 0.8, 0.0], [0.0, 0.2, 0.8], [0.8, 0.0, 0.2]])  # complex eigenvalues
    lam = np.sort(np.abs(np.linalg.eigvals(rot)))[::-1]
    assert abs(nb.spectral_gap(rot) - (1 - lam[1])) < 1e-12, "with complex eigenvalues, compare their moduli np.abs(...)"


def test_stage4_mixing_time_hand_computed(nb):
    # weather chain: from sunny the TV to pi after t steps is (1/6) 0.4^t, from rainy (5/6) 0.4^t
    for eps, want in ((0.25, 2), (0.01, 5)):
        got = nb.mixing_time(WEATHER, eps)
        assert got == want, (
            f"from the worst start (rainy) the distance after t steps is (5/6) 0.4^t; the first t with that <= "
            f"{eps} is {want}, got {got}. take the MAX over starting states.")
    assert nb.mixing_time(np.eye(3) * 0 + 1 / 3, 0.25) == 1, "a chain that jumps straight to uniform mixes in one step"
    assert nb.mixing_time(WEATHER, 1.0) == 0, (
        "with eps = 1 every distribution is already close enough at t = 0 (TV is never above 1): check t = 0 "
        "before taking any step")


def test_stage4_mixing_time_lazy_cycle(nb):
    P = _cycle(8)
    pi = np.full(8, 1 / 8)
    want, D = None, np.eye(8)
    for t in range(500):
        if 0.5 * np.abs(D - pi).sum(axis=1).max() <= 0.25:
            want = t
            break
        D = D @ P
    got = nb.mixing_time(P)
    assert got == want, f"the lazy walk on a cycle of 8 mixes (eps = 0.25) at t = {want}, got {got}"


def test_stage4_never_mixes(nb):
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert nb.mixing_time(flip, 0.25, max_steps=100) is None, (
        "the flip chain stays at TV distance 1/2 from (1/2, 1/2) forever: return None after max_steps")
    assert nb.mixing_time(flip, 0.5, max_steps=100) == 0, (
        "with eps = 1/2, the flip chain's distance of exactly 1/2 counts as mixed (the definition is "
        "d(t) <= eps), already at t = 0")


# ---------------------------------------------------------------- stage 5: Brownian motion


def test_stage5_shape_and_start(nb):
    B = np.asarray(nb.brownian_motion(7, 50, dt=0.02, rng=0))
    assert B.shape == (7, 51), f"7 paths of 50 steps include the starting point: (7, 51), got {B.shape}"
    assert np.all(B[:, 0] == 0), "every path starts at 0"
    assert np.array_equal(B, np.asarray(nb.brownian_motion(7, 50, dt=0.02, rng=0))), "same seed, same paths"


def test_stage5_increments(nb):
    dt = 0.01
    B = np.asarray(nb.brownian_motion(4000, 100, dt=dt, rng=1))
    inc = np.diff(B, axis=1)
    sd = inc.std()
    assert abs(sd - math.sqrt(dt)) < 0.003, (
        f"increments should be N(0, dt) with dt = {dt}: standard deviation sqrt(dt) = 0.1, got {sd:.4f}. "
        + ("you used dt as the standard deviation: the VARIANCE of an increment is dt." if abs(sd - dt) < 0.002 else ""))
    r = np.corrcoef(inc[:, :-1].ravel(), inc[:, 1:].ravel())[0, 1]
    assert abs(r) < 0.01, f"successive increments must be independent, but their correlation is {r:.3f}"


def test_stage5_variance_grows_linearly(nb):
    B = np.asarray(nb.brownian_motion(20_000, 200, dt=0.005, rng=2))
    for k, T in ((50, 0.25), (200, 1.0)):
        v = B[:, k].var()
        assert abs(v - T) < 0.06 * T, (
            f"Var(B_t) = t: at t = {T} the variance across paths should be {T}, got {v:.4f}. the path is the "
            "running SUM of the increments (np.cumsum), with a 0 in front.")
