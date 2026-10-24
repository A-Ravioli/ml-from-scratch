"""Checks for 28 · 07 alignment.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. all NumPy. the KL-regularised
policy against its formula and its optimality (no random policy scores higher on E[r] - beta KL), with
overflow-prone rewards; the statistics by hand; best-of-n against a Monte Carlo simulation, its closed form for
the best item, ties, and the log n - (n-1)/n KL limit; the corridor MDP's tables, the proxy-optimal policy
on both sides of the gaming threshold derived in the lesson, rollouts; the eval's pass rates, intervals and
paired comparison.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded


# ---------------------------------------------------------------- stage 1: KL-regularised optimisation


def test_stage1_formula(nb):
    ref = np.array([0.5, 0.3, 0.2])
    r = np.array([0.0, 1.0, 2.0])
    got = nb.kl_regularized_policy(ref, r, 0.5)
    want = ref * np.exp(r / 0.5)
    want /= want.sum()
    assert got.shape == (3,) and np.allclose(got, want), (
        f"pi(i) = ref(i) exp(r(i) / beta) / Z = {np.round(want, 4)}; got {np.round(got, 4)}")
    assert np.allclose(nb.kl_regularized_policy(ref, r, 1e6), ref, atol=1e-5), "beta -> infinity: stay at the reference"
    assert nb.kl_regularized_policy(ref, r, 1e-3)[2] > 0.999, "beta -> 0: all mass on the highest reward"
    big = nb.kl_regularized_policy(ref, np.array([1000.0, 1001.0, 999.0]), 0.1)
    assert np.isfinite(big).all() and np.isclose(big.sum(), 1) and big[1] > 0.99, (
        "large rewards / small beta overflow exp(): subtract the maximum log-weight first")
    zero = nb.kl_regularized_policy(np.array([0.0, 0.5, 0.5]), np.array([5.0, 0.0, 1.0]), 1.0)
    assert zero[0] == 0 and np.isclose(zero.sum(), 1), "items the reference never produces stay at probability 0"
    with pytest.raises(ValueError):
        nb.kl_regularized_policy(ref, r, 0.0)


def test_stage1_is_optimal(nb):
    rng = np.random.default_rng(0)
    ref = rng.dirichlet(np.ones(6))
    r = rng.normal(size=6)
    for beta in (0.3, 1.0, 3.0):
        pi = nb.kl_regularized_policy(ref, r, beta)
        best = pi @ r - beta * np.sum(pi * np.log(pi / ref))
        for _ in range(200):
            q = rng.dirichlet(np.ones(6))
            assert q @ r - beta * np.sum(q * np.log(q / ref)) <= best + 1e-9, (
                f"beta={beta}: some other policy scores higher on E[r] - beta KL(pi || ref); the closed form is the "
                "maximiser")


def test_stage1_stats(nb):
    pi, ref = np.array([0.5, 0.5, 0.0]), np.array([0.25, 0.25, 0.5])
    st = nb.policy_stats(pi, ref, proxy=np.array([1.0, 3.0, 10.0]), true=np.array([2.0, 0.0, -5.0]))
    assert set(st) >= {"kl", "proxy", "true"}
    assert np.isclose(st["kl"], math.log(2)), f"KL(pi || ref) = 2 * 0.5 log(0.5 / 0.25) = log 2 nats (0 log 0 = 0); got {st['kl']}"
    assert np.isclose(st["proxy"], 2.0) and np.isclose(st["true"], 1.0), "expectations of proxy and true under pi"
    assert all(isinstance(v, float) for v in st.values())


# ---------------------------------------------------------------- stage 2: best-of-n


def test_stage2_by_hand(nb):
    ref = np.array([0.5, 0.3, 0.2])
    s = np.array([1.0, 3.0, 2.0])
    assert np.allclose(nb.best_of_n_distribution(ref, s, 1), ref), "n = 1 is the reference distribution"
    got = nb.best_of_n_distribution(ref, s, 2)
    # best = item 1 unless both samples miss it: 1 - 0.7^2 = 0.51; item 2: 0.7^2 - 0.5^2 = 0.24; item 0: 0.25
    assert np.allclose(got, [0.25, 0.51, 0.24]), (
        f"n = 2: P(best is the top item) = 1 - (1 - 0.3)^2 = 0.51, then F(s)^n - F(s-)^n down the ranking: "
        f"[0.25, 0.51, 0.24]; got {np.round(got, 4)}")
    assert np.isclose(nb.best_of_n_distribution(ref, s, 10)[1], 1 - 0.7 ** 10)
    tie = nb.best_of_n_distribution(np.array([0.1, 0.3, 0.6]), np.array([1.0, 1.0, 0.0]), 3)
    top = 1 - 0.6 ** 3
    assert np.allclose(tie, [top * 0.25, top * 0.75, 0.6 ** 3]), (
        "items with equal scores share the probability of that score being the best, in proportion to ref "
        f"(0.1 : 0.3); got {np.round(tie, 4)}")
    with pytest.raises(ValueError):
        nb.best_of_n_distribution(ref, s, 0)


def test_stage2_simulation_and_kl(nb):
    rng = np.random.default_rng(1)
    ref = rng.dirichlet(np.ones(8))
    s = rng.normal(size=8)
    got = nb.best_of_n_distribution(ref, s, 4)
    draws = rng.choice(8, size=(100_000, 4), p=ref)
    picks = draws[np.arange(100_000), np.argmax(s[draws], axis=1)]
    freq = np.bincount(picks, minlength=8) / 100_000
    assert np.allclose(got, freq, atol=0.006) and np.isclose(got.sum(), 1), (
        f"compare with simulating best-of-4: {np.round(freq, 3)}; got {np.round(got, 3)}")
    many = np.full(20_000, 1 / 20_000)
    scores = rng.permutation(20_000).astype(float)
    for n in (2, 8, 64):
        b = nb.best_of_n_distribution(many, scores, n)
        kl = float(np.sum(b[b > 0] * np.log(b[b > 0] / many[b > 0])))
        assert abs(kl - (math.log(n) - (n - 1) / n)) < 0.01, (
            f"n={n}: with many distinct scores, KL(best-of-n || ref) -> log n - (n-1)/n (chapter 27's best_of_n_kl); "
            f"got {kl:.4f}")


# ---------------------------------------------------------------- stage 3: specification gaming


def test_stage3_tables(nb):
    P, Rp, Rt = nb.corridor_mdp(5, 2, 1.5, goal_reward=8.0)
    assert P.shape == (5, 2, 5) and Rp.shape == Rt.shape == (5, 2), "S = length states, 2 actions (0 left, 1 right)"
    assert np.allclose(P.sum(-1), 1), "each (s, a) row of P is a distribution"
    assert P[0, 0, 0] == 1 and P[0, 1, 1] == 1 and P[3, 0, 2] == 1, "left at 0 stays; right moves +1; left moves -1"
    assert np.all(P[4, :, 4] == 1) and np.all(Rt[4] == 0) and np.all(Rp[4] == 0), "the goal is absorbing, no reward"
    assert Rt[3, 1] == 8.0 and Rt.sum() == 8.0, "true reward: goal_reward for the step INTO the goal, nothing else"
    assert Rp[1, 1] == 1.5 and Rp[3, 0] == 1.5, "proxy bonus for entering the bonus cell, from either side"
    assert Rp[2, 0] == 0 and Rp[2, 1] == 0 and Rp[3, 1] == 8.0, "no bonus for leaving it; the proxy keeps the goal reward"
    with pytest.raises(ValueError):
        nb.corridor_mdp(5, 4, 1.0)


def test_stage3_threshold(nb):
    L, b, G, g = 8, 2, 10.0, 0.9
    threshold = G * g ** (L - 1 - b) * (1 - g ** 2) / g ** 2  # the lesson's derivation
    for B, games in ((0.9 * threshold, False), (1.1 * threshold, True)):
        P, Rp, Rt = nb.corridor_mdp(L, b, B, goal_reward=G)
        out = bounded(nb.gaming_gap, P, Rp, Rt, g, budget=20_000_000, what="gaming_gap")
        opt = G * g ** (L - 2)
        assert np.isclose(out["optimal_value"], opt, rtol=1e-6), f"the true optimum walks straight to the goal: {opt:.4f}"
        if games:
            assert np.isclose(out["true_value"], 0, atol=1e-6) and np.isclose(out["regret"], opt, rtol=1e-6), (
                f"bonus {B:.3f} above the threshold {threshold:.3f}: the proxy-optimal policy loops on the bonus and "
                f"never reaches the goal (true value 0); got {out['true_value']:.4f}")
        else:
            assert np.isclose(out["regret"], 0, atol=1e-6), (
                f"bonus {B:.3f} below the threshold {threshold:.3f}: the proxy-optimal policy still goes to the goal")
        assert out["policy"].shape == (L, 2), "return the proxy-optimal policy too"
    P, Rp, Rt = nb.corridor_mdp(L, b, 1.1 * threshold, goal_reward=G)
    near = nb.gaming_gap(P, Rp, Rt, g, start=L - 2)
    assert np.isclose(near["true_value"], G) and np.isclose(near["regret"], 0), (
        "start one step from the goal: even the proxy agent just takes the goal (start is the state you evaluate)")
    other = nb.gaming_gap(P, Rp, Rt, 0.8)
    assert np.isclose(other["regret"], 0, atol=1e-9), (
        "gamma is used: at gamma = 0.8 the threshold is G gamma^5 (1 - gamma^2) / gamma^2 = 1.84, above this bonus, "
        "so the proxy agent goes to the goal")


def test_stage3_rollout(nb):
    P, Rp, Rt = nb.corridor_mdp(6, 2, 50.0)
    pi = nb.gaming_gap(P, Rp, Rt, 0.95)["policy"]
    states = bounded(nb.rollout_states, P, pi, 0, 7, budget=500_000, what="rollout_states")
    assert len(states) == 8 and states[0] == 0, "[start] + one state per step"
    assert set(states[2:]) <= {1, 2, 3} and 5 not in states, f"a gaming agent shuttles around the bonus cell 2: {states}"
    honest = nb.gaming_gap(*nb.corridor_mdp(6, 2, 0.0), 0.95)["policy"]
    walk = bounded(nb.rollout_states, P, honest, 0, 7, budget=500_000, what="rollout_states")
    assert walk == [0, 1, 2, 3, 4, 5, 5, 5], "the honest agent walks to the goal and stays"


# ---------------------------------------------------------------- stage 4: a small eval


def _cases(n_a=10, n_b=30):
    cases = [("easy", lambda ag, i=i: ag["skill"] > i / n_a) for i in range(n_a)]
    cases += [("hard", lambda ag, i=i: ag["skill"] > 0.5 + i / (2 * n_b)) for i in range(n_b)]
    return cases


def test_stage4_eval_suite(nb):
    calls = []
    agent = {"skill": 0.75}
    cases = [(c, (lambda ag, f=f: calls.append(1) or f(ag))) for c, f in _cases()]
    out = bounded(nb.eval_suite, agent, cases, budget=5_000_000, what="eval_suite", rng=0)
    assert len(calls) == 40, "run every case's check exactly once"
    assert set(out) == {"easy", "hard", "overall"}, "one entry per category, plus 'overall'"
    assert np.isclose(out["easy"][0], 0.8) and np.isclose(out["hard"][0], 0.5) and np.isclose(out["overall"][0], 0.575), (
        f"pass rates: easy 8/10, hard 15/30, overall 23/40; got {out}")
    for k, (rate, lo, hi) in out.items():
        assert lo <= rate <= hi, f"{k}: the interval contains the pass rate"
    assert (out["easy"][2] - out["easy"][1]) > (out["overall"][2] - out["overall"][1]), "fewer cases, wider interval"
    again = nb.eval_suite(agent, _cases(), n_boot=20, rng=0)
    assert again == nb.eval_suite(agent, _cases(), n_boot=20, rng=0), "deterministic for a fixed rng (pass it on)"
    from mlfs.llm import accuracy_ci
    g = np.random.default_rng(3)  # one generator, shared in order: categories sorted, then overall
    want = {"easy": accuracy_ci(np.array([1.0] * 8 + [0.0] * 2), n_boot=20, rng=g),
            "hard": accuracy_ci(np.array([1.0] * 15 + [0.0] * 15), n_boot=20, rng=g)}
    want["overall"] = accuracy_ci(np.array([1.0] * 8 + [0.0] * 2 + [1.0] * 15 + [0.0] * 15), n_boot=20, rng=g)
    got = nb.eval_suite(agent, _cases(), n_boot=20, rng=3)
    assert all(np.allclose(got[k], want[k]) for k in want), (
        f"use ONE generator, as_rng(rng), for every interval (categories in sorted order, then 'overall'): expected "
        f"{want}, got {got}")
    seeds = {nb.eval_suite(agent, _cases(), n_boot=20, rng=s)["overall"] for s in range(10)}
    assert len(seeds) > 1, "the bootstrap draws come from rng (with 20 resamples, different seeds give different intervals)"
    with pytest.raises(ValueError):
        nb.eval_suite(agent, [])


def test_stage4_compare(nb):
    strong, weak = {"skill": 0.95}, {"skill": 0.55}
    diff, lo, hi = nb.compare_agents(strong, weak, _cases(), rng=0)
    want = (10 / 10 * 10 + 27 / 30 * 30) / 40 - (6 + 3) / 40
    assert np.isclose(diff, want), f"difference in pass rates over the same cases: {want:.3f}; got {diff:.3f}"
    assert lo > 0 and lo <= diff <= hi, "the paired interval excludes 0 for such different agents"
    same = nb.compare_agents(weak, weak, _cases(), rng=0)
    assert np.allclose(same, 0), "an agent against itself: (0, 0, 0) (paired, every case agrees)"
