"""Checks for exercise.py (bandits).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Randomness is
always seeded. Statistical checks compare a frequency with its exact probability and allow
about 5 standard errors, so a correct implementation fails less than once in a million runs,
while the plausible bugs (ties to the lowest arm, exploring only non-greedy arms, the wrong
UCB clock, Thompson playing the posterior mean) miss by 15 or more.
"""
import numpy as np
from scipy import integrate, stats

from mlfs.testing import bounded


class _ScriptedBandit:
    """deterministic bandit: arm a at round t pays 10 * a + t (so every reward is traceable)."""

    def __init__(self, means):
        self.means = np.asarray(means, dtype=float)
        self.n_arms = len(self.means)
        self.optimal_arm = int(np.argmax(self.means))
        self.t = 0

    def pull(self, arm):
        r = 10.0 * arm + self.t
        self.t += 1
        return r


class _CyclingAgent:
    def __init__(self, n_arms):
        self.n_arms, self.k, self.log = n_arms, 0, []

    def select_arm(self):
        self.log.append("select")
        return self.k % self.n_arms

    def update(self, arm, reward):
        self.log.append(("update", arm, reward))
        self.k += 1


def _freq(agent, n, n_arms):
    counts = np.bincount([agent.select_arm() for _ in range(n)], minlength=n_arms)
    return counts / n


# ---------------------------------------------------------------- stage 1: the testbed


def test_stage1_bernoulli_attributes_and_rewards(nb):
    b = nb.BernoulliBandit([0.2, 0.9, 0.5], rng=0)
    assert b.n_arms == 3 and b.optimal_arm == 1, (
        f"n_arms should be 3 and optimal_arm 1 (the arm with the largest mean); got {b.n_arms}, {b.optimal_arm}")
    assert np.allclose(b.means, [0.2, 0.9, 0.5]) and b.means.dtype == float, "means is a float array of the probs"
    assert isinstance(b.optimal_arm, int), "optimal_arm should be a plain int"
    r = [b.pull(1) for _ in range(20_000)]
    assert all(isinstance(x, float) for x in r[:10]) and set(r) <= {0.0, 1.0}, "pull returns floats 0.0 or 1.0"
    m = np.mean(r)
    assert abs(m - 0.9) < 5 * np.sqrt(0.09 / 20_000), (
        f"arm 1 pays 1 with probability 0.9: the average of 20,000 pulls should be 0.9 +- 0.01, got {m:.4f}")


def test_stage1_seeded_and_independent(nb):
    b1, b2 = nb.BernoulliBandit([0.5] * 2, rng=7), nb.BernoulliBandit([0.5] * 2, rng=7)
    s1, s2 = [b1.pull(0) for _ in range(50)], [b2.pull(0) for _ in range(50)]
    assert s1 == s2, "two bandits built with the same integer seed must produce the same rewards"
    b3 = nb.BernoulliBandit([0.5] * 2, rng=8)
    assert [b3.pull(0) for _ in range(50)] != s1, (
        "different seeds should give different reward sequences: build the generator with as_rng(rng)")
    g = np.random.default_rng(3)
    b4 = nb.BernoulliBandit([0.5] * 2, rng=g)
    assert b4.rng is g, "a Generator passed as rng should be used as is (as_rng)"


def test_stage1_gaussian_bandit(nb):
    b = nb.GaussianBandit([0.0, -1.0, 2.0], std=0.5, rng=1)
    assert b.n_arms == 3 and b.optimal_arm == 2 and np.isclose(b.std, 0.5), "store n_arms, optimal_arm and std"
    r = np.array([b.pull(2) for _ in range(20_000)])
    assert abs(r.mean() - 2.0) < 5 * 0.5 / np.sqrt(20_000) and abs(r.std() - 0.5) < 0.02, (
        f"arm 2 pays N(2, 0.5^2): sample mean {r.mean():.4f} and std {r.std():.4f}")


def test_stage1_run_bandit_loop(nb):
    bandit, agent = _ScriptedBandit([0.1, 0.5, 0.3]), _CyclingAgent(3)
    rewards, regret = bounded(nb.run_bandit, bandit, agent, 6, what="run_bandit")
    assert np.shape(rewards) == (6,) and np.shape(regret) == (6,), "return two arrays of shape (n_steps,)"
    assert np.allclose(rewards, [0, 11, 22, 3, 14, 25]), (
        f"rewards[t] is what bandit.pull returned at round t; got {np.asarray(rewards).tolist()}")
    want = np.cumsum([0.4, 0.0, 0.2, 0.4, 0.0, 0.2])
    assert np.allclose(regret, want), (
        f"cumulative PSEUDO-regret adds max(means) - means[arm] each round (not the realized reward gap): "
        f"expected {want.round(3).tolist()}, got {np.round(regret, 3).tolist()}")
    assert agent.log[:4] == ["select", ("update", 0, 0.0), "select", ("update", 1, 11.0)], (
        "each round: arm = agent.select_arm(), r = bandit.pull(arm), agent.update(arm, r)")


# ---------------------------------------------------------------- stage 2: epsilon-greedy


def test_stage2_sample_averages(nb):
    ag = nb.EpsilonGreedy(3, 0.1, rng=0)
    assert np.allclose(ag.Q, 0) and np.allclose(ag.N, 0) and ag.Q.shape == (3,), "Q and N start at zeros (3,)"
    for arm, r in [(0, 1.0), (0, 0.0), (2, 5.0), (0, 1.0), (2, 1.0)]:
        ag.update(arm, r)
    assert np.allclose(ag.Q, [2 / 3, 0.0, 3.0]) and np.allclose(ag.N, [3, 0, 2]), (
        f"after rewards (1, 0, 1) on arm 0 and (5, 1) on arm 2, Q should be the per-arm averages "
        f"[0.667, 0, 3] and N the counts [3, 0, 2]; got Q={np.round(ag.Q, 3).tolist()}, N={np.asarray(ag.N).tolist()}")


def test_stage2_greedy_breaks_ties_at_random(nb):
    f = _freq(nb.EpsilonGreedy(4, 0.0, rng=1), 4000, 4)
    assert np.all(np.abs(f - 0.25) < 5 * np.sqrt(0.25 * 0.75 / 4000)), (
        f"with epsilon 0 and all Q equal, every arm is greedy, so each should be picked about 1/4 of the time; "
        f"got {f.round(3).tolist()}. np.argmax always returns the FIRST maximum: pick at random among the ties")


def test_stage2_exploration_probability(nb):
    ag = nb.EpsilonGreedy(3, 0.1, rng=2)
    ag.Q[:] = [0.0, 1.0, 0.0]
    n = 20_000
    f = _freq(ag, n, 3)
    want = 0.9 + 0.1 / 3
    assert abs(f[1] - want) < 5 * np.sqrt(want * (1 - want) / n), (
        f"P(greedy arm) = (1 - eps) + eps / k = 0.9333: exploring picks uniformly among ALL arms, including the "
        f"greedy one. got {f[1]:.4f} (0.9 means you explored only the non-greedy arms)")
    f1 = _freq(nb.EpsilonGreedy(3, 1.0, rng=3), 6000, 3)
    assert np.all(np.abs(f1 - 1 / 3) < 0.04), f"epsilon 1 is uniformly random; got {f1.round(3).tolist()}"


def test_stage2_learns_and_is_reproducible(nb):
    runs = []
    for seed in range(3):
        bandit = nb.BernoulliBandit([0.2, 0.5, 0.8], rng=100 + seed)
        ag = nb.EpsilonGreedy(3, 0.1, rng=seed)
        bounded(nb.run_bandit, bandit, ag, 2000, budget=2_000_000, what="run_bandit")
        runs.append(ag.N[2] / 2000)
    assert np.mean(runs) > 0.85, (
        f"on means (0.2, 0.5, 0.8) epsilon-greedy (0.1) should play the best arm in more than 85% of 2,000 rounds; "
        f"got {np.round(runs, 3).tolist()}")
    a, b = nb.EpsilonGreedy(3, 0.3, rng=5), nb.EpsilonGreedy(3, 0.3, rng=5)
    assert [a.select_arm() for _ in range(30)] == [b.select_arm() for _ in range(30)], (
        "the same seed must give the same choices: draw everything from self.rng")


# ---------------------------------------------------------------- stage 3: UCB


def test_stage3_tries_every_arm_first(nb):
    ag = nb.UCB(4, c=2.0)
    seen = []
    for _ in range(4):
        a = ag.select_arm()
        seen.append(a)
        ag.update(a, 0.0)
    assert seen == [0, 1, 2, 3], f"UCB pulls each untried arm once, lowest index first; got {seen}"


def test_stage3_hand_example(nb):
    ag = nb.UCB(3, c=2.0)
    for arm, r in [(0, 1.0), (1, 0.0), (2, 0.0), (0, 1.0), (0, 0.0), (1, 1.0)]:
        ag.update(arm, r)
    # t = 6 pulls so far; Q = (2/3, 1/2, 0), N = (3, 2, 1)
    assert np.allclose(ag.Q, [2 / 3, 0.5, 0.0]) and np.allclose(ag.N, [3, 2, 1]), "sample averages, as in stage 2"
    bonus = np.sqrt(np.log(6) / np.array([3, 2, 1]))
    scores = np.array([2 / 3, 0.5, 0.0]) + 2.0 * bonus
    a = ag.select_arm()
    assert a == 2, (
        f"with Q = (2/3, 1/2, 0), N = (3, 2, 1), t = 6 pulls so far and c = 2 the upper bounds are "
        f"{scores.round(3).tolist()}: pick arm 2, got {a}")
    ag2 = nb.UCB(3, c=0.3)
    for arm, r in [(0, 1.0), (1, 0.0), (2, 0.0), (0, 1.0), (0, 0.0), (1, 1.0)]:
        ag2.update(arm, r)
    s2 = np.array([2 / 3, 0.5, 0.0]) + 0.3 * bonus
    assert ag2.select_arm() == int(np.argmax(s2)) == 0, (
        f"with c = 0.3 the bounds are {s2.round(4).tolist()}: arm 0. (are you using ln t with t = number of "
        f"pulls so far, and multiplying the bonus by c?)")


def test_stage3_clock_is_pulls_so_far(nb):
    ag = nb.UCB(2, c=1.0)
    for arm, r in [(0, 1.0), (1, 0.0), (0, 0.0)]:
        ag.update(arm, r)
    # N = (2, 1), t = 3. with Q = (0.33, 0) the bounds favour arm 0 at t = 3 but arm 1 at t = 4
    ag.Q[:] = [0.33, 0.0]
    t3 = [0.33 + np.sqrt(np.log(3) / 2), np.sqrt(np.log(3))]
    t4 = [0.33 + np.sqrt(np.log(4) / 2), np.sqrt(np.log(4))]
    assert int(np.argmax(t3)) != int(np.argmax(t4))
    got = ag.select_arm()
    assert got == int(np.argmax(t3)), (
        f"t in ln t is the number of pulls SO FAR (sum of N = 3 here), not the round about to be played: "
        f"bounds {np.round(t3, 4).tolist()} -> arm {int(np.argmax(t3))}; got arm {got}")


def test_stage3_logarithmic_regret(nb):
    finals = []
    for seed in range(5):
        bandit = nb.BernoulliBandit([0.3, 0.5, 0.7], rng=seed)
        _, reg = bounded(nb.run_bandit, bandit, nb.UCB(3, c=1.0), 4000, budget=4_000_000, what="run_bandit")
        finals.append((reg[999], reg[-1]))
    r1k, r4k = np.median(finals, axis=0)
    assert r4k < 80 and r4k - r1k < 1.5 * r1k, (
        f"UCB's regret should grow like log t: on means (0.3, 0.5, 0.7) with c = 1, about 50 by t = 4,000, and "
        f"rounds 1,000-4,000 should add well under 1.5 times what the first 1,000 did (linear regret would add 3 times). got {r1k:.1f} at 1,000 and {r4k:.1f} at 4,000")


# ---------------------------------------------------------------- stage 4: Thompson sampling


def test_stage4_posterior_updates(nb):
    ag = nb.ThompsonSampling(3, prior=(2.0, 3.0), rng=0)
    assert np.allclose(ag.alpha, 2) and np.allclose(ag.beta, 3) and ag.alpha.shape == (3,), (
        "alpha and beta start as arrays filled with prior[0] and prior[1]")
    for arm, r in [(0, 1.0), (0, 0.0), (1, 0.25), (0, 1.0)]:
        ag.update(arm, r)
    assert np.allclose(ag.alpha, [4, 2.25, 2]) and np.allclose(ag.beta, [4, 3.75, 3]), (
        f"a reward r adds r to alpha and 1 - r to beta (fractional rewards allowed); got alpha="
        f"{ag.alpha.tolist()}, beta={ag.beta.tolist()}")


def test_stage4_probability_matching(nb):
    ag = nb.ThompsonSampling(2, rng=1)
    ag.alpha[:] = [3.0, 2.0]
    ag.beta[:] = [2.0, 3.0]
    p0 = integrate.quad(lambda x: stats.beta(3, 2).pdf(x) * stats.beta(2, 3).cdf(x), 0, 1)[0]
    n = 20_000
    f = _freq(ag, n, 2)
    assert abs(f[0] - p0) < 5 * np.sqrt(p0 * (1 - p0) / n), (
        f"Thompson sampling plays each arm with the posterior probability that it's the best: with posteriors "
        f"Beta(3, 2) and Beta(2, 3) that's {p0:.4f} for arm 0, got {f[0]:.4f}. (1.0 means you played the "
        f"posterior MEAN's argmax: sample theta from the posterior instead)")


def test_stage4_beats_epsilon_greedy(nb):
    ts = []
    for seed in range(5):
        _, r1 = bounded(nb.run_bandit, nb.BernoulliBandit([0.3, 0.5, 0.7], rng=seed),
                        nb.ThompsonSampling(3, rng=10 + seed), 3000, budget=4_000_000, what="run_bandit")
        ts.append(r1[-1])
    assert np.median(ts) < 25, (
        f"Thompson sampling on means (0.3, 0.5, 0.7) should have cumulative regret around 10-15 after 3,000 "
        f"rounds (median over 5 seeds); got {np.round(ts, 1).tolist()}")
    a, b = nb.ThompsonSampling(3, rng=4), nb.ThompsonSampling(3, rng=4)
    assert [a.select_arm() for _ in range(20)] == [b.select_arm() for _ in range(20)], "seeded: same rng, same arms"


def test_stage4_lai_robbins(nb):
    m = [0.3, 0.5, 0.7]
    kl = lambda p, q: p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))  # noqa: E731
    want = 0.4 / kl(0.3, 0.7) + 0.2 / kl(0.5, 0.7)
    got = nb.lai_robbins_constant(m)
    assert np.isclose(got, want), (
        f"sum over suboptimal arms of gap / KL(mu_a, mu*) with the Bernoulli KL "
        f"p ln(p/q) + (1-p) ln((1-p)/(1-q)): expected {want:.4f}, got {got}")
    assert np.isclose(nb.lai_robbins_constant([0.6, 0.6, 0.2]), 0.4 / kl(0.2, 0.6)), (
        "arms tied for best cost nothing (skip them, don't divide by a zero KL)")
    assert np.isclose(nb.lai_robbins_constant([0.0, 0.5]), 0.5 / np.log(2)), "KL(0, q) = -ln(1 - q)"
