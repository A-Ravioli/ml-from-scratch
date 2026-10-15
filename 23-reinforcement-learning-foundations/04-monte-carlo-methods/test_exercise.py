"""Checks for exercise.py (monte carlo methods).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Blackjack's rules are
checked with a scripted deck (a stand-in for env.np_random whose integers() returns chosen cards).
Learning checks use tiny environments with exact answers, fixed seeds, and an environment wrapper
that fails the test after a fixed number of steps, so a loop that never ends can't hang.
"""
import numpy as np
import pytest

from mlfs.rl import Discrete, Env, GridWorld  # the reference protocol from topic 02
from mlfs.testing import bounded


class _Deck:
    """a scripted card source: integers() returns the next card, whatever the bounds."""

    def __init__(self, cards):
        self.cards = list(cards)

    def integers(self, low, high=None, size=None):
        assert self.cards, "the scripted deck ran out: your code drew more cards than the rules allow"
        return self.cards.pop(0)


class _Capped:
    """wraps an env; fails the test after max_steps calls to step."""

    def __init__(self, env, max_steps):
        self.env, self.max_steps, self.steps, self.resets = env, max_steps, 0, 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *, seed=None):
        self.resets += 1
        return self.env.reset(seed=seed)

    def step(self, action):
        self.steps += 1
        assert self.steps <= self.max_steps, (
            f"more than {self.max_steps:,} env.step calls: does your loop stop at n_episodes, and does each episode "
            f"end when terminated or truncated is True?")
        return self.env.step(action)


class _Loop(Env):
    """deterministic: obs 0 -> 1 -> 0 -> 2 (terminal) with rewards 1, 2, 3, whatever the action."""

    def __init__(self, n_actions=1):
        self.observation_space, self.action_space = Discrete(3), Discrete(n_actions)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.k = 0
        return 0, {}

    def step(self, action):
        self.k += 1
        obs, r = [(1, 1.0), (0, 2.0), (2, 3.0)][self.k - 1]
        return obs, r, self.k == 3, False, {}


class _TwoStep(Env):
    """state 0 -> (any action, reward 0) -> state 1 -> action 0: +1, action 1: 0 -> terminal 2."""

    def __init__(self):
        self.observation_space, self.action_space = Discrete(3), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.s = 0
        return 0, {}

    def step(self, action):
        if self.s == 0:
            self.s = 1
            return 1, 0.0, False, False, {}
        self.s = 2
        return 2, (1.0 if action == 0 else 0.0), True, False, {}


def _scripted(nb, cards):
    env = nb.Blackjack()
    env.np_random = _Deck(cards)
    obs, info = env.reset()
    return env, obs


# ---------------------------------------------------------------- stage 1: blackjack


def test_stage1_spaces_and_encoding(nb):
    env = nb.Blackjack()
    assert isinstance(env, Env), "Blackjack subclasses Env (import it from mlfs.rl)"
    assert env.observation_space.n == 360 and env.action_space.n == 2, "Discrete(360) observations, Discrete(2) actions"
    assert nb.Blackjack.encode(4, 1, False) == 0 and nb.Blackjack.encode(21, 10, True) == 359, (
        "encode(player_sum, dealer_card, usable_ace) = (player_sum - 4) * 20 + (dealer_card - 1) * 2 + usable_ace")
    assert nb.Blackjack.encode(13, 2, True) == 183
    seen = set()
    for ps in range(4, 22):
        for dc in range(1, 11):
            for ua in (False, True):
                o = nb.Blackjack.encode(ps, dc, ua)
                assert nb.Blackjack.decode(o) == (ps, dc, ua), f"decode(encode{(ps, dc, ua)}) should round-trip"
                seen.add(o)
    assert seen == set(range(360)), "the encoding is a bijection onto 0..359"
    d = nb.Blackjack.decode(183)
    assert type(d[0]) is int and type(d[1]) is int and type(d[2]) is bool, "decode returns (int, int, bool)"


def test_stage1_hand_values(nb):
    hv = nb.Blackjack.hand_value
    cases = {(1, 5): (16, True), (1, 5, 10): (16, False), (1, 1): (12, True), (10, 10, 5): (25, False),
             (1, 10): (21, True), (1, 1, 9): (21, True), (7, 8): (15, False)}
    for cards, want in cases.items():
        assert tuple(hv(list(cards))) == want, (
            f"hand_value({list(cards)}) should be {want}: an ace counts 11 if that doesn't bust (that's a usable ace)")


def test_stage1_deal_and_hit(nb):
    env, obs = _scripted(nb, [1, 6, 10, 9, 5])
    assert env.player == [1, 6] and env.dealer == [10, 9], (
        f"deal order: player, player, dealer, dealer (dealer[0] is the face-up card); got player={env.player}, "
        f"dealer={env.dealer}")
    assert obs == nb.Blackjack.encode(17, 10, True), f"obs after the deal is (17, 10, usable ace); got {nb.Blackjack.decode(obs)}"
    obs, r, term, trunc, info = env.step(1)
    assert nb.Blackjack.decode(obs) == (12, 10, False) and r == 0.0 and not term and not trunc, (
        f"hitting a 5 on (ace, 6) gives 1 + 6 + 5 = 12 with no usable ace (11 would bust); got "
        f"{nb.Blackjack.decode(obs)}, r={r}, terminated={term}")
    assert isinstance(info, dict)
    env, obs = _scripted(nb, [10, 6, 4, 7, 9])
    obs, r, term, trunc, _ = env.step(1)
    assert r == -1.0 and term and not trunc, "busting (10 + 6 + 9 = 25) ends the episode with reward -1"
    assert nb.Blackjack.decode(obs) == (21, 4, False), "after a bust the obs clips the player sum to 21"
    env, obs = _scripted(nb, [12, 13, 11, 1])
    assert env.player == [10, 10] and env.dealer == [10, 1], "jack, queen, king (11, 12, 13) all count 10"


def test_stage1_stick_and_dealer(nb):
    cases = [
        ([10, 10, 10, 6, 5], -1.0, "dealer 16 draws a 5 -> 21 beats your 20"),
        ([10, 10, 10, 6, 10], 1.0, "dealer 16 draws a 10 and busts: +1"),
        ([10, 8, 10, 8], 0.0, "18 against the dealer's 18 is a draw: 0"),
        ([10, 9, 1, 6], 1.0, "the dealer's ace + 6 is a soft 17, and the dealer sticks on 17: your 19 wins"),
    ]
    for cards, want, why in cases:
        env, _ = _scripted(nb, cards)
        obs, r, term, trunc, _ = env.step(0)
        assert r == want and term and not trunc, f"{why}; got reward {r}, terminated={term}"
    env, _ = _scripted(nb, [10, 7, 2, 3, 1, 1])
    obs, r, term, _, _ = env.step(0)
    assert env.dealer == [2, 3, 1, 1] and r == 0.0, (
        f"dealer 2, 3, ace = soft 16 must draw; the next ace makes a soft 17 and the dealer sticks: 17 vs your 17 is a "
        f"draw. got dealer {env.dealer}, reward {r}")
    with pytest.raises(ValueError):
        _scripted(nb, [10, 7, 2, 3])[0].step(2)


def test_stage1_seeded_cards(nb):
    env = nb.Blackjack()
    obs_a = [env.reset(seed=s)[0] for s in range(5)]
    env2 = nb.Blackjack()
    assert obs_a == [env2.reset(seed=s)[0] for s in range(5)], "reset(seed) must make the deal reproducible"
    env.reset(seed=0)
    cards = []
    for _ in range(3000):
        env.reset()
        cards += env.player + env.dealer
    f10 = np.mean(np.array(cards) == 10)
    assert abs(f10 - 4 / 13) < 0.02 and set(cards) == set(range(1, 11)), (
        f"an infinite deck: each of 13 ranks equally likely, so a 10-valued card has probability 4/13 = 0.308; "
        f"got {f10:.3f}")


# ---------------------------------------------------------------- stage 2: epsilon-greedy + prediction


def test_stage2_epsilon_greedy(nb):
    rng = np.random.default_rng(0)
    a = nb.epsilon_greedy([0.0, 1.0, 0.0], 0.0, rng=rng)
    assert a == 1 and type(a) is int, "with epsilon 0, the greedy action (a Python int)"
    n = 12_000
    picks = np.array([nb.epsilon_greedy(np.array([0.0, 1.0, 0.0]), 0.3, rng=rng) for _ in range(n)])
    p = 0.7 + 0.3 / 3
    assert abs(np.mean(picks == 1) - p) < 5 * np.sqrt(p * (1 - p) / n), (
        f"P(greedy) = (1 - eps) + eps / n_actions = 0.8; got {np.mean(picks == 1):.4f}")
    ties = np.array([nb.epsilon_greedy(np.zeros(4), 0.0, rng=rng) for _ in range(4000)])
    f = np.bincount(ties, minlength=4) / 4000
    assert np.all(np.abs(f - 0.25) < 0.04), f"ties are broken at random; got frequencies {f.round(3).tolist()}"


def test_stage2_first_vs_every_visit(nb):
    V1 = bounded(nb.mc_prediction, _Loop(), lambda o: 0, 3, gamma=1.0, rng=0, what="mc_prediction")
    Ve = bounded(nb.mc_prediction, _Loop(), lambda o: 0, 3, gamma=1.0, first_visit=False, rng=0, what="mc_prediction")
    assert np.allclose(V1, [6.0, 5.0, 0.0]), (
        f"episode 0 -> 1 -> 0 -> end with rewards 1, 2, 3: first-visit V(0) = G_0 = 6, V(1) = 5; got {np.round(V1, 3).tolist()}")
    assert np.allclose(Ve, [4.5, 5.0, 0.0]), (
        f"every-visit V(0) averages both visits' returns, (6 + 3) / 2 = 4.5; got {np.round(Ve, 3).tolist()}")
    Vg = bounded(nb.mc_prediction, _Loop(), lambda o: 0, 2, gamma=0.5, rng=0, what="mc_prediction")
    assert np.allclose(Vg, [2.75, 3.5, 0.0]), (
        f"with gamma 0.5: G_0 = 1 + 0.5 * 2 + 0.25 * 3 = 2.75, G_1 = 2 + 0.5 * 3 = 3.5; got {np.round(Vg, 3).tolist()}")


def test_stage2_random_walk_values(nb):
    prng = np.random.default_rng(5)
    env = _Capped(GridWorld(1, 3), 200_000)
    V = nb.mc_prediction(env, lambda o: int(prng.integers(4)), 3000, gamma=1.0, rng=1)
    assert np.allclose(V, [-12.0, -8.0, 0.0], atol=1.2), (
        f"averaging 3,000 random-walk episodes on the 1x3 corridor should give V close to (-12, -8, 0) "
        f"(topic 02's exact values); got {np.round(V, 2).tolist()}")
    assert env.resets == 3000, f"one env.reset per episode; got {env.resets}"


def test_stage2_seeds_the_env_from_rng(nb):
    pol = lambda o: 1  # noqa: E731
    a = nb.mc_prediction(_Capped(GridWorld(2, 3, slip=0.5), 50_000), pol, 200, rng=3)
    b = nb.mc_prediction(_Capped(GridWorld(2, 3, slip=0.5), 50_000), pol, 200, rng=3)
    c = nb.mc_prediction(_Capped(GridWorld(2, 3, slip=0.5), 50_000), pol, 200, rng=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c), (
        "the same rng must give the same estimate: seed the env from rng on the first episode "
        "(env.reset(seed=int(rng.integers(2**31))))")


# ---------------------------------------------------------------- stage 3: control


def test_stage3_first_visit_flag(nb):
    Q1, rets = bounded(nb.mc_control, _Loop(), 2, rng=0, what="mc_control")
    Qe, _ = bounded(nb.mc_control, _Loop(), 2, first_visit=False, rng=0, what="mc_control")
    assert np.shape(Q1) == (3, 1), "Q has shape (n_states, n_actions)"
    assert np.allclose(Q1[:, 0], [6, 5, 0]) and np.allclose(Qe[:, 0], [4.5, 5, 0]), (
        f"first-visit Q(0, 0) = 6, every-visit 4.5 (as for mc_prediction, now per (state, action)); "
        f"got {np.round(Q1[:, 0], 3).tolist()} and {np.round(Qe[:, 0], 3).tolist()}")
    assert rets == [6.0, 6.0] and all(type(x) is float for x in rets), (
        f"episode_returns is a list with each episode's undiscounted sum of rewards; got {rets}")
    Qg, rg = bounded(nb.mc_control, _Loop(), 2, gamma=0.5, rng=0, what="mc_control")
    assert np.allclose(Qg[:, 0], [2.75, 3.5, 0.0]), (
        f"with gamma 0.5 the returns are discounted backwards, G = R + 0.5 G: Q(0, 0) = 1 + 0.5 (2 + 0.5 * 3) = 2.75, "
        f"Q(1, 0) = 3.5, as for mc_prediction; got {np.round(Qg[:, 0], 4).tolist()}")
    assert rg == [6.0, 6.0], f"episode_returns stay undiscounted (the sum of rewards) whatever gamma is; got {rg}"


def test_stage3_learns_a_gridworld(nb):
    env = _Capped(GridWorld(3, 4, walls=[(1, 1)]), 300_000)
    Q, rets = nb.mc_control(env, 1500, epsilon=0.1, rng=0)
    assert len(rets) == 1500, "one return per episode"
    acts = np.argmax(Q, axis=1)
    g, s, steps = GridWorld(3, 4, walls=[(1, 1)]), 0, 0
    while not g.terminal[s] and steps < 30:
        s = int(np.argmax(g.P[s, acts[s]]))
        steps += 1
    assert steps == 5, (
        f"after 1,500 episodes the greedy policy for Q should walk from (0, 0) to (2, 3) in 5 steps; took {steps} "
        f"(30 means it never arrived)")
    assert np.mean(rets[-200:]) > -6.5 and np.mean(rets[:20]) < np.mean(rets[-200:]) - 3, (
        f"returns should improve to about -5.6 (5 steps, plus epsilon's detours): first 20 episodes average "
        f"{np.mean(rets[:20]):.1f}, last 200 {np.mean(rets[-200:]):.1f}")


def test_stage3_blackjack_sanity(nb):
    sticks, q21s, late = [], [], []
    for seed in range(3):
        Q, rets = nb.mc_control(_Capped(nb.Blackjack(), 300_000), 15_000, epsilon=0.1, rng=seed)
        high = [s for s in range(360) if nb.Blackjack.decode(s)[0] >= 20 and (Q[s] != 0).all()]
        sticks.append(np.mean(np.argmax(Q[high], 1) == 0))
        q21s.append(np.mean([Q[nb.Blackjack.encode(21, d, u), 0] for d in range(1, 11) for u in (False, True)]))
        late.append(np.mean(rets[-5000:]))
    assert np.median(sticks) > 0.9, (
        f"after 15,000 episodes, sticking on 20 or 21 should be greedy almost everywhere (both actions tried); "
        f"median over 3 seeds: {np.median(sticks):.0%}")
    assert np.median(q21s) > 0.5, (
        f"sticking on 21 wins most of the time: average Q(21, stick) should be well above 0.5; median {np.median(q21s):.2f}")
    assert -0.2 < np.median(late) < 0.0, (
        f"average return late in training should be about -0.1; median over 3 seeds {np.median(late):.3f}")
    Q2, _ = nb.mc_control(_Capped(nb.Blackjack(), 20_000), 500, rng=7)
    Q3, _ = nb.mc_control(_Capped(nb.Blackjack(), 20_000), 500, rng=7)
    assert np.array_equal(Q2, Q3), "the same rng must give the same Q"


# ---------------------------------------------------------------- stage 4: off-policy prediction


def test_stage4_weighted_and_ordinary(nb):
    target = np.array([[1.0, 0.0]] * 3)  # always action 0
    behavior = np.full((3, 2), 0.5)
    Qw = nb.off_policy_mc_prediction(_Capped(_TwoStep(), 20_000), target, behavior, 2000, rng=0)
    assert np.shape(Qw) == (3, 2), "Q has shape (S, A)"
    assert np.allclose(Qw[0], [1.0, 1.0]) and np.allclose(Qw[1], [1.0, 0.0]), (
        f"target 'always 0' from state 1 earns +1, so Q(0, a) = 1 for both a, Q(1, 0) = 1, Q(1, 1) = 0. weighted IS "
        f"recovers these exactly (each weighted average only mixes returns of 1). got {np.round(Qw[:2], 3).tolist()}")
    Qo = nb.off_policy_mc_prediction(_Capped(_TwoStep(), 20_000), target, behavior, 2000, weighted=False, rng=0)
    assert np.allclose(Qo[1], [1.0, 0.0]) and np.allclose(Qo[0], [1.0, 1.0], atol=0.15), (
        f"ordinary IS averages W * G over ALL visits: W = 2 (behaviour took action 0 at state 1) or 0, so Q(0, a) "
        f"is about 1. got {np.round(Qo[0], 3).tolist()}. (about 2 means you skipped the zero-weight visits: in "
        f"ordinary IS they still count in the average)")
    assert not np.allclose(Qo[0], 1.0, atol=1e-9), "ordinary IS is noisy: it shouldn't be exactly 1"


def test_stage4_discount_and_seed(nb):
    target = np.array([[1.0, 0.0]] * 3)
    behavior = np.array([[0.5, 0.5], [0.8, 0.2], [0.5, 0.5]])
    Q = nb.off_policy_mc_prediction(_Capped(_TwoStep(), 20_000), target, behavior, 3000, gamma=0.9, rng=1)
    assert np.allclose(Q[0], [0.9, 0.9]), f"with gamma 0.9 the +1 arrives one step later: Q(0, a) = 0.9; got {Q[0]}"
    a = nb.off_policy_mc_prediction(_Capped(_TwoStep(), 5_000), target, behavior, 300, weighted=False, rng=5)
    b = nb.off_policy_mc_prediction(_Capped(_TwoStep(), 5_000), target, behavior, 300, weighted=False, rng=5)
    assert np.array_equal(a, b), "the same rng must give the same estimate (behaviour actions drawn with rng)"


@pytest.mark.slow  # ~7 s: live checks skip it; pytest runs it
def test_stage4_matches_exact_values(nb):
    g = GridWorld(1, 4, slip=0.2, max_steps=None)
    target = np.tile([0.1, 0.7, 0.1, 0.1], (4, 1))  # mostly right
    behavior = np.full((4, 4), 0.25)
    P_pi = np.einsum("sa,sat->st", target, g.P)
    keep = [0, 1, 2]
    V = np.zeros(4)
    V[keep] = np.linalg.solve(np.eye(3) - P_pi[np.ix_(keep, keep)], (target * g.R).sum(1)[keep])
    Qx = g.R + g.P @ V
    errs = [np.max(np.abs(nb.off_policy_mc_prediction(_Capped(g, 200_000), target, behavior, 2000, rng=s)[:3]
                          - Qx[:3])) for s in range(7)]
    assert np.median(errs) < 1.3, (
        f"weighted IS on a slippery 1x4 corridor (target 'mostly right', behaviour uniform) should get Q^target within "
        f"1.3 of the exact values (-1.4 to -6.6) after 2,000 episodes; median largest error over 7 seeds "
        f"{np.median(errs):.2f}")
