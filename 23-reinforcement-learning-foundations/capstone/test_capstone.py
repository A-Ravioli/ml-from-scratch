"""Checks for capstone.py (design your own environment).

Small, seeded setups only. Catch is checked against hand-computed transitions; check_env must pass
a correct environment and catch six deliberately broken ones; evaluate_policy is checked where the
answer is known exactly; the features are checked for the invariance they're designed to have and
for letting linear_sarsa (the reference, under pytest) learn. Loops run under mlfs.testing.bounded.
"""
import numpy as np

from mlfs.rl import linear_sarsa, value_iteration
from mlfs.testing import bounded


def _catch(nb, **kw):
    return nb.Catch(**kw)


# ---------------------------------------------------------------- stage 1: the game


def test_stage1_layout_and_model(nb):
    g = nb.Catch()
    assert (g.rows, g.cols, g.wind) == (6, 5, 0.2), "defaults: rows=6, cols=5, wind=0.2"
    assert (g.n_states, g.n_actions) == (150, 3) and g.observation_space.n == 150 and g.action_space.n == 3, (
        "6 rows x 5 ball columns x 5 paddle columns = 150 states, 3 actions")
    assert g.encode(2, 3, 1) == (2 * 5 + 3) * 5 + 1 and g.decode(g.encode(2, 3, 1)) == (2, 3, 1), (
        "obs = (ball_row * cols + ball_col) * cols + paddle, and decode inverts it")
    assert g.P.shape == (150, 3, 150) and g.R.shape == (150, 3) and np.allclose(g.P.sum(2), 1)
    term = np.flatnonzero(g.terminal)
    assert len(term) == 25 and all(g.decode(s)[0] == 5 for s in term), "terminal = the ball on the bottom row"
    assert all(np.allclose(g.P[s, :, s], 1) and np.allclose(g.R[s], 0) for s in term), "terminals absorb, reward 0"


def test_stage1_transitions(nb):
    g = nb.Catch(wind=0.0)
    s = g.encode(0, 2, 2)
    assert np.isclose(g.P[s, 0, g.encode(1, 2, 1)], 1) and np.isclose(g.P[s, 2, g.encode(1, 2, 3)], 1), (
        "without wind: action 0 moves the paddle left, 2 right, and the ball falls one row")
    s = g.encode(0, 0, 0)
    assert np.isclose(g.P[s, 0, g.encode(1, 0, 0)], 1), "the paddle stops at the wall"
    g = nb.Catch(wind=0.2)
    s = g.encode(1, 0, 4)
    p = g.P[s, 1]
    assert np.isclose(p[g.encode(2, 0, 4)], 0.9) and np.isclose(p[g.encode(2, 1, 4)], 0.1), (
        "wind moves the ball one column left or right with probability wind / 2 each; at the wall the push "
        "left keeps it in place (0.8 + 0.1 = 0.9 for column 0)")
    s = g.encode(4, 2, 2)  # about to land, paddle under the ball
    assert np.isclose(g.R[s, 1], 0.8 - 0.2), (
        f"R is the expected landing reward: staying under the ball catches it with probability 0.8 (+1) and misses "
        f"with 0.2 (-1): R = 0.6; got {g.R[s, 1]}")
    assert np.isclose(g.R[g.encode(3, 2, 2), 1], 0.0), "no reward before landing"


def test_stage1_episodes(nb):
    g = nb.Catch()
    cols = []
    for k in range(400):
        obs, info = g.reset(seed=k)
        r, c, p = g.decode(obs)
        assert r == 0 and p == 2 and isinstance(info, dict), "reset: ball in row 0, paddle in the middle (cols // 2)"
        cols.append(c)
    assert set(cols) == set(range(5)) and max(np.bincount(cols)) < 120, "the ball's column is uniform on 0..4"
    g.reset(seed=1)
    rewards, flags = [], []
    for _ in range(5):
        obs, r, term, trunc, info = g.step(1)
        rewards.append(r)
        flags.append(term)
        assert isinstance(r, float) and type(term) is bool and type(trunc) is bool and not trunc
    assert flags == [False] * 4 + [True] and rewards[:4] == [0.0] * 4 and rewards[4] in (1.0, -1.0), (
        f"an episode lasts rows - 1 = 5 steps; the reward is 0 until the landing step, then +1 (caught) or -1; "
        f"got rewards {rewards}, terminated {flags}")
    a = [g.reset(seed=9)[0]] + [g.step(k % 3)[0] for k in range(5)]
    b = [g.reset(seed=9)[0]] + [g.step(k % 3)[0] for k in range(5)]
    assert a == b, "reset(seed) makes the episode reproducible (draw with self.np_random)"


def test_stage1_value_iteration_optimum(nb):
    g = nb.Catch()
    pi, V = value_iteration(g.P, g.R, gamma=1.0)
    v = np.mean([V[g.encode(0, c, 2)] for c in range(5)])
    assert abs(v - 0.68) < 0.005, (
        f"value iteration on your Catch(6, 5, wind=0.2) model should give an optimal expected return of 0.680 from "
        f"the start; got {v:.4f}. (check the wind probabilities and the landing reward)")
    assert np.isclose(nb.Catch(wind=0.0).R[nb.Catch(wind=0.0).encode(4, 1, 1), 1], 1.0)


# ---------------------------------------------------------------- stage 2: the checker


def _broken(nb, kind):
    base = nb.Catch

    class Broken(base):
        def __init__(self):
            super().__init__(wind=0.2)
            if kind == "P":  # an unreachable row (row 1, paddle 0) with a negative entry; it still sums to 1
                s = self.encode(1, 0, 0)
                s2 = int(np.argmax(self.P[s, 0]))
                self.P[s, 0, s2] += 0.2
                self.P[s, 0, (s2 + 1) % self.n_states] -= 0.2
            if kind == "absorbing":
                for s in np.flatnonzero(self.terminal):
                    self.P[s] = 0.0
                    self.P[s, :, 0] = 1.0

        def reset(self, *, seed=None):
            obs, info = super().reset(seed=seed)
            if kind == "global":
                self.ball[1] = int(np.random.randint(self.cols))
                obs = self.encode(self.ball[0], self.ball[1], self.paddle)
            if kind == "obs":
                return self.n_states + 3, info
            return obs, info

        def step(self, action):
            if kind == "dynamics":
                wind, self.wind = self.wind, 0.0
                out = super().step(action)
                self.wind = wind
                return out
            obs, r, term, trunc, info = super().step(action)
            if kind == "flags":
                return obs, r, np.bool_(term), trunc, info
            if kind == "int reward":
                return obs, int(r), term, trunc, info
            if kind == "reward" and term:  # the landing reward ignores where the paddle is
                return obs, 1.0, term, trunc, info
            return obs, r, term, trunc, info

    return Broken()


def test_stage2_passes_a_correct_env(nb):
    problems = bounded(nb.check_env, nb.Catch(), budget=20_000_000, what="check_env")
    assert problems == [], f"check_env should find no problems in a correct Catch; it reported {problems}"
    assert bounded(nb.check_env, nb.Catch(rows=4, cols=3, wind=0.5), n_steps=5000, budget=10_000_000,
                   what="check_env") == [], "nor in a smaller, windier one"


def test_stage2_catches_broken_envs(nb):
    for kind, what in [("P", "a P row with a negative entry (it still sums to 1)"),
                       ("int reward", "rewards returned as Python ints instead of floats"),
                       ("reward", "step's rewards disagreeing with R (transitions are right)"),
                       ("absorbing", "terminal states that aren't absorbing"),
                       ("flags", "terminated returned as np.bool_ instead of a Python bool"),
                       ("obs", "reset returning an observation outside the observation space"),
                       ("global", "randomness from the global np.random instead of self.np_random (not reproducible)"),
                       ("dynamics", "step ignoring the wind that P includes (dynamics disagree with the model)")]:
        problems = bounded(nb.check_env, _broken(nb, kind), budget=20_000_000, what="check_env")
        assert isinstance(problems, list) and len(problems) > 0 and all(isinstance(p, str) for p in problems), (
            f"check_env should report a problem for {what}; it returned {problems!r}")


# ---------------------------------------------------------------- stage 3: evaluation


def test_stage3_evaluate_policy(nb):
    g = nb.Catch(wind=0.0)

    def chase(obs):
        r, c, p = g.decode(obs)
        return 1 + int(np.sign(c - p))

    m, se = bounded(nb.evaluate_policy, nb.Catch(wind=0.0), chase, 50, budget=2_000_000, what="evaluate_policy")
    assert m == 1.0 and se == 0.0, f"without wind, chasing the ball always catches it: mean 1.0, s.e. 0; got {m}, {se}"
    m, se = bounded(nb.evaluate_policy, nb.Catch(wind=0.0), lambda obs: 1, 2000, seed=3, budget=20_000_000,
                    what="evaluate_policy")
    assert abs(m - (-0.6)) < 4 * se and 0.01 < se < 0.03, (
        f"never moving catches only balls that start in the middle column (1 in 5): expected return 0.2 - 0.8 = "
        f"-0.6, with standard error std / sqrt(n) = 0.8 / sqrt(2000) = 0.018; got {m:.3f} +- {se:.4f}")
    a = nb.evaluate_policy(nb.Catch(), lambda obs: 1, 100, seed=4)
    b = nb.evaluate_policy(nb.Catch(), lambda obs: 1, 100, seed=4)
    assert a == b and all(isinstance(x, float) for x in a), "returns (mean, se) as floats, reproducible from seed"


class _Alternate:
    """episodes of one step; the k-th episode pays k % 2. no randomness."""

    def __init__(self, nb):
        from mlfs.rl import Discrete
        self.observation_space, self.action_space, self.k = Discrete(1), Discrete(1), 0

    def reset(self, *, seed=None):
        self.k += 1
        return 0, {}

    def step(self, action):
        return 0, float(self.k % 2), True, False, {}


def test_stage3_standard_error_and_seeding(nb):
    m, se = nb.evaluate_policy(_Alternate(nb), lambda obs: 0, 2)
    assert np.isclose(m, 0.5) and np.isclose(se, 0.5), (
        f"returns (1, 0): mean 0.5, standard error std(ddof=1) / sqrt(n) = 0.7071 / 1.4142 = 0.5; got {m}, {se} "
        f"(0.354 means you used the population std, ddof=0)")
    pol = lambda obs: 1  # noqa: E731
    env = nb.Catch()
    rets = []
    for k in range(40):
        env.reset(seed=4) if k == 0 else env.reset()
        total, done = 0.0, False
        while not done:
            _, r, term, trunc, _ = env.step(pol(None))
            total, done = total + r, term or trunc
        rets.append(total)
    m, _ = nb.evaluate_policy(nb.Catch(), pol, 40, seed=4)
    assert np.isclose(m, np.mean(rets)), (
        f"seed only the FIRST episode (reset(seed=seed)); later episodes continue the env's stream (reset()). "
        f"expected mean {np.mean(rets):.3f}, got {m:.3f}")


def test_stage3_q_policy(nb):
    Q = np.array([[0.0, 2.0, 1.0], [3.0, 3.0, 0.0]])
    pol = nb.q_policy(Q)
    assert pol(0) == 1 and pol(1) == 0 and type(pol(0)) is int, "greedy action as a Python int, ties to the lowest"
    Q[0, 2] = 10.0
    assert pol(0) == 1, "q_policy should copy Q: later changes to the table must not change the policy"


# ---------------------------------------------------------------- stage 4: features


def test_stage4_features(nb):
    g = nb.Catch()
    f = nb.CatchFeatures(g)
    assert f.n_features == 6 * 9 + 9, "rows * (2 cols - 1) offset features plus (2 cols - 1) row-free ones = 63"
    seen = set()
    for s in range(g.n_states):
        idx = np.asarray(f(s))
        assert idx.shape == (2,) and idx.dtype.kind == "i" and np.all((idx >= 0) & (idx < f.n_features))
        seen.update(idx.tolist())
    assert seen == set(range(63)), "every feature is used by some state"
    assert list(f(g.encode(3, 1, 0))) == list(f(g.encode(3, 4, 3))), (
        "states with the same ball row and the same ball - paddle offset share their features (that's the point)")
    assert list(f(g.encode(3, 1, 0)))[1] == list(f(g.encode(0, 1, 0)))[1] and \
        list(f(g.encode(3, 1, 0)))[0] != list(f(g.encode(0, 1, 0)))[0], (
        "the first index depends on (row, offset), the second on the offset alone")


def test_stage4_features_learn_fast(nb):
    g = nb.Catch()
    f = nb.CatchFeatures(g)
    scores = []
    for seed in range(5):
        w, _ = bounded(linear_sarsa, nb.Catch(), f, 150, alpha=0.2, epsilon=0.1, rng=seed, budget=20_000_000,
                       what="linear_sarsa with your features")
        pol = lambda obs, w=w: int(np.argmax(w[f(obs)].sum(axis=0)))  # noqa: E731
        env = nb.Catch()
        total = 0.0
        env.reset(seed=11 + seed)
        for k in range(300):
            obs, _ = env.reset()
            done = False
            while not done:
                obs, r, term, trunc, _ = env.step(pol(obs))
                done = term or trunc
            total += r
        scores.append(total / 300)
    assert np.median(scores) > 0.45, (
        f"with features that generalize across positions, linear SARSA (alpha 0.2) should play well after only 150 episodes "
        f"(the optimum is 0.68; tabular Q-learning is still below 0.2 at this point): median greedy return over "
        f"5 seeds {np.median(scores):.3f} ({np.round(scores, 2).tolist()})")
