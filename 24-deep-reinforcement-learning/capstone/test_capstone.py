"""Checks for capstone.py: the acrobot against an independent implementation of the published
dynamics, evaluation and statistics by hand, and a small multi-seed run with a metric threshold."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from mlfs.deeprl import CartPole, train_a2c, train_ppo
from mlfs.rl import Discrete, Env
from mlfs.testing import bounded

torch.set_num_threads(1)  # tiny networks: one thread is fastest, and avoids oversubscribing a busy machine


def _dsdt(s, a):
    """Sutton & Barto's acrobot ('book' dynamics): m = l = 1, lc = 0.5, I = 1, g = 9.8."""
    t1, t2, d1, d2 = s
    D1 = 0.25 + (1 + 0.25 + 2 * 0.5 * np.cos(t2)) + 2.0
    D2 = 0.25 + 0.5 * np.cos(t2) + 1.0
    p2 = 0.5 * 9.8 * np.cos(t1 + t2 - np.pi / 2)
    p1 = -0.5 * d2**2 * np.sin(t2) - 2 * 0.5 * d2 * d1 * np.sin(t2) + 1.5 * 9.8 * np.cos(t1 - np.pi / 2) + p2
    dd2 = (a + D2 / D1 * p1 - 0.5 * d1**2 * np.sin(t2) - p2) / (0.25 + 1.0 - D2**2 / D1)
    dd1 = -(D2 * dd2 + p1) / D1
    return np.array([d1, d2, dd1, dd2])


def _acro_step(s, action):
    a, h = action - 1.0, 0.2
    k1 = _dsdt(s, a)
    k2 = _dsdt(s + h / 2 * k1, a)
    k3 = _dsdt(s + h / 2 * k2, a)
    k4 = _dsdt(s + h * k3, a)
    s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    s[:2] = (s[:2] + np.pi) % (2 * np.pi) - np.pi
    s[2], s[3] = np.clip(s[2], -4 * np.pi, 4 * np.pi), np.clip(s[3], -9 * np.pi, 9 * np.pi)
    return s


class _BoxSpace:
    def __init__(self, n):
        self.shape = (n,)


class _Scripted(Env):
    """one-step episodes whose reward is the episode index k = 0, 1, 2, ... (since the last seeded reset)."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(1), Discrete(2)
        self.k = -1

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.k = 0 if seed is not None else self.k + 1
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), float(self.k), True, False, {}


class _SeedEcho(Env):
    """one-step episodes paying the seed passed to the most recent seeded reset."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(1), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed0 = seed
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), float(self.seed0), True, False, {}


class _TimeLimit(Env):
    """never terminates; reward 1 per step; TRUNCATED after 3 steps."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(1), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1), {}

    def step(self, action):
        self.t += 1
        return np.zeros(1), 1.0, False, self.t >= 3, {}


class _Contextual(Env):
    """one-step episodes: obs is a random one-hot context in {0, 1}; action == context pays 1, else 0."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(2), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.c = int(self.np_random.integers(2))
        return np.eye(2)[self.c], {}

    def step(self, action):
        return np.eye(2)[self.c], float(action == self.c), True, False, {}


class _FixedAC(nn.Module):
    """an actor-critic stand-in whose logits favour action 1 (probability 0.7)."""

    def forward(self, obs):
        logits = torch.tensor([[0.0, np.log(0.7 / 0.3)]]).expand(obs.shape[0], 2)
        return Categorical(logits=logits), torch.zeros(obs.shape[0])


# ---------------------------------------------------------------- stage 1: Acrobot


def test_stage1_spaces_and_reset(nb):
    env = nb.Acrobot()
    assert tuple(env.observation_space.shape) == (6,) and env.action_space.n == 3, "obs (6,), three torques"
    assert env.max_steps == 500, "the default time limit is 500 steps (Acrobot-v1)"
    obs, info = env.reset(seed=1)
    s = np.asarray(env.state)
    assert s.shape == (4,) and np.all(np.abs(s) <= 0.1), "start state ~ U(-0.1, 0.1)^4"
    want = [np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]]
    assert np.allclose(obs, want), "obs = (cos th1, sin th1, cos th2, sin th2, dth1, dth2)"
    assert np.allclose(nb.Acrobot().reset(seed=1)[0], obs), "reset(seed) is reproducible"


def test_stage1_dynamics_match_the_equations(nb):
    env = nb.Acrobot()
    env.reset(seed=0)
    for s0, a in [([0.1, -0.05, 0.0, 0.2], 2), ([1.0, 2.0, -3.0, 5.0], 0), ([3.0, -2.5, 12.0, -27.0], 1)]:
        env.state, env.t = np.array(s0, dtype=float), 0
        obs, r, term, trunc, _ = env.step(a)
        want = _acro_step(np.array(s0, dtype=float), a)
        assert np.allclose(env.state, want, atol=1e-9), (
            f"from {s0} with action {a} (torque {a - 1}): expected {np.round(want, 5).tolist()}, got "
            f"{np.round(np.asarray(env.state), 5).tolist()}. check the 'book' equations, one RK4 step of dt = 0.2, "
            f"angle wrapping into [-pi, pi) and velocity clipping (4 pi, 9 pi)")
        assert type(r) is float and type(term) is bool and type(trunc) is bool, "Python float reward, bool flags"


def test_stage1_termination_and_truncation(nb):
    env = nb.Acrobot()
    env.reset(seed=0)
    env.state, env.t = np.array([np.pi - 0.05, 0.0, 0.0, 0.0]), 0
    _, r, term, trunc, _ = env.step(1)
    assert term and not trunc and r == 0.0, (
        "the tip is above the bar when -cos(th1) - cos(th1 + th2) > 1: the episode terminates, and that step pays 0 "
        "(every other step pays -1)")
    env = nb.Acrobot(max_steps=5)
    env.reset(seed=2)
    out = [env.step(1)[1:4] for _ in range(5)]
    assert [o[0] for o in out] == [-1.0] * 5 and out[-1][1:] == (False, True), (
        "hanging near the bottom with no torque: reward -1 per step, truncated at max_steps")


# ---------------------------------------------------------------- stage 2: evaluation


def test_stage2_greedy(nb):
    act = nb.greedy(_FixedAC())
    picks = [act(np.zeros(3)) for _ in range(200)]
    assert all(type(a) is int for a in picks), "greedy returns Python ints"
    assert picks == [1] * 200, (
        f"greedy must ALWAYS pick the most likely action (probability 0.7 here), never sample: over 200 calls it "
        f"picked action 0 {picks.count(0)} times")


def test_stage2_evaluate_policy(nb):
    mean, se = bounded(nb.evaluate_policy, _Scripted, lambda o: 0, 5, seed=3, budget=2_000_000,
                       what="evaluate_policy")
    r = np.arange(5.0)
    assert abs(mean - 2.0) < 1e-12 and abs(se - r.std(ddof=1) / np.sqrt(5)) < 1e-12, (
        f"returns 0, 1, 2, 3, 4 (the env pays the episode index; only the first reset is seeded): mean 2, standard "
        f"error std(ddof=1) / sqrt(5) = {r.std(ddof=1) / np.sqrt(5):.4f}; got ({mean}, {se})")
    assert isinstance(mean, float) and isinstance(se, float), "return Python floats"
    mean, se = bounded(nb.evaluate_policy, _TimeLimit, lambda o: 0, 4, seed=0, budget=2_000_000,
                       what="evaluate_policy", hint="an episode also ends when it is truncated (a time limit).")
    assert mean == 3.0 and se == 0.0, f"episodes truncated after 3 steps return 3 each; got ({mean}, {se})"


# ---------------------------------------------------------------- stage 3: curves and bands


def test_stage3_learning_curves(nb):
    h1 = {"episode_returns": [1.0, 2.0, 3.0, 4.0], "episode_ends": [10, 20, 30, 40]}
    h2 = {"episode_returns": [10.0, 20.0], "episode_ends": [25, 45]}
    c = nb.learning_curves([h1, h2], np.array([5, 20, 35, 50]), window=2)
    want = np.array([[np.nan, 1.5, 2.5, 3.5], [np.nan, np.nan, 10.0, 15.0]])
    assert c.shape == (2, 4) and np.allclose(c, want, equal_nan=True), (
        f"at each grid step, the mean of the last `window` episodes that had ENDED by then (nan before the first); "
        f"expected\n{want}\ngot\n{c}")


def test_stage3_mean_ci(nb):
    same = np.tile([1.0, 2.0, 3.0], (5, 1))
    m, lo, hi = nb.mean_ci(same, rng=0)
    assert np.allclose(m, [1, 2, 3]) and np.allclose(lo, m) and np.allclose(hi, m), (
        "identical runs: the band collapses onto the mean")
    rng = np.random.default_rng(0)
    few, many = rng.normal(0, 1, (5, 20)), rng.normal(0, 1, (80, 20))
    m5, lo5, hi5 = nb.mean_ci(few, rng=1)
    m80, lo80, hi80 = nb.mean_ci(many, rng=1)
    assert np.all(lo5 <= m5) and np.all(m5 <= hi5), "the band contains the mean"
    assert np.mean(hi80 - lo80) < 0.5 * np.mean(hi5 - lo5), "more runs give a narrower band (on average over columns)"
    assert abs(np.mean(hi80 - lo80) / (2 * 1.96 / np.sqrt(80)) - 1) < 0.15, (
        "the percentile bootstrap band of a mean is about ±1.96 standard errors wide")
    a, b = nb.mean_ci(few, rng=7), nb.mean_ci(few, rng=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "the same rng gives the same band"
    m, lo, hi = nb.mean_ci(np.array([[np.nan, 1.0], [2.0, 3.0]]), rng=0)
    assert np.isclose(m[0], 2.0) and np.isclose(m[1], 2.0), "nan entries (runs without data yet) are ignored"


# ---------------------------------------------------------------- stage 4: comparing seeds


def test_stage4_compare_seeds(nb):
    a = np.array([10.0, 11.0, 9.0, 10.5, 9.5, 10.0])
    out = nb.compare_seeds(a, a.copy(), n_boot=4000, rng=0)
    assert set(out) >= {"diff", "ci", "p"}, "compare_seeds returns a dict with 'diff', 'ci' and 'p'"
    assert out["diff"] == 0 and out["ci"][0] <= 0 <= out["ci"][1] and out["p"] > 0.5, (
        f"two identical samples: difference 0, an interval around 0, a large p-value; got {out}")
    b = a - 5
    out = nb.compare_seeds(a, b, n_boot=4000, rng=0)
    assert abs(out["diff"] - 5.0) < 1e-12 and 4 < out["ci"][0] < 5 < out["ci"][1] < 6 and out["p"] < 0.01, (
        f"a shift of 5 with little spread: difference 5, an interval tightly around it, p < 0.01; got {out}")
    rng = np.random.default_rng(3)
    x, y = rng.normal(1.0, 2.0, 7), rng.normal(0.0, 2.0, 9)
    out = nb.compare_seeds(x, y, n_boot=4000, rng=0)
    assert np.isclose(out["diff"], x.mean() - y.mean()), "diff is the difference of the two sample means"
    se = np.sqrt(x.var(ddof=1) / 7 + y.var(ddof=1) / 9)
    assert abs((out["ci"][1] - out["ci"][0]) / (2 * 1.96 * se) - 1) < 0.35, (
        "the interval resamples the two groups SEPARATELY (they're independent seeds): its width should be about "
        "2 x 1.96 x sqrt(var_a / n_a + var_b / n_b)")


# ---------------------------------------------------------------- stage 5: multi-seed runs


def test_stage5_run_seeds(nb):
    calls = []

    def fake_train(make_env, total_steps, *, seed, lr):
        calls.append((seed, total_steps, lr))
        return _FixedAC(), {"episode_returns": [float(seed)], "episode_ends": [total_steps]}

    out = nb.run_seeds(_Scripted, fake_train, 7, [3, 4], eval_episodes=4, lr=0.5)
    assert calls == [(3, 7, 0.5), (4, 7, 0.5)], "train_fn(make_env, total_steps, seed=s, **kwargs) once per seed"
    assert [h["episode_returns"] for h in out["histories"]] == [[3.0], [4.0]], "histories in seed order"
    assert isinstance(out["final"], np.ndarray) and np.allclose(out["final"], 1.5), (
        "final holds each seed's greedy evaluation over eval_episodes (here returns 0..3, mean 1.5)")
    out = nb.run_seeds(_SeedEcho, fake_train, 7, [3, 4], eval_episodes=2, lr=0.5)
    assert np.allclose(out["final"], [10_003, 10_004]), (
        f"evaluate seed s with evaluate_policy(..., seed=10_000 + s), never with the training seed itself; the "
        f"evaluation env echoes its seed and gave {out['final'].tolist()}")


@pytest.mark.slow
def test_stage5_metric_threshold(nb):
    out = bounded(nb.run_seeds, _Contextual, train_ppo, 1600, range(3), eval_episodes=50, n_envs=4, n_steps=16,
                  minibatch_size=32, n_epochs=4, hidden_sizes=(16,), lr=1e-2, budget=200_000_000, what="run_seeds")
    assert np.median(out["final"]) > 0.9, f"PPO on a contextual task: greedy success > 0.9; got {out['final']}"
    out = nb.run_seeds(CartPole, train_ppo, 20_000, range(3), eval_episodes=5, lr=1e-3)
    assert np.median(out["final"]) >= 150, (
        f"PPO on CartPole for 20,000 steps (lr 1e-3): median greedy return over 3 seeds should be at least 150; "
        f"got {np.round(out['final'], 0).tolist()}")
    out = nb.run_seeds(_Contextual, train_a2c, 2000, range(2), eval_episodes=20, n_envs=4, n_steps=5, lr=1e-2,
                       hidden_sizes=(16,))
    assert out["final"].shape == (2,), "run_seeds works with any train_fn of the same shape (here train_a2c)"
