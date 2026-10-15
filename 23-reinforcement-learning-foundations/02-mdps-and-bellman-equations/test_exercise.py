"""Checks for exercise.py (MDPs & bellman equations).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The environment
protocol checks are strict because chapter 24 builds on them. Every episode loop runs under
mlfs.testing.bounded, so a loop that never ends fails with a message instead of hanging.
Value checks compare with an exact linear solve.
"""
import numpy as np
import pytest

from mlfs.testing import bounded


def _uniform(S, A=4):
    return np.full((S, A), 1.0 / A)


def _exact_v(P, R, pi, gamma, keep=None):
    S = len(R)
    P_pi = np.einsum("sa,sat->st", pi, P)
    R_pi = (pi * R).sum(1)
    if keep is None:
        return np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)
    V = np.zeros(S)
    k = np.asarray(keep)
    V[k] = np.linalg.solve(np.eye(len(k)) - gamma * P_pi[np.ix_(k, k)], R_pi[k])
    return V


# ---------------------------------------------------------------- stage 1: spaces and Env


def test_stage1_discrete(nb):
    d = nb.Discrete(5)
    assert d.n == 5 and isinstance(d.n, int), "Discrete(5).n should be the int 5"
    xs = [d.sample(rng=np.random.default_rng(0)) for _ in range(3)]
    assert len(set(xs)) == 1, "sample(rng) must use the generator it's given: the same seed, the same sample"
    rng = np.random.default_rng(1)
    draws = [d.sample(rng) for _ in range(5000)]
    assert all(isinstance(x, int) for x in draws[:20]), "sample returns a plain Python int"
    f = np.bincount(draws, minlength=5) / 5000
    assert set(draws) == set(range(5)) and np.all(np.abs(f - 0.2) < 0.03), (
        f"samples should be uniform on 0..4; frequencies {f.round(3).tolist()}")
    good, bad = [0, 4, np.int64(2), np.array(3)], [5, -1, 1.0, 2.5, True, "2", None, np.array([1])]
    for x in good:
        assert d.contains(x), f"Discrete(5).contains({x!r}) should be True"
    for x in bad:
        assert not d.contains(x), (
            f"Discrete(5).contains({x!r}) should be False: only integers 0..n-1 (bools and floats are not actions)")


def test_stage1_box(nb):
    b = nb.Box(np.array([-1.0, 0.0]), np.array([1.0, 5.0]))
    assert b.shape == (2,) and isinstance(b.shape, tuple), f"shape is inferred from low/high: (2,), got {b.shape}"
    b3 = nb.Box(-2, 2, shape=(3,))
    assert b3.shape == (3,) and np.allclose(b3.low, [-2, -2, -2]) and np.allclose(b3.high, 2), (
        "scalar low/high with shape=(3,) are broadcast to float arrays of that shape")
    assert b3.low.dtype == float, "low and high are float arrays"
    xs = np.array([b.sample(np.random.default_rng(i)) for i in range(500)])
    assert xs.shape == (500, 2) and np.all(xs >= b.low) and np.all(xs <= b.high), "samples lie in the box"
    assert xs[:, 1].max() > 4.5 and xs[:, 1].min() < 0.5, "samples should cover the box uniformly"
    assert b.contains(np.array([0.0, 2.0])) and b.contains([1.0, 5.0]), "points inside (and on the boundary) are in"
    assert not b.contains(np.array([0.0, 6.0])) and not b.contains(np.array([0.0, 1.0, 2.0])), (
        "points outside the box, or of the wrong shape, are not")


def _make_coin(nb):
    """a test environment built on your Env: obs is the number of heads so far."""
    class Coin(nb.Env):
        def __init__(self):
            self.observation_space = nb.Discrete(11)
            self.action_space = nb.Discrete(1)

        def reset(self, *, seed=None):
            super().reset(seed=seed)
            self.heads = 0
            return self.heads, {}

        def step(self, action):
            self.heads += int(self.np_random.random() < 0.5)
            return self.heads, 0.0, self.heads >= 10, False, {}

    return Coin()


def test_stage1_env_seeding(nb):
    env = _make_coin(nb)
    env.reset()
    assert isinstance(env.np_random, np.random.Generator), "the first reset() creates self.np_random, a Generator"
    env.reset(seed=3)
    a = [env.np_random.random() for _ in range(3)]
    env.reset(seed=3)
    b = [env.np_random.random() for _ in range(3)]
    assert a == b, "reset(seed=3) must reseed: np.random.default_rng(3)"
    assert np.allclose(a, np.random.default_rng(3).random(3)), "seed with np.random.default_rng(seed)"
    env.reset()
    c = env.np_random.random()
    assert np.isclose(c, np.random.default_rng(3).random(4)[3]), (
        "reset() WITHOUT a seed must keep the existing generator (the stream continues), as in gymnasium; "
        "only reset(seed=...) or the very first reset creates a new one")
    sentinel = np.random.default_rng(99)
    env.np_random = sentinel
    env.reset()
    assert env.np_random is sentinel, "reset() without a seed must not replace an existing self.np_random"


# ---------------------------------------------------------------- stage 2: the gridworld MDP


def test_stage2_shapes_and_spaces(nb):
    g = nb.GridWorld()
    assert (g.n_states, g.n_actions) == (16, 4), "a 4x4 grid has 16 states and 4 actions"
    assert g.P.shape == (16, 4, 16) and g.R.shape == (16, 4), f"P is (S, A, S'), R is (S, A); got {g.P.shape}, {g.R.shape}"
    assert isinstance(g.observation_space, nb.Discrete) and g.observation_space.n == 16
    assert isinstance(g.action_space, nb.Discrete) and g.action_space.n == 4
    assert isinstance(g, nb.Env), "GridWorld subclasses Env"
    assert np.allclose(g.P.sum(axis=2), 1.0), "every row P[s, a, :] is a probability distribution"
    assert g.start_state == 0 and g.terminal.dtype == bool and np.flatnonzero(g.terminal).tolist() == [15], (
        "start_state is 0 (cell (0, 0)); terminal is a bool mask, True only at the goal (3, 3) = 15")


def test_stage2_moves(nb):
    g = nb.GridWorld()
    nxt = lambda s, a: int(np.argmax(g.P[s, a]))  # noqa: E731
    got = [nxt(5, a) for a in range(4)]
    assert got == [1, 6, 9, 4], (
        f"from state 5 = (1, 1): up -> 1, right -> 6, down -> 9, left -> 4 (obs = row * width + col); got {got}")
    assert [nxt(0, 0), nxt(0, 3), nxt(3, 1), nxt(12, 2)] == [0, 0, 3, 12], "moves off the grid leave you where you are"
    assert np.allclose(g.P[15, :, 15], 1.0) and np.allclose(g.R[15], 0.0), (
        "the goal is absorbing (every action loops to it) with zero reward")
    assert np.allclose(np.delete(g.R, 15, axis=0), -1.0), (
        "every action in a non-terminal state costs step_reward = -1, including the step INTO the goal")


def test_stage2_walls_goal_and_layout(nb):
    g = nb.GridWorld(3, 4, start=(2, 0), goal=(0, 3), walls=[(1, 1)], step_reward=-2.0)
    assert g.n_states == 12 and g.start_state == 8, "obs = row * width + col: (2, 0) -> 8 in a 3x4 grid"
    nxt = lambda s, a: int(np.argmax(g.P[s, a]))  # noqa: E731
    assert nxt(1, 2) == 1 and nxt(4, 1) == 4 and nxt(6, 3) == 6 and nxt(9, 0) == 9, (
        "you can't walk into a wall: moving into (1, 1) leaves you where you are")
    assert g.terminal[3] and g.terminal.sum() == 1, "goal (0, 3) is state 3"
    assert np.allclose(g.R[3], 0) and np.allclose(g.R[0], -2.0), "R uses step_reward"
    assert np.allclose(g.P[5, :, 5], 1.0) and np.allclose(g.R[5], 0), (
        "wall cells are unreachable states; make them absorbing with zero reward so P stays a valid model")
    with pytest.raises(ValueError):
        nb.GridWorld(3, 3, start=(1, 1), walls=[(1, 1)])


def test_stage2_slip(nb):
    g = nb.GridWorld(slip=0.2)
    row = g.P[5, 1]
    want = np.zeros(16)
    want[[1, 6, 9, 4]] = 0.05
    want[6] += 0.8
    assert np.allclose(row, want), (
        f"with slip 0.2 the action is replaced by a uniformly random one (any of the 4) with probability 0.2: "
        f"P[5, right] = 0.85 at 6 and 0.05 at 1, 9, 4. got nonzeros at {np.flatnonzero(row).tolist()} "
        f"= {row[row > 0].round(3).tolist()}")
    corner = g.P[0, 0]
    assert np.isclose(corner[0], 0.8 + 0.1) and np.isclose(corner[1], 0.05) and np.isclose(corner[4], 0.05), (
        "slip moves that hit the edge also stay put: P[0, up] = 0.8 + 0.05 (up) + 0.05 (left) at 0")


# ---------------------------------------------------------------- stage 3: reset, step, rollout


def test_stage3_reset_and_step(nb):
    g = nb.GridWorld()
    out = g.reset(seed=0)
    assert isinstance(out, tuple) and len(out) == 2 and out[0] == 0 and isinstance(out[1], dict), (
        "reset returns (obs, info) with obs = start_state and info a dict")
    step = g.step(1)
    assert isinstance(step, tuple) and len(step) == 5, "step returns (obs, reward, terminated, truncated, info)"
    obs, r, term, trunc, info = step
    assert obs == 1 and r == -1.0 and term is False and trunc is False and isinstance(info, dict), (
        f"moving right from 0 gives (1, -1.0, False, False, {{}}); got {step}")
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool), (
        "reward is a float and terminated/truncated are Python bools")
    assert g.observation_space.contains(obs), "obs must be in observation_space (a plain int)"
    for a in [1, 1, 2, 2]:
        obs, r, term, trunc, _ = g.step(a)
    assert obs == 11 and not term, f"right, right, down, down from state 1 ends at 11; got {obs}"
    obs, r, term, trunc, _ = g.step(2)
    assert obs == 15 and r == -1.0 and term and not trunc, (
        f"the step into the goal costs step_reward and sets terminated=True; got obs={obs}, r={r}, term={term}")
    with pytest.raises(ValueError):
        g.step(4)


def test_stage3_truncation(nb):
    g = nb.GridWorld(max_steps=3)
    g.reset(seed=0)
    flags = [g.step(0)[2:4] for _ in range(3)]
    assert flags == [(False, False), (False, False), (False, True)], (
        f"with max_steps=3, the third step returns truncated=True (and terminated=False: the goal wasn't reached); "
        f"got (terminated, truncated) = {flags}")
    g.reset()
    assert g.step(0)[3] is False, "reset() restarts the step counter"
    g = nb.GridWorld(1, 3, max_steps=2)
    g.reset(seed=0)
    flags = [g.step(1)[2:4] for _ in range(2)]
    assert flags == [(False, False), (True, False)], (
        f"reaching the goal on the very step the time limit runs out is a termination, not a truncation: at most one "
        f"of the two flags is True (truncated means 'stopped early, with a future'). got {flags}")


def test_stage3_samples_from_P(nb):
    g = nb.GridWorld(slip=0.4)
    g.reset(seed=1)
    counts = np.zeros(16)
    n = 4000
    for _ in range(n):
        g.state, g.t = 5, 0
        counts[g.step(1)[0]] += 1
    f = counts / n
    p = g.P[5, 1]
    assert np.all(np.abs(f - p) < 5 * np.sqrt(p * (1 - p) / n) + 1e-9), (
        f"next states must be drawn from P[s, a] with self.np_random: frequencies {f[p > 0].round(3).tolist()} "
        f"vs P {p[p > 0].round(3).tolist()} (states {np.flatnonzero(p).tolist()}). (step reads the current "
        f"state from self.state and the step count from self.t.)")


def test_stage3_seeded_episodes(nb):
    def run(seed):
        g = nb.GridWorld(slip=0.5)
        g.reset(seed=seed)
        return [g.step(a)[0] for a in [1, 2, 1, 2, 0, 3, 1, 2]]

    assert run(4) == run(4), "the same reset seed must give the same trajectory (use self.np_random only)"
    assert any(run(4) != run(s) for s in range(5, 10)), "different seeds should give different trajectories"


def test_stage3_rollout(nb):
    g = nb.GridWorld(1, 4)
    out = bounded(nb.rollout, g, lambda obs: 1, what="rollout")
    want = {"observations": [0, 1, 2, 3], "actions": [1, 1, 1], "rewards": [-1.0, -1.0, -1.0],
            "terminated": True, "truncated": False, "return": -3.0}
    for k, v in want.items():
        assert k in out and out[k] == v, f"rollout()['{k}'] should be {v!r} on a 1x4 corridor going right; got {out.get(k)!r}"
    out = bounded(nb.rollout, nb.GridWorld(max_steps=None), lambda obs: 0, max_steps=10, what="rollout")
    assert len(out["actions"]) == 10 and len(out["observations"]) == 11 and out["truncated"] and not out["terminated"], (
        "rollout(max_steps=10) stops after 10 actions and reports truncated=True")
    out = bounded(nb.rollout, nb.GridWorld(max_steps=5), lambda obs: 0, what="rollout")
    assert len(out["actions"]) == 5 and out["truncated"], "the env's own truncation also ends the episode"


def test_stage3_rollout_seed_and_any_env(nb):
    rng_policy = lambda obs: int(obs) % 4  # noqa: E731
    a = bounded(nb.rollout, nb.GridWorld(slip=0.3), rng_policy, seed=11, what="rollout")
    b = bounded(nb.rollout, nb.GridWorld(slip=0.3), rng_policy, seed=11, what="rollout")
    assert a["observations"] == b["observations"], "rollout(seed=...) passes the seed to env.reset: reproducible"

    class Walk(nb.Env):
        def __init__(self):
            self.observation_space = nb.Box(-10.0, 10.0, shape=(1,))
            self.action_space = nb.Discrete(2)

        def reset(self, *, seed=None):
            super().reset(seed=seed)
            self.x = np.zeros(1)
            return self.x.copy(), {}

        def step(self, a):
            self.x = self.x + (1.0 if a == 1 else -1.0)
            return self.x.copy(), 0.5, bool(abs(self.x[0]) >= 3), False, {}

    out = bounded(nb.rollout, Walk(), lambda obs: 1, what="rollout")
    assert len(out["actions"]) == 3 and np.isclose(out["return"], 1.5) and out["terminated"], (
        "rollout must work with any Env (here observations are arrays): 3 steps of +1, return 1.5")
    assert isinstance(out["return"], float), "the return is a float (the undiscounted sum of rewards)"


# ---------------------------------------------------------------- stage 4: policy evaluation


def test_stage4_hand_example(nb):
    g = nb.GridWorld(1, 3)
    V = bounded(nb.policy_evaluation, g.P, g.R, _uniform(3), gamma=1.0, what="policy_evaluation")
    assert np.allclose(V, [-12.0, -8.0, 0.0], atol=1e-5), (
        f"uniform random policy on a 1x3 corridor (goal on the right), gamma 1: V = (-12, -8, 0) "
        f"(see the worked example); got {np.round(V, 4).tolist()}")
    V9 = bounded(nb.policy_evaluation, g.P, g.R, _uniform(3), gamma=0.9, what="policy_evaluation")
    want = _exact_v(g.P, g.R, _uniform(3), 0.9)
    assert np.allclose(V9, want, atol=1e-5), f"gamma 0.9: expected {want.round(4).tolist()}, got {np.round(V9, 4).tolist()}"


def test_stage4_matches_linear_solve(nb):
    g = nb.GridWorld(4, 5, walls=[(1, 1), (2, 3)], slip=0.2)
    pi = 0.5 * np.random.default_rng(0).dirichlet(np.ones(4), size=g.n_states) + 0.5 / 4  # every action >= 1/8
    V = bounded(nb.policy_evaluation, g.P, g.R, pi, gamma=0.95, what="policy_evaluation")
    want = _exact_v(g.P, g.R, pi, 0.95)
    assert V.shape == (20,) and np.allclose(V, want, atol=1e-5), (
        f"V^pi solves V = R_pi + gamma P_pi V; largest error vs a linear solve: {np.max(np.abs(V - want)):.2e}")
    V1 = bounded(nb.policy_evaluation, g.P, g.R, pi, gamma=1.0, what="policy_evaluation")
    keep = [s for s in range(20) if not g.terminal[s] and s not in (6, 13)]
    want1 = _exact_v(g.P, g.R, pi, 1.0, keep)
    assert np.allclose(V1, want1, atol=1e-4), "with gamma = 1 the values are still finite: every policy here ends"


def test_stage4_respects_max_iters(nb):
    g = nb.GridWorld(1, 3)
    pi = np.zeros((3, 4))
    pi[:, 3] = 1.0  # always left: never reaches the goal
    V = bounded(nb.policy_evaluation, g.P, g.R, pi, gamma=1.0, max_iters=50, what="policy_evaluation")
    assert -51 <= V[0] <= -49, (
        f"a policy that never terminates has V = -infinity at gamma = 1: iteration must stop after max_iters "
        f"(= 50 here, so V[0] = -50); got {V[0]}")


def test_stage4_q_from_v(nb):
    g = nb.GridWorld(1, 3)
    V = np.array([-12.0, -8.0, 0.0])
    Q = nb.q_from_v(g.P, g.R, V, gamma=1.0)
    assert Q.shape == (3, 4), "Q has shape (S, A)"
    want = np.array([[-13, -9, -13, -13], [-9, -1, -9, -13], [0, 0, 0, 0]], dtype=float)
    assert np.allclose(Q, want), (
        f"Q(s, a) = R(s, a) + gamma sum_s' P(s'|s, a) V(s'): e.g. Q(1, right) = -1 + 0 = -1, Q(0, up) = -1 + V(0) "
        f"= -13. expected {want.tolist()}, got {np.round(Q, 3).tolist()}")
    g2 = nb.GridWorld(slip=0.3)
    pi = _uniform(16)
    V2 = _exact_v(g2.P, g2.R, pi, 0.9)
    Q2 = nb.q_from_v(g2.P, g2.R, V2, gamma=0.9)
    assert np.allclose((pi * Q2).sum(1), V2), "averaging Q^pi over the policy's actions gives back V^pi"
