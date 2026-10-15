"""Checks for exercise.py (function approximation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Tile coding and the
mountain car dynamics are checked against hand calculations; learning uses a fixed seed and a
modest threshold; the TD matrices are checked on tabular features (where the TD fixed point is the
exact V) and on Baird's counterexample (where off-policy semi-gradient TD diverges). Environments
are wrapped so that a loop that never ends fails after a fixed number of steps.
"""
import numpy as np
import pytest

from mlfs.rl import Box, Discrete, Env  # the reference protocol from topic 02


class _Capped:
    def __init__(self, env, max_steps):
        self.env, self.max_steps, self.steps = env, max_steps, 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        self.steps += 1
        assert self.steps <= self.max_steps, (
            f"more than {self.max_steps:,} env.step calls: does your loop stop at n_episodes, and does each episode "
            f"end when terminated or truncated is True?")
        return self.env.step(action)


def _baird(gamma=0.99):
    Phi = np.zeros((7, 8))
    for i in range(6):
        Phi[i, i], Phi[i, 7] = 2.0, 1.0
    Phi[6, 6], Phi[6, 7] = 1.0, 2.0
    P = np.zeros((7, 7))
    P[:, 6] = 1.0  # the target policy always takes the solid action, to state 7
    return Phi, P, np.zeros(7)


# ---------------------------------------------------------------- stage 1: tile coding


def test_stage1_hand_example(nb):
    tc = nb.TileCoder(0.0, 1.0, n_tilings=2, tiles_per_dim=5)
    assert tc.n_features == 10, "n_features = n_tilings * tiles_per_dim ** d = 2 * 5 = 10"
    assert np.allclose(tc.width, 0.25), "the tile width is (high - low) / (tiles_per_dim - 1) = 0.25"
    assert np.allclose(tc.offsets, [[0.0], [0.5]]), (
        f"default offsets: tiling i is shifted by (i * (2j + 1) / n_tilings) mod 1 tile widths in dimension j: "
        f"(0, 0.5) here; got {np.round(tc.offsets, 3).tolist()}")
    a, b = tc(np.array([0.3])), tc(np.array([0.4]))
    assert np.asarray(a).dtype.kind == "i" and np.shape(a) == (2,), "returns an int array with one index per tiling"
    assert list(a) == [1, 6] and list(b) == [1, 7], (
        f"x = 0.3 is 1.2 widths in: tiling 0 tile floor(1.2) = 1, tiling 1 tile floor(1.2 + 0.5) = 1 -> index 5 + 1 = 6; "
        f"x = 0.4 (1.6 widths): tiles 1 and floor(2.1) = 2 -> (1, 7). got {list(a)} and {list(b)}")
    assert list(tc(np.array([-3.0]))) == list(tc(np.array([0.0]))) and list(tc(np.array([9.0]))) == list(tc(np.array([1.0]))), (
        "observations outside the box are clipped to it")
    assert list(tc(0.3)) == [1, 6], "a scalar observation works for a 1-D coder"


def test_stage1_two_dimensional_layout(nb):
    tc = nb.TileCoder([-1.2, -0.07], [0.5, 0.07], n_tilings=4, tiles_per_dim=6)
    assert tc.n_features == 4 * 36, "4 tilings of 6 x 6 tiles"
    want = np.array([[(i * (2 * j + 1) / 4) % 1 for j in range(2)] for i in range(4)])
    assert np.allclose(tc.offsets, want), f"asymmetric default offsets (1, 3) / n_tilings per tiling; got {tc.offsets}"
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-1.2, 0.5, 300), rng.uniform(-0.07, 0.07, 300)])
    for p in pts:
        idx = np.asarray(tc(p))
        assert np.all(idx // 36 == np.arange(4)), "the index of tiling i lies in [i * 36, (i + 1) * 36)"
    x = np.array([-0.5, 0.01])
    w = tc.width
    s0 = tc(x)
    s_close = tc(x + 0.1 * w)
    s_far = tc(x + 2.0 * w)
    shared = lambda u, v: len(set(np.asarray(u).tolist()) & set(np.asarray(v).tolist()))  # noqa: E731
    assert shared(s0, s_close) >= 2 and shared(s0, s_far) == 0, (
        "nearby points share most of their tiles (generalization), points two tile widths apart share none")
    idx = np.asarray(tc(np.array([0.5, 0.07])))
    assert np.all(idx < tc.n_features) and np.all(idx >= 0), "the top corner of the box must stay inside the index range"


def test_stage1_random_offsets(nb):
    a = nb.TileCoder([0, 0], [1, 1], n_tilings=8, tiles_per_dim=4, rng=0)
    b = nb.TileCoder([0, 0], [1, 1], n_tilings=8, tiles_per_dim=4, rng=0)
    c = nb.TileCoder([0, 0], [1, 1], n_tilings=8, tiles_per_dim=4)
    assert a.offsets.shape == (8, 2) and np.all((a.offsets >= 0) & (a.offsets < 1)), "random offsets in [0, 1) widths"
    assert np.array_equal(a.offsets, b.offsets) and not np.allclose(a.offsets, c.offsets), (
        "with rng given, offsets are random (as_rng(rng).random((n_tilings, d))) and reproducible; without, the "
        "deterministic asymmetric ones")


# ---------------------------------------------------------------- stage 2: mountain car


def test_stage2_spaces_and_reset(nb):
    env = nb.MountainCar()
    assert isinstance(env, Env), "MountainCar subclasses Env"
    assert isinstance(env.observation_space, Box) and np.allclose(env.observation_space.low, [-1.2, -0.07]) and \
        np.allclose(env.observation_space.high, [0.5, 0.07]), "observation_space = Box([-1.2, -0.07], [0.5, 0.07])"
    assert isinstance(env.action_space, Discrete) and env.action_space.n == 3, "3 actions"
    starts = []
    for s in range(30):
        obs, info = env.reset(seed=s)
        assert isinstance(obs, np.ndarray) and obs.shape == (2,) and obs[1] == 0.0 and isinstance(info, dict)
        starts.append(obs[0])
    assert -0.6 <= min(starts) and max(starts) < -0.4 and np.std(starts) > 0.03, (
        "the start position is uniform on [-0.6, -0.4), the velocity 0")
    assert np.array_equal(env.reset(seed=3)[0], nb.MountainCar().reset(seed=3)[0]), "reset(seed) is reproducible"


def test_stage2_dynamics(nb):
    env = nb.MountainCar()
    env.reset(seed=0)
    env.state, env.t = np.array([-0.5, 0.0]), 0
    obs, r, term, trunc, info = env.step(2)
    v = 0.001 - 0.0025 * np.cos(-1.5)
    assert np.allclose(obs, [-0.5 + v, v]), (
        f"v' = v + 0.001 (a - 1) - 0.0025 cos(3x) = {v:.8f}, then x' = x + v' (the NEW velocity); got {obs}")
    assert r == -1.0 and not term and not trunc, "every step costs -1"
    obs2, *_ = env.step(1)
    assert obs2 is not obs and np.allclose(obs, [-0.5 + v, v]), (
        "return a fresh array every step (state.copy()): replay buffers keep references to old observations")
    x_before = float(env.state[0])
    obs2[0] = 99.0
    assert float(env.state[0]) == x_before, (
        "the returned observation must be a copy: writing into it must not change the environment's own state")
    env.state = np.array([-1.19, -0.05])
    obs, *_ = env.step(0)
    assert obs[0] == -1.2 and obs[1] == 0.0, f"hitting the left wall clips x to -1.2 and sets the velocity to 0; got {obs}"
    env.state = np.array([-0.5, 0.0699])
    obs, *_ = env.step(2)
    assert np.isclose(obs[1], 0.07), "the velocity is clipped to [-0.07, 0.07]"
    env.state, env.t = np.array([0.49, 0.03]), 0
    obs, r, term, trunc, _ = env.step(1)
    assert term and obs[0] == 0.5 and not trunc, "reaching x = 0.5 terminates the episode"
    with pytest.raises(ValueError):
        env.step(3)


def test_stage2_truncation_and_energy_pumping(nb):
    env = nb.MountainCar(max_steps=5)
    env.reset(seed=1)
    flags = [env.step(1)[3] for _ in range(5)]
    assert flags == [False] * 4 + [True], f"truncated becomes True at max_steps = 5; got {flags}"
    env = _Capped(nb.MountainCar(), 1000)
    steps = []
    for s in range(5):
        obs, _ = env.reset(seed=s)
        for t in range(1, 201):
            obs, r, term, trunc, _ = env.step(2 if obs[1] >= 0 else 0)
            if term or trunc:
                break
        steps.append(t if term else None)
    assert all(s is not None and s < 150 for s in steps), (
        f"pushing in the direction of motion pumps energy and reaches the top in about 120 steps; got {steps} "
        f"(None = didn't make it in 200)")


# ---------------------------------------------------------------- stage 3: linear SARSA


class _TwoState(Env):
    """obs 0 -> (action a) -> obs 1 (reward 0) -> (action a) -> terminal (reward +1 if a == 1 else 0)."""

    def __init__(self):
        self.observation_space, self.action_space = Discrete(3), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.s = 0
        return np.array([0.0]), {}

    def step(self, a):
        if self.s == 0:
            self.s = 1
            return np.array([1.0]), 0.0, False, False, {}
        return np.array([2.0]), (1.0 if a == 1 else 0.0), True, False, {}


class _Feat:
    """state 0 -> features (0, 1), state 1 -> features (1, 2): they share feature 1."""

    n_features = 3

    def __call__(self, obs):
        return np.array([0, 1]) if obs[0] < 0.5 else np.array([1, 2])


def test_stage3_hand_update(nb):
    w, rets = nb.linear_sarsa(_Capped(_TwoState(), 10), _Feat(), 1, alpha=0.5, gamma=1.0, epsilon=0.0, rng=0)
    assert np.shape(w) == (3, 2), "w has shape (n_features, n_actions)"
    assert rets in ([0.0], [1.0]), f"one episode, return 0 or 1; got {rets}"
    total = rets[0]
    # first step: Q(s0, a) = Q(s1, a') = 0 -> delta 0, nothing changes. second step: delta = r - 0, and each of the
    # two active weights gets alpha / 2 * delta.
    nz = w[np.abs(w) > 0]
    assert (total == 0.0 and nz.size == 0) or (total == 1.0 and np.allclose(sorted(nz), [0.25, 0.25])), (
        f"the step size per weight is alpha / len(idx) = 0.25: after a +1 the two active weights of (s1, a) should be "
        f"0.25 each, and the first step (delta 0) changes nothing; got w = {w.tolist()}")


class _OneHot:
    """tabular features for an env with int observations: one active feature per state."""

    def __init__(self, n):
        self.n_features = n

    def __call__(self, obs):
        return np.array([int(np.ravel(obs)[0])])


class _Chain(Env):
    """one action; states 0 -> 1 -> ... -> len(rewards) (terminal), with the given rewards."""

    def __init__(self, rewards):
        self.rewards = list(rewards)
        self.observation_space, self.action_space = Discrete(len(rewards) + 1), Discrete(1)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.s = 0
        return 0, {}

    def step(self, action):
        r = self.rewards[self.s]
        self.s += 1
        return self.s, r, self.s == len(self.rewards), False, {}


class _Cutoff(Env):
    """episodes alternate: start in 1 (-> terminal, reward +1) or start in 0 (-> 1, reward 0, TRUNCATED)."""

    def __init__(self):
        self.observation_space, self.action_space, self.k = Discrete(3), Discrete(1), 0

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.k += 1
        self.s = self.k % 2
        return self.s, {}

    def step(self, action):
        if self.s == 1:
            return 2, 1.0, True, False, {}
        self.s = 1
        return 1, 0.0, False, True, {}


class _Fork(Env):
    """state 0 -> (either action, reward 0) -> state 1 -> action 0: +1, action 1: -1 -> terminal 2."""

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
        return 2, (1.0 if action == 0 else -1.0), True, False, {}


def test_stage3_tabular_special_case(nb):
    w, _ = nb.linear_sarsa(_Capped(_Chain([1.0, 2.0]), 100), _OneHot(3), 2, alpha=1.0, gamma=0.5, rng=0)
    assert np.allclose(w[:2, 0], [2.0, 2.0]), (
        f"with one-hot features linear SARSA is tabular SARSA. chain with rewards 1 then 2, alpha 1, gamma 0.5: after "
        f"two episodes Q(0) = 1 + 0.5 * 2 = 2, Q(1) = 2; got {np.round(w[:2, 0], 4).tolist()} (did you drop gamma?)")
    w, _ = nb.linear_sarsa(_Capped(_Cutoff(), 100), _OneHot(3), 4, alpha=1.0, gamma=0.9, rng=0)
    assert np.isclose(w[0, 0], 0.9), (
        f"episodes from state 0 are TRUNCATED in state 1 (worth 1): bootstrap, Q(0) = 0.9 Q(1) = 0.9; got "
        f"{w[0, 0]:.3f} (only `terminated` cuts the bootstrap)")
    q0s = []
    for seed in range(8):
        w, _ = nb.linear_sarsa(_Capped(_Fork(), 2000), _OneHot(3), 200, alpha=1.0, epsilon=1.0, rng=seed)
        assert np.allclose(w[1], [1, -1]), f"Q(1, .) = (1, -1) with alpha 1; got {np.round(w[1], 3).tolist()}"
        q0s.append(np.round(w[0], 9).tolist())
    assert any(q != [1.0, 1.0] for q in q0s), (
        f"SARSA's target uses the next action actually taken (random here), so over 8 seeds some Q(0, a) should end "
        f"at -1; got {q0s} (all (1, 1) means a max target: that's linear Q-learning)")


def test_stage3_generalization_through_shared_features(nb):
    gaps, diffs = [], []
    for seed in range(7):
        w, _ = nb.linear_sarsa(_Capped(_TwoState(), 4000), _Feat(), 1000, alpha=0.1, gamma=1.0, epsilon=0.2,
                               rng=seed)
        q0 = w[[0, 1]].sum(axis=0)
        q1 = w[[1, 2]].sum(axis=0)
        gaps.append(q1[1] - q1[0])
        diffs.append(abs(q0[1] - q1[1]))
    assert np.median(gaps) > 0.5, (
        f"Q(s1, a=1) (reward 1) should clearly beat Q(s1, a=0) (reward 0): median gap over 7 seeds "
        f"{np.median(gaps):.2f}")
    assert np.median(diffs) < 0.3, (
        f"Q(s0, .) bootstraps from Q(s1, .), so its value for the best path should be close to Q(s1, 1) too: "
        f"median |Q(s0, 1) - Q(s1, 1)| over 7 seeds {np.median(diffs):.2f}")


def test_stage3_mountain_car(nb):
    tc = nb.TileCoder([-1.2, -0.07], [0.5, 0.07], n_tilings=8, tiles_per_dim=8)
    w, rets = nb.linear_sarsa(_Capped(nb.MountainCar(), 30_000), tc, 100, alpha=0.5, rng=0)
    assert np.shape(w) == (tc.n_features, 3) and len(rets) == 100
    reached = sum(r > -200 for r in rets)
    assert np.mean(rets[:5]) == -200 and np.mean(rets[-10:]) > -180 and reached > 30, (
        f"with 8 tilings of 8 x 8, alpha 0.5 (0.5 / 8 per weight) and optimistic zero initial values, the car "
        f"should learn to reach the top: last 10 episodes average about -140 (got {np.mean(rets[-10:]):.0f}), and "
        f"well over 30 of the 100 episodes should succeed (got {reached})")
    w2, _ = nb.linear_sarsa(_Capped(nb.MountainCar(), 3000), tc, 10, alpha=0.5, rng=0)
    w3, _ = nb.linear_sarsa(_Capped(nb.MountainCar(), 3000), tc, 10, alpha=0.5, rng=0)
    assert np.array_equal(w2, w3), "the same rng must give the same weights"


# ---------------------------------------------------------------- stage 4: TD dynamics


def test_stage4_tabular_fixed_point_is_v(nb):
    P = np.array([[0.1, 0.6, 0.0, 0.3, 0.0], [0.0, 0.2, 0.5, 0.0, 0.3], [0.4, 0.0, 0.1, 0.5, 0.0],
                  [0.0, 0.3, 0.0, 0.2, 0.5], [0.5, 0.0, 0.25, 0.0, 0.25]])
    r = np.array([1.0, -0.5, 2.0, 0.0, -1.0])
    d = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    A, b = nb.td_matrices(np.eye(5), P, r, d, 0.9)
    assert A.shape == (5, 5) and b.shape == (5,)
    assert np.allclose(A, np.diag(d) @ (np.eye(5) - 0.9 * P)) and np.allclose(b, d * r), (
        "A = Phi^T D (Phi - gamma P Phi), b = Phi^T D r with D = diag(d)")
    V = np.linalg.solve(np.eye(5) - 0.9 * P, r)
    assert np.allclose(np.linalg.solve(A, b), V), "with one-hot (tabular) features the TD fixed point is exactly V"
    W = nb.expected_td(np.eye(5), P, r, d, 0.9, alpha=1.0, n_iters=3000)
    assert W.shape == (3001, 5) and np.allclose(W[0], 0) and np.allclose(W[-1], V, atol=1e-4), (
        "expected_td returns every iterate, starting with w0 = 0, and converges to V for tabular features")


def test_stage4_one_step(nb):
    Phi, P, r = _baird()
    d = np.full(7, 1 / 7)
    w0 = np.array([1, 1, 1, 1, 1, 1, 10, 1.0])
    W = nb.expected_td(Phi, P, r, d, 0.99, alpha=0.1, n_iters=2, w0=w0)
    A, b = nb.td_matrices(Phi, P, r, d, 0.99)
    assert np.allclose(W[1], w0 + 0.1 * (b - A @ w0)) and np.allclose(W[0], w0), (
        "each iterate is w + alpha (b - A w), starting from w0 (copied, not modified)")
    assert w0[6] == 10.0, "expected_td must not modify w0"


def test_stage4_bairds_counterexample(nb):
    Phi, P, r = _baird()
    w0 = np.array([1, 1, 1, 1, 1, 1, 10, 1.0])
    A_off, _ = nb.td_matrices(Phi, P, r, np.full(7, 1 / 7), 0.99)
    A_on, _ = nb.td_matrices(Phi, P, r, np.eye(7)[6], 0.99)
    assert np.linalg.eigvals(A_off).real.min() < -0.1 and np.linalg.eigvals(A_on).real.min() > -1e-9, (
        "off-policy (uniform d): A has an eigenvalue with negative real part, so w + alpha (b - A w) grows along it; "
        "on-policy (d on state 7) all real parts are >= 0")
    off = nb.expected_td(Phi, P, r, np.full(7, 1 / 7), 0.99, alpha=0.01, n_iters=1000, w0=w0)
    on = nb.expected_td(Phi, P, r, np.eye(7)[6], 0.99, alpha=0.01, n_iters=1000, w0=w0)
    assert np.linalg.norm(off[-1]) > 20 * np.linalg.norm(w0) and np.linalg.norm(on[-1]) < 2 * np.linalg.norm(w0), (
        f"Baird's counterexample: every reward is 0 (so w = 0 is a perfect solution), yet off-policy semi-gradient TD "
        f"diverges: |w| {np.linalg.norm(w0):.1f} -> {np.linalg.norm(off[-1]):.1f} in 1,000 expected updates, while the "
        f"on-policy distribution stays bounded ({np.linalg.norm(on[-1]):.1f})")
