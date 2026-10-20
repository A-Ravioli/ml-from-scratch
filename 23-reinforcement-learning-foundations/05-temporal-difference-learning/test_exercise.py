"""Checks for exercise.py (temporal-difference learning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Update rules are
checked exactly on tiny deterministic environments (with alpha = 1 or hand-computed steps);
learning behaviour is checked on the cliff with fixed seeds and results averaged over a few runs.
Every environment is wrapped so that a loop that never ends fails after a fixed number of steps.
"""
import numpy as np

from mlfs.rl import Discrete, Env, GridWorld  # the reference protocol from topic 02


class _Capped:
    """wraps an env; fails the test after max_steps calls to step."""

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


def _greedy_path(Q, max_len=60):
    g = GridWorld(4, 12, start=(3, 0), goal=(3, 11))
    cliff = set(range(37, 47))
    s, path = 36, [36]
    for _ in range(max_len):
        a = int(np.argmax(Q[s]))
        s2 = int(np.argmax(g.P[s, a]))
        if s2 in cliff:
            return path, True
        s = s2
        path.append(s)
        if s == 47:
            break
    return path, False


# ---------------------------------------------------------------- stage 1: cliff + TD(0)


def test_stage1_cliff_layout(nb):
    env = nb.CliffWalking()
    assert isinstance(env, GridWorld), "CliffWalking subclasses your GridWorld (so it has P, R, reset, step)"
    assert (env.n_states, env.n_actions, env.start_state) == (48, 4, 36), "4 x 12 = 48 states, start (3, 0) = 36"
    assert np.flatnonzero(env.terminal).tolist() == [47], "the goal (3, 11) = 47 is the only terminal state"
    assert np.allclose(env.P[36, 1, 36], 1.0) and env.R[36, 1] == -100.0, (
        "stepping right from the start walks into the cliff: reward -100 and back to the start (36)")
    assert np.allclose(env.P[25, 2, 36], 1.0) and env.R[25, 2] == -100.0, "stepping down from (2, 1) into the cliff, too"
    assert np.allclose(env.P[24, 2, 36], 1.0) and env.R[24, 2] == -1.0, "stepping down from (2, 0) is a normal move to 36"
    assert np.allclose(env.P[35, 2, 47], 1.0) and env.R[35, 2] == -1.0, "stepping down from (2, 11) reaches the goal"
    assert np.allclose(env.P.sum(2), 1.0), "every row of P is a distribution"
    assert env.max_steps is not None and env.max_steps >= 1000, "a generous time limit (max_steps = 1000)"
    env.reset(seed=0)
    total, done = 0.0, False
    for a in [0] + [1] * 11 + [2]:
        _, r, term, trunc, _ = env.step(a)
        total += r
    assert term and total == -13.0, f"the safe-edge path up, 11 x right, down returns -13 and terminates; got {total}"


def test_stage1_td0_hand_updates(nb):
    env = _Capped(GridWorld(1, 3), 100)
    V = nb.td0_prediction(env, lambda s: 1, 1, alpha=0.5, gamma=1.0, rng=0)
    assert np.allclose(V, [-0.5, -0.5, 0.0]), (
        f"one episode right along the 1x3 corridor with alpha 0.5: V(0) += 0.5 (-1 + V(1) - V(0)) = -0.5, then "
        f"V(1) += 0.5 (-1 + 0 - V(1)) = -0.5 (no bootstrap from the terminal state). got {np.round(V, 4).tolist()}")
    V = nb.td0_prediction(_Capped(GridWorld(1, 3), 100), lambda s: 1, 2, alpha=0.5, gamma=1.0, rng=0)
    assert np.allclose(V, [-1.0, -0.75, 0.0]), (
        f"a second episode: V(0) = -0.5 + 0.5 (-1 - 0.5 + 0.5) = -1, V(1) = -0.5 + 0.5 (-1 + 0.5) = -0.75; "
        f"got {np.round(V, 4).tolist()}")


def test_stage1_td0_converges_and_bootstraps_on_truncation(nb):
    Vs = []
    for seed in range(3):
        prng = np.random.default_rng(3 + seed)
        Vs.append(nb.td0_prediction(_Capped(GridWorld(1, 3), 300_000), lambda s: int(prng.integers(4)), 6000,
                                    alpha=0.01, gamma=1.0, rng=seed))
    V = np.mean(Vs, axis=0)
    assert np.allclose(V, [-12, -8, 0], atol=1.2), (
        f"the random policy on the corridor: V should approach (-12, -8, 0); got {np.round(V, 2).tolist()}")
    V = nb.td0_prediction(_Capped(_Cutoff(), 2000), lambda s: 0, 400, alpha=0.2, gamma=0.9, rng=0)
    assert abs(V[1] - 1.0) < 0.01 and abs(V[0] - 0.9) < 0.02, (
        f"state 0's episodes are TRUNCATED after one step into state 1 (worth 1): a time limit isn't a terminal state, "
        f"so bootstrap: V(0) -> 0 + 0.9 V(1) = 0.9. got V(0) = {V[0]:.3f} (0 means you treated truncated as terminated)")


# ---------------------------------------------------------------- stage 2: SARSA and Q-learning


def test_stage2_update_targets(nb):
    Q, rets = nb.q_learning(_Capped(_Fork(), 2000), 200, alpha=1.0, gamma=1.0, epsilon=1.0, rng=0)
    assert np.allclose(Q[1], [1, -1]) and np.allclose(Q[0], [1, 1]), (
        f"Q-learning (alpha 1, random behaviour) backs up the MAX of the next state's values: Q(0, .) = max(1, -1) = 1. "
        f"got Q(0) = {np.round(Q[0], 3).tolist()}, Q(1) = {np.round(Q[1], 3).tolist()}")
    assert len(rets) == 200 and set(rets) <= {1.0, -1.0}, "episode_returns: one undiscounted return per episode"
    q0s = []
    for seed in range(8):
        Q, _ = nb.sarsa(_Capped(_Fork(), 2000), 200, alpha=1.0, gamma=1.0, epsilon=1.0, rng=seed)
        assert np.allclose(Q[1], [1, -1]) and set(np.round(Q[0], 9)) <= {1.0, -1.0}, (
            f"SARSA with alpha 1: Q(1, .) = (1, -1), and each Q(0, a) is the value of the next action actually "
            f"taken, +1 or -1; got Q(0) = {np.round(Q[0], 3).tolist()}, Q(1) = {np.round(Q[1], 3).tolist()}")
        q0s.append(Q[0].tolist())
    assert any(q != [1.0, 1.0] for q in q0s), (
        f"SARSA backs up the value of the action it actually takes next (random here), not the max: over 8 seeds "
        f"some Q(0, a) should end at -1. got {q0s} (all (1, 1) means you used max_a Q(s', a): that's Q-learning)")


def test_stage2_discounting(nb):
    for name in ("sarsa", "q_learning"):
        Q, _ = getattr(nb, name)(_Capped(_Chain([1.0, 2.0]), 100), 2, alpha=1.0, gamma=0.5, epsilon=0.1, rng=0)
        assert np.allclose(Q[:2, 0], [2.0, 2.0]), (
            f"{name}, chain with rewards 1 then 2, alpha 1, gamma 0.5: episode 1 sets Q(0) = 1 + 0.5 * 0 = 1 and "
            f"Q(1) = 2; episode 2 sets Q(0) = 1 + 0.5 * 2 = 2. expected (2, 2), got {np.round(Q[:2, 0], 4).tolist()}")


def test_stage2_bootstrap_on_truncation(nb):
    for name in ("sarsa", "q_learning"):
        Q, _ = getattr(nb, name)(_Capped(_Cutoff(), 100), 4, alpha=1.0, gamma=0.9, epsilon=0.1, rng=0)
        assert np.isclose(Q[1, 0], 1.0) and np.isclose(Q[0, 0], 0.9), (
            f"{name}: episodes from state 0 are TRUNCATED in state 1 (worth 1). a time limit isn't a terminal state, "
            f"so the target is 0 + 0.9 Q(1) = 0.9; got Q(0) = {Q[0, 0]:.3f} (0 means you cut the bootstrap on "
            f"truncation: only `terminated` does that)")


def test_stage2_cliff_sarsa_vs_q_learning(nb):
    sar, ql = [], []
    for seed in range(2):
        Qs, rs = nb.sarsa(_Capped(nb.CliffWalking(), 200_000), 300, alpha=0.2, epsilon=0.1, rng=seed)
        Qq, rq = nb.q_learning(_Capped(nb.CliffWalking(), 200_000), 300, alpha=0.2, epsilon=0.1, rng=seed)
        sar.append(np.mean(rs[-100:]))
        ql.append(np.mean(rq[-100:]))
        path_q, fell_q = _greedy_path(Qq)
        assert not fell_q and len(path_q) == 14, (
            f"Q-learning's greedy policy should be the optimal path along the cliff edge (13 steps); got "
            f"{len(path_q) - 1} steps{' and a fall' if fell_q else ''}")
        path_s, fell_s = _greedy_path(Qs)
        assert not fell_s, "SARSA's greedy policy should keep away from the cliff edge; following it falls off"

    assert np.mean(sar) > np.mean(ql) + 10, (
        f"online (epsilon-greedy) returns over the last 100 episodes: SARSA about -25, Q-learning about -50, because "
        f"Q-learning walks the edge and exploration pushes it off. got SARSA {np.mean(sar):.1f}, Q-learning "
        f"{np.mean(ql):.1f}")


def test_stage2_reproducible(nb):
    a = nb.q_learning(_Capped(nb.CliffWalking(), 50_000), 30, rng=4)[0]
    b = nb.q_learning(_Capped(nb.CliffWalking(), 50_000), 30, rng=4)[0]
    c = nb.sarsa(_Capped(nb.CliffWalking(), 50_000), 30, rng=4)[0]
    d = nb.sarsa(_Capped(nb.CliffWalking(), 50_000), 30, rng=4)[0]
    assert np.array_equal(a, b) and np.array_equal(c, d), "the same rng must give the same Q"


# ---------------------------------------------------------------- stage 3: expected SARSA


def test_stage3_expected_target(nb):
    Q, _ = nb.expected_sarsa(_Capped(_Fork(), 3000), 300, alpha=1.0, gamma=1.0, epsilon=1.0, rng=0)
    assert np.allclose(Q[1], [1, -1]) and np.allclose(Q[0], [0, 0]), (
        f"expected SARSA backs up the policy's AVERAGE: with epsilon 1 (uniform), Q(0, .) = (1 + -1) / 2 = 0; "
        f"got {np.round(Q[0], 4).tolist()}")
    Q, _ = nb.expected_sarsa(_Capped(_Fork(), 3000), 300, alpha=1.0, gamma=1.0, epsilon=0.5, rng=0)
    assert np.allclose(Q[0], [0.5, 0.5]), (
        f"with epsilon 0.5 and greedy action 0 at state 1: pi = (0.75, 0.25), so Q(0, .) = 0.75 - 0.25 = 0.5; "
        f"got {np.round(Q[0], 4).tolist()}")
    Q, _ = nb.expected_sarsa(_Capped(_Fork(), 30), 1, alpha=1.0, gamma=1.0, epsilon=0.2, rng=0)
    assert np.allclose(Q[0], 0.0), "in the first episode Q(1, .) is still 0 when Q(0, a) is updated, so Q(0, a) = 0"


def test_stage3_discounting_and_truncation(nb):
    Q, _ = nb.expected_sarsa(_Capped(_Chain([1.0, 2.0]), 100), 2, alpha=1.0, gamma=0.5, epsilon=0.1, rng=0)
    assert np.allclose(Q[:2, 0], [2.0, 2.0]), (
        f"chain with rewards 1 then 2, alpha 1, gamma 0.5: after two episodes Q(0) = 1 + 0.5 * 2 = 2; got "
        f"{np.round(Q[:2, 0], 4).tolist()}")
    Q, _ = nb.expected_sarsa(_Capped(_Cutoff(), 100), 4, alpha=1.0, gamma=0.9, epsilon=0.1, rng=0)
    assert np.isclose(Q[0, 0], 0.9), (
        f"bootstrap on truncation (only `terminated` cuts it): Q(0) = 0.9 Q(1) = 0.9; got {Q[0, 0]:.3f}")


def test_stage3_cliff(nb):
    rets = []
    for seed in range(2):
        Q, r = nb.expected_sarsa(_Capped(nb.CliffWalking(), 200_000), 300, alpha=0.9, epsilon=0.1, rng=seed)
        rets.append(np.mean(r[-100:]))
        path, fell = _greedy_path(Q)
        assert not fell and path[-1] == 47, "expected SARSA's greedy policy reaches the goal without falling"
    assert np.mean(rets) > -30, (
        f"expected SARSA tolerates a large step size (alpha 0.9) because its target has no next-action noise: "
        f"online returns should be about -20 to -25; got {np.round(rets, 1).tolist()}")


# ---------------------------------------------------------------- stage 4: n-step SARSA


def test_stage4_n_step_returns(nb):
    Q, rets = nb.n_step_sarsa(_Capped(_Chain([1.0, 2.0, 3.0, 4.0]), 100), 1, n=2, alpha=1.0, gamma=0.5, rng=0)
    want = [1 + 0.5 * 2, 2 + 0.5 * 3, 3 + 0.5 * 4, 4.0]
    assert np.allclose(Q[:4, 0], want), (
        f"n = 2, gamma 0.5, alpha 1, one episode of rewards 1, 2, 3, 4: every Q(S_t) becomes R_t+1 + 0.5 R_t+2 "
        f"+ 0.25 Q(S_t+2) with Q still 0, and the last ones stop at the terminal. expected {want}, "
        f"got {np.round(Q[:4, 0], 4).tolist()}")
    assert rets == [10.0], f"episode_returns holds the undiscounted sum, 10; got {rets}"
    Q, _ = nb.n_step_sarsa(_Capped(_Chain([1.0, 2.0, 3.0, 4.0]), 100), 2, n=2, alpha=1.0, gamma=0.5, rng=0)
    assert np.isclose(Q[0, 0], 2.0 + 0.25 * 5.0), (
        f"second episode: Q(S_0) = 1 + 0.5 * 2 + 0.25 Q(S_2) = 2 + 0.25 * 5 = 3.25 (Q(S_2) = 5 from episode 1); "
        f"got {Q[0, 0]}")
    Q, _ = nb.n_step_sarsa(_Capped(_Chain([1.0, 2.0, 3.0]), 100), 1, n=10, alpha=1.0, gamma=1.0, rng=0)
    assert np.allclose(Q[:3, 0], [6, 5, 3]), "with n longer than the episode, n-step SARSA is Monte Carlo: (6, 5, 3)"


def test_stage4_n1_is_sarsa(nb):
    for make, n_ep in [(lambda: _Chain([1.0, -2.0, 3.0, 0.5]), 5), (_Cutoff, 6)]:
        a = nb.n_step_sarsa(_Capped(make(), 200), n_ep, n=1, alpha=0.5, gamma=0.9, epsilon=0.1, rng=3)
        b = nb.sarsa(_Capped(make(), 200), n_ep, alpha=0.5, gamma=0.9, epsilon=0.1, rng=3)
        assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]), (
            f"with n = 1, n-step SARSA is one-step SARSA: on a one-action chain (and on an env that truncates) the two "
            f"must give the same Q after {n_ep} episodes; got {np.round(a[0][:, 0], 4).tolist()} vs "
            f"{np.round(b[0][:, 0], 4).tolist()}")


def test_stage4_bootstraps_on_truncation(nb):
    Q, _ = nb.n_step_sarsa(_Capped(_Cutoff(), 3000), 400, n=3, alpha=0.2, gamma=0.9, epsilon=0.0, rng=0)
    assert abs(Q[1, 0] - 1.0) < 0.01 and abs(Q[0, 0] - 0.9) < 0.02, (
        f"an episode cut off by a time limit still bootstraps from Q(S_T, A_T): Q(0) -> 0.9 Q(1) = 0.9. got "
        f"{Q[0, 0]:.3f}")


def test_stage4_cliff(nb):
    rets = [np.mean(nb.n_step_sarsa(_Capped(nb.CliffWalking(), 200_000), 200, n=4, alpha=0.2, rng=s)[1][-50:])
            for s in range(7)]
    assert np.median(rets) > -50, f"4-step SARSA should learn the cliff in 200 episodes (returns around -20); got {rets}"


# ---------------------------------------------------------------- stage 5: TD(lambda)


def test_stage5_hand_traces(nb):
    for lam, want in [(1.0, [0.5, 0.5, 0.5]), (0.5, [0.125, 0.25, 0.5]), (0.0, [0.0, 0.0, 0.5])]:
        V = nb.td_lambda_prediction(_Capped(_Chain([0.0, 0.0, 1.0]), 50), lambda s: 0, 1, lam=lam, alpha=0.5,
                                    gamma=1.0, rng=0)
        assert np.allclose(V[:3], want), (
            f"lambda = {lam}: one episode 0 -> 1 -> 2 -> end with rewards 0, 0, 1. only the last TD error is nonzero "
            f"(delta = 1), and it updates every state in proportion to its trace (gamma lambda)^k: expected {want}, "
            f"got {np.round(V[:3], 4).tolist()}")


def test_stage5_lambda0_is_td0_and_lambda1_is_mc(nb):
    prng1, prng2 = np.random.default_rng(1), np.random.default_rng(1)
    a = nb.td_lambda_prediction(_Capped(GridWorld(1, 4), 50_000), lambda s: int(prng1.integers(4)), 200, lam=0.0,
                                alpha=0.1, gamma=0.9, rng=2)
    b = nb.td0_prediction(_Capped(GridWorld(1, 4), 50_000), lambda s: int(prng2.integers(4)), 200, alpha=0.1,
                          gamma=0.9, rng=2)
    assert np.allclose(a, b), "TD(0) is TD(lambda) with lambda = 0: same seeds, same values"
    V = nb.td_lambda_prediction(_Capped(_Chain([1.0, 2.0, 3.0]), 50), lambda s: 0, 1, lam=1.0, alpha=0.5,
                                gamma=1.0, rng=0)
    assert np.allclose(V[:3], [3.0, 2.5, 1.5]), (
        f"with lambda = 1, one episode of TD(lambda) through distinct states is exactly the Monte Carlo update: "
        f"returns (6, 5, 3), so V = 0.5 * (6, 5, 3) = (3, 2.5, 1.5); got {np.round(V[:3], 4).tolist()}")


class _Revisit(Env):
    """one action: 0 -> 1 -> 0 -> terminal, rewards 0, 0, 1."""

    def __init__(self):
        self.observation_space, self.action_space = Discrete(3), Discrete(1)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.k = 0
        return 0, {}

    def step(self, action):
        self.k += 1
        return [(1, 0.0, False), (0, 0.0, False), (2, 1.0, True)][self.k - 1] + (False, {})


def test_stage5_traces_reset_each_episode(nb):
    V = nb.td_lambda_prediction(_Capped(_Chain([0.0, 1.0]), 50), lambda s: 0, 2, lam=1.0, alpha=0.5, gamma=1.0, rng=0)
    assert np.allclose(V[:2], [0.75, 0.75]), (
        f"two episodes of a chain with rewards 0, 1 (lambda = gamma = 1, alpha 0.5): V = (0.5, 0.5) after the first, "
        f"(0.75, 0.75) after the second. got {np.round(V[:2], 4).tolist()} ((1, 1) means the traces leaked from one "
        f"episode into the next: zero them at every reset)")


def test_stage5_accumulating_traces(nb):
    V = nb.td_lambda_prediction(_Capped(_Revisit(), 50), lambda s: 0, 1, lam=1.0, alpha=0.5, gamma=1.0, rng=0)
    assert np.allclose(V[:2], [1.0, 0.5]), (
        f"0 -> 1 -> 0 -> end, rewards 0, 0, 1, lambda = gamma = 1, alpha 0.5. state 0 is visited twice, so its "
        f"ACCUMULATING trace is 1 + 1 = 2 when the only nonzero TD error (1) arrives: V = 0.5 * (2, 1) = (1, 0.5). "
        f"got {np.round(V[:2], 4).tolist()} ((0.5, 0.5) means a replacing trace: e[s] = 1 instead of e[s] += 1)")
