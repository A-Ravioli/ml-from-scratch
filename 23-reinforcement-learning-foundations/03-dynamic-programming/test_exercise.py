"""Checks for exercise.py (dynamic programming).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The oracle for
optimality is brute force: on small random MDPs, every deterministic policy is evaluated with an
exact linear solve and the best is kept. Gridworld checks use V*(s) = -(Manhattan distance to the
goal), which holds with gamma = 1 and no slip. Every solver runs under mlfs.testing.bounded.
"""
import itertools

import numpy as np

from mlfs.rl import GridWorld  # the reference gridworld from topic 02 (your own one in the notebook)
from mlfs.testing import bounded


def _random_mdp(S, A, seed):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S) * 0.5, size=(S, A))
    R = rng.normal(size=(S, A))
    return P, R


def _v_of(P, R, actions, gamma):
    S = len(actions)
    P_pi = P[np.arange(S), actions]
    R_pi = R[np.arange(S), actions]
    return np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)


def _brute_force(P, R, gamma):
    S, A = R.shape
    best = None
    for acts in itertools.product(range(A), repeat=S):
        V = _v_of(P, R, np.array(acts), gamma)
        best = V if best is None else np.maximum(best, V)
    return best  # V* is the elementwise max over deterministic policies


def _check_policy(pi, S, A, name):
    pi = np.asarray(pi)
    assert pi.shape == (S, A), f"{name} should return a policy of shape {(S, A)}, got {pi.shape}"
    assert np.allclose(pi.sum(1), 1) and set(np.unique(pi)) <= {0.0, 1.0}, f"{name}'s policy must be one-hot rows"
    return np.argmax(pi, axis=1)


def _grid_vstar(g):
    gr, gc = g.goal
    return np.array([-(abs(s // g.width - gr) + abs(s % g.width - gc)) for s in range(g.n_states)], dtype=float)


# ---------------------------------------------------------------- stage 1: greedy + backup


def test_stage1_greedy_policy(nb):
    Q = np.array([[1.0, 3.0, 2.0], [5.0, 5.0, 0.0], [-1.0, -2.0, -1.0], [0.0, 0.0, 0.0]])
    Q0 = Q.copy()
    pi = nb.greedy_policy(Q)
    want = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    assert np.array_equal(np.asarray(pi), want), (
        f"greedy_policy puts probability 1 on the argmax of each row, ties to the LOWEST action; "
        f"expected {want.tolist()}, got {np.asarray(pi).tolist()}")
    assert np.array_equal(Q, Q0), "greedy_policy must not modify Q"
    assert np.asarray(pi).dtype == float, "the policy is a float array of probabilities"


def test_stage1_bellman_optimality_hand(nb):
    g = GridWorld(1, 3)
    V = np.array([-12.0, -8.0, 0.0])
    TV = nb.bellman_optimality(g.P, g.R, V, gamma=1.0)
    assert np.shape(TV) == (3,) and np.allclose(TV, [-9.0, -1.0, 0.0]), (
        f"(T V)(s) = max_a [R(s, a) + gamma sum P V]: from the random policy's values (-12, -8, 0) one backup gives "
        f"(-9, -1, 0) on the 1x3 corridor (best moves: right, right). got {np.round(TV, 3).tolist()}")
    P, R = _random_mdp(5, 3, 0)
    V = np.random.default_rng(1).normal(size=5)
    want = np.max(R + 0.9 * P @ V, axis=1)
    assert np.allclose(nb.bellman_optimality(P, R, V, gamma=0.9), want), "matches max_a (R + gamma P V) on a random MDP"


# ---------------------------------------------------------------- stage 2: policy iteration


def test_stage2_gridworld(nb):
    g = GridWorld(4, 5, goal=(1, 3))
    pi, V = bounded(nb.policy_iteration, g.P, g.R, gamma=1.0, what="policy_iteration")
    _check_policy(pi, 20, 4, "policy_iteration")
    want = _grid_vstar(g)
    assert np.allclose(V, want, atol=1e-6), (
        f"with -1 per step and gamma = 1, V*(s) = -(Manhattan distance to the goal). largest error "
        f"{np.max(np.abs(V - want)):.3g}: did you stop before the policy was stable?")
    acts = np.argmax(pi, axis=1)
    s, steps = g.start_state, 0
    while not g.terminal[s] and steps < 50:
        s = int(np.argmax(g.P[s, acts[s]]))
        steps += 1
    assert steps == 4, f"following the returned policy from (0, 0) should reach (1, 3) in 4 steps; took {steps}"


def test_stage2_random_mdps_brute_force(nb):
    for seed in range(4):
        P, R = _random_mdp(5, 3, seed)
        vstar = _brute_force(P, R, 0.9)
        pi, V = bounded(nb.policy_iteration, P, R, gamma=0.9, what="policy_iteration")
        acts = _check_policy(pi, 5, 3, "policy_iteration")
        assert np.allclose(V, vstar, atol=1e-6), (
            f"random MDP {seed}: V should equal V* found by trying all 243 deterministic policies; "
            f"largest error {np.max(np.abs(V - vstar)):.3g}")
        assert np.allclose(_v_of(P, R, acts, 0.9), vstar, atol=1e-6), "the returned policy must be optimal"


def test_stage2_slippery_grid(nb):
    g = GridWorld(5, 5, walls=[(1, 1), (2, 2), (3, 1)], slip=0.3)
    pi, V = bounded(nb.policy_iteration, g.P, g.R, gamma=0.95, what="policy_iteration")
    acts = _check_policy(pi, 25, 4, "policy_iteration")
    Vq = V.copy()
    for _ in range(3000):  # exact V* by long value iteration
        Vq = np.max(g.R + 0.95 * g.P @ Vq, axis=1)
    assert np.allclose(V, Vq, atol=1e-5), f"slippery 5x5 with walls: V should be V*; error {np.max(np.abs(V - Vq)):.3g}"
    assert np.allclose(_v_of(g.P, g.R, acts, 0.95), Vq, atol=1e-5), "and the returned policy must achieve it"


# ---------------------------------------------------------------- stage 3: value iteration


def test_stage3_gridworld_and_brute_force(nb):
    g = GridWorld(4, 5, goal=(1, 3), walls=[(2, 2)])
    pi, V = bounded(nb.value_iteration, g.P, g.R, gamma=1.0, what="value_iteration")
    _check_policy(pi, 20, 4, "value_iteration")
    Vq = np.zeros(20)
    for _ in range(200):
        Vq = np.max(g.R + g.P @ Vq, axis=1)
    assert np.allclose(V, Vq, atol=1e-6), f"V should be V* (-distance around the wall); got {np.round(V, 3).tolist()}"
    for seed in range(4, 7):
        P, R = _random_mdp(5, 3, seed)
        vstar = _brute_force(P, R, 0.8)
        pi, V = bounded(nb.value_iteration, P, R, gamma=0.8, tol=1e-10, what="value_iteration")
        acts = _check_policy(pi, 5, 3, "value_iteration")
        assert np.allclose(V, vstar, atol=1e-8), f"random MDP {seed}: V should be V*; error {np.max(np.abs(V - vstar)):.3g}"
        assert np.allclose(_v_of(P, R, acts, 0.8), vstar, atol=1e-8), "and its greedy policy optimal"


def test_stage3_iteration_semantics(nb):
    P = np.ones((1, 2, 1))  # one state, two actions, both loop
    R = np.array([[1.0, 2.0]])
    pi, V = bounded(nb.value_iteration, P, R, gamma=1.0, max_iters=20, what="value_iteration")
    assert 39 <= V[0] <= 41, (
        f"one state, best reward 2 per step forever, gamma = 1: V_k = 2k never converges, so stop after "
        f"max_iters = 20 updates (V = 40); got {V[0]}")
    assert np.array_equal(np.asarray(pi), [[0.0, 1.0]]), "the greedy policy takes the reward-2 action"
    pi, V = bounded(nb.value_iteration, P, R, gamma=0.5, tol=1e-12, what="value_iteration")
    assert np.isclose(V[0], 4.0), f"with gamma 0.5: V* = 2 / (1 - 0.5) = 4; got {V[0]}"


# ---------------------------------------------------------------- stage 4: modified policy iteration


def test_stage4_m1_is_value_iteration(nb):
    g = GridWorld(5, 5, slip=0.2)
    pi1, V1, n1 = bounded(nb.modified_policy_iteration, g.P, g.R, gamma=0.9, m=1, what="modified_policy_iteration")
    V, n = np.zeros(25), 0
    while True:
        n += 1
        Vn = np.max(g.R + 0.9 * g.P @ V, axis=1)
        if np.max(np.abs(Vn - V)) < 1e-8:
            V = Vn
            break
        V = Vn
    assert isinstance(n1, (int, np.integer)), "n_iters is an int: the number of greedy (improvement) steps"
    assert n1 == n and np.allclose(V1, V, atol=1e-12), (
        f"with m = 1, modified policy iteration IS value iteration: the same {n} iterations and the same V. "
        f"got {n1} iterations")


def test_stage4_larger_m_fewer_iterations(nb):
    g = GridWorld(6, 6, walls=[(1, 1), (2, 3), (4, 2)], slip=0.2)
    counts = {}
    for m in [1, 5, 50]:
        pi, V, n = bounded(nb.modified_policy_iteration, g.P, g.R, gamma=0.99, m=m, what="modified_policy_iteration")
        counts[m] = n
        Vq = np.zeros(36)
        for _ in range(4000):
            Vq = np.max(g.R + 0.99 * g.P @ Vq, axis=1)
        assert np.allclose(V, Vq, atol=1e-5), f"m = {m}: V should converge to V*; error {np.max(np.abs(V - Vq)):.3g}"
        assert np.allclose(_v_of(g.P, g.R, np.argmax(pi, 1), 0.99), Vq, atol=1e-5), f"m = {m}: the policy is optimal"
    assert counts[1] > 2 * counts[5] and counts[5] > counts[50], (
        f"more evaluation sweeps per improvement means far fewer improvement steps: got {counts}")
    pi, V, n = bounded(nb.modified_policy_iteration, g.P, g.R, gamma=0.99, m=1, max_iters=7, what="modified_policy_iteration")
    assert n == 7, f"stop after max_iters improvement steps; got n_iters = {n}"
