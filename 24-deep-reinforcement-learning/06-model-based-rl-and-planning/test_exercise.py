"""Checks for exercise.py (model-based RL and planning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Dyna-Q is checked
against Q-learning (no planning) and against the exact fixed point of its own model (lots of
planning); Connect Four against an independent implementation over random games; MCTS on tiny
games with known minimax values and on Connect Four tactics; the AlphaZero pieces exactly and by
their invariants. Loops are capped so a bug fails instead of hanging.
"""
import numpy as np
import torch
from torch import nn

from mlfs.rl import Discrete, Env, GridWorld, q_learning
from mlfs.testing import bounded

torch.set_num_threads(1)  # tiny networks: one thread is fastest, and avoids oversubscribing a busy machine


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


class _Recorder(_Capped):
    """also records every (state, action, reward, next_state, terminated) transition taken."""

    def reset(self, *, seed=None):
        self.s, info = self.env.reset(seed=seed)
        return self.s, info

    def step(self, action):
        out = super().step(action)
        self.log = getattr(self, "log", [])
        self.log.append((self.s, action, out[1], out[0], out[2]))
        self.s = out[0]
        return out


class _Chain(Env):
    """states 0..4; action 1 moves right, action 0 left (0 stays); entering 4 pays +1 and terminates."""

    def __init__(self):
        self.observation_space, self.action_space = Discrete(5), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.s = 0
        return 0, {}

    def step(self, action):
        self.s = min(self.s + 1, 4) if action == 1 else max(self.s - 1, 0)
        return self.s, float(self.s == 4), self.s == 4, False, {}


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


class _Trap:
    """a two-ply game: root (+1) plays 0 or 1. after 0, the opponent has 3 replies: reply 0 wins for the opponent,
    replies 1 and 2 lose for it. after 1, the only reply draws. random play favours 0 (+1/3); minimax says 1 (0)."""

    n_actions = 3

    def initial_state(self):
        return ()

    def current_player(self, s):
        return 1 if len(s) % 2 == 0 else -1

    def is_terminal(self, s):
        return len(s) == 2

    def legal_actions(self, s):
        return [] if len(s) == 2 else ([0, 1] if len(s) == 0 else ([0, 1, 2] if s[0] == 0 else [0]))

    def next_state(self, s, a):
        assert a in self.legal_actions(s), f"illegal move {a} in state {s}"
        return s + (a,)

    def terminal_value(self, s):  # the player to move at a terminal state is +1
        return {(0, 0): -1.0, (0, 1): 1.0, (0, 2): 1.0}.get(s, 0.0)

    def encode(self, s):
        return np.zeros((1, 2), np.float32)


class _Flat:
    """one move to a non-terminal child, then one more move to a draw; 3 actions everywhere."""

    n_actions = 3

    def initial_state(self):
        return ()

    def current_player(self, s):
        return 1 if len(s) % 2 == 0 else -1

    def is_terminal(self, s):
        return len(s) == 2

    def legal_actions(self, s):
        return [] if len(s) == 2 else [0, 1, 2]

    def next_state(self, s, a):
        return s + (a,)

    def terminal_value(self, s):
        return 0.0

    def encode(self, s):
        return np.zeros((1, 2), np.float32)


class _Lucky:
    """root (+1) plays 0 or 1; each leads to a position with a single forced move: after 0 it ends in a win for +1
    (terminal_value is from the view of the player to move there, +1, so +1), after 1 in a draw."""

    n_actions = 2

    def initial_state(self):
        return ()

    def current_player(self, s):
        return 1 if len(s) % 2 == 0 else -1

    def is_terminal(self, s):
        return len(s) == 2

    def legal_actions(self, s):
        return [] if len(s) == 2 else ([0, 1] if len(s) == 0 else [0])

    def next_state(self, s, a):
        assert a in self.legal_actions(s), f"illegal move {a} in state {s}"
        return s + (a,)

    def terminal_value(self, s):
        return 1.0 if s == (0, 0) else 0.0

    def encode(self, s):
        return np.zeros((1, 2), np.float32)


def _ref_winner(board):
    """an independent full-board scan: +1, -1 or 0."""
    R, C = board.shape
    for r in range(R):
        for c in range(C):
            p = board[r, c]
            if p == 0:
                continue
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                cells = [(r + k * dr, c + k * dc) for k in range(4)]
                if all(0 <= rr < R and 0 <= cc < C and board[rr, cc] == p for rr, cc in cells):
                    return int(p)
    return 0


def _play(game, moves):
    s = game.initial_state()
    for m in moves:
        s = game.next_state(s, m)
    return s


# ---------------------------------------------------------------- stage 1: Dyna-Q


def test_stage1_no_planning_is_q_learning(nb):
    for make, n_ep in [(lambda: GridWorld(3, 4, slip=0.2), 30), (_Chain, 20), (lambda: GridWorld(3, 4, max_steps=4), 30)]:
        a = nb.dyna_q(_Capped(make(), 50_000), n_ep, n_planning=0, alpha=0.5, gamma=0.9, epsilon=0.2, rng=7)
        b = q_learning(_Capped(make(), 50_000), n_ep, alpha=0.5, gamma=0.9, epsilon=0.2, rng=7)
        assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]), (
            "with n_planning = 0, Dyna-Q is exactly Q-learning: same seeding (env.reset(seed=int(rng.integers(2**31))) "
            "on the first episode), same epsilon_greedy draws, same update (bootstrapping through a time limit: only "
            "`terminated` cuts it). your Q differs from mlfs.rl.q_learning's")


def test_stage1_planning_solves_the_model(nb):
    env = _Recorder(_Chain(), 10_000)
    Q, rets = bounded(nb.dyna_q, env, 1, n_planning=500, alpha=1.0, gamma=0.9, epsilon=1.0, rng=0, budget=30_000_000,
                      what="dyna_q", hint="the planning loop runs n_planning times per real step.")
    assert rets == [1.0], f"one episode of the chain earns 1; episode_returns {rets}"
    model = {(s, a): (r, s2, t) for s, a, r, s2, t in env.log}
    want = np.zeros((5, 2))
    for _ in range(200):  # value iteration on the learned model, unobserved pairs stay 0
        for (s, a), (r, s2, t) in model.items():
            want[s, a] = r if t else r + 0.9 * want[s2].max()
    got = np.array([Q[s, a] for s, a in model])
    exp = np.array([want[s, a] for s, a in model])
    assert np.allclose(got, exp, atol=1e-6), (
        f"with alpha = 1 and 500 planning updates per step, Q on every pair seen in the real episode must "
        f"converge to the fixed point of the learned (deterministic) model, e.g. Q(3, right) = 1, Q(2, right) = 0.9. "
        f"max error {np.max(np.abs(got - exp)):.4f}. does planning sample previously SEEN (s, a) pairs and replay "
        f"the stored (r, s', terminated)?")


def test_stage1_model_keeps_terminated(nb):
    Q, _ = nb.dyna_q(_Capped(_Cutoff(), 200), 6, n_planning=20, alpha=0.5, gamma=0.9, epsilon=0.0, rng=0)
    assert abs(Q[1, 0] - 1.0) < 0.01 and abs(Q[0, 0] - 0.9) < 0.02, (
        f"episodes from state 0 are TRUNCATED in state 1 (worth 1): the model must store `terminated` (False here), "
        f"so planning keeps bootstrapping: Q(0) -> 0.9 Q(1) = 0.9. got Q(0) = {Q[0, 0]:.3f}")


def test_stage1_planning_speeds_up_learning(nb):
    maze = dict(start=(2, 0), goal=(0, 8), walls=[(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)],
                step_reward=0.0, max_steps=2000)

    class _Goal(GridWorld):
        def step(self, action):
            s2, r, term, trunc, info = super().step(action)
            return s2, float(term), term, trunc, info

    class _Episodes(_Capped):
        def reset(self, *, seed=None):
            self.starts = getattr(self, "starts", []) + [self.steps]
            return super().reset(seed=seed)

    steps = {}
    for n in (0, 20):
        later = []
        for seed in range(5):
            env = _Episodes(_Goal(6, 9, **maze), 200_000)
            bounded(nb.dyna_q, env, 15, n_planning=n, alpha=0.1, gamma=0.95, epsilon=0.1, rng=seed,
                    budget=50_000_000, what="dyna_q")
            later.append(env.steps - env.starts[1])  # real steps in episodes 2..15 (episode 1 is pure search)
        steps[n] = np.median(later)
    assert steps[20] < 0.5 * steps[0], (
        f"Sutton & Barto's Dyna maze: after the first episode finds the goal, 20 planning updates per real step should "
        f"spread its value through the model, so episodes 2-15 need far fewer real steps (median over 5 seeds, at "
        f"most half). got {steps[0]:.0f} steps without planning, {steps[20]:.0f} with")


# ---------------------------------------------------------------- stage 2: Connect Four


def test_stage2_rules(nb):
    g = nb.ConnectFour()
    assert (g.rows, g.cols, g.n_actions) == (6, 7, 7), "the default board is 6 rows x 7 columns, 7 actions"
    s0 = g.initial_state()
    assert g.current_player(s0) == 1 and g.legal_actions(s0) == list(range(7)) and not g.is_terminal(s0), (
        "player +1 moves first on an empty board; every column is legal (ascending order)")
    s1 = g.next_state(s0, 3)
    assert g.current_player(s1) == -1 and g.current_player(s0) == 1, "players alternate; states are never mutated"
    x = g.encode(s1)
    assert x.shape == (2, 6, 7) and x.dtype == np.float32, "encode returns float32 planes (2, rows, cols)"
    assert x[1, 5, 3] == 1 and x.sum() == 1, (
        "plane 0 = the player to move's stones, plane 1 = the opponent's; row 0 is the top, so the first stone in "
        "column 3 lands at [5, 3] and belongs to the opponent of the player now to move")
    s = _play(g, [0] * 6)
    assert 0 not in g.legal_actions(s), "a full column is not a legal action"
    for bad in (-1, 7):
        try:
            g.next_state(s0, bad)
        except ValueError:
            continue
        raise AssertionError(f"next_state(state, {bad}) must raise ValueError: columns are 0..6")
    win = _play(g, [0, 0, 1, 1, 2, 2, 3])
    assert g.is_terminal(win) and g.terminal_value(win) == -1.0 and g.legal_actions(win) == [], (
        "four in a row ends the game; terminal_value is from the view of the player to move at the terminal state, "
        "who has just lost: -1; no legal actions remain")


def test_stage2_matches_an_independent_implementation(nb):
    rng = np.random.default_rng(0)
    for rows, cols in [(6, 7), (4, 5), (5, 4)]:
        g = nb.ConnectFour(rows, cols)
        for _ in range(60):
            s, board, player = g.initial_state(), np.zeros((rows, cols), int), 1
            for _ in range(rows * cols + 1):
                legal = [c for c in range(cols) if board[0, c] == 0]
                w = _ref_winner(board)
                done = w != 0 or not legal
                assert g.is_terminal(s) == done, (
                    f"is_terminal disagrees with a full-board check on\n{board}\n(winner {w}); yours says "
                    f"{g.is_terminal(s)}. check all four directions, including both diagonals")
                if done:
                    want = 0.0 if w == 0 else (1.0 if w == player else -1.0)
                    assert g.terminal_value(s) == want, f"terminal_value: expected {want} on\n{board}"
                    break
                assert g.legal_actions(s) == legal, f"legal_actions: expected {legal}, got {g.legal_actions(s)}"
                c = int(rng.choice(legal))
                r = max(np.flatnonzero(board[:, c] == 0))
                board[r, c] = player
                player = -player
                s = g.next_state(s, c)
                x = g.encode(s)
                assert np.array_equal(x[0], board == player) and np.array_equal(x[1], board == -player), (
                    "encode's planes don't match the board (plane 0: player to move, plane 1: opponent; row 0 top)")


def test_stage2_draw(nb):
    g = nb.ConnectFour(2, 3)
    s = _play(g, [0, 0, 1, 1, 2, 2])
    assert g.is_terminal(s) and g.terminal_value(s) == 0.0, "a full board without four in a row is a draw (value 0)"


# ---------------------------------------------------------------- stage 3: MCTS


def _search(nb, game, state, **kw):
    return bounded(nb.MCTS(game, **kw).search, state, budget=30_000_000, what="MCTS.search",
                   hint="search must stop after n_simulations simulations, and rollouts must reach a terminal state.")


def test_stage3_output_and_reproducibility(nb):
    g = nb.ConnectFour(4, 4)
    s = _play(g, [0, 0, 0, 0])  # column 0 is full
    pi = _search(nb, g, s, n_simulations=60, rng=0)
    assert isinstance(pi, np.ndarray) and pi.shape == (4,) and abs(pi.sum() - 1) < 1e-9, (
        "search returns the root's visit counts normalized to a distribution over all n_actions")
    assert pi[0] == 0, "illegal actions (a full column) get no visits"
    assert np.allclose(pi * 60, np.round(pi * 60)), "with 60 simulations, every entry is a multiple of 1/60"
    assert np.array_equal(pi, _search(nb, g, s, n_simulations=60, rng=0)), "the same rng gives the same search"


def test_stage3_sees_past_the_opponents_reply(nb):
    picks = [int(np.argmax(_search(nb, _Trap(), (), n_simulations=300, c_puct=1.0, rng=seed))) for seed in range(5)]
    assert sum(p == 1 for p in picks) >= 4, (
        f"in the trap game, random play makes move 0 look good (+1/3 on average) but the opponent's best reply to it "
        f"wins: the minimax-correct move is 1 (a draw). with 300 simulations MCTS should prefer it; argmax over 5 "
        f"seeds: {picks}. check the backup: each level's value is from the view of the player to move there, so it "
        f"flips sign on the way up")


def test_stage3_rollout_values_change_point_of_view(nb):
    picks = [int(np.argmax(_search(nb, _Lucky(), (), n_simulations=20, c_puct=0.1, rng=seed))) for seed in range(3)]
    assert picks == [0, 0, 0], (
        f"after move 0 the rollout (one forced move) ends in a position worth +1 to the player to move there, who is "
        f"the ROOT player: so the leaf, where the opponent is to move, is worth -1 to its player, and move 0 is worth "
        f"+1 to the root. the search should pick move 0; argmax over 3 seeds {picks}. convert a rollout's terminal "
        f"value with current_player(end) * current_player(leaf)")


def test_stage3_uses_evaluate(nb):
    def prior_only(state):
        return np.array([0.1, 0.1, 0.8]), 0.0

    pi = _search(nb, _Flat(), (), n_simulations=20, evaluate=prior_only, rng=0)
    assert np.argmax(pi) == 2 and pi[2] > 0.5, (
        f"with every value 0, PUCT's visits follow the prior (0.1, 0.1, 0.8); got visit distribution {pi.round(2)}")

    def child_values(state):  # the value is for the player to move at `state`
        v = {(0,): 0.9, (1,): -0.9, (2,): 0.0}.get(state, 0.0)
        return np.ones(3) / 3, v

    pi = _search(nb, _Flat(), (), n_simulations=60, evaluate=child_values, c_puct=1.0, rng=0)
    assert np.argmax(pi) == 1, (
        f"evaluate says the position after move 1 is worth -0.9 to the player who must move there (the opponent): "
        f"so move 1 is the best for the root, and move 0 (worth +0.9 to the opponent) the worst. visits "
        f"{pi.round(2)}: did you forget to flip the sign of a child's value for the parent?")


def test_stage3_connect_four_tactics(nb):
    g = nb.ConnectFour()
    win = _play(g, [0, 0, 1, 1, 2, 2])
    picks = [int(np.argmax(_search(nb, g, win, n_simulations=200, rng=s))) for s in range(3)]
    assert picks.count(3) >= 2, f"+1 has three in a row on the bottom: column 3 wins at once. picks over seeds {picks}"
    block = _play(g, [6, 0, 5, 1, 6, 2])
    picks = [int(np.argmax(_search(nb, g, block, n_simulations=400, rng=s))) for s in range(3)]
    assert picks.count(3) >= 2, f"-1 threatens four on the bottom row: +1 must block column 3. picks {picks}"


# ---------------------------------------------------------------- stage 4: AlphaZero-lite


def test_stage4_policy_value_net(nb):
    torch.manual_seed(0)
    g = nb.ConnectFour(4, 5)
    net = nb.PolicyValueNet(g, hidden_sizes=(32, 32))
    assert isinstance(net, nn.Module), "PolicyValueNet is an nn.Module"
    x = torch.as_tensor(np.stack([g.encode(_play(g, m)) for m in ([], [1], [1, 2])]))
    logits, value = net(x)
    assert logits.shape == (3, 5) and value.shape == (3,), f"(logits (B, n_actions), value (B,)); got {logits.shape}"
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(30.0)
        _, big = net(x)
    assert torch.all(big.abs() <= 1), "the value head ends in tanh: values stay in [-1, 1] however large the weights"
    net = nb.PolicyValueNet(g, hidden_sizes=(32, 32))
    prior, v = net.evaluate(_play(g, [1]))
    assert isinstance(prior, np.ndarray) and prior.shape == (5,) and abs(prior.sum() - 1) < 1e-5, (
        "evaluate(state) returns the softmax prior as an ndarray over all actions")
    assert isinstance(v, float) and -1 <= v <= 1, "and the value as a Python float"


def test_stage4_loss_by_hand(nb):
    logits = torch.tensor([[0.0, 0.0], [2.0, 0.0]], requires_grad=True)
    values = torch.tensor([0.5, -0.5], requires_grad=True)
    pi = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    z = torch.tensor([1.0, -1.0])
    loss = nb.alphazero_loss(logits, values, pi, z)
    lsm = torch.log_softmax(logits.detach(), 1)
    ce = -(pi * lsm).sum(1).mean()
    want = ce + ((values.detach() - z) ** 2).mean()
    assert abs(loss.item() - want.item()) < 1e-5, (
        f"loss = mean cross-entropy(pi, softmax(logits)) + mean (v - z)^2 = {want.item():.4f}; got {loss.item():.4f}")
    loss.backward()
    assert torch.allclose(logits.grad, (torch.softmax(logits.detach(), 1) - pi) / 2, atol=1e-6), (
        "the policy term's gradient on the logits is (softmax - pi) / B")


def test_stage4_self_play_records(nb):
    g = nb.ConnectFour(4, 4)
    mcts = nb.MCTS(g, n_simulations=12, rng=0)
    X, P, Z = bounded(nb.self_play, g, mcts, temperature=1.0, rng=1, budget=30_000_000, what="self_play")
    T = len(Z)
    assert X.shape == (T, 2, 4, 4) and P.shape == (T, 4) and Z.shape == (T,), (
        f"self_play returns one record per move: encodings (T, 2, 4, 4), visit distributions (T, 4), outcomes (T,); "
        f"got {X.shape}, {P.shape}, {Z.shape}")
    assert np.allclose(P.sum(1), 1), "each record's pi is the search's visit distribution"
    full = X.sum(1)[:, 0, :] > 0  # a column is full when its top cell is occupied
    assert np.all(P[full] == 0), "pi is 0 on full columns (illegal moves)"
    assert np.all(X[1:].sum((1, 2, 3)) - X[:-1].sum((1, 2, 3)) == 1), "consecutive records are one move apart"
    if np.all(Z == 0):
        assert X[-1].sum() == 15, "an all-zero outcome means a draw, which needs a full board"
    else:
        assert Z[-1] == 1 and np.all(Z[1:] == -Z[:-1]), (
            f"z is the final result from the view of the player to move at each record: it alternates in sign, and the "
            f"last mover won, so the last record's z is +1. got {Z.tolist()}")
    again = nb.self_play(g, nb.MCTS(g, n_simulations=12, rng=0), temperature=1.0, rng=1)
    assert np.array_equal(again[2], Z) and np.array_equal(again[0], X), "the same seeds give the same game"
    _, P0, _ = nb.self_play(g, nb.MCTS(g, n_simulations=12, rng=0), temperature=0, rng=1)
    assert len(P0) > 0, "temperature 0 plays the most-visited move"


class _Masked(_Flat):
    """like _Flat, but action 2 is illegal at the root."""

    def legal_actions(self, s):
        return [] if len(s) == 2 else ([0, 1] if len(s) == 0 else [0, 1, 2])


def test_stage3_prior_is_renormalized_over_legal_moves(nb):
    def evaluate(state):  # most of the prior on an illegal move; the opponent is worse off after move 1
        return np.array([0.04, 0.04, 0.92]), {(1,): -0.5}.get(state, 0.0)

    pi = _search(nb, _Masked(), (), n_simulations=20, evaluate=evaluate, c_puct=1.0, rng=0)
    assert pi[2] == 0, "the illegal move gets no visits"
    assert pi[0] >= 0.35, (
        f"the prior must be renormalized over the LEGAL moves, giving (0.5, 0.5, 0), so the search keeps exploring "
        f"move 0 against move 1's early lead; with the raw prior (0.04, 0.04) the exploration bonus is ten times too "
        f"small and move 0 is starved. visits {pi.round(3)}")


class _OneMove:
    """one move ends the game: move 0 wins for the mover, move 1 draws. the outcome reveals the move."""

    n_actions = 2

    def initial_state(self):
        return ()

    def current_player(self, s):
        return 1 if len(s) == 0 else -1

    def is_terminal(self, s):
        return len(s) == 1

    def legal_actions(self, s):
        return [] if s else [0, 1]

    def next_state(self, s, a):
        return s + (a,)

    def terminal_value(self, s):
        return -1.0 if s == (0,) else 0.0

    def encode(self, s):
        return np.zeros((1, 2), np.float32)


class _FixedSearch:
    def search(self, state):
        return np.array([0.6, 0.4])


def test_stage4_self_play_temperature(nb):
    rng = np.random.default_rng(0)
    first = [nb.self_play(_OneMove(), _FixedSearch(), temperature=0.5, rng=int(rng.integers(2**31)))[2][0] == 1
             for _ in range(3000)]
    p = np.mean(first)
    want = 0.36 / 0.52
    assert abs(p - want) < 0.03, (
        f"with temperature 0.5 the move is drawn from pi^(1/T) / sum = (0.36, 0.16) / 0.52: move 0 with probability "
        f"{want:.3f} (T = 1 would give 0.6). measured {p:.3f} over 3,000 games")
