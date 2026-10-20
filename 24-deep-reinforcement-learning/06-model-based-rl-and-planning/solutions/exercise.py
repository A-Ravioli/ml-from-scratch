import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.rl import epsilon_greedy  # yours, from ch 23 · topic 04
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # model-based RL & planning · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `dyna_q`: Q-learning that also learns from a model of its own experience → `mlfs.deeprl` |
    | 2 | `ConnectFour`: a two-player game behind the protocol MCTS searches → `mlfs.deeprl` |
    | 3 | `MCTS`: PUCT Monte Carlo tree search, with rollouts or a learned evaluator → `mlfs.deeprl` |
    | 4 | `PolicyValueNet`, `alphazero_loss`, `self_play`: the pieces of AlphaZero-lite |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def dyna_q(env, n_episodes, *, n_planning=10, alpha=0.1, gamma=0.95, epsilon=0.1, rng=None):
    """stage 1: tabular Dyna-Q: each real Q-learning step is followed by n_planning updates replayed from the model."""
    rng = as_rng(rng)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    model = {}
    seen = []
    episode_returns = []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(2**31))) if ep == 0 else env.reset()
        done, total = False, 0.0
        while not done:
            a = epsilon_greedy(Q[s], epsilon, rng=rng)
            s2, r, terminated, truncated, _ = env.step(a)
            total += r
            Q[s, a] += alpha * ((r if terminated else r + gamma * np.max(Q[s2])) - Q[s, a])
            if (s, a) not in model:
                seen.append((s, a))
            model[(s, a)] = (r, s2, terminated)
            for _ in range(n_planning):
                ps, pa = seen[int(rng.integers(len(seen)))]
                pr, ps2, pterm = model[(ps, pa)]
                Q[ps, pa] += alpha * ((pr if pterm else pr + gamma * np.max(Q[ps2])) - Q[ps, pa])
            s, done = s2, terminated or truncated
        episode_returns.append(total)
    return Q, episode_returns


@app.cell(hide_code=True)
def _():
    checks.run(1, dyna_q=dyna_q)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ConnectFour:
    """Connect Four on a rows x cols board; states are immutable values."""

    def __init__(self, rows=6, cols=7):
        """stage 2: store rows, cols; n_actions = cols."""
        self.rows, self.cols = int(rows), int(cols)
        self.n_actions = self.cols

    def initial_state(self):
        """stage 2: the empty board with player +1 to move."""
        return (bytes(self.rows * self.cols), 1, 0, 0)

    def current_player(self, state):
        """stage 2: +1 or -1, the player to move."""
        return state[1]

    def _board(self, state):
        return np.frombuffer(state[0], dtype=np.int8).reshape(self.rows, self.cols)

    def legal_actions(self, state):
        """stage 2: the columns that aren't full (empty list at a terminal state), ascending."""
        if self.is_terminal(state):
            return []
        board = self._board(state)
        return [c for c in range(self.cols) if board[0, c] == 0]

    def next_state(self, state, action):
        """stage 2: drop the mover's stone into column `action`; returns a NEW state."""
        if not (isinstance(action, (int, np.integer)) and 0 <= action < self.cols):
            raise ValueError(f"illegal action {action!r}: columns are 0..{self.cols - 1}")
        board = self._board(state).copy()
        player = state[1]
        free = np.flatnonzero(board[:, action] == 0)
        if self.is_terminal(state) or len(free) == 0:
            raise ValueError(f"illegal action {action}")
        r = free[-1]
        board[r, action] = player
        winner = player if self._wins(board, r, action, player) else 0
        return (board.tobytes(), -player, winner, state[3] + 1)

    def _wins(self, board, r, c, p):
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            n = 1
            for sign in (1, -1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < self.rows and 0 <= cc < self.cols and board[rr, cc] == p:
                    n += 1
                    rr, cc = rr + sign * dr, cc + sign * dc
            if n >= 4:
                return True
        return False

    def is_terminal(self, state):
        """stage 2: someone has four in a row, or the board is full."""
        return state[2] != 0 or state[3] == self.rows * self.cols

    def terminal_value(self, state):
        """stage 2: from the view of the player to move at the terminal state: -1 (they lost) or 0 (draw)."""
        return 0.0 if state[2] == 0 else (1.0 if state[2] == state[1] else -1.0)

    def encode(self, state):
        """stage 2: float32 planes (2, rows, cols): the mover's stones, then the opponent's."""
        board = self._board(state)
        p = state[1]
        return np.stack([board == p, board == -p]).astype(np.float32)


@app.cell(hide_code=True)
def _():
    checks.run(2, ConnectFour=ConnectFour)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class MCTS:
    """PUCT tree search over any game with the ConnectFour protocol; a fresh tree per search."""

    def __init__(self, game, *, n_simulations=100, c_puct=1.0, evaluate=None, rng=None):
        """stage 3: store the arguments; rng (as_rng) drives the random rollouts used when evaluate is None."""
        self.game, self.n_simulations, self.c_puct = game, int(n_simulations), float(c_puct)
        self.evaluate = evaluate
        self.rng = as_rng(rng)

    def _evaluate(self, state):
        """(prior over all actions, value for the player to move) at a non-terminal state."""
        legal = self.game.legal_actions(state)
        if self.evaluate is not None:
            prior, value = self.evaluate(state)
            prior = np.asarray(prior, dtype=float).copy()
            mask = np.zeros(self.game.n_actions, bool)
            mask[legal] = True
            prior[~mask] = 0.0
            prior = prior / prior.sum() if prior.sum() > 0 else mask / mask.sum()
            return prior, float(value)
        prior = np.zeros(self.game.n_actions)
        prior[legal] = 1.0 / len(legal)
        s = state
        while not self.game.is_terminal(s):
            moves = self.game.legal_actions(s)
            s = self.game.next_state(s, moves[int(self.rng.integers(len(moves)))])
        sign = self.game.current_player(s) * self.game.current_player(state)
        return prior, sign * self.game.terminal_value(s)

    def search(self, state):
        """stage 3: run n_simulations simulations from state; returns visit counts / total, shape (n_actions,)."""
        root = {"children": {}}
        self._expand(root, state)
        for _ in range(self.n_simulations):
            node, s, path = root, state, []
            while node.get("prior") is not None and not node["terminal"]:
                a = self._select(node)
                path.append((node, a))
                s = self.game.next_state(s, a)
                node = node["children"].setdefault(a, {"children": {}})
            value = self._expand(node, s) if node.get("prior") is None else node["value"]
            for parent, a in reversed(path):  # value is for the player to move at the child: flip it
                value = -value
                parent["N"][a] += 1
                parent["W"][a] += value
        visits = root["N"].copy()
        return visits / visits.sum() if visits.sum() > 0 else visits

    def _expand(self, node, state):
        node["N"] = np.zeros(self.game.n_actions)
        node["W"] = np.zeros(self.game.n_actions)
        node["terminal"] = self.game.is_terminal(state)
        if node["terminal"]:
            node["prior"], node["value"] = np.zeros(self.game.n_actions), self.game.terminal_value(state)
        else:
            node["prior"], node["value"] = self._evaluate(state)
            node["legal"] = np.asarray(self.game.legal_actions(state))
        return node["value"]

    def _select(self, node):
        legal = node["legal"]
        N, W, P = node["N"][legal], node["W"][legal], node["prior"][legal]
        Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
        U = self.c_puct * P * np.sqrt(N.sum()) / (1 + N)
        return int(legal[int(np.argmax(Q + U))])


@app.cell(hide_code=True)
def _():
    checks.run(3, MCTS=MCTS, ConnectFour=ConnectFour)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class PolicyValueNet(nn.Module):
    """AlphaZero-lite's network: encoded planes -> (move logits (B, n_actions), value (B,) in [-1, 1])."""

    def __init__(self, game, *, hidden_sizes=(128, 128)):
        """stage 4: a shared MLP trunk on the flattened encoding, a policy head and a tanh value head."""
        super().__init__()
        self.game = game
        n_in = int(np.prod(game.encode(game.initial_state()).shape))
        self.trunk = nn.Sequential(MLP(n_in, tuple(hidden_sizes[:-1]), hidden_sizes[-1]), nn.ReLU())
        self.policy_head = nn.Linear(hidden_sizes[-1], game.n_actions)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        """stage 4: x (B, *encoding shape) float -> (logits (B, n_actions), value (B,))."""
        h = self.trunk(x.reshape(x.shape[0], -1))
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)

    def evaluate(self, state):
        """stage 4: MCTS's evaluate hook: (softmax prior over actions as an ndarray, value float), no gradients."""
        with torch.no_grad():
            logits, value = self(torch.as_tensor(self.game.encode(state))[None])
        return torch.softmax(logits[0], -1).numpy(), float(value[0])


@app.function
def alphazero_loss(logits, values, target_pi, target_z):
    """stage 4: cross-entropy between target_pi and softmax(logits), plus MSE between values and target_z (means)."""
    policy_loss = -(target_pi * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    value_loss = ((values - target_z) ** 2).mean()
    return policy_loss + value_loss


@app.function
def self_play(game, mcts, *, temperature=1.0, rng=None):
    """stage 4: one game of mcts against itself; returns (encodings (T, ...), visit distributions (T, A), z (T,))."""
    rng = as_rng(rng)
    state = game.initial_state()
    encodings, pis, players = [], [], []
    while not game.is_terminal(state):
        pi = mcts.search(state)
        encodings.append(game.encode(state))
        pis.append(pi)
        players.append(game.current_player(state))
        if temperature == 0:
            a = int(np.argmax(pi))
        else:
            p = pi ** (1.0 / temperature)
            a = int(rng.choice(len(p), p=p / p.sum()))
        state = game.next_state(state, a)
    final = game.terminal_value(state) * game.current_player(state)  # +1 if player +1 won
    z = np.array([final * p for p in players], dtype=np.float32)
    return np.array(encodings), np.array(pis, dtype=np.float32), z


@app.cell(hide_code=True)
def _():
    checks.run(4, PolicyValueNet=PolicyValueNet, alphazero_loss=alphazero_loss, self_play=self_play,
               ConnectFour=ConnectFour, MCTS=MCTS)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_alphazero(game, n_iterations, *, games_per_iteration=10, n_simulations=50, c_puct=1.5,
                    temperature=1.0, epochs=4, batch_size=64, lr=1e-3, buffer_games=200, hidden_sizes=(128, 128),
                    seed=None):
    """AlphaZero-lite: alternate self-play with the current net and training on recent games; returns (net, history)."""
    rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        net = PolicyValueNet(game, hidden_sizes=hidden_sizes)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        games, history = [], {"loss": [], "game_lengths": []}
        for _ in range(n_iterations):
            net.eval()
            mcts = MCTS(game, n_simulations=n_simulations, c_puct=c_puct, evaluate=net.evaluate,
                        rng=int(rng.integers(2**31)))
            for _ in range(games_per_iteration):
                games.append(self_play(game, mcts, temperature=temperature, rng=int(rng.integers(2**31))))
                history["game_lengths"].append(len(games[-1][2]))
            games = games[-buffer_games:]
            X = torch.as_tensor(np.concatenate([g[0] for g in games]))
            P = torch.as_tensor(np.concatenate([g[1] for g in games]))
            Z = torch.as_tensor(np.concatenate([g[2] for g in games]))
            net.train()
            for _ in range(epochs):
                perm = torch.as_tensor(rng.permutation(len(X)))
                for i in range(0, len(X), batch_size):
                    idx = perm[i:i + batch_size]
                    logits, values = net(X[idx])
                    loss = alphazero_loss(logits, values, P[idx], Z[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            history["loss"].append(loss.item())
        net.eval()
    return net, history


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `dyna_q`, `ConnectFour` and `MCTS` are part of `mlfs.deeprl`
    (the capstone's stretch goal builds AlphaZero-lite for full-size Connect Four on them).
    open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
