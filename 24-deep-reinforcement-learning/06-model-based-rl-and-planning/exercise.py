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
    # model-based RL & planning · build

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
    raise NotImplementedError("stage1: implement dyna_q")


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
        raise NotImplementedError("stage2: implement ConnectFour.__init__")

    def initial_state(self):
        """stage 2: the empty board with player +1 to move."""
        raise NotImplementedError("stage2: implement ConnectFour.initial_state")

    def current_player(self, state):
        """stage 2: +1 or -1, the player to move."""
        raise NotImplementedError("stage2: implement ConnectFour.current_player")

    def legal_actions(self, state):
        """stage 2: the columns that aren't full (empty list at a terminal state), ascending."""
        raise NotImplementedError("stage2: implement ConnectFour.legal_actions")

    def next_state(self, state, action):
        """stage 2: drop the mover's stone into column `action`; returns a NEW state."""
        raise NotImplementedError("stage2: implement ConnectFour.next_state")

    def is_terminal(self, state):
        """stage 2: someone has four in a row, or the board is full."""
        raise NotImplementedError("stage2: implement ConnectFour.is_terminal")

    def terminal_value(self, state):
        """stage 2: from the view of the player to move at the terminal state: -1 (they lost) or 0 (draw)."""
        raise NotImplementedError("stage2: implement ConnectFour.terminal_value")

    def encode(self, state):
        """stage 2: float32 planes (2, rows, cols): the mover's stones, then the opponent's."""
        raise NotImplementedError("stage2: implement ConnectFour.encode")


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
        raise NotImplementedError("stage3: implement MCTS.__init__")

    def search(self, state):
        """stage 3: run n_simulations simulations from state; returns visit counts / total, shape (n_actions,)."""
        raise NotImplementedError("stage3: implement MCTS.search")


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
        raise NotImplementedError("stage4: implement PolicyValueNet.__init__")

    def forward(self, x):
        """stage 4: x (B, *encoding shape) float -> (logits (B, n_actions), value (B,))."""
        raise NotImplementedError("stage4: implement PolicyValueNet.forward")

    def evaluate(self, state):
        """stage 4: MCTS's evaluate hook: (softmax prior over actions as an ndarray, value float), no gradients."""
        raise NotImplementedError("stage4: implement PolicyValueNet.evaluate")


@app.function
def alphazero_loss(logits, values, target_pi, target_z):
    """stage 4: cross-entropy between target_pi and softmax(logits), plus MSE between values and target_z (means)."""
    raise NotImplementedError("stage4: implement alphazero_loss")


@app.function
def self_play(game, mcts, *, temperature=1.0, rng=None):
    """stage 4: one game of mcts against itself; returns (encodings (T, ...), visit distributions (T, A), z (T,))."""
    raise NotImplementedError("stage4: implement self_play")


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
