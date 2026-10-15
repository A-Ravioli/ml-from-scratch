import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.rl import Box, Discrete, Env, epsilon_greedy  # yours, from ch 23 · topics 02 and 04
    from mlfs.torchnn import MLP, ema_update  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # deep Q-networks · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. this chapter is spec-driven: the lesson's **spec** boxes are the
    contract (names, arguments, conventions), and the internal structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `CartPole`: the classic cart–pole balancing task as an `mlfs.rl.Env` |
    | 2 | `ReplayBuffer`: a preallocated ring buffer of transitions → `mlfs.deeprl` |
    | 3 | `QNetwork`, `dqn_loss`: the network (plain and dueling) and the (double) DQN loss → `mlfs.deeprl` |
    | 4 | `train_dqn`: the full agent: ε-schedule, replay, target network |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class CartPole(Env):
    """Barto, Sutton & Anderson's cart-pole: obs (x, x_dot, theta, theta_dot); actions 0 push left, 1 push right."""

    gravity, masscart, masspole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x_threshold, theta_threshold = 2.4, 12 * 2 * np.pi / 360

    def __init__(self, *, max_steps=500):
        """stage 1: store max_steps; observation_space Box(4), action_space Discrete(2)."""
        raise NotImplementedError("stage1: implement CartPole.__init__")

    def reset(self, *, seed=None):
        """stage 1: state ~ U(-0.05, 0.05)^4 from self.np_random, step counter 0; returns (obs copy, {})."""
        raise NotImplementedError("stage1: implement CartPole.reset")

    def step(self, action):
        """stage 1: one Euler step of the frictionless dynamics; reward 1.0; terminate past 2.4 m or 12 degrees."""
        raise NotImplementedError("stage1: implement CartPole.step")


@app.cell(hide_code=True)
def _():
    checks.run(1, CartPole=CartPole)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ReplayBuffer:
    """a fixed-size ring buffer of transitions (s, a, r, s', terminated), stored in preallocated NumPy arrays."""

    def __init__(self, capacity, obs_shape, *, action_shape=(), action_dtype="int64", seed=None):
        """stage 2: preallocate the arrays (float32 except actions); position and size 0; a Generator from seed."""
        raise NotImplementedError("stage2: implement ReplayBuffer.__init__")

    def add(self, obs, action, reward, next_obs, terminated):
        """stage 2: write the transition at the current position, overwriting the oldest once full."""
        raise NotImplementedError("stage2: implement ReplayBuffer.add")

    def sample(self, batch_size):
        """stage 2: batch_size transitions drawn uniformly (with replacement); dict of torch tensors."""
        raise NotImplementedError("stage2: implement ReplayBuffer.sample")

    def __len__(self):
        """stage 2: the number of transitions stored (at most capacity)."""
        raise NotImplementedError("stage2: implement ReplayBuffer.__len__")


@app.cell(hide_code=True)
def _():
    checks.run(2, ReplayBuffer=ReplayBuffer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class QNetwork(nn.Module):
    """obs (B, obs_dim) -> Q-values (B, n_actions); dueling = separate value and advantage streams."""

    def __init__(self, obs_dim, n_actions, *, hidden_sizes=(128, 128), dueling=False):
        """stage 3: an MLP; if dueling, a shared trunk with a value head (1) and an advantage head (n_actions)."""
        raise NotImplementedError("stage3: implement QNetwork.__init__")

    def forward(self, obs):
        """stage 3: Q (B, n_actions); dueling: V + A - mean_a A."""
        raise NotImplementedError("stage3: implement QNetwork.forward")


@app.function
def dqn_loss(q_net, target_net, batch, *, gamma=0.99, double=False, huber=True):
    """stage 3: TD loss between Q(s, a) and r + gamma (1 - terminated) Q_target(s', a*), no gradient into the target."""
    raise NotImplementedError("stage3: implement dqn_loss")


@app.cell(hide_code=True)
def _():
    checks.run(3, QNetwork=QNetwork, dqn_loss=dqn_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def linear_schedule(start, end, duration, t):
    """stage 4: start + (end - start) * min(t / duration, 1): a linear ramp held at `end` after `duration` steps."""
    raise NotImplementedError("stage4: implement linear_schedule")


@app.function
def train_dqn(env, n_steps, *, gamma=0.99, lr=1e-3, buffer_size=50_000, batch_size=64, learning_starts=1000,
              train_freq=1, target_update=500, tau=None, eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000,
              hidden_sizes=(64, 64), double=False, dueling=False, huber=True, seed=None):
    """stage 4: DQN for n_steps env steps; returns (q_net, history dict with 'episode_returns' and 'episode_ends')."""
    raise NotImplementedError("stage4: implement train_dqn")


@app.cell(hide_code=True)
def _():
    checks.run(4, train_dqn=train_dqn, linear_schedule=linear_schedule, CartPole=CartPole)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `ReplayBuffer`, `QNetwork` and `dqn_loss` are part of
    `mlfs.deeprl` (try `from mlfs.deeprl import dqn_loss`), and `CartPole` is the
    environment the next three topics learn on. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
