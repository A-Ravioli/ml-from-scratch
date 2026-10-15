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
    from torch.distributions import Normal

    from mlfs import checks, lesson
    from mlfs.deeprl import ReplayBuffer  # yours, from topic 01
    from mlfs.rl import Box, Env  # yours, from ch 23 · topic 02
    from mlfs.torchnn import MLP, ema_update  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # continuous control · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `Pendulum`: the classic torque-limited swing-up task as an `mlfs.rl.Env` |
    | 2 | `SquashedGaussianActor`, `TwinQ`: SAC's policy and critics → `mlfs.deeprl` |
    | 3 | `sac_critic_target`, `sac_actor_loss`, `sac_alpha_loss`: soft actor-critic's three losses |
    | 4 | `train_sac`: the full agent, with a fixed or learned temperature |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def angle_normalize(x):
    """stage 1: x wrapped into [-pi, pi)."""
    raise NotImplementedError("stage1: implement angle_normalize")


@app.class_definition
class Pendulum(Env):
    """swing a pendulum up and hold it: obs (cos th, sin th, th_dot), action a torque in [-2, 2]."""

    max_speed, max_torque, dt, g, m, l = 8.0, 2.0, 0.05, 10.0, 1.0, 1.0

    def __init__(self, *, max_steps=200):
        """stage 1: store max_steps; observation_space Box(3), action_space Box(-2, 2, (1,))."""
        raise NotImplementedError("stage1: implement Pendulum.__init__")

    def _obs(self):
        th, thdot = self.state
        return np.array([np.cos(th), np.sin(th), thdot])

    def reset(self, *, seed=None):
        """stage 1: (th, th_dot) ~ U([-pi, pi] x [-1, 1]) from self.np_random; step counter 0; returns (obs, {})."""
        raise NotImplementedError("stage1: implement Pendulum.reset")

    def step(self, action):
        """stage 1: clip the torque, pay the quadratic cost, one semi-implicit Euler step; truncate at max_steps."""
        raise NotImplementedError("stage1: implement Pendulum.step")


@app.cell(hide_code=True)
def _():
    checks.run(1, Pendulum=Pendulum, angle_normalize=angle_normalize)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class SquashedGaussianActor(nn.Module):
    """a = act_limit * tanh(u), u ~ N(mu(s), sigma(s)^2): a Gaussian squashed into the action box."""

    def __init__(self, obs_dim, act_dim, *, hidden_sizes=(256, 256), act_limit=1.0):
        """stage 2: an MLP from obs_dim to 2 * act_dim outputs (mean and log-std); store act_limit."""
        raise NotImplementedError("stage2: implement SquashedGaussianActor.__init__")

    def forward(self, obs, deterministic=False):
        """stage 2: (action (B, act_dim) in (-act_limit, act_limit), log_prob (B,) with the tanh correction)."""
        raise NotImplementedError("stage2: implement SquashedGaussianActor.forward")


@app.class_definition
class TwinQ(nn.Module):
    """two independent Q-networks Q1(s, a), Q2(s, a) on the concatenated observation and action."""

    def __init__(self, obs_dim, act_dim, *, hidden_sizes=(256, 256)):
        """stage 2: two separate MLPs from obs_dim + act_dim to 1."""
        raise NotImplementedError("stage2: implement TwinQ.__init__")

    def forward(self, obs, act):
        """stage 2: (q1 (B,), q2 (B,))."""
        raise NotImplementedError("stage2: implement TwinQ.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, SquashedGaussianActor=SquashedGaussianActor, TwinQ=TwinQ)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def sac_critic_target(rewards, terminated, next_q1, next_q2, next_log_prob, *, gamma=0.99, alpha=0.2):
    """stage 3: y = r + gamma (1 - terminated) (min(Q1', Q2') - alpha log pi(a'|s')), without gradients."""
    raise NotImplementedError("stage3: implement sac_critic_target")


@app.function
def sac_actor_loss(q1, q2, log_prob, alpha):
    """stage 3: mean(alpha log pi(a|s) - min(Q1, Q2)(s, a)) for a ~ pi (reparameterized)."""
    raise NotImplementedError("stage3: implement sac_actor_loss")


@app.function
def sac_alpha_loss(log_alpha, log_prob, target_entropy):
    """stage 3: -mean(exp(log_alpha) * (log_prob + target_entropy)), with log_prob a constant."""
    raise NotImplementedError("stage3: implement sac_alpha_loss")


@app.cell(hide_code=True)
def _():
    checks.run(3, sac_critic_target=sac_critic_target, sac_actor_loss=sac_actor_loss, sac_alpha_loss=sac_alpha_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_sac(env, n_steps, *, gamma=0.99, tau=0.005, lr=3e-4, batch_size=256, buffer_size=100_000,
              learning_starts=1000, alpha=0.2, target_entropy=None, hidden_sizes=(256, 256), seed=None):
    """stage 4: SAC for n_steps env steps (alpha a float or 'auto'); returns (actor, history)."""
    raise NotImplementedError("stage4: implement train_sac")


@app.cell(hide_code=True)
def _():
    checks.run(4, train_sac=train_sac, Pendulum=Pendulum)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `SquashedGaussianActor` and `TwinQ` are part of
    `mlfs.deeprl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
