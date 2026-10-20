import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Categorical, Independent, Normal

    from mlfs import checks, lesson
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # actor-critic · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `compute_gae`: generalized advantage estimation over a rollout → `mlfs.deeprl` |
    | 2 | `ActorCritic`: a policy and a value function in one module, discrete or continuous → `mlfs.deeprl` |
    | 3 | `RolloutCollector`: synchronous rollouts from N environments, with time limits handled correctly |
    | 4 | `a2c_loss`, `train_a2c`: advantage actor-critic |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def compute_gae(rewards, values, terminated, last_value, *, gamma=0.99, lam=0.95):
    """stage 1: GAE(gamma, lambda) advantages and returns = advantages + values; (T,) or (T, N); no autograd graph."""
    raise NotImplementedError("stage1: implement compute_gae")


@app.cell(hide_code=True)
def _():
    checks.run(1, compute_gae=compute_gae)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ActorCritic(nn.Module):
    """a policy network (actor) and a value network (critic); discrete (Categorical) or continuous (diagonal Normal)."""

    def __init__(self, obs_dim, act_dim, *, hidden_sizes=(64, 64), continuous=False):
        """stage 2: separate tanh MLPs for the actor (act_dim outputs) and the critic (1 output); log_std = 0 if continuous."""
        raise NotImplementedError("stage2: implement ActorCritic.__init__")

    def forward(self, obs):
        """stage 2: obs (B, obs_dim) -> (distribution with log_prob/entropy of shape (B,), value (B,))."""
        raise NotImplementedError("stage2: implement ActorCritic.forward")

    def act(self, obs):
        """stage 2: one ndarray observation -> (action, log_prob float, value float), sampled without gradients."""
        raise NotImplementedError("stage2: implement ActorCritic.act")


@app.cell(hide_code=True)
def _():
    checks.run(2, ActorCritic=ActorCritic)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class RolloutCollector:
    """steps N environments in lockstep with a model; auto-resets them and bootstraps time limits into the rewards."""

    def __init__(self, envs, model, *, gamma=0.99, seed=None):
        """stage 3: store envs, model, gamma; reset env i with the i-th of N seeds drawn from default_rng(seed)."""
        raise NotImplementedError("stage3: implement RolloutCollector.__init__")

    def collect(self, n_steps):
        """stage 3: n_steps steps of every env; returns a dict of tensors with a leading (T, N) shape, plus last_value (N,)."""
        raise NotImplementedError("stage3: implement RolloutCollector.collect")


@app.cell(hide_code=True)
def _():
    checks.run(3, RolloutCollector=RolloutCollector, compute_gae=compute_gae)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def a2c_loss(dist, actions, advantages, returns, values, *, vf_coef=0.5, ent_coef=0.01):
    """stage 4: policy-gradient loss + vf_coef * value MSE - ent_coef * entropy; returns (loss, info dict of floats)."""
    raise NotImplementedError("stage4: implement a2c_loss")


@app.function
def train_a2c(make_env, total_steps, *, n_envs=8, n_steps=16, gamma=0.99, lam=0.95, lr=7e-4, vf_coef=0.5,
              ent_coef=0.01, max_grad_norm=0.5, hidden_sizes=(64, 64), continuous=False, seed=None):
    """stage 4: A2C with GAE on n_envs copies of make_env(); returns (model, history)."""
    raise NotImplementedError("stage4: implement train_a2c")


@app.cell(hide_code=True)
def _():
    checks.run(4, a2c_loss=a2c_loss, train_a2c=train_a2c)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `compute_gae` and `ActorCritic` are part of `mlfs.deeprl`
    (the next topic's PPO is built from them and your `RolloutCollector`). open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
