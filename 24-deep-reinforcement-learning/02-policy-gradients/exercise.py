import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.distributions import Categorical

    from mlfs import checks, lesson
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # policy gradients · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `discounted_returns`: the return from every step of an episode → `mlfs.deeprl` |
    | 2 | `CategoricalPolicy`: a network that outputs a distribution over actions → `mlfs.deeprl` |
    | 3 | `reinforce_loss`: the score-function surrogate whose gradient is the policy gradient → `mlfs.deeprl` |
    | 4 | `train_reinforce`: REINFORCE with an optional learned baseline |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def discounted_returns(rewards, gamma):
    """stage 1: G_t = r_t + gamma G_{t+1} for one episode's 1-D rewards; same type (ndarray or tensor) and shape."""
    raise NotImplementedError("stage1: implement discounted_returns")


@app.cell(hide_code=True)
def _():
    checks.run(1, discounted_returns=discounted_returns)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class CategoricalPolicy(nn.Module):
    """pi(a | s): an MLP from observations to the logits of a Categorical distribution over actions."""

    def __init__(self, obs_dim, n_actions, *, hidden_sizes=(64, 64)):
        """stage 2: an MLP (tanh hidden layers) from obs_dim to n_actions logits."""
        raise NotImplementedError("stage2: implement CategoricalPolicy.__init__")

    def forward(self, obs):
        """stage 2: obs (B, obs_dim) -> torch.distributions.Categorical with batch shape (B,)."""
        raise NotImplementedError("stage2: implement CategoricalPolicy.forward")

    def act(self, obs):
        """stage 2: one observation (ndarray) -> a sampled action, a Python int; no autograd graph."""
        raise NotImplementedError("stage2: implement CategoricalPolicy.act")


@app.cell(hide_code=True)
def _():
    checks.run(2, CategoricalPolicy=CategoricalPolicy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def reinforce_loss(log_probs, returns, *, baseline=None):
    """stage 3: -mean((returns - baseline) * log_probs), with returns and baseline treated as constants."""
    raise NotImplementedError("stage3: implement reinforce_loss")


@app.cell(hide_code=True)
def _():
    checks.run(3, reinforce_loss=reinforce_loss, CategoricalPolicy=CategoricalPolicy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_reinforce(env, n_episodes, *, gamma=0.99, lr=1e-2, batch_episodes=1, hidden_sizes=(32,),
                    baseline=None, lr_value=1e-2, seed=None):
    """stage 4: REINFORCE with reward-to-go and an optional value baseline; returns (policy, history)."""
    raise NotImplementedError("stage4: implement train_reinforce")


@app.cell(hide_code=True)
def _():
    checks.run(4, train_reinforce=train_reinforce, CategoricalPolicy=CategoricalPolicy)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `discounted_returns`, `CategoricalPolicy` and
    `reinforce_loss` are part of `mlfs.deeprl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
