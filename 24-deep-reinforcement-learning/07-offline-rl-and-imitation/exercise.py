import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson
    from mlfs.deeprl import CategoricalPolicy, QNetwork, ReplayBuffer, dqn_loss  # yours, from topics 01-02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # offline RL & imitation · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `behavior_cloning`: supervised learning of an expert's actions |
    | 2 | `dagger`: imitation that asks the expert about the states the learner actually visits |
    | 3 | `cql_penalty`, `train_offline_q`: Q-learning from a fixed dataset, with and without conservatism → `mlfs.deeprl` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def behavior_cloning(policy, obs, actions, *, epochs=50, lr=1e-2, batch_size=64, seed=None):
    """stage 1: train policy (obs -> Categorical) by minibatch cross-entropy on (obs, actions); returns epoch losses."""
    raise NotImplementedError("stage1: implement behavior_cloning")


@app.function
def run_policy(env, act, n_episodes, *, seed=None):
    """stage 1: n_episodes with act(obs) -> action; returns (list of episode returns, list of visited observations)."""
    raise NotImplementedError("stage1: implement run_policy")


@app.cell(hide_code=True)
def _():
    checks.run(1, behavior_cloning=behavior_cloning, run_policy=run_policy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def dagger(env, expert, policy, *, n_iterations=5, episodes_per_iteration=5, epochs=30, lr=1e-2, batch_size=64,
           seed=None):
    """stage 2: DAgger: iteration 0 imitates expert rollouts; later ones add the learner's own states, labelled by the expert."""
    raise NotImplementedError("stage2: implement dagger")


@app.cell(hide_code=True)
def _():
    checks.run(2, dagger=dagger, behavior_cloning=behavior_cloning, run_policy=run_policy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def cql_penalty(q_values, actions, *, alpha=1.0):
    """stage 3: alpha * mean(logsumexp_a Q(s, a) - Q(s, a_data)) for q_values (B, n_actions), actions (B,)."""
    raise NotImplementedError("stage3: implement cql_penalty")


@app.function
def train_offline_q(buffer, obs_dim, n_actions, n_updates, *, gamma=0.99, lr=1e-3, batch_size=64,
                    target_update=100, cql_alpha=0.0, hidden_sizes=(64, 64), double=True, seed=None):
    """stage 3: (double) DQN on a fixed replay buffer plus cql_alpha * cql_penalty; returns (q_net, history)."""
    raise NotImplementedError("stage3: implement train_offline_q")


@app.cell(hide_code=True)
def _():
    checks.run(3, cql_penalty=cql_penalty, train_offline_q=train_offline_q)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `cql_penalty` is part of `mlfs.deeprl`. open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
