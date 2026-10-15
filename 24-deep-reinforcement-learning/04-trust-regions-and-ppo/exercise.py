import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Categorical

    from mlfs import checks, lesson
    from mlfs.deeprl import ActorCritic, RolloutCollector, compute_gae  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # trust regions & PPO · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `ppo_clip_loss`: the clipped surrogate objective, with its diagnostics → `mlfs.deeprl` |
    | 2 | `explained_variance`: how much of the returns' variance the critic explains → `mlfs.deeprl` |
    | 3 | `conjugate_gradient`, `fisher_vector_product`, `natural_gradient_step`: TRPO's trust-region step |
    | 4 | `ppo_update`: several epochs of minibatch updates on one batch |
    | 5 | `train_ppo`: the full PPO agent |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def ppo_clip_loss(log_probs, old_log_probs, advantages, *, clip_eps=0.2):
    """stage 1: -mean(min(r A, clip(r, 1 - eps, 1 + eps) A)) with r = exp(log_probs - old); (loss, info)."""
    raise NotImplementedError("stage1: implement ppo_clip_loss")


@app.cell(hide_code=True)
def _():
    checks.run(1, ppo_clip_loss=ppo_clip_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def explained_variance(values, returns):
    """stage 2: 1 - Var(returns - values) / Var(returns), a Python float (nan if Var(returns) = 0)."""
    raise NotImplementedError("stage2: implement explained_variance")


@app.cell(hide_code=True)
def _():
    checks.run(2, explained_variance=explained_variance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def conjugate_gradient(Avp, b, *, n_iters=10, tol=1e-10):
    """stage 3: approximately solve A x = b for symmetric positive-definite A, given only v -> A v; returns x."""
    raise NotImplementedError("stage3: implement conjugate_gradient")


@app.function
def fisher_vector_product(dist_fn, params, v, *, damping=0.0):
    """stage 3: (F + damping I) v, F the Hessian of mean KL(pi_old || pi_theta) at theta = theta_old; v flat."""
    raise NotImplementedError("stage3: implement fisher_vector_product")


@app.function
def natural_gradient_step(g, fvp, max_kl, *, cg_iters=10):
    """stage 3: the step along F^-1 g whose quadratic KL estimate 0.5 s^T F s equals max_kl."""
    raise NotImplementedError("stage3: implement natural_gradient_step")


@app.cell(hide_code=True)
def _():
    checks.run(3, conjugate_gradient=conjugate_gradient, fisher_vector_product=fisher_vector_product,
               natural_gradient_step=natural_gradient_step)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ppo_update(model, optimizer, batch, *, n_epochs=10, minibatch_size=64, clip_eps=0.2, vf_coef=0.5,
               ent_coef=0.0, max_grad_norm=0.5, norm_adv=True, target_kl=None, seed=None):
    """stage 4: n_epochs passes of shuffled minibatch steps on the PPO loss; returns a dict of diagnostics."""
    raise NotImplementedError("stage4: implement ppo_update")


@app.cell(hide_code=True)
def _():
    checks.run(4, ppo_update=ppo_update)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_ppo(make_env, total_steps, *, n_envs=8, n_steps=128, gamma=0.99, lam=0.95, lr=3e-4, anneal_lr=True,
              n_epochs=10, minibatch_size=256, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
              norm_adv=True, target_kl=None, hidden_sizes=(64, 64), continuous=False, seed=None):
    """stage 5: PPO with GAE on n_envs copies of make_env(); returns (model, history)."""
    raise NotImplementedError("stage5: implement train_ppo")


@app.cell(hide_code=True)
def _():
    checks.run(5, train_ppo=train_ppo)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `ppo_clip_loss` and `explained_variance` are part of
    `mlfs.deeprl` (chapter 27's RLHF uses your `ppo_clip_loss` and `compute_gae`). open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
