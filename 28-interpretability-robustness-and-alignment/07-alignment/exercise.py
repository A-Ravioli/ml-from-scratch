import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.llm import accuracy_ci, paired_accuracy_diff  # yours, from ch 27 · 06
    from mlfs.rl import policy_evaluation, value_iteration  # yours, from ch 23 · 02-03
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # alignment · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. nothing in this topic becomes part of `mlfs`: it's a
    workbench for studying what goes wrong when you optimise a proxy.

    | stage | you build |
    |---|---|
    | 1 | `kl_regularized_policy`, `policy_stats`: optimising a proxy reward with a KL leash, in closed form |
    | 2 | `best_of_n_distribution`: exactly what best-of-n sampling does to a distribution |
    | 3 | `corridor_mdp`, `gaming_gap`, `rollout_states`: specification gaming in a tiny MDP |
    | 4 | `eval_suite`, `compare_agents`: a small behavioural eval with intervals |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def kl_regularized_policy(ref_probs, rewards, beta):
    """stage 1: pi(i) proportional to ref_probs(i) exp(rewards(i) / beta), the maximiser of E[r] - beta KL(pi || ref)."""
    raise NotImplementedError("stage1: implement kl_regularized_policy")


@app.function
def policy_stats(policy, ref_probs, proxy, true):
    """stage 1: {'kl': KL(policy || ref) in nats, 'proxy': E_policy[proxy], 'true': E_policy[true]}."""
    raise NotImplementedError("stage1: implement policy_stats")


@app.cell(hide_code=True)
def _():
    checks.run(1, kl_regularized_policy=kl_regularized_policy, policy_stats=policy_stats)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def best_of_n_distribution(ref_probs, scores, n):
    """stage 2: P(best-of-n picks item i) for n samples from ref_probs, ties split in proportion to ref_probs."""
    raise NotImplementedError("stage2: implement best_of_n_distribution")


@app.cell(hide_code=True)
def _():
    checks.run(2, best_of_n_distribution=best_of_n_distribution)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def corridor_mdp(length, bonus_pos, bonus, *, goal_reward=10.0):
    """stage 3: (P (S, 2, S), R_proxy (S, 2), R_true (S, 2)) for a corridor 0..length-1, actions left/right, goal at the end."""
    raise NotImplementedError("stage3: implement corridor_mdp")


@app.function
def gaming_gap(P, R_proxy, R_true, gamma=0.95, start=0):
    """stage 3: the proxy-optimal policy, its TRUE value at start, the true-optimal value at start, and the regret."""
    raise NotImplementedError("stage3: implement gaming_gap")


@app.function
def rollout_states(P, policy, start, horizon):
    """stage 3: the states visited by a deterministic policy in a deterministic MDP: [start, s1, ..., s_horizon]."""
    raise NotImplementedError("stage3: implement rollout_states")


@app.cell(hide_code=True)
def _():
    checks.run(3, corridor_mdp=corridor_mdp, gaming_gap=gaming_gap, rollout_states=rollout_states)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def eval_suite(agent, cases, *, n_boot=2000, rng=None):
    """stage 4: {category: (pass rate, lo, hi)} plus 'overall', over cases (category, check); check(agent) -> bool;
    one shared generator, used for the categories in sorted order, then 'overall'."""
    raise NotImplementedError("stage4: implement eval_suite")


@app.function
def compare_agents(agent_a, agent_b, cases, *, n_boot=2000, rng=None):
    """stage 4: paired pass-rate difference a - b over the same cases, with a bootstrap interval: (diff, lo, hi)."""
    raise NotImplementedError("stage4: implement compare_agents")


@app.cell(hide_code=True)
def _():
    checks.run(4, eval_suite=eval_suite, compare_agents=compare_agents)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. then the chapter's
    capstone: train a small transformer on modular addition, watch it grok, and use this
    chapter's tools to read the algorithm it found.
    """)
    return


if __name__ == "__main__":
    app.run()
