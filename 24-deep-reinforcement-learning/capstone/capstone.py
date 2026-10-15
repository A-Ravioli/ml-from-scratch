import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import checks, datasets
    from mlfs.deeprl import CartPole, train_a2c, train_ppo  # yours, from this chapter
    from mlfs.rl import Box, Discrete, Env  # yours, from ch 23
    from mlfs.stats import bootstrap_ci, permutation_test  # yours, from ch 05

    torch.set_num_threads(1)  # small networks: one thread is fastest

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    SEEDS = 2 if SMALL else 10
    CARTPOLE_STEPS = 8_000 if SMALL else 100_000
    ACROBOT_STEPS = 8_000 if SMALL else 150_000
    EVAL_EPISODES = 3 if SMALL else 20


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 24 · capstone: PPO, reproducibly

    read [`brief.md`](brief.md) first. you'll build a harder control task (the acrobot), the
    evaluation and statistics a deep RL result needs (greedy evaluation with error bars,
    learning curves with bootstrap bands over seeds, a two-sample comparison), and a
    multi-seed experiment runner; then you use them to show that your PPO solves CartPole and
    the acrobot, compare it with A2C, and test one implementation detail of your choice.

    | stage | you build |
    |---|---|
    | 1 | `Acrobot`: the two-link swing-up task, with the published dynamics and RK4 integration |
    | 2 | `greedy`, `evaluate_policy`: deterministic evaluation with a standard error |
    | 3 | `learning_curves`, `mean_ci`: align runs on a step grid; bootstrap bands over seeds |
    | 4 | `compare_seeds`: a difference of means with a bootstrap interval and a permutation p-value |
    | 5 | `run_seeds`: train and evaluate one configuration over many seeds |

    when they're green, scroll down: the experiments run on your functions, and the write-up
    prompts are at the end.
    """)
    return


@app.class_definition
class Acrobot(Env):
    """Sutton's acrobot (Gymnasium Acrobot-v1): two links, torque on the elbow; swing the tip above the bar."""

    m1 = m2 = 1.0
    l1 = 1.0
    lc1 = lc2 = 0.5
    I1 = I2 = 1.0
    g = 9.8
    dt = 0.2
    max_vel_1, max_vel_2 = 4 * np.pi, 9 * np.pi

    def __init__(self, *, max_steps=500):
        """stage 1: store max_steps; observation_space Box(6), action_space Discrete(3) (torques -1, 0, +1)."""
        raise NotImplementedError("stage1: implement Acrobot.__init__")

    def _obs(self):
        t1, t2, d1, d2 = self.state
        return np.array([np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2), d1, d2])

    def reset(self, *, seed=None):
        """stage 1: state (theta1, theta2, dtheta1, dtheta2) ~ U(-0.1, 0.1)^4 from self.np_random; returns (obs, {})."""
        raise NotImplementedError("stage1: implement Acrobot.reset")

    def dsdt(self, s, torque):
        """stage 1: time derivative (dtheta1, dtheta2, ddtheta1, ddtheta2) of the state s for a torque (the book dynamics)."""
        raise NotImplementedError("stage1: implement Acrobot.dsdt")

    def step(self, action):
        """stage 1: one RK4 step of dt = 0.2; wrap angles, clip velocities; reward -1, 0 on the step that terminates."""
        raise NotImplementedError("stage1: implement Acrobot.step")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", Acrobot=Acrobot)
    return


@app.function
def greedy(model):
    """stage 2: a callable obs -> action for an ActorCritic: the most likely action (or the Gaussian mean)."""
    raise NotImplementedError("stage2: implement greedy")


@app.function
def evaluate_policy(make_env, act, n_episodes, *, seed=0):
    """stage 2: (mean, standard error) of the undiscounted returns of act over n_episodes of a fresh make_env()."""
    raise NotImplementedError("stage2: implement evaluate_policy")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", greedy=greedy, evaluate_policy=evaluate_policy)
    return


@app.function
def learning_curves(histories, grid, *, window=10):
    """stage 3: (n_runs, len(grid)): for each run, the mean return of the last `window` episodes finished by each grid step."""
    raise NotImplementedError("stage3: implement learning_curves")


@app.function
def mean_ci(curves, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 3: per column, (mean over runs, percentile-bootstrap lo, hi) resampling runs; nan entries are ignored."""
    raise NotImplementedError("stage3: implement mean_ci")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", learning_curves=learning_curves, mean_ci=mean_ci)
    return


@app.function
def compare_seeds(a, b, *, n_boot=10000, rng=None):
    """stage 4: dict(diff = mean(a) - mean(b), ci = bootstrap (lo, hi) resampling a and b separately, p = permutation p)."""
    raise NotImplementedError("stage4: implement compare_seeds")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", compare_seeds=compare_seeds)
    return


@app.function
def run_seeds(make_env, train_fn, total_steps, seeds, *, eval_episodes=20, **kwargs):
    """stage 5: train_fn(make_env, total_steps, seed=s, **kwargs) per seed; dict(histories, final) with greedy evaluations."""
    raise NotImplementedError("stage5: implement run_seeds")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", run_seeds=run_seeds, greedy=greedy, evaluate_policy=evaluate_policy)
    return


@app.function
def plot_runs(ax, results, total_steps, label, color=None):
    """learning curve (mean and 95% bootstrap band over seeds) on a 50-point step grid."""
    grid = np.linspace(total_steps / 50, total_steps, 50)
    m, lo, hi = mean_ci(learning_curves(results["histories"], grid), rng=0)
    ax.plot(grid, m, label=label, color=color)
    ax.fill_between(grid, lo, hi, alpha=0.25, color=color)


@app.function
def summary_row(name, final):
    lo, hi = bootstrap_ci(final, np.mean, n_boot=5000, rng=0) if len(final) > 1 else (np.nan, np.nan)
    return f"| {name} | {np.mean(final):.1f} [{lo:.1f}, {hi:.1f}] | {np.min(final):.0f} | {np.max(final):.0f} |"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A: PPO solves CartPole

    your `train_ppo` (8 environments × 128 steps, 10 epochs, minibatches of 256, Adam lr
    $10^{-3}$) on `CartPole`, over many seeds. the band is a 95% bootstrap interval over
    seeds for the mean training return; the table gives the greedy evaluation after
    training (mean over seeds, with a 95% bootstrap interval). CartPole-v1 counts as solved
    at an average of 475.
    """)
    return


@app.cell
def _():
    ppo_cartpole = run_seeds(CartPole, train_ppo, CARTPOLE_STEPS, range(SEEDS), eval_episodes=EVAL_EPISODES,
                             lr=1e-3)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    plot_runs(_ax, ppo_cartpole, CARTPOLE_STEPS, "PPO")
    _ax.axhline(475, color="k", ls=":", lw=1)
    _ax.set(xlabel="environment steps", ylabel="training return (10-episode mean)", title=f"CartPole, {SEEDS} seeds")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| agent | greedy return [95% CI over seeds] | worst seed | best seed |\n|---|---|---|---|\n"
                           + summary_row("PPO", ppo_cartpole["final"]))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B: the acrobot, PPO against A2C

    the same PPO and your `train_a2c` (8 environments × 16 steps, Adam lr $7\cdot10^{-4}$)
    on `Acrobot`, with the same number of environment steps and seeds. a random policy
    scores $-500$ (it never swings up within the time limit); a good one about $-80$.
    """)
    return


@app.cell
def _():
    ppo_acro = run_seeds(Acrobot, train_ppo, ACROBOT_STEPS, range(SEEDS), eval_episodes=EVAL_EPISODES, lr=1e-3)
    a2c_acro = run_seeds(Acrobot, train_a2c, ACROBOT_STEPS, range(SEEDS), eval_episodes=EVAL_EPISODES)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    plot_runs(_ax, ppo_acro, ACROBOT_STEPS, "PPO", "C0")
    plot_runs(_ax, a2c_acro, ACROBOT_STEPS, "A2C", "C1")
    _ax.set(xlabel="environment steps", ylabel="training return (10-episode mean)", title=f"Acrobot, {SEEDS} seeds")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _c = compare_seeds(ppo_acro["final"], a2c_acro["final"], rng=0)
    mo.vstack([_fig, mo.md(
        "| agent | greedy return [95% CI over seeds] | worst seed | best seed |\n|---|---|---|---|\n"
        + summary_row("PPO", ppo_acro["final"]) + "\n" + summary_row("A2C", a2c_acro["final"])
        + f"\n\nPPO − A2C: **{_c['diff']:.1f}**, 95% CI [{_c['ci'][0]:.1f}, {_c['ci'][1]:.1f}], "
        f"permutation p = {_c['p']:.4f}")])
    return (ppo_acro,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C: one implementation detail, tested properly

    pick one detail of PPO and test it with the same seeds. the example removes advantage
    normalization. replace `ABLATION` with your own choice (the clip, the number of epochs,
    the entropy bonus, λ, the learning-rate schedule, the network size, ...), and state your
    hypothesis in the write-up *before* you run it.
    """)
    return


@app.cell
def _(ppo_acro):
    ABLATION = dict(lr=1e-3, norm_adv=False)
    ablated = run_seeds(Acrobot, train_ppo, ACROBOT_STEPS, range(SEEDS), eval_episodes=EVAL_EPISODES, **ABLATION)
    _c = compare_seeds(ppo_acro["final"], ablated["final"], rng=1)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    plot_runs(_ax, ppo_acro, ACROBOT_STEPS, "PPO", "C0")
    plot_runs(_ax, ablated, ACROBOT_STEPS, f"ablated: {ABLATION}", "C3")
    _ax.set(xlabel="environment steps", ylabel="training return (10-episode mean)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| agent | greedy return [95% CI over seeds] | worst seed | best seed |\n|---|---|---|---|\n"
        + summary_row("PPO", ppo_acro["final"]) + "\n" + summary_row("ablated", ablated["final"])
        + f"\n\nPPO − ablated: **{_c['diff']:.1f}**, 95% CI [{_c['ci'][0]:.1f}, {_c['ci'][1]:.1f}], "
        f"permutation p = {_c['p']:.4f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer each prompt with numbers from the experiments above (full config: run with
    `MLFS_OFFLINE` unset). see [`rubric.md`](rubric.md).

    1. **the environment.** how did you check that your `Acrobot` implements the published
       dynamics? what do a random policy and your best policy score, and what does a good
       policy actually do (watch a few episodes)?
    2. **CartPole.** did PPO solve it on every seed? report the greedy return with its
       interval, and say how much the training curve (with exploration) understates it.
    3. **PPO versus A2C.** which is better on the acrobot at this budget, by how much, and
       how sure are you (interval and p-value)? explain the difference with the chapter's
       ideas (sample reuse, the clip, variance).
    4. **your ablation.** the hypothesis you wrote down before running it, the result with
       uncertainty, and whether the evidence supports it. how many seeds would you need to
       detect a difference half as large?
    5. **reproducibility.** list everything someone would need to reproduce your numbers
       exactly, and one thing that would still differ on their machine.
    """)
    return


if __name__ == "__main__":
    app.run()
