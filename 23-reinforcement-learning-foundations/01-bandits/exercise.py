import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bandits · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    classes to write, and checks them live as you edit. by the end you have a bandit testbed
    and the three classic ways to trade exploration against exploitation, all in `mlfs.rl`.

    | stage | you build |
    |---|---|
    | 1 | `BernoulliBandit`, `GaussianBandit`, `run_bandit`: the testbed and the loop that measures regret → `mlfs.rl` |
    | 2 | `EpsilonGreedy`: sample averages, explore with probability ε → `mlfs.rl` |
    | 3 | `UCB`: optimism in the face of uncertainty → `mlfs.rl` |
    | 4 | `ThompsonSampling`, `lai_robbins_constant`: sample from the posterior, and the best any algorithm can do → `mlfs.rl` |

    the lesson's spec boxes give the exact conventions (tie-breaking, the UCB clock, the
    order of random draws). you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class BernoulliBandit:
    """a k-armed bandit whose arm a pays 1 with probability probs[a], else 0."""

    def __init__(self, probs, *, rng=None):
        """stage 1: store means (float array (k,)), n_arms, optimal_arm and a Generator self.rng."""
        raise NotImplementedError("stage1: implement BernoulliBandit.__init__")

    def pull(self, arm):
        """stage 1: one reward from `arm`, a float 0.0 or 1.0."""
        raise NotImplementedError("stage1: implement BernoulliBandit.pull")


@app.class_definition
class GaussianBandit:
    """a k-armed bandit whose arm a pays a N(means[a], std^2) reward."""

    def __init__(self, means, *, std=1.0, rng=None):
        """stage 1: store means, std, n_arms, optimal_arm and a Generator self.rng."""
        raise NotImplementedError("stage1: implement GaussianBandit.__init__")

    def pull(self, arm):
        """stage 1: one Gaussian reward from `arm`, a float."""
        raise NotImplementedError("stage1: implement GaussianBandit.pull")


@app.function
def run_bandit(bandit, agent, n_steps):
    """stage 1: play n_steps rounds; returns (rewards (n_steps,), cumulative pseudo-regret (n_steps,))."""
    raise NotImplementedError("stage1: implement run_bandit")


@app.cell(hide_code=True)
def _():
    checks.run(1, BernoulliBandit=BernoulliBandit, GaussianBandit=GaussianBandit, run_bandit=run_bandit)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class EpsilonGreedy:
    """ε-greedy with sample-average estimates Q (n_arms,) and pull counts N (n_arms,)."""

    def __init__(self, n_arms, epsilon=0.1, *, rng=None):
        """stage 2: store n_arms, epsilon, self.rng; Q and N start at zeros."""
        raise NotImplementedError("stage2: implement EpsilonGreedy.__init__")

    def select_arm(self):
        """stage 2: a uniformly random arm with probability epsilon, else a greedy arm (random ties)."""
        raise NotImplementedError("stage2: implement EpsilonGreedy.select_arm")

    def update(self, arm, reward):
        """stage 2: incremental sample average: N[arm] += 1, Q[arm] += (reward - Q[arm]) / N[arm]."""
        raise NotImplementedError("stage2: implement EpsilonGreedy.update")


@app.cell(hide_code=True)
def _():
    checks.run(2, EpsilonGreedy=EpsilonGreedy, BernoulliBandit=BernoulliBandit, run_bandit=run_bandit)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class UCB:
    """UCB1: pull every arm once, then argmax of Q + c sqrt(ln t / N), t = pulls so far."""

    def __init__(self, n_arms, c=2.0):
        """stage 3: store n_arms and c; Q and N start at zeros."""
        raise NotImplementedError("stage3: implement UCB.__init__")

    def select_arm(self):
        """stage 3: the lowest-index untried arm if any, else the arm with the largest upper bound (ties to the lowest)."""
        raise NotImplementedError("stage3: implement UCB.select_arm")

    def update(self, arm, reward):
        """stage 3: the same sample-average update as EpsilonGreedy."""
        raise NotImplementedError("stage3: implement UCB.update")


@app.cell(hide_code=True)
def _():
    checks.run(3, UCB=UCB, BernoulliBandit=BernoulliBandit, EpsilonGreedy=EpsilonGreedy, run_bandit=run_bandit)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class ThompsonSampling:
    """Beta-Bernoulli Thompson sampling: posterior Beta(alpha[a], beta[a]) per arm."""

    def __init__(self, n_arms, *, prior=(1.0, 1.0), rng=None):
        """stage 4: store n_arms, self.rng, and alpha, beta float arrays (n_arms,) filled from prior."""
        raise NotImplementedError("stage4: implement ThompsonSampling.__init__")

    def select_arm(self):
        """stage 4: draw theta ~ Beta(alpha, beta) for all arms at once (one rng.beta call); play its argmax."""
        raise NotImplementedError("stage4: implement ThompsonSampling.select_arm")

    def update(self, arm, reward):
        """stage 4: alpha[arm] += reward, beta[arm] += 1 - reward (reward in [0, 1])."""
        raise NotImplementedError("stage4: implement ThompsonSampling.update")


@app.function
def lai_robbins_constant(means):
    """stage 4: sum over suboptimal Bernoulli arms of gap / KL(mu_a, mu*): regret >= this * ln T."""
    raise NotImplementedError("stage4: implement lai_robbins_constant")


@app.cell(hide_code=True)
def _():
    checks.run(4, ThompsonSampling=ThompsonSampling, lai_robbins_constant=lai_robbins_constant,
               BernoulliBandit=BernoulliBandit, UCB=UCB, run_bandit=run_bandit)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your testbed and the three agents are part of `mlfs.rl`
    (try `from mlfs.rl import UCB`). open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
