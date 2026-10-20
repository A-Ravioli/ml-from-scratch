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
    # bandits · build (reference solution)

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
        self.means = np.asarray(probs, dtype=float)
        self.n_arms = len(self.means)
        self.optimal_arm = int(np.argmax(self.means))
        self.rng = as_rng(rng)

    def pull(self, arm):
        """stage 1: one reward from `arm`, a float 0.0 or 1.0."""
        return float(self.rng.random() < self.means[arm])


@app.class_definition
class GaussianBandit:
    """a k-armed bandit whose arm a pays a N(means[a], std^2) reward."""

    def __init__(self, means, *, std=1.0, rng=None):
        """stage 1: store means, std, n_arms, optimal_arm and a Generator self.rng."""
        self.means = np.asarray(means, dtype=float)
        self.std = float(std)
        self.n_arms = len(self.means)
        self.optimal_arm = int(np.argmax(self.means))
        self.rng = as_rng(rng)

    def pull(self, arm):
        """stage 1: one Gaussian reward from `arm`, a float."""
        return float(self.rng.normal(self.means[arm], self.std))


@app.function
def run_bandit(bandit, agent, n_steps):
    """stage 1: play n_steps rounds; returns (rewards (n_steps,), cumulative pseudo-regret (n_steps,))."""
    rewards = np.zeros(n_steps)
    gaps = np.zeros(n_steps)
    best = np.max(bandit.means)
    for t in range(n_steps):
        arm = agent.select_arm()
        r = bandit.pull(arm)
        agent.update(arm, r)
        rewards[t] = r
        gaps[t] = best - bandit.means[arm]
    return rewards, np.cumsum(gaps)


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
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = as_rng(rng)
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms, dtype=int)

    def select_arm(self):
        """stage 2: a uniformly random arm with probability epsilon, else a greedy arm (random ties)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        best = np.flatnonzero(self.Q == self.Q.max())
        return int(self.rng.choice(best))

    def update(self, arm, reward):
        """stage 2: incremental sample average: N[arm] += 1, Q[arm] += (reward - Q[arm]) / N[arm]."""
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


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
        self.n_arms = n_arms
        self.c = c
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms, dtype=int)

    def select_arm(self):
        """stage 3: the lowest-index untried arm if any, else the arm with the largest upper bound (ties to the lowest)."""
        untried = np.flatnonzero(self.N == 0)
        if len(untried):
            return int(untried[0])
        t = self.N.sum()
        return int(np.argmax(self.Q + self.c * np.sqrt(np.log(t) / self.N)))

    def update(self, arm, reward):
        """stage 3: the same sample-average update as EpsilonGreedy."""
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


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
        self.n_arms = n_arms
        self.rng = as_rng(rng)
        self.alpha = np.full(n_arms, float(prior[0]))
        self.beta = np.full(n_arms, float(prior[1]))

    def select_arm(self):
        """stage 4: draw theta ~ Beta(alpha, beta) for all arms at once (one rng.beta call); play its argmax."""
        theta = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(theta))

    def update(self, arm, reward):
        """stage 4: alpha[arm] += reward, beta[arm] += 1 - reward (reward in [0, 1])."""
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward


@app.function
def lai_robbins_constant(means):
    """stage 4: sum over suboptimal Bernoulli arms of gap / KL(mu_a, mu*): regret >= this * ln T."""
    mu = np.asarray(means, dtype=float)
    best = mu.max()
    total = 0.0
    for m in mu:
        if m >= best:
            continue
        with np.errstate(divide="ignore"):  # best = 1 makes the KL infinite: that arm costs nothing
            kl = (m * np.log(m / best) if m > 0 else 0.0) + (1 - m) * np.log((1 - m) / (1 - best))
        total += (best - m) / kl
    return float(total)


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
