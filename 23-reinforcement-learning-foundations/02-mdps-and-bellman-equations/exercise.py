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
    # MDPs & bellman equations · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    classes and functions to write, and checks them live as you edit. the protocol you build
    here (spaces, `Env.reset`/`Env.step`, `rollout`) is the one every environment and agent in
    chapters 23 and 24 speaks, so the checks are strict about it.

    | stage | you build |
    |---|---|
    | 1 | `Discrete`, `Box`, `Env`: action and observation spaces, and the environment base class → `mlfs.rl` |
    | 2 | `GridWorld.__init__`: the MDP as arrays: `P` (S, A, S'), `R` (S, A) → `mlfs.rl` |
    | 3 | `GridWorld.reset`, `GridWorld.step`, `rollout`: the gym-style interaction loop → `mlfs.rl` |
    | 4 | `policy_evaluation`, `q_from_v`: solve the Bellman expectation equation → `mlfs.rl` |

    the lesson's spec boxes give the exact conventions. you can also run the checks from a
    terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Discrete:
    """the integers {0, 1, ..., n - 1}."""

    def __init__(self, n):
        """stage 1: store n (an int) and shape = ()."""
        raise NotImplementedError("stage1: implement Discrete.__init__")

    def sample(self, rng=None):
        """stage 1: a uniformly random element, a Python int."""
        raise NotImplementedError("stage1: implement Discrete.sample")

    def contains(self, x):
        """stage 1: True iff x is an integer (not a bool, not a float) in [0, n)."""
        raise NotImplementedError("stage1: implement Discrete.contains")

    def __repr__(self):
        return f"Discrete({self.n})"


@app.class_definition
class Box:
    """a box of real vectors, low <= x <= high elementwise."""

    def __init__(self, low, high, shape=None):
        """stage 1: store shape (a tuple) and low, high as float arrays broadcast to it."""
        raise NotImplementedError("stage1: implement Box.__init__")

    def sample(self, rng=None):
        """stage 1: a uniform draw from the box, a float array of shape self.shape."""
        raise NotImplementedError("stage1: implement Box.sample")

    def contains(self, x):
        """stage 1: True iff x has shape self.shape and lies in the box."""
        raise NotImplementedError("stage1: implement Box.contains")

    def __repr__(self):
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape})"


@app.class_definition
class Env:
    """base class of every environment: gymnasium's reset/step protocol.

    subclasses set observation_space and action_space, and define reset (calling
    Env.reset first, to seed) and step(action) -> (obs, reward, terminated, truncated, info).
    """

    observation_space = None
    action_space = None

    def reset(self, *, seed=None):
        """stage 1: (re)seed: self.np_random = np.random.default_rng(seed) if seed is given or none exists yet."""
        raise NotImplementedError("stage1: implement Env.reset")

    def step(self, action):
        """advance one step; every subclass defines this."""
        raise TypeError(f"{type(self).__name__} doesn't define step(action)")


@app.cell(hide_code=True)
def _():
    checks.run(1, Discrete=Discrete, Box=Box, Env=Env)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class GridWorld(Env):
    """a tabular gridworld: obs = row * width + col; actions 0 up, 1 right, 2 down, 3 left."""

    def __init__(self, height=4, width=4, *, start=(0, 0), goal=None, walls=(), slip=0.0,
                 step_reward=-1.0, max_steps=100):
        """stage 2: store the arguments; build n_states, n_actions, spaces, start_state, terminal (S,), P (S, A, S'), R (S, A)."""
        raise NotImplementedError("stage2: implement GridWorld.__init__")

    def reset(self, *, seed=None):
        """stage 3: seed via Env.reset, put the agent on start_state, zero the step counter; returns (obs, {})."""
        raise NotImplementedError("stage3: implement GridWorld.reset")

    def step(self, action):
        """stage 3: sample s' ~ P[s, action] with self.np_random; returns (s', R[s, action], terminated, truncated, {}) (truncated only if not terminated)."""
        raise NotImplementedError("stage3: implement GridWorld.step")


@app.cell(hide_code=True)
def _():
    checks.run(2, GridWorld=GridWorld, Discrete=Discrete, Env=Env)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def rollout(env, policy, *, max_steps=None, seed=None):
    """stage 3: one episode with policy(obs) -> action; returns dict(observations, actions, rewards, terminated, truncated, return)."""
    raise NotImplementedError("stage3: implement rollout")


@app.cell(hide_code=True)
def _():
    checks.run(3, GridWorld=GridWorld, rollout=rollout, Env=Env, Discrete=Discrete, Box=Box)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def policy_evaluation(P, R, policy, gamma=0.99, *, tol=1e-08, max_iters=10000):
    """stage 4: V^pi (S,) by iterating V <- R_pi + gamma P_pi V until the largest change is below tol."""
    raise NotImplementedError("stage4: implement policy_evaluation")


@app.function
def q_from_v(P, R, V, gamma=0.99):
    """stage 4: Q (S, A) = R + gamma * sum_s' P[s, a, s'] V[s']."""
    raise NotImplementedError("stage4: implement q_from_v")


@app.cell(hide_code=True)
def _():
    checks.run(4, policy_evaluation=policy_evaluation, q_from_v=q_from_v, GridWorld=GridWorld)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your spaces, `Env`, `GridWorld`, `rollout` and policy
    evaluation are part of `mlfs.rl` (try `from mlfs.rl import GridWorld`). open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
