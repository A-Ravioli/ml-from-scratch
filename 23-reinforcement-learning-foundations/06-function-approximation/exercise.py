import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.rl import Box, Discrete, Env, epsilon_greedy  # yours, from topics 02 and 04
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # function approximation · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. tables stop working when states are
    continuous; here you replace them with a linear function of features, learn to drive an
    underpowered car up a hill, and then see exactly why the same idea can blow up.

    | stage | you build |
    |---|---|
    | 1 | `TileCoder`: overlapping grids that turn a continuous state into a few active features → `mlfs.rl` |
    | 2 | `MountainCar`: the classic continuous-state control problem → `mlfs.rl` |
    | 3 | `linear_sarsa`: semi-gradient SARSA with a linear Q → `mlfs.rl` |
    | 4 | `td_matrices`, `expected_td`: the deterministic dynamics of semi-gradient TD, and the deadly triad |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class TileCoder:
    """n_tilings offset grids over the box [low, high]; obs (d,) -> active feature indices (n_tilings,)."""

    def __init__(self, low, high, *, n_tilings=8, tiles_per_dim=8, rng=None):
        """stage 1: store low, high (float (d,)), n_tilings, tiles_per_dim, width, offsets (n_tilings, d) and n_features."""
        raise NotImplementedError("stage1: implement TileCoder.__init__")

    def __call__(self, obs):
        """stage 1: the index of the active tile in each tiling, an int array (n_tilings,)."""
        raise NotImplementedError("stage1: implement TileCoder.__call__")


@app.cell(hide_code=True)
def _():
    checks.run(1, TileCoder=TileCoder)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class MountainCar(Env):
    """Sutton & Barto's mountain car (Example 10.1): obs (position, velocity), actions 0 reverse, 1 coast, 2 forward."""

    def __init__(self, *, max_steps=200):
        """stage 2: store max_steps; observation_space = Box([-1.2, -0.07], [0.5, 0.07]), action_space = Discrete(3)."""
        raise NotImplementedError("stage2: implement MountainCar.__init__")

    def reset(self, *, seed=None):
        """stage 2: position ~ U[-0.6, -0.4) from self.np_random, velocity 0, step counter 0; returns (obs copy, {})."""
        raise NotImplementedError("stage2: implement MountainCar.reset")

    def step(self, action):
        """stage 2: v += 0.001 (a - 1) - 0.0025 cos(3 x) (clipped), x += v (clipped, v = 0 at the left wall); reward -1."""
        raise NotImplementedError("stage2: implement MountainCar.step")


@app.cell(hide_code=True)
def _():
    checks.run(2, MountainCar=MountainCar)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def linear_sarsa(env, featurizer, n_episodes, *, alpha=0.1, gamma=1.0, epsilon=0.0, rng=None):
    """stage 3: semi-gradient SARSA with Q(s, a) = w[featurizer(s), a].sum(); returns (w (n_features, n_actions), returns)."""
    raise NotImplementedError("stage3: implement linear_sarsa")


@app.cell(hide_code=True)
def _():
    checks.run(3, linear_sarsa=linear_sarsa, TileCoder=TileCoder, MountainCar=MountainCar)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def td_matrices(Phi, P, r, d, gamma):
    """stage 4: (A, b) with A = Phi^T D (Phi - gamma P Phi), b = Phi^T D r, D = diag(d): expected TD(0) is w += alpha (b - A w)."""
    raise NotImplementedError("stage4: implement td_matrices")


@app.function
def expected_td(Phi, P, r, d, gamma, *, alpha=0.01, n_iters=1000, w0=None):
    """stage 4: iterate w <- w + alpha (b - A w) from w0 (default zeros); returns the weights (n_iters + 1, k)."""
    raise NotImplementedError("stage4: implement expected_td")


@app.cell(hide_code=True)
def _():
    checks.run(4, td_matrices=td_matrices, expected_td=expected_td)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `TileCoder`, `MountainCar` and `linear_sarsa` are part of
    `mlfs.rl`, and chapter 23's library is complete. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
