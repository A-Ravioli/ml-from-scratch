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
    # function approximation · build (reference solution)

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
        self.low = np.atleast_1d(np.asarray(low, dtype=float))
        self.high = np.atleast_1d(np.asarray(high, dtype=float))
        self.n_tilings, self.tiles_per_dim = int(n_tilings), int(tiles_per_dim)
        d = len(self.low)
        self.width = (self.high - self.low) / (self.tiles_per_dim - 1)
        if rng is None:
            i, j = np.arange(self.n_tilings)[:, None], np.arange(d)[None, :]
            self.offsets = (i * (2 * j + 1) / self.n_tilings) % 1.0
        else:
            self.offsets = as_rng(rng).random((self.n_tilings, d))
        self.n_features = self.n_tilings * self.tiles_per_dim**d

    def __call__(self, obs):
        """stage 1: the index of the active tile in each tiling, an int array (n_tilings,)."""
        x = np.clip(np.atleast_1d(np.asarray(obs, dtype=float)), self.low, self.high)
        u = (x - self.low) / self.width
        coords = np.minimum(np.floor(u + self.offsets).astype(int), self.tiles_per_dim - 1)
        flat = np.ravel_multi_index(coords.T, (self.tiles_per_dim,) * len(self.low))
        return np.arange(self.n_tilings) * self.tiles_per_dim ** len(self.low) + flat


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
        self.max_steps = max_steps
        self.observation_space = Box(np.array([-1.2, -0.07]), np.array([0.5, 0.07]))
        self.action_space = Discrete(3)

    def reset(self, *, seed=None):
        """stage 2: position ~ U[-0.6, -0.4) from self.np_random, velocity 0, step counter 0; returns (obs copy, {})."""
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(-0.6, -0.4), 0.0])
        self.t = 0
        return self.state.copy(), {}

    def step(self, action):
        """stage 2: v += 0.001 (a - 1) - 0.0025 cos(3 x) (clipped), x += v (clipped, v = 0 at the left wall); reward -1."""
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}: 0, 1 or 2")
        x, v = self.state
        v = float(np.clip(v + 0.001 * (action - 1) - 0.0025 * np.cos(3 * x), -0.07, 0.07))
        x = float(np.clip(x + v, -1.2, 0.5))
        if x <= -1.2:
            v = 0.0
        self.state = np.array([x, v])
        self.t += 1
        terminated = bool(x >= 0.5)
        truncated = bool(not terminated and self.max_steps is not None and self.t >= self.max_steps)
        return self.state.copy(), -1.0, terminated, truncated, {}


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
    rng = as_rng(rng)
    w = np.zeros((featurizer.n_features, env.action_space.n))
    episode_returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31))) if ep == 0 else env.reset()
        idx = featurizer(obs)
        a = epsilon_greedy(w[idx].sum(axis=0), epsilon, rng=rng)
        done, total = False, 0.0
        while not done:
            obs2, r, terminated, truncated, _ = env.step(a)
            total += r
            if terminated:
                target = r
            else:
                idx2 = featurizer(obs2)
                a2 = epsilon_greedy(w[idx2].sum(axis=0), epsilon, rng=rng)
                target = r + gamma * w[idx2, a2].sum()
            delta = target - w[idx, a].sum()
            w[idx, a] += alpha / len(idx) * delta
            done = terminated or truncated
            if not done:
                idx, a = idx2, a2
        episode_returns.append(total)
    return w, episode_returns


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
    Phi = np.asarray(Phi, dtype=float)
    D = np.diag(np.asarray(d, dtype=float))
    A = Phi.T @ D @ (Phi - gamma * P @ Phi)
    b = Phi.T @ D @ np.asarray(r, dtype=float)
    return A, b


@app.function
def expected_td(Phi, P, r, d, gamma, *, alpha=0.01, n_iters=1000, w0=None):
    """stage 4: iterate w <- w + alpha (b - A w) from w0 (default zeros); returns the weights (n_iters + 1, k)."""
    A, b = td_matrices(Phi, P, r, d, gamma)
    w = np.zeros(A.shape[0]) if w0 is None else np.array(w0, dtype=float)
    out = [w]
    for _ in range(n_iters):
        w = w + alpha * (b - A @ w)
        out.append(w)
    return np.array(out)


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
