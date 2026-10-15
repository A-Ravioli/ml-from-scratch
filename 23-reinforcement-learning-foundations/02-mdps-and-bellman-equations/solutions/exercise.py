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
    # MDPs & bellman equations · build (reference solution)

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
        self.n = int(n)
        self.shape = ()

    def sample(self, rng=None):
        """stage 1: a uniformly random element, a Python int."""
        return int(as_rng(rng).integers(self.n))

    def contains(self, x):
        """stage 1: True iff x is an integer (not a bool, not a float) in [0, n)."""
        if isinstance(x, (bool, np.bool_)):
            return False
        if isinstance(x, np.ndarray):
            if x.shape != () or not np.issubdtype(x.dtype, np.integer):
                return False
            x = int(x)
        if not isinstance(x, (int, np.integer)):
            return False
        return 0 <= int(x) < self.n

    def __repr__(self):
        return f"Discrete({self.n})"


@app.class_definition
class Box:
    """a box of real vectors, low <= x <= high elementwise."""

    def __init__(self, low, high, shape=None):
        """stage 1: store shape (a tuple) and low, high as float arrays broadcast to it."""
        if shape is None:
            shape = np.broadcast(np.asarray(low), np.asarray(high)).shape
        self.shape = (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)
        self.low = np.broadcast_to(np.asarray(low, dtype=float), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=float), self.shape).copy()

    def sample(self, rng=None):
        """stage 1: a uniform draw from the box, a float array of shape self.shape."""
        return as_rng(rng).uniform(self.low, self.high)

    def contains(self, x):
        """stage 1: True iff x has shape self.shape and lies in the box."""
        x = np.asarray(x)
        if x.shape != self.shape or not np.issubdtype(x.dtype, np.number):
            return False
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

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
        if seed is not None or getattr(self, "np_random", None) is None:
            self.np_random = np.random.default_rng(seed)

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
        self.height, self.width = int(height), int(width)
        self.start = tuple(start)
        self.goal = (self.height - 1, self.width - 1) if goal is None else tuple(goal)
        self.walls = tuple(tuple(w) for w in walls)
        self.slip = float(slip)
        self.step_reward = float(step_reward)
        self.max_steps = max_steps
        for name, cell in [("start", self.start), ("goal", self.goal)]:
            if not (0 <= cell[0] < self.height and 0 <= cell[1] < self.width) or cell in self.walls:
                raise ValueError(f"{name} {cell} must be a free cell inside the grid")
        if self.start == self.goal:
            raise ValueError("start and goal must differ")

        S, A = self.height * self.width, 4
        self.n_states, self.n_actions = S, A
        self.observation_space, self.action_space = Discrete(S), Discrete(A)
        self.start_state = self.start[0] * self.width + self.start[1]
        goal_state = self.goal[0] * self.width + self.goal[1]
        blocked = {r * self.width + c for r, c in self.walls}
        self.terminal = np.zeros(S, dtype=bool)
        self.terminal[goal_state] = True

        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        nxt = np.zeros((S, A), dtype=int)
        for s in range(S):
            r, c = divmod(s, self.width)
            for a, (dr, dc) in enumerate(moves):
                rr, cc = r + dr, c + dc
                inside = 0 <= rr < self.height and 0 <= cc < self.width
                if s == goal_state or s in blocked or not inside or rr * self.width + cc in blocked:
                    nxt[s, a] = s
                else:
                    nxt[s, a] = rr * self.width + cc
        det = np.zeros((S, A, S))
        det[np.arange(S)[:, None], np.arange(A)[None, :], nxt] = 1.0
        self.P = (1 - self.slip) * det + self.slip * det.mean(axis=1, keepdims=True)
        self.R = np.full((S, A), self.step_reward)
        absorbing = self.terminal.copy()
        absorbing[list(blocked)] = True
        for s in np.flatnonzero(absorbing):  # terminal (and wall) states loop to themselves
            self.P[s] = 0.0
            self.P[s, :, s] = 1.0
        self.R[absorbing] = 0.0

    def reset(self, *, seed=None):
        """stage 3: seed via Env.reset, put the agent on start_state, zero the step counter; returns (obs, {})."""
        super().reset(seed=seed)
        self.state = self.start_state
        self.t = 0
        return self.state, {}

    def step(self, action):
        """stage 3: sample s' ~ P[s, action] with self.np_random; returns (s', R[s, action], terminated, truncated, {}) (truncated only if not terminated)."""
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}: expected an int in [0, {self.n_actions})")
        s = self.state
        cdf = np.cumsum(self.P[s, action])
        s2 = int(min(np.searchsorted(cdf, self.np_random.random(), side="right"), self.n_states - 1))
        self.state = s2
        self.t += 1
        terminated = bool(self.terminal[s2])
        truncated = bool(not terminated and self.max_steps is not None and self.t >= self.max_steps)
        return s2, float(self.R[s, action]), terminated, truncated, {}


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
    obs, _ = env.reset(seed=seed)
    observations, actions, rewards = [obs], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        if max_steps is not None and len(actions) >= max_steps:
            truncated = True
            break
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
    return {"observations": observations, "actions": actions, "rewards": rewards,
            "terminated": bool(terminated), "truncated": bool(truncated), "return": float(sum(rewards))}


@app.cell(hide_code=True)
def _():
    checks.run(3, GridWorld=GridWorld, rollout=rollout, Env=Env, Discrete=Discrete)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def policy_evaluation(P, R, policy, gamma=0.99, *, tol=1e-08, max_iters=10000):
    """stage 4: V^pi (S,) by iterating V <- R_pi + gamma P_pi V until the largest change is below tol."""
    P_pi = np.einsum("sa,sat->st", policy, P)
    R_pi = np.sum(policy * R, axis=1)
    V = np.zeros(len(R_pi))
    for _ in range(max_iters):
        V_new = R_pi + gamma * P_pi @ V
        done = np.max(np.abs(V_new - V)) < tol
        V = V_new
        if done:
            break
    return V


@app.function
def q_from_v(P, R, V, gamma=0.99):
    """stage 4: Q (S, A) = R + gamma * sum_s' P[s, a, s'] V[s']."""
    return R + gamma * P @ V


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
