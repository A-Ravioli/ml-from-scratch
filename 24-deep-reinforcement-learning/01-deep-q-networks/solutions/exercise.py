import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.rl import Box, Discrete, Env, epsilon_greedy  # yours, from ch 23 · topics 02 and 04
    from mlfs.torchnn import MLP, ema_update  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # deep Q-networks · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. this chapter is spec-driven: the lesson's **spec** boxes are the
    contract (names, arguments, conventions), and the internal structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `CartPole`: the classic cart–pole balancing task as an `mlfs.rl.Env` |
    | 2 | `ReplayBuffer`: a preallocated ring buffer of transitions → `mlfs.deeprl` |
    | 3 | `QNetwork`, `dqn_loss`: the network (plain and dueling) and the (double) DQN loss → `mlfs.deeprl` |
    | 4 | `train_dqn`: the full agent: ε-schedule, replay, target network |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class CartPole(Env):
    """Barto, Sutton & Anderson's cart-pole: obs (x, x_dot, theta, theta_dot); actions 0 push left, 1 push right."""

    gravity, masscart, masspole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x_threshold, theta_threshold = 2.4, 12 * 2 * np.pi / 360

    def __init__(self, *, max_steps=500):
        """stage 1: store max_steps; observation_space Box(4), action_space Discrete(2)."""
        self.max_steps = max_steps
        high = np.array([2 * self.x_threshold, np.inf, 2 * self.theta_threshold, np.inf])
        self.observation_space = Box(-high, high)
        self.action_space = Discrete(2)
        self.state = None

    def reset(self, *, seed=None):
        """stage 1: state ~ U(-0.05, 0.05)^4 from self.np_random, step counter 0; returns (obs copy, {})."""
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.copy(), {}

    def step(self, action):
        """stage 1: one Euler step of the frictionless dynamics; reward 1.0; terminate past 2.4 m or 12 degrees."""
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}: 0 (left) or 1 (right)")
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        total_mass = self.masscart + self.masspole
        polemass_length = self.masspole * self.length
        cos, sin = np.cos(theta), np.sin(theta)
        temp = (force + polemass_length * theta_dot**2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos**2 / total_mass))
        x_acc = temp - polemass_length * theta_acc * cos / total_mass
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * x_acc
        theta, theta_dot = theta + self.tau * theta_dot, theta_dot + self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.t += 1
        terminated = bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
        truncated = bool(not terminated and self.max_steps is not None and self.t >= self.max_steps)
        return self.state.copy(), 1.0, terminated, truncated, {}


@app.cell(hide_code=True)
def _():
    checks.run(1, CartPole=CartPole)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ReplayBuffer:
    """a fixed-size ring buffer of transitions (s, a, r, s', terminated), stored in preallocated NumPy arrays."""

    def __init__(self, capacity, obs_shape, *, action_shape=(), action_dtype="int64", seed=None):
        """stage 2: preallocate the arrays (float32 except actions); position and size 0; a Generator from seed."""
        self.capacity = int(capacity)
        obs_shape = (int(obs_shape),) if np.isscalar(obs_shape) else tuple(obs_shape)
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *tuple(action_shape)), dtype=action_dtype)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)
        self.pos, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        """stage 2: write the transition at the current position, overwriting the oldest once full."""
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """stage 2: batch_size transitions drawn uniformly (with replacement); dict of torch tensors."""
        if self.size == 0:
            raise ValueError("can't sample from an empty replay buffer")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {"obs": torch.as_tensor(self.obs[idx]), "actions": torch.as_tensor(self.actions[idx]),
                "rewards": torch.as_tensor(self.rewards[idx]), "next_obs": torch.as_tensor(self.next_obs[idx]),
                "terminated": torch.as_tensor(self.terminated[idx])}

    def __len__(self):
        """stage 2: the number of transitions stored (at most capacity)."""
        return self.size


@app.cell(hide_code=True)
def _():
    checks.run(2, ReplayBuffer=ReplayBuffer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class QNetwork(nn.Module):
    """obs (B, obs_dim) -> Q-values (B, n_actions); dueling = separate value and advantage streams."""

    def __init__(self, obs_dim, n_actions, *, hidden_sizes=(128, 128), dueling=False):
        """stage 3: an MLP; if dueling, a shared trunk with a value head (1) and an advantage head (n_actions)."""
        super().__init__()
        self.dueling = dueling
        hidden_sizes = tuple(hidden_sizes)
        if not dueling:
            self.net = MLP(obs_dim, hidden_sizes, n_actions)
        else:
            self.trunk = nn.Sequential(MLP(obs_dim, hidden_sizes[:-1], hidden_sizes[-1]), nn.ReLU())
            self.value = nn.Linear(hidden_sizes[-1], 1)
            self.advantage = nn.Linear(hidden_sizes[-1], n_actions)

    def forward(self, obs):
        """stage 3: Q (B, n_actions); dueling: V + A - mean_a A."""
        if not self.dueling:
            return self.net(obs)
        h = self.trunk(obs)
        a = self.advantage(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)


@app.function
def dqn_loss(q_net, target_net, batch, *, gamma=0.99, double=False, huber=True):
    """stage 3: TD loss between Q(s, a) and r + gamma (1 - terminated) Q_target(s', a*), no gradient into the target."""
    q = q_net(batch["obs"]).gather(1, batch["actions"].long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(batch["next_obs"])
        if double:
            best = q_net(batch["next_obs"]).argmax(dim=1, keepdim=True)
        else:
            best = next_q.argmax(dim=1, keepdim=True)
        target = batch["rewards"] + gamma * (1.0 - batch["terminated"]) * next_q.gather(1, best).squeeze(1)
    return F.smooth_l1_loss(q, target) if huber else F.mse_loss(q, target)


@app.cell(hide_code=True)
def _():
    checks.run(3, QNetwork=QNetwork, dqn_loss=dqn_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def linear_schedule(start, end, duration, t):
    """stage 4: start + (end - start) * min(t / duration, 1): a linear ramp held at `end` after `duration` steps."""
    return end if duration <= 0 else start + (end - start) * min(t / duration, 1.0)


@app.function
def train_dqn(env, n_steps, *, gamma=0.99, lr=1e-3, buffer_size=50_000, batch_size=64, learning_starts=1000,
              train_freq=1, target_update=500, tau=None, eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000,
              hidden_sizes=(64, 64), double=False, dueling=False, huber=True, seed=None):
    """stage 4: DQN for n_steps env steps; returns (q_net, history dict with 'episode_returns' and 'episode_ends')."""
    rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = env.action_space.n
        q_net = QNetwork(obs_dim, n_actions, hidden_sizes=hidden_sizes, dueling=dueling)
        target_net = copy.deepcopy(q_net)
        for p in target_net.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(q_net.parameters(), lr=lr)
        buf = ReplayBuffer(buffer_size, (obs_dim,), seed=int(rng.integers(2**31)))

        history = {"episode_returns": [], "episode_ends": []}
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        ep_return, n_updates = 0.0, 0
        for t in range(n_steps):
            eps = linear_schedule(eps_start, eps_end, eps_decay_steps, t)
            with torch.no_grad():
                q = q_net(torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1))[0].numpy()
            action = epsilon_greedy(q, eps, rng=rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buf.add(np.ravel(obs), action, reward, np.ravel(next_obs), terminated)  # terminated, NOT done
            ep_return += reward
            obs = next_obs
            if terminated or truncated:
                history["episode_returns"].append(ep_return)
                history["episode_ends"].append(t + 1)
                obs, _ = env.reset()
                ep_return = 0.0

            if t + 1 >= learning_starts and (t + 1) % train_freq == 0:
                loss = dqn_loss(q_net, target_net, buf.sample(batch_size), gamma=gamma, double=double, huber=huber)
                opt.zero_grad()
                loss.backward()
                opt.step()
                n_updates += 1
                if tau is not None:
                    ema_update(target_net, q_net, decay=1.0 - tau)
                elif n_updates % target_update == 0:
                    target_net.load_state_dict(q_net.state_dict())
    return q_net, history


@app.cell(hide_code=True)
def _():
    checks.run(4, train_dqn=train_dqn, linear_schedule=linear_schedule, CartPole=CartPole)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `ReplayBuffer`, `QNetwork` and `dqn_loss` are part of
    `mlfs.deeprl` (try `from mlfs.deeprl import dqn_loss`), and `CartPole` is the
    environment the next three topics learn on. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
