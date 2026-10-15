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
    from torch.distributions import Normal

    from mlfs import checks, lesson
    from mlfs.deeprl import ReplayBuffer  # yours, from topic 01
    from mlfs.rl import Box, Env  # yours, from ch 23 · topic 02
    from mlfs.torchnn import MLP, ema_update  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # continuous control · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `Pendulum`: the classic torque-limited swing-up task as an `mlfs.rl.Env` |
    | 2 | `SquashedGaussianActor`, `TwinQ`: SAC's policy and critics → `mlfs.deeprl` |
    | 3 | `sac_critic_target`, `sac_actor_loss`, `sac_alpha_loss`: soft actor-critic's three losses |
    | 4 | `train_sac`: the full agent, with a fixed or learned temperature |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def angle_normalize(x):
    """stage 1: x wrapped into [-pi, pi)."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


@app.class_definition
class Pendulum(Env):
    """swing a pendulum up and hold it: obs (cos th, sin th, th_dot), action a torque in [-2, 2]."""

    max_speed, max_torque, dt, g, m, l = 8.0, 2.0, 0.05, 10.0, 1.0, 1.0

    def __init__(self, *, max_steps=200):
        """stage 1: store max_steps; observation_space Box(3), action_space Box(-2, 2, (1,))."""
        self.max_steps = max_steps
        self.observation_space = Box(np.array([-1.0, -1.0, -self.max_speed]), np.array([1.0, 1.0, self.max_speed]))
        self.action_space = Box(-self.max_torque, self.max_torque, (1,))
        self.state = None

    def _obs(self):
        th, thdot = self.state
        return np.array([np.cos(th), np.sin(th), thdot])

    def reset(self, *, seed=None):
        """stage 1: (th, th_dot) ~ U([-pi, pi] x [-1, 1]) from self.np_random; step counter 0; returns (obs, {})."""
        super().reset(seed=seed)
        self.state = self.np_random.uniform([-np.pi, -1.0], [np.pi, 1.0])
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        """stage 1: clip the torque, pay the quadratic cost, one semi-implicit Euler step; truncate at max_steps."""
        th, thdot = self.state
        u = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -self.max_torque, self.max_torque))
        cost = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * u**2
        thdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l**2) * u) * self.dt
        thdot = float(np.clip(thdot, -self.max_speed, self.max_speed))
        th = th + thdot * self.dt
        self.state = np.array([th, thdot])
        self.t += 1
        truncated = bool(self.max_steps is not None and self.t >= self.max_steps)
        return self._obs(), -float(cost), False, truncated, {}


@app.cell(hide_code=True)
def _():
    checks.run(1, Pendulum=Pendulum, angle_normalize=angle_normalize)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class SquashedGaussianActor(nn.Module):
    """a = act_limit * tanh(u), u ~ N(mu(s), sigma(s)^2): a Gaussian squashed into the action box."""

    def __init__(self, obs_dim, act_dim, *, hidden_sizes=(256, 256), act_limit=1.0):
        """stage 2: an MLP from obs_dim to 2 * act_dim outputs (mean and log-std); store act_limit."""
        super().__init__()
        self.net = MLP(obs_dim, tuple(hidden_sizes), 2 * act_dim)
        self.act_limit = float(act_limit)

    def forward(self, obs, deterministic=False):
        """stage 2: (action (B, act_dim) in (-act_limit, act_limit), log_prob (B,) with the tanh correction)."""
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        std = log_std.clamp(-20, 2).exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        log_prob = Normal(mu, std).log_prob(u).sum(-1)
        log_prob = log_prob - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return self.act_limit * torch.tanh(u), log_prob


@app.class_definition
class TwinQ(nn.Module):
    """two independent Q-networks Q1(s, a), Q2(s, a) on the concatenated observation and action."""

    def __init__(self, obs_dim, act_dim, *, hidden_sizes=(256, 256)):
        """stage 2: two separate MLPs from obs_dim + act_dim to 1."""
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, tuple(hidden_sizes), 1)
        self.q2 = MLP(obs_dim + act_dim, tuple(hidden_sizes), 1)

    def forward(self, obs, act):
        """stage 2: (q1 (B,), q2 (B,))."""
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


@app.cell(hide_code=True)
def _():
    checks.run(2, SquashedGaussianActor=SquashedGaussianActor, TwinQ=TwinQ)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def sac_critic_target(rewards, terminated, next_q1, next_q2, next_log_prob, *, gamma=0.99, alpha=0.2):
    """stage 3: y = r + gamma (1 - terminated) (min(Q1', Q2') - alpha log pi(a'|s')), without gradients."""
    with torch.no_grad():
        soft_value = torch.min(next_q1, next_q2) - alpha * next_log_prob
        return rewards + gamma * (1.0 - terminated) * soft_value


@app.function
def sac_actor_loss(q1, q2, log_prob, alpha):
    """stage 3: mean(alpha log pi(a|s) - min(Q1, Q2)(s, a)) for a ~ pi (reparameterized)."""
    return (alpha * log_prob - torch.min(q1, q2)).mean()


@app.function
def sac_alpha_loss(log_alpha, log_prob, target_entropy):
    """stage 3: -mean(exp(log_alpha) * (log_prob + target_entropy)), with log_prob a constant."""
    return -(log_alpha.exp() * (log_prob.detach() + target_entropy)).mean()


@app.cell(hide_code=True)
def _():
    checks.run(3, sac_critic_target=sac_critic_target, sac_actor_loss=sac_actor_loss, sac_alpha_loss=sac_alpha_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_sac(env, n_steps, *, gamma=0.99, tau=0.005, lr=3e-4, batch_size=256, buffer_size=100_000,
              learning_starts=1000, alpha=0.2, target_entropy=None, hidden_sizes=(256, 256), seed=None):
    """stage 4: SAC for n_steps env steps (alpha a float or 'auto'); returns (actor, history)."""
    rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        act_limit = float(np.max(env.action_space.high))
        actor = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes=hidden_sizes, act_limit=act_limit)
        critic = TwinQ(obs_dim, act_dim, hidden_sizes=hidden_sizes)
        critic_target = copy.deepcopy(critic)
        for p in critic_target.parameters():
            p.requires_grad_(False)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
        auto = alpha == "auto"
        if auto:
            target_entropy = -float(act_dim) if target_entropy is None else target_entropy
            log_alpha = torch.zeros(1, requires_grad=True)
            alpha_opt = torch.optim.Adam([log_alpha], lr=lr)
        buf = ReplayBuffer(buffer_size, (obs_dim,), action_shape=(act_dim,), action_dtype="float32",
                           seed=int(rng.integers(2**31)))
        history = {"episode_returns": [], "episode_ends": [], "alpha": [], "critic": critic,
                   "critic_target": critic_target}
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        ep_return = 0.0
        for t in range(n_steps):
            if t < learning_starts:
                action = env.action_space.sample(rng)
            else:
                with torch.no_grad():
                    action = actor(torch.as_tensor(np.ravel(obs), dtype=torch.float32)[None])[0][0].numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buf.add(np.ravel(obs), action, reward, np.ravel(next_obs), terminated)
            ep_return += reward
            obs = next_obs
            if terminated or truncated:
                history["episode_returns"].append(ep_return)
                history["episode_ends"].append(t + 1)
                obs, _ = env.reset()
                ep_return = 0.0
            if t + 1 < learning_starts:
                continue
            a_coef = log_alpha.exp().item() if auto else float(alpha)
            b = buf.sample(batch_size)
            with torch.no_grad():
                a2, lp2 = actor(b["next_obs"])
                q1t, q2t = critic_target(b["next_obs"], a2)
            y = sac_critic_target(b["rewards"], b["terminated"], q1t, q2t, lp2, gamma=gamma, alpha=a_coef)
            q1, q2 = critic(b["obs"], b["actions"])
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            a_new, lp = actor(b["obs"])
            q1n, q2n = critic(b["obs"], a_new)
            actor_loss = sac_actor_loss(q1n, q2n, lp, a_coef)
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            if auto:
                alpha_loss = sac_alpha_loss(log_alpha, lp, target_entropy)
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
            ema_update(critic_target, critic, decay=1.0 - tau)
            if terminated or truncated:
                history["alpha"].append(a_coef)
    return actor, history


@app.cell(hide_code=True)
def _():
    checks.run(4, train_sac=train_sac, Pendulum=Pendulum)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `SquashedGaussianActor` and `TwinQ` are part of
    `mlfs.deeprl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
