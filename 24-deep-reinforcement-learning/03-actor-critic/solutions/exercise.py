import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Categorical, Independent, Normal

    from mlfs import checks, lesson
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # actor-critic · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `compute_gae`: generalized advantage estimation over a rollout → `mlfs.deeprl` |
    | 2 | `ActorCritic`: a policy and a value function in one module, discrete or continuous → `mlfs.deeprl` |
    | 3 | `RolloutCollector`: synchronous rollouts from N environments, with time limits handled correctly |
    | 4 | `a2c_loss`, `train_a2c`: advantage actor-critic |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def compute_gae(rewards, values, terminated, last_value, *, gamma=0.99, lam=0.95):
    """stage 1: GAE(gamma, lambda) advantages and returns = advantages + values; (T,) or (T, N); no autograd graph."""
    with torch.no_grad():
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        values = torch.as_tensor(values, dtype=torch.float32).detach()
        nonterminal = 1.0 - torch.as_tensor(terminated, dtype=torch.float32)
        next_value = torch.as_tensor(last_value, dtype=torch.float32).detach()
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + gamma * nonterminal[t] * next_value - values[t]
            gae = delta + gamma * lam * nonterminal[t] * gae
            advantages[t] = gae
            next_value = values[t]
        return advantages, advantages + values


@app.cell(hide_code=True)
def _():
    checks.run(1, compute_gae=compute_gae)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ActorCritic(nn.Module):
    """a policy network (actor) and a value network (critic); discrete (Categorical) or continuous (diagonal Normal)."""

    def __init__(self, obs_dim, act_dim, *, hidden_sizes=(64, 64), continuous=False):
        """stage 2: separate tanh MLPs for the actor (act_dim outputs) and the critic (1 output); log_std = 0 if continuous."""
        super().__init__()
        self.continuous = continuous
        self.actor = MLP(obs_dim, tuple(hidden_sizes), act_dim, activation="tanh")
        self.critic = MLP(obs_dim, tuple(hidden_sizes), 1, activation="tanh")
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        """stage 2: obs (B, obs_dim) -> (distribution with log_prob/entropy of shape (B,), value (B,))."""
        out = self.actor(obs)
        if self.continuous:
            dist = Independent(Normal(out, self.log_std.exp().expand_as(out)), 1)
        else:
            dist = Categorical(logits=out)
        return dist, self.critic(obs).squeeze(-1)

    def act(self, obs):
        """stage 2: one ndarray observation -> (action, log_prob float, value float), sampled without gradients."""
        with torch.no_grad():
            dist, value = self(torch.as_tensor(np.asarray(obs), dtype=torch.float32).reshape(1, -1))
            a = dist.sample()
            lp = float(dist.log_prob(a)[0])
        action = a[0].numpy() if self.continuous else int(a[0])
        return action, lp, float(value[0])


@app.cell(hide_code=True)
def _():
    checks.run(2, ActorCritic=ActorCritic)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class RolloutCollector:
    """steps N environments in lockstep with a model; auto-resets them and bootstraps time limits into the rewards."""

    def __init__(self, envs, model, *, gamma=0.99, seed=None):
        """stage 3: store envs, model, gamma; reset env i with the i-th of N seeds drawn from default_rng(seed)."""
        self.envs, self.model, self.gamma = list(envs), model, gamma
        seeds = np.random.default_rng(seed).integers(2**31, size=len(self.envs))
        self.obs = [np.asarray(env.reset(seed=int(s))[0], dtype=np.float32) for env, s in zip(self.envs, seeds)]
        self.running = np.zeros(len(self.envs))
        self.episode_returns = []
        self.total_steps = 0
        self.episode_ends = []

    def _forward(self, obs):
        return self.model(torch.as_tensor(np.stack(obs), dtype=torch.float32))

    def collect(self, n_steps):
        """stage 3: n_steps steps of every env; returns a dict of tensors with a leading (T, N) shape, plus last_value (N,)."""
        keys = ("obs", "actions", "log_probs", "values", "rewards", "dones")
        out = {k: [] for k in keys}
        for _ in range(n_steps):
            with torch.no_grad():
                dist, value = self._forward(self.obs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            rewards = np.zeros(len(self.envs), dtype=np.float32)
            dones = np.zeros(len(self.envs), dtype=np.float32)
            next_obs = []
            self.total_steps += len(self.envs)
            for i, env in enumerate(self.envs):
                a = actions[i].numpy() if actions.ndim > 1 else int(actions[i])
                o, r, terminated, truncated, _ = env.step(a)
                o = np.asarray(o, dtype=np.float32)
                self.running[i] += r
                rewards[i] = r
                if terminated or truncated:
                    if truncated and not terminated:  # a time limit: bootstrap from the state we were cut off in
                        with torch.no_grad():
                            rewards[i] += self.gamma * float(self._forward([o])[1][0])
                    dones[i] = 1.0
                    self.episode_returns.append(float(self.running[i]))
                    self.episode_ends.append(self.total_steps)
                    self.running[i] = 0.0
                    o = np.asarray(env.reset()[0], dtype=np.float32)
                next_obs.append(o)
            out["obs"].append(torch.as_tensor(np.stack(self.obs)))
            out["actions"].append(actions)
            out["log_probs"].append(log_probs)
            out["values"].append(value)
            out["rewards"].append(torch.as_tensor(rewards))
            out["dones"].append(torch.as_tensor(dones))
            self.obs = next_obs
        batch = {k: torch.stack(v) for k, v in out.items()}
        with torch.no_grad():
            batch["last_value"] = self._forward(self.obs)[1]
        return batch


@app.cell(hide_code=True)
def _():
    checks.run(3, RolloutCollector=RolloutCollector, compute_gae=compute_gae)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def a2c_loss(dist, actions, advantages, returns, values, *, vf_coef=0.5, ent_coef=0.01):
    """stage 4: policy-gradient loss + vf_coef * value MSE - ent_coef * entropy; returns (loss, info dict of floats)."""
    policy_loss = -(advantages.detach() * dist.log_prob(actions)).mean()
    value_loss = ((values - returns.detach()) ** 2).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return loss, {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item()}


@app.function
def train_a2c(make_env, total_steps, *, n_envs=8, n_steps=16, gamma=0.99, lam=0.95, lr=7e-4, vf_coef=0.5,
              ent_coef=0.01, max_grad_norm=0.5, hidden_sizes=(64, 64), continuous=False, seed=None):
    """stage 4: A2C with GAE on n_envs copies of make_env(); returns (model, history)."""
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        envs = [make_env() for _ in range(n_envs)]
        obs_dim = int(np.prod(envs[0].observation_space.shape))
        act_dim = int(np.prod(envs[0].action_space.shape)) if continuous else envs[0].action_space.n
        model = ActorCritic(obs_dim, act_dim, hidden_sizes=hidden_sizes, continuous=continuous)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        collector = RolloutCollector(envs, model, gamma=gamma, seed=seed)
        history = {"policy_loss": [], "value_loss": [], "entropy": []}
        for _ in range(max(1, total_steps // (n_envs * n_steps))):
            batch = collector.collect(n_steps)
            adv, ret = compute_gae(batch["rewards"], batch["values"], batch["dones"], batch["last_value"],
                                   gamma=gamma, lam=lam)
            dist, values = model(batch["obs"].reshape(-1, obs_dim))
            actions = batch["actions"].reshape((-1, act_dim) if continuous else (-1,))
            loss, info = a2c_loss(dist, actions, adv.reshape(-1), ret.reshape(-1), values, vf_coef=vf_coef,
                                  ent_coef=ent_coef)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            for k, v in info.items():
                history[k].append(v)
        history["episode_returns"] = collector.episode_returns
        history["episode_ends"] = collector.episode_ends
    return model, history


@app.cell(hide_code=True)
def _():
    checks.run(4, a2c_loss=a2c_loss, train_a2c=train_a2c)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `compute_gae` and `ActorCritic` are part of `mlfs.deeprl`
    (the next topic's PPO is built from them and your `RolloutCollector`). open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
