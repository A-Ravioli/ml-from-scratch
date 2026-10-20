import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.distributions import Categorical

    from mlfs import checks, lesson
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # policy gradients · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `discounted_returns`: the return from every step of an episode → `mlfs.deeprl` |
    | 2 | `CategoricalPolicy`: a network that outputs a distribution over actions → `mlfs.deeprl` |
    | 3 | `reinforce_loss`: the score-function surrogate whose gradient is the policy gradient → `mlfs.deeprl` |
    | 4 | `train_reinforce`: REINFORCE with an optional learned baseline |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def discounted_returns(rewards, gamma):
    """stage 1: G_t = r_t + gamma G_{t+1} for one episode's 1-D rewards; same type (ndarray or tensor) and shape."""
    if torch.is_tensor(rewards):
        out = torch.zeros_like(rewards, dtype=torch.result_type(rewards, 1.0))
    else:
        rewards = np.asarray(rewards)
        out = np.zeros(rewards.shape, dtype=np.result_type(rewards.dtype, np.float32))
    acc = 0.0
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, discounted_returns=discounted_returns)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class CategoricalPolicy(nn.Module):
    """pi(a | s): an MLP from observations to the logits of a Categorical distribution over actions."""

    def __init__(self, obs_dim, n_actions, *, hidden_sizes=(64, 64)):
        """stage 2: an MLP (tanh hidden layers) from obs_dim to n_actions logits."""
        super().__init__()
        self.net = MLP(obs_dim, tuple(hidden_sizes), n_actions, activation="tanh")

    def forward(self, obs):
        """stage 2: obs (B, obs_dim) -> torch.distributions.Categorical with batch shape (B,)."""
        return Categorical(logits=self.net(obs))

    def act(self, obs):
        """stage 2: one observation (ndarray) -> a sampled action, a Python int; no autograd graph."""
        with torch.no_grad():
            dist = self(torch.as_tensor(np.asarray(obs), dtype=torch.float32).reshape(1, -1))
            return int(dist.sample()[0])


@app.cell(hide_code=True)
def _():
    checks.run(2, CategoricalPolicy=CategoricalPolicy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def reinforce_loss(log_probs, returns, *, baseline=None):
    """stage 3: -mean((returns - baseline) * log_probs), with returns and baseline treated as constants."""
    weights = torch.as_tensor(returns, dtype=log_probs.dtype).detach()
    if baseline is not None:
        weights = weights - torch.as_tensor(baseline, dtype=log_probs.dtype).detach()
    return -(weights * log_probs).mean()


@app.cell(hide_code=True)
def _():
    checks.run(3, reinforce_loss=reinforce_loss, CategoricalPolicy=CategoricalPolicy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_reinforce(env, n_episodes, *, gamma=0.99, lr=1e-2, batch_episodes=1, hidden_sizes=(32,),
                    baseline=None, lr_value=1e-2, seed=None):
    """stage 4: REINFORCE with reward-to-go and an optional value baseline; returns (policy, history)."""
    if baseline not in (None, "value"):
        raise ValueError(f"baseline must be None or 'value', got {baseline!r}")
    rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        obs_dim = int(np.prod(env.observation_space.shape))
        policy = CategoricalPolicy(obs_dim, env.action_space.n, hidden_sizes=hidden_sizes)
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        if baseline == "value":
            value_net = MLP(obs_dim, tuple(hidden_sizes), 1, activation="tanh")
            value_opt = torch.optim.Adam(value_net.parameters(), lr=lr_value)
        history = {"episode_returns": []}
        if baseline == "value":
            history["value_net"] = value_net
        first = True
        for start in range(0, n_episodes, batch_episodes):
            all_obs, all_actions, all_returns = [], [], []
            for _ in range(min(batch_episodes, n_episodes - start)):
                obs, _ = env.reset(seed=int(rng.integers(2**31))) if first else env.reset()
                first = False
                ep_obs, ep_actions, ep_rewards = [], [], []
                while True:
                    a = policy.act(obs)
                    next_obs, r, terminated, truncated, _ = env.step(a)
                    ep_obs.append(np.ravel(obs))
                    ep_actions.append(a)
                    ep_rewards.append(float(r))
                    obs = next_obs
                    if terminated or truncated:
                        break
                history["episode_returns"].append(float(sum(ep_rewards)))
                rewards = np.array(ep_rewards)
                if truncated and not terminated and baseline == "value":  # a time limit: bootstrap the tail
                    with torch.no_grad():
                        v_last = value_net(torch.as_tensor(np.ravel(obs), dtype=torch.float32)[None]).item()
                    rewards[-1] += gamma * v_last
                all_obs += ep_obs
                all_actions += ep_actions
                all_returns.append(discounted_returns(rewards, gamma))
            obs_t = torch.as_tensor(np.array(all_obs), dtype=torch.float32)
            act_t = torch.as_tensor(all_actions)
            ret_t = torch.as_tensor(np.concatenate(all_returns), dtype=torch.float32)
            log_probs = policy(obs_t).log_prob(act_t)
            if baseline == "value":
                values = value_net(obs_t).squeeze(-1)
                loss = reinforce_loss(log_probs, ret_t, baseline=values)
                value_loss = F.mse_loss(values, ret_t)
                value_opt.zero_grad()
                value_loss.backward()
                value_opt.step()
            else:
                loss = reinforce_loss(log_probs, ret_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy, history


@app.cell(hide_code=True)
def _():
    checks.run(4, train_reinforce=train_reinforce, CategoricalPolicy=CategoricalPolicy)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `discounted_returns`, `CategoricalPolicy` and
    `reinforce_loss` are part of `mlfs.deeprl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
