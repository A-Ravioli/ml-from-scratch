import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson
    from mlfs.deeprl import CategoricalPolicy, QNetwork, ReplayBuffer, dqn_loss  # yours, from topics 01-02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # offline RL & imitation · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `behavior_cloning`: supervised learning of an expert's actions |
    | 2 | `dagger`: imitation that asks the expert about the states the learner actually visits |
    | 3 | `cql_penalty`, `train_offline_q`: Q-learning from a fixed dataset, with and without conservatism → `mlfs.deeprl` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def behavior_cloning(policy, obs, actions, *, epochs=50, lr=1e-2, batch_size=64, seed=None):
    """stage 1: train policy (obs -> Categorical) by minibatch cross-entropy on (obs, actions); returns epoch losses."""
    rng = np.random.default_rng(seed)
    obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    actions = torch.as_tensor(np.asarray(actions), dtype=torch.int64)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(len(obs))
        total = 0.0
        for i in range(0, len(obs), batch_size):
            idx = torch.as_tensor(perm[i:i + batch_size])
            loss = -policy(obs[idx]).log_prob(actions[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(obs))
    return losses


@app.function
def run_policy(env, act, n_episodes, *, seed=None):
    """stage 1: n_episodes with act(obs) -> action; returns (list of episode returns, list of visited observations)."""
    returns, visited = [], []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed if k == 0 else None)
        total = 0.0
        while True:
            visited.append(np.asarray(obs, dtype=np.float32))
            obs, r, terminated, truncated, _ = env.step(act(obs))
            total += r
            if terminated or truncated:
                break
        returns.append(total)
    return returns, visited


@app.cell(hide_code=True)
def _():
    checks.run(1, behavior_cloning=behavior_cloning, run_policy=run_policy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def dagger(env, expert, policy, *, n_iterations=5, episodes_per_iteration=5, epochs=30, lr=1e-2, batch_size=64,
           seed=None):
    """stage 2: DAgger: iteration 0 imitates expert rollouts; later ones add the learner's own states, labelled by the expert."""
    rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        data_obs, data_act = [], []
        history = {"returns": [], "dataset_size": []}
        for it in range(n_iterations):
            if it == 0:
                rets, visited = run_policy(env, expert, episodes_per_iteration, seed=int(rng.integers(2**31)))
            else:
                rets, visited = run_policy(env, policy.act, episodes_per_iteration, seed=int(rng.integers(2**31)))
            data_obs += visited
            data_act += [int(expert(o)) for o in visited]
            behavior_cloning(policy, np.array(data_obs), np.array(data_act), epochs=epochs, lr=lr,
                             batch_size=batch_size, seed=int(rng.integers(2**31)))
            history["returns"].append(float(np.mean(rets)))
            history["dataset_size"].append(len(data_obs))
    return policy, history


@app.cell(hide_code=True)
def _():
    checks.run(2, dagger=dagger, behavior_cloning=behavior_cloning, run_policy=run_policy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def cql_penalty(q_values, actions, *, alpha=1.0):
    """stage 3: alpha * mean(logsumexp_a Q(s, a) - Q(s, a_data)) for q_values (B, n_actions), actions (B,)."""
    q_data = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    return alpha * (torch.logsumexp(q_values, dim=1) - q_data).mean()


@app.function
def train_offline_q(buffer, obs_dim, n_actions, n_updates, *, gamma=0.99, lr=1e-3, batch_size=64,
                    target_update=100, cql_alpha=0.0, hidden_sizes=(64, 64), double=True, seed=None):
    """stage 3: (double) DQN on a fixed replay buffer plus cql_alpha * cql_penalty; returns (q_net, history)."""
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        q_net = QNetwork(obs_dim, n_actions, hidden_sizes=hidden_sizes)
        target_net = copy.deepcopy(q_net)
        opt = torch.optim.Adam(q_net.parameters(), lr=lr)
        history = {"td_loss": [], "cql": []}
        for k in range(n_updates):
            batch = buffer.sample(batch_size)
            td = dqn_loss(q_net, target_net, batch, gamma=gamma, double=double)
            cql = cql_penalty(q_net(batch["obs"]), batch["actions"], alpha=cql_alpha) if cql_alpha > 0 else 0.0 * td
            loss = td + cql
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (k + 1) % target_update == 0:
                target_net.load_state_dict(q_net.state_dict())
            history["td_loss"].append(td.item())
            history["cql"].append(cql.item())
    return q_net, history


@app.cell(hide_code=True)
def _():
    checks.run(3, cql_penalty=cql_penalty, train_offline_q=train_offline_q)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `cql_penalty` is part of `mlfs.deeprl`. open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
