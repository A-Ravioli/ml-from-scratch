"""Checks for exercise.py (actor-critic).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. GAE and the A2C loss
are checked exactly by hand; the rollout collector is checked step by step on scripted
environments with a scripted model (including an episode cut off by a time limit); learning is
checked on tiny tasks with known optima over several seeds. Environments are wrapped so a loop
that never ends fails after a fixed number of steps.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal

from mlfs.rl import Discrete, Env
from mlfs.testing import bounded

torch.set_num_threads(1)  # tiny networks: one thread is fastest, and avoids oversubscribing a busy machine


class _Capped:
    """wraps an env; fails the test after max_steps calls to step."""

    def __init__(self, env, max_steps):
        self.env, self.max_steps, self.steps = env, max_steps, 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        self.steps += 1
        assert self.steps <= self.max_steps, (
            f"more than {self.max_steps:,} env.step calls: does your loop stop after total_steps environment steps?")
        return self.env.step(action)


class _BoxSpace:
    def __init__(self, n):
        self.shape = (n,)


class _Counter(Env):
    """obs [t]; reward 1 per step; terminates (kind 'term') or is truncated (kind 'trunc') at step `length`."""

    def __init__(self, kind, length):
        self.kind, self.length = kind, length
        self.observation_space, self.action_space = _BoxSpace(1), Discrete(2)
        self.seeds = []

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.t = 0
        return np.array([0.0]), {}

    def step(self, action):
        assert action == 0, "the scripted model always picks action 0"
        self.t += 1
        end = self.t >= self.length
        return np.array([float(self.t)]), 1.0, end and self.kind == "term", end and self.kind == "trunc", {}


class _ScriptedModel(nn.Module):
    """always action 0 (probability 1); V(obs) = 10 * obs[0]."""

    def forward(self, obs):
        logits = torch.full((obs.shape[0], 2), -1e9)
        logits[:, 0] = 0.0
        return Categorical(logits=logits), 10.0 * obs[:, 0]


class _Contextual(Env):
    """one-step episodes: obs is a random one-hot context in {0, 1}; action == context pays 1, else 0."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(2), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.c = int(self.np_random.integers(2))
        return np.eye(2)[self.c], {}

    def step(self, action):
        return np.eye(2)[self.c], float(action == self.c), True, False, {}


class _Delayed(Env):
    """6 steps (one-hot step obs); the FIRST action decides a reward of 1 that only arrives at the last step."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(6), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.t, self.first = 0, None
        return np.eye(6)[0], {}

    def step(self, action):
        if self.t == 0:
            self.first = action
        self.t += 1
        done = self.t >= 6
        return np.eye(6)[min(self.t, 5)], float(done and self.first == 1), done, False, {}


class _Target(Env):
    """continuous one-step task: obs a random target x in [-1, 1]; reward -(a - x)^2."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(1), _BoxSpace(1)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.x = float(self.np_random.uniform(-1, 1))
        return np.array([self.x]), {}

    def step(self, action):
        a = float(np.asarray(action).reshape(-1)[0])
        return np.array([self.x]), -((a - self.x) ** 2), True, False, {}


def _lp(dist, a):
    lp = dist.log_prob(a)
    return lp.sum(-1) if lp.ndim == 2 else lp


# ---------------------------------------------------------------- stage 1: GAE


def test_stage1_hand_example(nb):
    r = torch.tensor([1.0, 0.0, 2.0])
    v = torch.tensor([0.5, 1.0, -1.0])
    # deltas with gamma 0.9: 1 + 0.9 * 1 - 0.5 = 1.4;  0 + 0.9 * -1 - 1 = -1.9;  2 + 0.9 * 3 - (-1) = 5.7
    # lambda 0.5 (gamma lambda = 0.45): A2 = 5.7, A1 = -1.9 + 0.45 * 5.7 = 0.665, A0 = 1.4 + 0.45 * 0.665 = 1.69925
    adv, ret = nb.compute_gae(r, v, torch.zeros(3), torch.tensor(3.0), gamma=0.9, lam=0.5)
    assert torch.allclose(adv, torch.tensor([1.69925, 0.665, 5.7]), atol=1e-5), (
        f"deltas (1.4, -1.9, 5.7) (the last bootstraps from last_value = 3), then A_t = delta_t + 0.45 A_t+1: "
        f"expected (1.69925, 0.665, 5.7); got {adv.tolist()}")
    assert torch.allclose(ret, adv + v), "returns = advantages + values"


def test_stage1_lambda_limits(nb):
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    v = torch.tensor([0.5, -0.5, 1.0, 2.0])
    term = torch.tensor([0.0, 0.0, 0.0, 1.0])
    adv0, _ = nb.compute_gae(r, v, term, torch.tensor(7.0), gamma=0.5, lam=0.0)
    delta = torch.tensor([1 + 0.5 * -0.5 - 0.5, 2 + 0.5 * 1 + 0.5, 3 + 0.5 * 2 - 1, 4 - 2.0])
    assert torch.allclose(adv0, delta), (
        f"lambda = 0: the advantage is the one-step TD error delta_t; the last step terminated, so last_value (7) is "
        f"NOT used: expected {delta.tolist()}, got {adv0.tolist()}")
    adv1, ret1 = nb.compute_gae(r, v, term, torch.tensor(7.0), gamma=0.5, lam=1.0)
    G = torch.tensor([1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 4, 2 + 0.5 * 3 + 0.25 * 4, 3 + 0.5 * 4, 4.0])
    assert torch.allclose(ret1, G), (
        f"lambda = 1 on an episode that terminates: returns are the Monte Carlo returns {G.tolist()}; "
        f"got {ret1.tolist()}")


def test_stage1_termination_cuts_both(nb):
    r = torch.tensor([1.0, 1.0, 1.0])
    v = torch.tensor([2.0, 3.0, 4.0])
    adv, _ = nb.compute_gae(r, v, torch.tensor([0.0, 1.0, 0.0]), torch.tensor(5.0), gamma=1.0, lam=1.0)
    # step 2 bootstraps from last_value: delta 1 + 5 - 4 = 2.  step 1 ended an episode: 1 - 3 = -2 (no V(s_2), no A_2)
    # step 0: 1 + 3 - 2 = 2, plus A_1 = -2 -> 0
    assert torch.allclose(adv, torch.tensor([0.0, -2.0, 2.0])), (
        f"terminated[1] = 1: step 1's target is r_1 alone, and the accumulation from step 2 (a new episode) stops "
        f"there too. expected (0, -2, 2); got {adv.tolist()}")


def test_stage1_batched_and_constant(nb):
    torch.manual_seed(0)
    r, v = torch.randn(6, 3), torch.randn(6, 3, requires_grad=True)
    term = (torch.rand(6, 3) < 0.3).float()
    last = torch.randn(3)
    adv, ret = nb.compute_gae(r, v, term, last, gamma=0.9, lam=0.8)
    assert adv.shape == (6, 3) and ret.shape == (6, 3), "(T, N) inputs give (T, N) advantages and returns"
    for n in range(3):
        a1, _ = nb.compute_gae(r[:, n], v[:, n].detach(), term[:, n], last[n], gamma=0.9, lam=0.8)
        assert torch.allclose(adv[:, n], a1, atol=1e-6), f"column {n} of a (T, N) batch must equal the (T,) result"
    assert not adv.requires_grad and not ret.requires_grad, (
        "advantages and returns are regression targets: compute them without an autograd graph (torch.no_grad or "
        "detach the values)")


# ---------------------------------------------------------------- stage 2: actor-critic module


def test_stage2_discrete(nb):
    torch.manual_seed(1)
    m = nb.ActorCritic(4, 3, hidden_sizes=(16,))
    assert isinstance(m, nn.Module), "ActorCritic is an nn.Module"
    dist, value = m(torch.randn(5, 4))
    assert isinstance(dist, Categorical) and dist.batch_shape == (5,), "discrete: a Categorical with batch shape (B,)"
    assert value.shape == (5,), f"values have shape (B,), not (B, 1); got {tuple(value.shape)}"
    (dist.log_prob(torch.tensor([0, 1, 2, 0, 1])).sum() + value.sum()).backward()
    assert all(p.grad is not None for p in m.parameters()), "the policy and value outputs must reach every parameter"
    obs = np.array([0.1, -0.2, 0.3, 0.0])
    a, lp, v = m.act(obs)
    assert type(a) is int and isinstance(lp, float) and isinstance(v, float), "act returns (int, float, float)"
    d, val = m(torch.as_tensor(obs, dtype=torch.float32)[None])
    assert abs(lp - d.log_prob(torch.tensor([a])).item()) < 1e-5 and abs(v - val.item()) < 1e-5, (
        "act's log_prob and value must be those of the model's forward pass for that observation and action")


def test_stage2_continuous(nb):
    torch.manual_seed(2)
    m = nb.ActorCritic(3, 2, hidden_sizes=(16,), continuous=True)
    obs = torch.randn(5, 3)
    dist, value = m(obs)
    assert value.shape == (5,), "values have shape (B,)"
    a = dist.sample()
    assert a.shape == (5, 2), f"actions have shape (B, act_dim) = (5, 2); got {tuple(a.shape)}"
    lp = dist.log_prob(a)
    assert lp.shape == (5,), (
        f"log_prob must sum over the action dimensions and give (B,) (wrap the Normal in "
        f"torch.distributions.Independent(..., 1)); got shape {tuple(lp.shape)}")
    assert torch.allclose(dist.stddev, torch.ones(5, 2)), "the state-independent log_std starts at 0 (std 1)"
    assert any(p.shape == (2,) for p in m.parameters()), "log_std is a learnable parameter of shape (act_dim,)"
    act, alp, v = m.act(np.array([0.5, 0.5, -1.0]))
    assert isinstance(act, np.ndarray) and act.shape == (2,), "continuous act returns an ndarray action (act_dim,)"
    d, _ = m(torch.tensor([[0.5, 0.5, -1.0]]))
    assert abs(alp - _lp(d, torch.as_tensor(act, dtype=torch.float32)[None]).item()) < 1e-4, (
        "act's log_prob must match the distribution's log-density of the returned action (summed over dimensions)")


# ---------------------------------------------------------------- stage 3: rollouts


def test_stage3_rollout_by_hand(nb):
    envs = [_Counter("term", 2), _Counter("trunc", 3)]
    col = nb.RolloutCollector([_Capped(e, 4) for e in envs], _ScriptedModel(), gamma=0.9, seed=0)
    b = col.collect(4)
    assert set(b) >= {"obs", "actions", "log_probs", "values", "rewards", "dones", "last_value"}, f"keys: {sorted(b)}"
    assert b["obs"].shape == (4, 2, 1) and b["rewards"].shape == (4, 2) and b["last_value"].shape == (2,), (
        "obs (T, N, obs_dim); actions, log_probs, values, rewards, dones (T, N); last_value (N,)")
    assert torch.allclose(b["obs"][:, :, 0], torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [1.0, 0.0]])), (
        f"obs[t] is the observation the action was taken in; after an episode ends the env is reset (obs 0). "
        f"got {b['obs'][:, :, 0].tolist()}")
    assert torch.allclose(b["values"], 10 * b["obs"][:, :, 0]), "values[t] = V(obs[t])"
    want_r = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0 + 0.9 * 30.0], [1.0, 1.0]])
    assert torch.allclose(b["rewards"], want_r), (
        f"env 0 TERMINATES at its 2nd step: reward 1, nothing added. env 1 is TRUNCATED at its 3rd step in a state "
        f"with V = 30: its reward becomes 1 + gamma V(s_cut) = 1 + 0.9 * 30 = 28 (so GAE can treat the step as an "
        f"episode end without losing the future). expected {want_r.tolist()}; got {b['rewards'].tolist()}")
    assert torch.allclose(b["dones"], torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])), (
        f"dones (the flag that cuts GAE) is 1 when an episode ended for either reason; got {b['dones'].tolist()}")
    assert torch.allclose(b["last_value"], torch.tensor([0.0, 10.0])), (
        f"last_value is V of each env's observation after the last step (env 0 was just reset: 0; env 1 is at 1: 10); "
        f"got {b['last_value'].tolist()}")
    assert col.episode_returns == [2.0, 3.0, 2.0], (
        f"episode_returns: undiscounted, WITHOUT the bootstrap, in the order episodes finished (env index breaks "
        f"ties); expected [2, 3, 2], got {col.episode_returns}")
    assert col.episode_ends == [4, 6, 8], (
        f"episode_ends counts ENVIRONMENT steps over all N envs (N = 2 per lockstep), so the episodes ending at "
        f"locksteps 2, 3 and 4 end at env steps 4, 6, 8 (the x-axis of every learning curve); got {col.episode_ends}")
    assert envs[0].seeds[0] is not None and envs[0].seeds[0] != envs[1].seeds[0] and envs[0].seeds[1] is None, (
        "seed each env once, with its own seed from default_rng(seed); later resets take no seed")


def test_stage3_continues_across_calls(nb):
    envs = [_Counter("term", 3)]
    col = nb.RolloutCollector(envs, _ScriptedModel(), gamma=0.9, seed=0)
    b1, b2 = col.collect(2), col.collect(2)
    assert b1["obs"][:, 0, 0].tolist() == [0.0, 1.0] and b2["obs"][:, 0, 0].tolist() == [2.0, 0.0], (
        "a second collect() continues the running episodes where the first stopped (don't reset between calls)")
    assert col.episode_returns == [3.0], "the episode that spans two collect() calls counts once, with its full return"


def test_stage3_gae_on_a_rollout(nb):
    col = nb.RolloutCollector([_Counter("trunc", 3)], _ScriptedModel(), gamma=0.5, seed=0)
    b = col.collect(3)
    adv, ret = nb.compute_gae(b["rewards"], b["values"], b["dones"], b["last_value"], gamma=0.5, lam=1.0)
    # the episode is cut at V(s_3) = 30: the right returns are 1 + .5 + .25 + .125 * 30 = 5.5, 1 + .5 + .25 * 30 = 9, 1 + 15 = 16
    assert torch.allclose(ret[:, 0], torch.tensor([5.5, 9.0, 16.0])), (
        f"with the bootstrap folded into the truncated step's reward, GAE (lambda 1) gives the returns of the "
        f"untruncated problem: (5.5, 9, 16); got {ret[:, 0].tolist()}")


# ---------------------------------------------------------------- stage 4: A2C


def test_stage4_a2c_loss_by_hand(nb):
    logits = torch.tensor([[0.0, 0.0], [1.0, -1.0]], requires_grad=True)
    dist = Categorical(logits=logits)
    actions = torch.tensor([0, 1])
    adv = torch.tensor([2.0, -1.0])
    values = torch.tensor([0.5, 1.5], requires_grad=True)
    returns = torch.tensor([1.5, 1.0])
    loss, info = nb.a2c_loss(dist, actions, adv, returns, values, vf_coef=0.5, ent_coef=0.1)
    lp = torch.log_softmax(logits.detach(), 1)[[0, 1], [0, 1]]
    ent = -(torch.softmax(logits.detach(), 1) * torch.log_softmax(logits.detach(), 1)).sum(1).mean()
    pg = -(adv * lp).mean()
    vl = ((values.detach() - returns) ** 2).mean()
    want = pg + 0.5 * vl - 0.1 * ent
    assert abs(loss.item() - want.item()) < 1e-5, (
        f"loss = -mean(A log pi) + vf_coef * mean((V - R)^2) - ent_coef * mean(entropy) = {want.item():.5f}; got "
        f"{loss.item():.5f}")
    assert set(info) >= {"policy_loss", "value_loss", "entropy"} and all(isinstance(x, float) for x in info.values()), (
        "info holds floats 'policy_loss', 'value_loss', 'entropy'")
    assert abs(info["value_loss"] - vl.item()) < 1e-6 and abs(info["entropy"] - ent.item()) < 1e-6, (
        "info's value_loss is the MSE (without vf_coef) and entropy the mean entropy")
    loss.backward()
    assert torch.allclose(values.grad, 0.5 * 2 * (values.detach() - returns) / 2), "d loss / dV = vf_coef * 2 (V - R) / B"

    a_req = torch.tensor([2.0, -1.0], requires_grad=True)
    nb.a2c_loss(Categorical(logits=torch.zeros(2, 2, requires_grad=True)), actions, a_req, returns,
                values.detach())[0].backward()
    assert a_req.grad is None or torch.all(a_req.grad == 0), "advantages are constants in the policy loss (detach them)"

    cont = Independent(Normal(torch.zeros(2, 3), torch.ones(2, 3)), 1)
    loss_c, _ = nb.a2c_loss(cont, torch.zeros(2, 3), adv, returns, values.detach(), vf_coef=0.0, ent_coef=0.0)
    assert abs(loss_c.item() - (-(adv * (-1.5 * np.log(2 * np.pi))).mean().item())) < 1e-4, (
        "for a continuous policy, the log-probability of an action is the distribution's log_prob (already summed over "
        "action dimensions by Independent)")


def _train(nb, make, total_steps, **kw):
    capped = []

    def make_env():
        e = _Capped(make(), total_steps + 200)
        capped.append(e)
        return e

    out = bounded(nb.train_a2c, make_env, total_steps, budget=40_000_000, what="train_a2c",
                  hint="it should run about total_steps environment steps.", **kw)
    steps = sum(e.steps for e in capped)
    assert steps <= total_steps + 64, f"train_a2c took {steps} env steps for total_steps={total_steps}"
    return out


def test_stage4_interface_and_reproducibility(nb):
    kw = dict(n_envs=4, n_steps=4, lr=1e-2, hidden_sizes=(16,))
    torch.manual_seed(5)
    m1, h1 = _train(nb, _Contextual, 320, seed=0, **kw)
    torch.manual_seed(6)
    m2, h2 = _train(nb, _Contextual, 320, seed=0, **kw)
    assert isinstance(m1, nn.Module) and len(h1["episode_returns"]) == 320, (
        "train_a2c returns (model, history); one-step episodes: 320 steps finish 320 episodes")
    assert h1["episode_returns"] == h2["episode_returns"] and all(
        torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values())), (
        "the same seed must give the same run whatever torch's global RNG state was")
    _, h3 = _train(nb, _Contextual, 320, seed=1, **kw)
    assert h3["episode_returns"] != h1["episode_returns"], "different seeds should give different runs"


def test_stage4_learns_discrete(nb):
    finals = []
    for seed in range(5):
        _, h = _train(nb, _Contextual, 2000, n_envs=4, n_steps=5, lr=1e-2, hidden_sizes=(16,), seed=seed)
        finals.append(np.mean(h["episode_returns"][-200:]))
    assert np.median(finals) > 0.85, (
        f"A2C on a one-step contextual task (reward 1 for matching the context): success over the last 200 of 2,000 "
        f"episodes should approach 1 (median of 5 seeds > 0.85); got {np.round(finals, 2).tolist()}")


def test_stage4_learns_continuous(nb):
    finals = []
    for seed in range(5):
        _, h = _train(nb, _Target, 3000, n_envs=4, n_steps=5, lr=1e-2, ent_coef=0.0, hidden_sizes=(16,),
                      continuous=True, seed=seed)
        finals.append(np.mean(h["episode_returns"][-300:]))
    assert np.median(finals) > -0.4, (
        f"continuous one-step task, reward -(a - x)^2 for a target x in [-1, 1]: a Gaussian policy starts with "
        f"std 1 (expected reward about -1.3) and should learn mean ~ x and a smaller std. median return over the last "
        f"300 episodes: {np.median(finals):.2f} (want > -0.4); per seed {np.round(finals, 2).tolist()}")


def test_stage4_gae_carries_a_delayed_reward(nb):
    finals = []
    for seed in range(5):
        _, h = _train(nb, _Delayed, 1200, n_envs=4, n_steps=6, lam=0.95, gamma=1.0, lr=1e-2, ent_coef=0.0,
                      hidden_sizes=(16,), seed=seed)
        finals.append(np.mean(h["episode_returns"][-40:]))
    assert np.median(finals) > 0.9, (
        f"the first action earns a reward 5 steps later. with lam = 0.95 the GAE advantage of that action contains the "
        f"real reward right away, so A2C should pick the right first action within 1,200 steps (median success over "
        f"the last 40 episodes > 0.9, 5 seeds); got {np.round(finals, 2).tolist()}. (about 0.6 means the advantages "
        f"don't use `lam`: pass it to compute_gae)")


def test_stage4_gradient_clipping(nb):
    def flat(m):
        return torch.cat([p.detach().reshape(-1) for p in m.parameters()])

    kw = dict(n_envs=4, n_steps=4, lr=1e-2, hidden_sizes=(16,), seed=0)
    a, _ = _train(nb, _Contextual, 16, max_grad_norm=1e-12, **kw)
    b, _ = _train(nb, _Contextual, 160, max_grad_norm=1e-12, **kw)
    c, _ = _train(nb, _Contextual, 160, max_grad_norm=0.5, **kw)
    assert (flat(a) - flat(b)).abs().max() < 1e-4, (
        "with max_grad_norm = 1e-12 the clipped gradients are far below Adam's epsilon, so 10 updates should barely "
        "move the model compared with 1. yours moved: clip the global gradient norm with "
        "nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) between backward() and step()")
    assert (flat(a) - flat(c)).abs().max() > 1e-3, "with max_grad_norm = 0.5 the model should train"
