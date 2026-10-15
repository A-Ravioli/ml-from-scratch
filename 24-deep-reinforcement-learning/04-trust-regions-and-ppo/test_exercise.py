"""Checks for exercise.py (trust regions and PPO).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The clipped loss,
explained variance and the trust-region step are checked exactly by hand or against explicit
matrices; the PPO update is checked against a hand-computed first step (where PPO equals the
vanilla policy gradient); learning is checked on tiny tasks over several seeds. Environments are
wrapped so a loop that never ends fails after a fixed number of steps.
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

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


class _TinyAC(nn.Module):
    """linear actor (3 actions) and linear critic on 3-D observations."""

    def __init__(self):
        super().__init__()
        self.actor, self.critic = nn.Linear(3, 3), nn.Linear(3, 1)

    def forward(self, obs):
        return Categorical(logits=self.actor(obs)), self.critic(obs).squeeze(-1)


def _batch(model, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, 3, generator=g)
    with torch.no_grad():
        dist, values = model(obs)
        actions = torch.multinomial(dist.probs, 1, generator=g).squeeze(1)
        log_probs = dist.log_prob(actions)
    return {"obs": obs, "actions": actions, "log_probs": log_probs, "values": values,
            "advantages": torch.randn(n, generator=g), "returns": torch.randn(n, generator=g)}


# ---------------------------------------------------------------- stage 1: the clipped loss


def test_stage1_hand_example(nb):
    r = torch.tensor([1.5, 0.5, 0.5, 1.5, 1.1])
    adv = torch.tensor([2.0, 1.0, -1.0, -2.0, 1.0])
    old = torch.tensor([-1.0, -0.5, -2.0, -0.3, -1.2], requires_grad=True)
    lp = (old.detach() + r.log()).clone().requires_grad_(True)
    loss, info = nb.ppo_clip_loss(lp, old, adv, clip_eps=0.2)
    # min(rA, clip(r)A): min(3, 2.4) = 2.4;  min(.5, .8) = .5;  min(-.5, -.8) = -.8;  min(-3, -2.4) = -3;  1.1
    assert loss.ndim == 0 and abs(loss.item() - (-0.04)) < 1e-5, (
        f"surrogates (2.4, 0.5, -0.8, -3, 1.1): the min picks the clipped term when it's SMALLER (more pessimistic). "
        f"loss = -mean = -0.04; got {loss.item():.5f}")
    loss.backward()
    want = torch.tensor([0.0, -0.1, 0.0, 0.6, -0.22])
    assert torch.allclose(lp.grad, want, atol=1e-5), (
        f"d loss / d log_prob: 0 where the clipped term is active (samples 1 and 3), -r A / n elsewhere: "
        f"expected {want.tolist()}, got {lp.grad.tolist()}")
    assert old.grad is None, "old_log_probs are constants: no gradient may flow into them"
    assert set(info) >= {"clip_frac", "approx_kl"} and all(isinstance(v, float) for v in info.values()), (
        "info has Python floats 'clip_frac' and 'approx_kl'")
    assert abs(info["clip_frac"] - 0.8) < 1e-6, (
        f"clip_frac: the fraction of samples with |r - 1| > clip_eps (4 of 5); got {info['clip_frac']}")
    kl = float(((r - 1) - r.log()).mean())
    assert abs(info["approx_kl"] - kl) < 1e-5, (
        f"approx_kl: mean((r - 1) - log r) = {kl:.5f} (the low-variance estimator of KL(old || new)); got "
        f"{info['approx_kl']:.5f}")


def test_stage1_ratio_one_is_the_policy_gradient(nb):
    lp = torch.tensor([-0.7, -1.3, -0.2], requires_grad=True)
    adv = torch.tensor([1.0, -2.0, 0.5])
    loss, info = nb.ppo_clip_loss(lp, lp.detach(), adv, clip_eps=0.1)
    loss.backward()
    assert torch.allclose(lp.grad, -adv / 3) and abs(loss.item() + adv.mean().item()) < 1e-6, (
        "at the old policy (r = 1) the clipped objective's value is -mean(A) and its gradient is the vanilla policy "
        f"gradient -A / n; got loss {loss.item():.4f}, grad {lp.grad.tolist()}")
    assert info["clip_frac"] == 0.0 and abs(info["approx_kl"]) < 1e-7, "at r = 1 nothing is clipped and KL is 0"


# ---------------------------------------------------------------- stage 2: explained variance


def test_stage2_explained_variance(nb):
    r = np.array([1.0, 2.0, 3.0, 6.0])
    assert abs(nb.explained_variance(r, r) - 1.0) < 1e-9, "a perfect critic explains all the variance: 1"
    assert abs(nb.explained_variance(np.full(4, r.mean()), r)) < 1e-9, "predicting the mean explains nothing: 0"
    assert abs(nb.explained_variance(-r, r) - (1 - np.var(2 * r) / np.var(r))) < 1e-9, (
        "a critic worse than the mean gives a negative value: here 1 - Var(2G) / Var(G) = -3")
    got = nb.explained_variance(torch.tensor([0.5, 2.5, 2.0, 5.0]), torch.tensor(r))
    want = 1 - np.var(r - np.array([0.5, 2.5, 2.0, 5.0])) / np.var(r)
    assert isinstance(got, float) and abs(got - want) < 1e-6, f"tensors work too and give a float: {want:.4f}, got {got}"
    assert np.isnan(nb.explained_variance(np.array([1.0, 2.0]), np.array([3.0, 3.0]))), (
        "constant returns have zero variance: return nan rather than dividing by zero")


# ---------------------------------------------------------------- stage 3: the trust-region step


def test_stage3_conjugate_gradient(nb):
    g = torch.Generator().manual_seed(0)
    M = torch.randn(6, 6, generator=g, dtype=torch.float64)
    A = M @ M.T + 0.5 * torch.eye(6, dtype=torch.float64)
    b = torch.randn(6, generator=g, dtype=torch.float64)
    calls = []

    def Avp(v):
        calls.append(1)
        assert len(calls) <= 50, "conjugate_gradient called Avp more than 50 times: does it stop after n_iters?"
        return A @ v

    x = bounded(nb.conjugate_gradient, Avp, b, n_iters=6, tol=1e-20, what="conjugate_gradient")
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-6), (
        "in exact arithmetic CG solves an n x n SPD system in n iterations: with n_iters=6 on a 6 x 6 system x "
        f"should match torch.linalg.solve; max error {float((x - torch.linalg.solve(A, b)).abs().max()):.2e}")
    assert len(calls) <= 7, f"n_iters=6 should need at most 7 matrix-vector products; used {len(calls)}"
    calls.clear()
    x2 = nb.conjugate_gradient(Avp, b, n_iters=2)
    assert len(calls) <= 3, f"n_iters=2 should need at most 3 matrix-vector products; used {len(calls)}"
    xs = torch.linalg.solve(A, b)
    e0, e2 = xs @ A @ xs, (x2 - xs) @ A @ (x2 - xs)
    assert e2 < e0, (
        "two CG iterations must reduce the A-norm of the error, (x - x*)^T A (x - x*), below its value at x = 0 (CG "
        "minimizes exactly that over a growing subspace)")
    z = nb.conjugate_gradient(Avp, torch.zeros(6, dtype=torch.float64), n_iters=5)
    assert torch.all(z == 0), "for b = 0 the solution is 0 (and the loop must not divide by zero)"


def test_stage3_fisher_vector_product(nb):
    g = torch.Generator().manual_seed(1)
    obs = torch.randn(4, 2, generator=g, dtype=torch.float64)
    W = torch.randn(2, 3, generator=g, dtype=torch.float64, requires_grad=True)
    v = torch.randn(6, generator=g, dtype=torch.float64)
    got = nb.fisher_vector_product(lambda: Categorical(logits=obs @ W), [W], v)
    F = torch.zeros(6, 6, dtype=torch.float64)  # F = mean_s sum_a pi(a|s) g g^T, g = d log pi(a|s) / dW
    with torch.no_grad():
        P = torch.softmax(obs @ W, 1)
    for s in range(4):
        for a in range(3):
            gw = torch.outer(obs[s], torch.eye(3, dtype=torch.float64)[a] - P[s]).reshape(-1)
            F += P[s, a] * torch.outer(gw, gw) / 4
    assert got.shape == (6,) and torch.allclose(got, F @ v, atol=1e-8), (
        f"the Hessian of the mean KL(old || new) at the old parameters is the Fisher information F; the product F v "
        f"(computed by differentiating (grad KL) . v) should match the explicit F @ v. max error "
        f"{float((got - F @ v).abs().max()):.2e}")
    damped = nb.fisher_vector_product(lambda: Categorical(logits=obs @ W), [W], v, damping=0.1)
    assert torch.allclose(damped, F @ v + 0.1 * v, atol=1e-8), "damping adds damping * v"


def test_stage3_natural_gradient_step(nb):
    F = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    gvec = torch.tensor([1.0, -1.0], dtype=torch.float64)
    s = nb.natural_gradient_step(gvec, lambda v: F @ v, 0.01, cg_iters=10)
    nat = torch.linalg.solve(F, gvec)
    want = torch.sqrt(2 * 0.01 / (gvec @ nat)) * nat
    assert torch.allclose(s, want, atol=1e-8), (
        f"the natural-gradient step is sqrt(2 delta / g^T F^-1 g) F^-1 g = {want.tolist()}; got {s.tolist()}")
    assert abs(0.5 * float(s @ F @ s) - 0.01) < 1e-8, "its quadratic KL estimate 0.5 s^T F s must equal max_kl"


# ---------------------------------------------------------------- stage 4: the PPO update


def _sgd_expected(model, batch, lr, *, norm_adv=False, vf_coef=0.5):
    m = copy.deepcopy(model)
    dist, values = m(batch["obs"])
    adv = batch["advantages"]
    if norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    loss = -(adv * dist.log_prob(batch["actions"])).mean() + vf_coef * ((values - batch["returns"]) ** 2).mean()
    m.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in m.parameters():
            p -= lr * p.grad
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


def _flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def test_stage4_first_step_is_the_policy_gradient(nb):
    torch.manual_seed(0)
    model = _TinyAC()
    batch = _batch(model)
    want = _sgd_expected(model, batch, 0.1)
    before = _flat(model)
    stats = nb.ppo_update(model, torch.optim.SGD(model.parameters(), lr=0.1), batch, n_epochs=1, minibatch_size=16,
                          vf_coef=0.5, ent_coef=0.0, max_grad_norm=None, norm_adv=False, seed=0)
    got = _flat(model)
    assert torch.allclose(got, want, atol=1e-6), (
        f"one epoch, one full-batch minibatch, plain SGD: at the old policy the ratio is 1, so the step must equal "
        f"the vanilla policy gradient plus 0.5 * the value MSE gradient. max parameter error "
        f"{float((got - want).abs().max()):.2e} (moved {float((got - before).abs().max()):.2e}). use the batch's "
        f"stored log_probs as the old log-probabilities, and the MSE (no factor 1/2) for the value loss")
    assert isinstance(stats, dict) and stats.get("n_updates") == 1, "stats['n_updates'] counts optimizer steps (1 here)"
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_variance"):
        assert key in stats and isinstance(stats[key], float), f"stats needs a float '{key}'"


def test_stage4_advantage_normalization(nb):
    torch.manual_seed(1)
    model = _TinyAC()
    batch = _batch(model, seed=1)
    batch["advantages"] = batch["advantages"] * 50 + 20
    want = _sgd_expected(model, batch, 0.05, norm_adv=True, vf_coef=0.0) - _flat(model)
    before = _flat(model)
    nb.ppo_update(model, torch.optim.SGD(model.parameters(), lr=0.05), batch, n_epochs=1, minibatch_size=16,
                  vf_coef=0.0, ent_coef=0.0, max_grad_norm=None, norm_adv=True, seed=0)
    got = _flat(model) - before
    cos = float(got @ want / (got.norm() * want.norm()))
    assert cos > 0.999 and 0.9 < float(got.norm() / want.norm()) < 1.1, (
        f"with norm_adv=True the minibatch's advantages are standardized (mean 0, std 1) before the loss, so "
        f"advantages scaled by 50 and shifted by 20 give the same step as standardized ones: cosine {cos:.4f}, "
        f"norm ratio {float(got.norm() / want.norm()):.3f}")


def test_stage4_epochs_minibatches_and_early_stop(nb):
    torch.manual_seed(2)
    model = _TinyAC()
    batch = _batch(model, n=32, seed=2)
    stats = bounded(nb.ppo_update, model, torch.optim.Adam(model.parameters(), lr=0.05), batch, n_epochs=3,
                    minibatch_size=8, seed=0, what="ppo_update")
    assert stats["n_updates"] == 12, f"3 epochs of 32 samples in minibatches of 8 is 12 steps; got {stats['n_updates']}"
    model = _TinyAC()
    batch = _batch(model, n=32, seed=3)
    stats = nb.ppo_update(model, torch.optim.Adam(model.parameters(), lr=0.5), batch, n_epochs=20, minibatch_size=8,
                          max_grad_norm=None, seed=0)
    assert stats["clip_frac"] > 0.05, (
        f"20 epochs at a large step size move the policy far from the old one, so many ratios leave [0.8, 1.2]: "
        f"mean clip_frac should be well above 0; got {stats['clip_frac']:.3f}. are you computing the ratio against "
        f"the batch's OLD log-probabilities (not recomputing them each epoch)?")
    model = _TinyAC()
    batch = _batch(model, n=32, seed=4)
    stats = nb.ppo_update(model, torch.optim.Adam(model.parameters(), lr=0.05), batch, n_epochs=10,
                          minibatch_size=8, target_kl=1e-12, seed=0)
    assert stats["n_updates"] == 4, (
        f"with target_kl tiny, the first epoch's mean approx_kl exceeds it and the update stops after that epoch "
        f"(4 minibatch steps); got {stats['n_updates']}")


# ---------------------------------------------------------------- stage 5: PPO


def _train(nb, make, total_steps, **kw):
    capped = []

    def make_env():
        e = _Capped(make(), total_steps + 200)
        capped.append(e)
        return e

    out = bounded(nb.train_ppo, make_env, total_steps, budget=60_000_000, what="train_ppo",
                  hint="it should run about total_steps environment steps.", **kw)
    steps = sum(e.steps for e in capped)
    assert steps <= total_steps + 64, f"train_ppo took {steps} env steps for total_steps={total_steps}"
    return out


SMALL = dict(n_envs=4, n_steps=16, minibatch_size=32, n_epochs=4, hidden_sizes=(16,))


def test_stage5_interface_and_annealing(nb):
    torch.manual_seed(5)
    m1, h1 = _train(nb, _Contextual, 640, lr=1e-2, seed=0, **SMALL)
    torch.manual_seed(6)
    m2, h2 = _train(nb, _Contextual, 640, lr=1e-2, seed=0, **SMALL)
    assert isinstance(m1, nn.Module) and len(h1["episode_returns"]) == 640, (
        "train_ppo returns (model, history); one-step episodes: 640 steps finish 640 episodes")
    assert h1["episode_returns"] == h2["episode_returns"] and all(
        torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values())), (
        "the same seed must give the same run whatever torch's global RNG state was")
    assert np.allclose(h1["lr"], [1e-2 * (1 - k / 10) for k in range(10)]), (
        f"anneal_lr: 640 / (4 * 16) = 10 updates, update k uses lr * (1 - k / 10); got {np.round(h1['lr'], 5)}")
    _, h3 = _train(nb, _Contextual, 640, lr=1e-2, anneal_lr=False, seed=0, **SMALL)
    assert np.allclose(h3["lr"], 1e-2), "anneal_lr=False keeps the learning rate constant"
    for key in ("approx_kl", "clip_frac", "explained_variance"):
        assert len(h1[key]) == 10, f"history['{key}'] has one entry per update"


def test_stage5_learns_discrete(nb):
    finals = []
    for seed in range(5):
        _, h = _train(nb, _Contextual, 1600, lr=1e-2, seed=seed, **SMALL)
        finals.append(np.mean(h["episode_returns"][-200:]))
    assert np.median(finals) > 0.85, (
        f"PPO on a one-step contextual task: success over the last 200 of 1,600 episodes should approach 1 (median "
        f"of 5 seeds > 0.85); got {np.round(finals, 2).tolist()}")


def test_stage5_learns_continuous(nb):
    finals = []
    for seed in range(5):
        _, h = _train(nb, _Target, 2400, lr=1e-2, continuous=True, seed=seed, **SMALL)
        finals.append(np.mean(h["episode_returns"][-300:]))
    assert np.median(finals) > -0.4, (
        f"continuous one-step task, reward -(a - x)^2: the Gaussian policy should learn mean ~ x and shrink its std. "
        f"median return over the last 300 episodes: {np.median(finals):.2f} (want > -0.4); per seed "
        f"{np.round(finals, 2).tolist()}")


def test_stage5_gae_carries_a_delayed_reward(nb):
    finals = []
    for seed in range(5):
        _, h = _train(nb, _Delayed, 1200, n_envs=4, n_steps=6, minibatch_size=24, n_epochs=4, lam=0.95, gamma=1.0,
                      lr=1e-2, hidden_sizes=(16,), seed=seed)
        finals.append(np.mean(h["episode_returns"][-40:]))
    assert np.median(finals) > 0.9, (
        f"the first action earns a reward 5 steps later. with lam = 0.95 the GAE advantage of that action contains the "
        f"real reward right away, so PPO should pick the right first action within 1,200 steps (median success over "
        f"the last 40 episodes > 0.9, 5 seeds); got {np.round(finals, 2).tolist()}. (check that `gamma` and `lam` "
        f"reach compute_gae)")


def test_stage4_minibatches_are_shuffled(nb):
    def run(seed):
        torch.manual_seed(3)
        model = _TinyAC()
        batch = _batch(model, n=32, seed=5)
        nb.ppo_update(model, torch.optim.SGD(model.parameters(), lr=0.5), batch, n_epochs=2, minibatch_size=8,
                      max_grad_norm=None, seed=seed)
        return _flat(model)

    assert torch.equal(run(0), run(0)), "the same seed must give the same update"
    assert not torch.allclose(run(0), run(1)), (
        "different seeds should visit the minibatches in a different order (and so end somewhere else): shuffle "
        "each epoch with np.random.default_rng(seed).permutation")
