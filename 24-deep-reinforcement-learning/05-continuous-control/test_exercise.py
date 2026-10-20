"""Checks for exercise.py (continuous control).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Pendulum is checked
against an independent implementation of the published equations; the squashed Gaussian's
log-probability is checked against a histogram of its own samples; the SAC losses are checked
exactly by hand; learning is checked on tiny tasks over several seeds. Environments are wrapped
so a loop that never ends fails after a fixed number of steps.
"""
import numpy as np
import torch
from torch import nn

from mlfs.rl import Box, Env
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
            f"more than {self.max_steps:,} env.step calls: does your loop stop after n_steps environment steps?")
        return self.env.step(action)


class _BoxSpace:
    def __init__(self, n):
        self.shape = (n,)


class _Target(Env):
    """continuous one-step task: obs a random target x in [-0.8, 0.8]; reward -(a - x)^2 for a in [-1, 1]."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(1), Box(-1.0, 1.0, (1,))

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.x = float(self.np_random.uniform(-0.8, 0.8))
        return np.array([self.x]), {}

    def step(self, action):
        a = float(np.asarray(action).reshape(-1)[0])
        return np.array([self.x]), -((a - self.x) ** 2), True, False, {}


class _Constant(Env):
    """one state (obs [1.0]), actions in [-1, 1] irrelevant, reward 1 per step, TRUNCATED after 5 steps."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(1), Box(-1.0, 1.0, (1,))

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([1.0]), {}

    def step(self, action):
        self.t += 1
        return np.array([1.0]), 1.0, False, self.t >= 5, {}


def _pendulum_step(th, thdot, u):
    """Gymnasium's Pendulum-v1: g = 10, m = l = 1, dt = 0.05, |u| <= 2, |th_dot| <= 8."""
    u = np.clip(u, -2.0, 2.0)
    norm = ((th + np.pi) % (2 * np.pi)) - np.pi
    cost = norm**2 + 0.1 * thdot**2 + 0.001 * u**2
    thdot2 = np.clip(thdot + (3 * 10.0 / 2 * np.sin(th) + 3.0 * u) * 0.05, -8.0, 8.0)
    return th + thdot2 * 0.05, thdot2, -cost


# ---------------------------------------------------------------- stage 1: Pendulum


def test_stage1_spaces_and_reset(nb):
    env = nb.Pendulum()
    assert tuple(env.observation_space.shape) == (3,) and tuple(env.action_space.shape) == (1,), (
        "obs (cos th, sin th, th_dot): shape (3,); action: one torque, shape (1,)")
    assert np.allclose(env.action_space.low, -2.0) and np.allclose(env.action_space.high, 2.0), "torque in [-2, 2]"
    assert env.max_steps == 200, "the default time limit is 200 steps (Pendulum-v1)"
    obs, info = env.reset(seed=3)
    th, thdot = env.state
    assert -np.pi <= th <= np.pi and -1 <= thdot <= 1, f"start state ~ U([-pi, pi] x [-1, 1]); got {env.state}"
    assert np.allclose(obs, [np.cos(th), np.sin(th), thdot]), "obs = (cos th, sin th, th_dot)"
    again, _ = nb.Pendulum().reset(seed=3)
    assert np.allclose(obs, again), "reset(seed=3) twice must give the same start (draw it from self.np_random)"
    got = [nb.angle_normalize(x) for x in (0.5, np.pi + 0.5, -np.pi - 0.5, 3 * np.pi)]
    assert np.allclose(got, [0.5, -np.pi + 0.5, np.pi - 0.5, -np.pi]), f"angle_normalize wraps into [-pi, pi); got {got}"


def test_stage1_dynamics_and_reward(nb):
    env = nb.Pendulum()
    env.reset(seed=0)
    for th, thdot, u in [(0.3, -0.5, 1.0), (np.pi - 0.1, 2.0, -0.7), (1.0, 7.9, 2.0), (-1.0, 0.0, 5.0), (4.0, 0.5, 0.3),
                           (2 * np.pi + 0.3, -1.0, -1.5)]:
        env.state, env.t = np.array([th, thdot]), 0
        obs, r, term, trunc, info = env.step(np.array([u]))
        th2, thdot2, r_want = _pendulum_step(th, thdot, u)
        assert np.allclose(env.state, [th2, thdot2], atol=1e-10) and np.allclose(
            obs, [np.cos(th2), np.sin(th2), thdot2], atol=1e-10), (
            f"from (th, th_dot) = ({th:.2f}, {thdot:.2f}) with torque {u}: expected state ({th2:.5f}, {thdot2:.5f}), got "
            f"{np.round(env.state, 5).tolist()}. update th_dot first (clipped to [-8, 8]), then th with the NEW th_dot; "
            f"clip the torque to [-2, 2]")
        assert type(r) is float and abs(r - r_want) < 1e-10, (
            f"reward = -(angle_normalize(th)^2 + 0.1 th_dot^2 + 0.001 u^2) at the state BEFORE the step (u clipped): "
            f"expected {r_want:.6f}, got {r}")
        assert term is False and trunc is False and isinstance(info, dict), "Pendulum never terminates"


def test_stage1_truncation(nb):
    env = nb.Pendulum(max_steps=200)
    env.reset(seed=1)
    flags = [env.step(np.array([0.0]))[2:4] for _ in range(200)]
    assert all(f == (False, False) for f in flags[:-1]) and flags[-1] == (False, True), (
        "the 200th step is truncated (time limit), never terminated")


# ---------------------------------------------------------------- stage 2: actor and critics


def test_stage2_actor_shapes_and_bounds(nb):
    torch.manual_seed(0)
    actor = nb.SquashedGaussianActor(3, 2, hidden_sizes=(32,), act_limit=2.0)
    assert isinstance(actor, nn.Module), "SquashedGaussianActor is an nn.Module"
    obs = torch.randn(64, 3)
    a, lp = actor(obs)
    assert a.shape == (64, 2) and lp.shape == (64,), f"(action (B, act_dim), log_prob (B,)); got {a.shape}, {lp.shape}"
    assert torch.all(a.abs() < 2.0), "actions lie strictly inside (-act_limit, act_limit)"
    a.sum().backward()
    assert all(p.grad is not None for p in actor.parameters()), (
        "the sampled action must be differentiable in the actor's parameters (reparameterize: mu + std * noise, "
        "e.g. Normal.rsample, not .sample)")
    d1, _ = actor(obs, deterministic=True)
    d2, _ = actor(obs, deterministic=True)
    assert torch.equal(d1, d2), "deterministic=True returns act_limit * tanh(mean), with no noise"
    n_default = sum(p.numel() for p in nb.SquashedGaussianActor(3, 1).parameters())
    assert n_default == 3 * 256 + 256 + 256 * 256 + 256 + 256 * 2 + 2, (
        "default hidden_sizes (256, 256), and the network outputs a mean and a log-std per action dimension")


def test_stage2_log_prob_is_the_density(nb):
    torch.manual_seed(1)
    actor = nb.SquashedGaussianActor(2, 1, hidden_sizes=(16,), act_limit=1.0)
    obs = torch.tensor([[0.3, -0.4]]).expand(200_000, 2)
    with torch.no_grad():
        a, lp = actor(obs)
        det, _ = actor(obs[:1], deterministic=True)
    a, dens = a[:, 0].numpy(), lp.exp().numpy()
    assert abs(np.median(a) - det.item()) < 0.02, (
        f"tanh is monotonic, so the median action is tanh(mean) = the deterministic action ({det.item():.3f}); the "
        f"samples' median is {np.median(a):.3f}")
    edges = np.linspace(-0.9, 0.9, 19)
    hist, _ = np.histogram(a, bins=edges)
    emp = hist / (len(a) * np.diff(edges))
    idx = np.digitize(a, edges) - 1
    model = np.array([dens[idx == k].mean() for k in range(len(edges) - 1)])
    ok = emp > 0.05
    err = np.max(np.abs(emp[ok] - model[ok]) / model[ok])
    assert err < 0.08, (
        f"exp(log_prob) must be the density of the returned action. compared with a histogram of 200,000 samples it "
        f"is off by up to {100 * err:.0f}%: check the tanh correction, log pi(a) = log N(u; mu, sigma) - "
        f"sum log(1 - tanh(u)^2), with u the PRE-squash sample")


def test_stage2_saturation_and_act_limit(nb):
    torch.manual_seed(2)
    actor = nb.SquashedGaussianActor(2, 1, hidden_sizes=(8,), act_limit=1.0)
    with torch.no_grad():
        for p in actor.parameters():
            p.mul_(100.0)
    a, lp = actor(torch.randn(256, 2) * 10)
    lp.sum().backward()
    assert torch.all(torch.isfinite(lp)) and all(torch.all(torch.isfinite(p.grad)) for p in actor.parameters()), (
        "with huge pre-squash values (tanh saturates) the log-probability and its gradient must stay finite: compute "
        "log(1 - tanh(u)^2) stably, as 2 (log 2 - u - softplus(-2u)), and clamp log_std")
    torch.manual_seed(3)
    one = nb.SquashedGaussianActor(2, 1, hidden_sizes=(8,), act_limit=1.0)
    two = nb.SquashedGaussianActor(2, 1, hidden_sizes=(8,), act_limit=2.0)
    two.load_state_dict(one.state_dict())
    obs = torch.randn(16, 2)
    torch.manual_seed(4)
    a1, lp1 = one(obs)
    torch.manual_seed(4)
    a2, lp2 = two(obs)
    assert torch.allclose(a2, 2 * a1, atol=1e-6), "act_limit scales the squashed action: a = act_limit * tanh(u)"
    d = (lp2 - lp1).detach()
    assert torch.allclose(d, torch.zeros(16), atol=1e-5), (
        "the spec follows Spinning Up: log_prob is the log-density of tanh(u), WITHOUT the constant -log(act_limit) "
        f"per dimension; with act_limit 2 instead of 1 it must not change. got a shift of {d.mean().item():.4f}")


def test_stage2_twin_q(nb):
    torch.manual_seed(5)
    q = nb.TwinQ(3, 2, hidden_sizes=(16,))
    obs, act = torch.randn(7, 3), torch.randn(7, 2)
    q1, q2 = q(obs, act)
    assert q1.shape == (7,) and q2.shape == (7,), f"(q1 (B,), q2 (B,)); got {q1.shape}, {q2.shape}"
    assert not torch.allclose(q1, q2), "the two critics are independently initialized networks"
    q1.sum().backward()
    touched = [p for p in q.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert 0 < len(touched) < len(list(q.parameters())), (
        "Q1 and Q2 share no parameters: the gradient of q1 must not reach the second network")
    n = sum(p.numel() for p in nb.TwinQ(3, 1).parameters())
    assert n == 2 * (4 * 256 + 256 + 256 * 256 + 256 + 256 + 1), "default hidden_sizes (256, 256) for each critic"


# ---------------------------------------------------------------- stage 3: SAC losses


def test_stage3_critic_target(nb):
    q1n = torch.tensor([2.0, 3.0], requires_grad=True)
    y = nb.sac_critic_target(torch.tensor([1.0, 0.5]), torch.tensor([0.0, 1.0]), q1n, torch.tensor([1.5, 4.0]),
                             torch.tensor([-1.0, 0.5]), gamma=0.9, alpha=0.2)
    assert torch.allclose(y, torch.tensor([1 + 0.9 * (1.5 + 0.2), 0.5])), (
        f"y = r + gamma (1 - terminated) (min(Q1', Q2') - alpha log pi(a'|s')): (1 + 0.9 * (1.5 - 0.2 * -1), 0.5) = "
        f"(2.53, 0.5); got {y.tolist()}")
    assert not y.requires_grad, "the target is a label: compute it without gradients"


def test_stage3_actor_loss(nb):
    q1 = torch.tensor([1.0, 3.0], requires_grad=True)
    q2 = torch.tensor([2.0, 2.0], requires_grad=True)
    lp = torch.tensor([-1.0, 0.5], requires_grad=True)
    loss = nb.sac_actor_loss(q1, q2, lp, 0.1)
    assert abs(loss.item() - (-1.525)) < 1e-6, (
        f"mean(alpha log pi - min(Q1, Q2)) = mean(0.1 * -1 - 1, 0.1 * 0.5 - 2) = -1.525; got {loss.item()}")
    loss.backward()
    assert torch.allclose(lp.grad, torch.tensor([0.05, 0.05])), "d/d log_prob = alpha / B"
    assert torch.allclose(q1.grad, torch.tensor([-0.5, 0.0])) and torch.allclose(q2.grad, torch.tensor([0.0, -0.5])), (
        "the gradient goes through the smaller of the two Q-values only")


def test_stage3_alpha_loss(nb):
    log_alpha = torch.tensor([np.log(0.5)], requires_grad=True)
    lp = torch.tensor([-1.0, -3.0], requires_grad=True)
    loss = nb.sac_alpha_loss(log_alpha, lp, -1.0)
    assert abs(loss.item() - 1.5) < 1e-6, (
        f"-mean(alpha (log pi + target_entropy)) = -0.5 * mean(-2, -4) = 1.5; got {loss.item()}")
    loss.backward()
    assert abs(log_alpha.grad.item() - 1.5) < 1e-6, (
        "the policy's entropy (mean -log pi = 2) exceeds the target (1), so the gradient on log_alpha is positive "
        f"and descent LOWERS the temperature; got gradient {log_alpha.grad.item()}")
    assert lp.grad is None, "log_prob is a constant in the temperature loss (detach it)"


# ---------------------------------------------------------------- stage 4: SAC


def _run(nb, env, n_steps, **kw):
    return bounded(nb.train_sac, _Capped(env, n_steps), n_steps, budget=40_000_000, what="train_sac",
                   hint="the loop should run exactly n_steps environment steps.", **kw)


SMALL = dict(hidden_sizes=(32, 32), batch_size=64, lr=3e-3, tau=0.05)


def test_stage4_interface_and_reproducibility(nb):
    torch.manual_seed(7)
    a1, h1 = _run(nb, _Target(), 150, learning_starts=50, seed=0, **SMALL)
    torch.manual_seed(8)
    a2, h2 = _run(nb, _Target(), 150, learning_starts=50, seed=0, **SMALL)
    assert isinstance(a1, nn.Module) and len(h1["episode_returns"]) == 150, (
        "train_sac returns (actor, history); one-step episodes: 150 steps finish 150 episodes")
    assert isinstance(h1.get("critic"), nn.Module), "history['critic'] holds the trained TwinQ"
    assert h1["episode_returns"] == h2["episode_returns"] and all(
        torch.equal(x, y) for x, y in zip(a1.state_dict().values(), a2.state_dict().values())), (
        "the same seed must give the same run whatever torch's global RNG state was")
    _, h3 = _run(nb, _Target(), 150, learning_starts=50, seed=1, **SMALL)
    assert h3["episode_returns"] != h1["episode_returns"], "different seeds should give different runs"
    assert isinstance(h1.get("critic_target"), nn.Module), "history['critic_target'] holds the target TwinQ"
    b1, _ = _run(nb, _Target(), 40, learning_starts=1000, seed=0, **SMALL)
    b2, _ = _run(nb, _Target(), 90, learning_starts=1000, seed=0, **SMALL)
    assert all(torch.equal(x, y) for x, y in zip(b1.state_dict().values(), b2.state_dict().values())), (
        "no gradient step may happen before learning_starts env steps (actions come from action_space.sample)")


def test_stage4_bootstraps_through_truncation(nb):
    vals = []
    for seed in range(3):
        _, h = _run(nb, _Constant(), 400, gamma=0.8, alpha=0.0, learning_starts=50, seed=seed, **SMALL)
        with torch.no_grad():
            q1, q2 = h["critic"](torch.ones(5, 1), torch.linspace(-0.9, 0.9, 5)[:, None])
        vals.append(float(torch.min(q1, q2).mean()))
    v = float(np.median(vals))
    assert abs(v - 5.0) < 0.6, (
        f"one state, reward 1 per step, gamma 0.8, episodes TRUNCATED after 5 steps, alpha = 0: Q = 1 / (1 - 0.8) = 5 "
        f"for every action. median learned Q over 3 seeds: {v:.2f} (about 2.8 means you stored `terminated or "
        f"truncated` in the buffer: store `terminated` only)")


def test_stage4_learns_a_continuous_target(nb):
    errs = []
    xs = torch.linspace(-0.8, 0.8, 9)[:, None]
    for seed in range(5):
        actor, _ = _run(nb, _Target(), 300, learning_starts=100, alpha=0.05, seed=seed, **SMALL)
        with torch.no_grad():
            a, _ = actor(xs, deterministic=True)
        errs.append(float(((a - xs) ** 2).mean()))
    assert np.median(errs) < 0.03, (
        f"one-step task with reward -(a - x)^2: SAC's deterministic action should track the target x. median squared "
        f"error over 5 seeds after 300 steps: {np.median(errs):.3f} (want < 0.03); per seed {np.round(errs, 3).tolist()}")


class _CountingBox(Box):
    """a Box that counts calls to sample()."""

    calls = 0

    def sample(self, rng=None):
        _CountingBox.calls += 1
        return super().sample(rng)


def test_stage4_warmup_samples_the_action_space(nb):
    env = _Target()
    env.action_space = _CountingBox(-1.0, 1.0, (1,))
    _CountingBox.calls = 0
    _run(nb, env, 80, learning_starts=50, seed=0, **SMALL)
    assert _CountingBox.calls == 50, (
        f"for the first learning_starts steps the action must come from env.action_space.sample(rng) (uniform over the "
        f"box), and from the actor afterwards: with learning_starts=50 and 80 steps, sample() should be called 50 "
        f"times; it was called {_CountingBox.calls} times")


def test_stage4_soft_target_update(nb):
    kw = dict(hidden_sizes=(16,), batch_size=32, lr=3e-3, seed=0)
    _, h0 = _run(nb, _Target(), 20, learning_starts=1000, **kw)  # no updates: the critic as initialized
    init = [p.detach().clone() for p in h0["critic"].parameters()]
    _, h = _run(nb, _Target(), 120, learning_starts=40, tau=0.0, **kw)
    tgt = list(h["critic_target"].parameters())
    crit = list(h["critic"].parameters())
    assert all(torch.equal(a, b) for a, b in zip(tgt, init)), (
        "with tau = 0 the target critic must never move: target <- (1 - tau) target + tau critic, i.e. "
        "ema_update(target, critic, decay=1 - tau). yours changed (did you pass decay=tau?)")
    assert not all(torch.equal(a, b) for a, b in zip(crit, init)), "the critic itself must have been trained"
