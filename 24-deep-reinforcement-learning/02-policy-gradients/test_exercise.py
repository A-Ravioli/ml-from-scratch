"""Checks for exercise.py (policy gradients).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Returns and the loss
are checked exactly by hand; the score-function estimator is checked against the exact gradient of
a one-step problem; learning is checked on tiny tasks with known optima over several seeds. Every
environment is wrapped so a loop that never ends fails after a fixed number of steps.
"""
import numpy as np
import torch
from torch import nn

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
            f"more than {self.max_steps:,} env.step calls: does your loop stop after n_episodes episodes, and does "
            f"each episode end when terminated or truncated is True?")
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


class _Constant(Env):
    """one state (obs [1.0]), two identical actions, reward 1 per step, TRUNCATED after 5 steps."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(1), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([1.0]), {}

    def step(self, action):
        self.t += 1
        return np.array([1.0]), 1.0, False, self.t >= 5, {}


class _Terminating(_Constant):
    """like _Constant, but the episode TERMINATES after 5 steps."""

    def step(self, action):
        self.t += 1
        return np.array([1.0]), 1.0, self.t >= 5, False, {}


class _NoisyStart(Env):
    """two steps: step 0 pays +-10 at random whatever you do; at step 1, action 1 pays +1 and action 0 pays 0."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(2), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.t = 0
        return np.eye(2)[0], {}

    def step(self, action):
        self.t += 1
        if self.t == 1:
            return np.eye(2)[1], float(self.np_random.choice([-10.0, 10.0])), False, False, {}
        return np.eye(2)[1], float(action == 1), True, False, {}


class _Delayed(Env):
    """two steps: the first action decides everything, the reward arrives at the end. obs = one-hot step index."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(2), Discrete(2)

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.t, self.first = 0, None
        return np.eye(2)[0], {}

    def step(self, action):
        self.t += 1
        if self.t == 1:
            self.first = action
            return np.eye(2)[1], 0.0, False, False, {}
        return np.eye(2)[1], float(self.first == 1), True, False, {}


# ---------------------------------------------------------------- stage 1: discounted returns


def test_stage1_hand_example(nb):
    got = nb.discounted_returns(np.array([1.0, 2.0, 3.0]), 0.5)
    assert isinstance(got, np.ndarray), f"an ndarray in gives an ndarray out; got {type(got).__name__}"
    assert np.allclose(got, [2.75, 3.5, 3.0]), (
        f"rewards (1, 2, 3), gamma 0.5: G_2 = 3, G_1 = 2 + 0.5 * 3 = 3.5, G_0 = 1 + 0.5 * 3.5 = 2.75; got {got}")
    t = nb.discounted_returns(torch.tensor([1.0, 2.0, 3.0]), 0.5)
    assert torch.is_tensor(t) and t.shape == (3,), "a tensor in gives a tensor out, of the same shape"
    assert torch.allclose(t, torch.tensor([2.75, 3.5, 3.0])), f"tensor version: expected (2.75, 3.5, 3); got {t}"


def test_stage1_edge_cases(nb):
    r = np.array([1.0, -1.0, 2.0, 0.5])
    assert np.allclose(nb.discounted_returns(r, 1.0), [2.5, 1.5, 2.5, 0.5]), "gamma = 1: reversed cumulative sums"
    assert np.allclose(nb.discounted_returns(r, 0.0), r), "gamma = 0: each G_t is just r_t"
    before = r.copy()
    nb.discounted_returns(r, 0.9)
    assert np.array_equal(r, before), "don't modify the rewards you were given"
    ints = nb.discounted_returns(np.array([1, 1, 1]), 0.5)
    assert np.allclose(ints, [1.75, 1.5, 1.0]), f"integer rewards still give fractional returns; got {ints}"
    long = nb.discounted_returns(np.ones(2000), 0.99)
    assert np.isclose(long[0], (1 - 0.99**2000) / 0.01) and np.isclose(long[-1], 1.0), "2,000 ones, gamma 0.99"


# ---------------------------------------------------------------- stage 2: the policy


def test_stage2_distribution(nb):
    torch.manual_seed(0)
    pi = nb.CategoricalPolicy(4, 3, hidden_sizes=(16,))
    assert isinstance(pi, nn.Module), "CategoricalPolicy is an nn.Module"
    dist = pi(torch.randn(5, 4))
    assert isinstance(dist, torch.distributions.Categorical), "forward returns a torch.distributions.Categorical"
    assert dist.batch_shape == (5,) and dist.probs.shape == (5, 3), (
        f"a batch of 5 observations gives batch_shape (5,) and probs (5, 3); got {dist.batch_shape}")
    assert torch.allclose(dist.probs.sum(1), torch.ones(5)), "each row of probs sums to 1"
    dist.log_prob(torch.tensor([0, 1, 2, 0, 1])).sum().backward()
    assert all(p.grad is not None for p in pi.parameters()), "log-probabilities must be differentiable in every parameter"
    n = sum(p.numel() for p in nb.CategoricalPolicy(4, 2).parameters())
    assert n == 4 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2, f"default hidden_sizes (64, 64): {n} parameters"


def test_stage2_act_samples(nb):
    torch.manual_seed(1)
    pi = nb.CategoricalPolicy(3, 4, hidden_sizes=(8,))
    obs = np.array([0.5, -1.0, 2.0])
    with torch.no_grad():
        probs = pi(torch.as_tensor(obs, dtype=torch.float32)[None]).probs[0].numpy()
    a = pi.act(obs)
    assert type(a) is int and 0 <= a < 4, f"act returns a Python int in [0, n_actions); got {a!r}"
    counts = np.bincount([pi.act(obs) for _ in range(4000)], minlength=4) / 4000
    assert np.all(np.abs(counts - probs) < 5 * np.sqrt(probs * (1 - probs) / 4000) + 1e-3), (
        f"act must SAMPLE from the policy's distribution (not take the argmax): action frequencies {counts.round(3)} "
        f"vs probabilities {probs.round(3)}")


# ---------------------------------------------------------------- stage 3: the REINFORCE loss


def test_stage3_hand_example(nb):
    lp = torch.tensor([-0.5, -1.0, -2.0], requires_grad=True)
    loss = nb.reinforce_loss(lp, torch.tensor([1.0, 2.0, -1.0]))
    assert loss.ndim == 0 and abs(loss.item() - (-(-0.5 - 2.0 + 2.0) / 3)) < 1e-6, (
        f"-(1 * -0.5 + 2 * -1 + -1 * -2) / 3 = {0.5 / 3:.4f}; got {loss.item():.4f} (check the sign and that you "
        f"take the MEAN)")
    loss.backward()
    assert torch.allclose(lp.grad, torch.tensor([-1.0, -2.0, 1.0]) / 3), (
        f"d loss / d log_prob_i = -(G_i - b) / n; got {lp.grad.tolist()}")
    b = nb.reinforce_loss(torch.tensor([-0.5, -1.0, -2.0]), np.array([1.0, 2.0, -1.0]), baseline=1.0)
    assert abs(b.item() - (-(0.0 - 1.0 + 4.0) / 3)) < 1e-6, (
        f"baseline 1: weights (0, 1, -2), loss -(0 - 1 + 4) / 3 = -1; got {b.item():.4f}")


def test_stage3_returns_and_baseline_are_constants(nb):
    lp = torch.tensor([-0.3, -0.7], requires_grad=True)
    G = torch.tensor([1.0, 3.0], requires_grad=True)
    v = torch.tensor([0.5, 0.5], requires_grad=True)
    nb.reinforce_loss(lp, G, baseline=v).backward()
    assert G.grad is None and v.grad is None, (
        "no gradient may flow into the returns or the baseline (detach them): a value network used as the baseline "
        "must be trained by its own loss, not pushed around by the policy's")
    assert torch.allclose(lp.grad, torch.tensor([-0.25, -1.25])), f"weights (0.5, 2.5) / 2; got {lp.grad.tolist()}"


def test_stage3_score_function_is_unbiased(nb):
    torch.manual_seed(2)
    pi = nb.CategoricalPolicy(2, 3, hidden_sizes=(8,))
    obs = torch.tensor([[0.3, -0.6]])
    r = torch.tensor([1.0, -0.5, 2.0])
    exact = torch.autograd.grad((pi(obs).probs[0] * r).sum(), list(pi.parameters()))
    torch.manual_seed(3)
    n = 40_000
    dist = pi(obs.expand(n, 2))
    a = dist.sample()
    est = torch.autograd.grad(-nb.reinforce_loss(dist.log_prob(a), r[a]), list(pi.parameters()))
    e, g = torch.cat([x.flatten() for x in exact]), torch.cat([x.flatten() for x in est])
    cos = torch.dot(e, g) / (e.norm() * g.norm())
    assert cos > 0.95 and abs(g.norm() / e.norm() - 1) < 0.15, (
        f"on a one-step problem, minus the gradient of reinforce_loss over many samples must match the exact gradient "
        f"of E[r] = sum_a pi(a) r(a): cosine {cos:.3f}, norm ratio {g.norm() / e.norm():.3f}. a cosine near -1 means a "
        f"sign error; a norm ratio far from 1 means you summed instead of averaging")


# ---------------------------------------------------------------- stage 4: REINFORCE


def _run(nb, env, n_episodes, max_steps, **kw):
    return bounded(nb.train_reinforce, _Capped(env, max_steps), n_episodes, budget=30_000_000,
                   what="train_reinforce", hint="the loop should run exactly n_episodes episodes.", **kw)


def test_stage4_interface_and_reproducibility(nb):
    torch.manual_seed(123)
    p1, h1 = _run(nb, _Delayed(), 30, 60, lr=0.05, batch_episodes=4, seed=0)
    torch.manual_seed(456)
    p2, h2 = _run(nb, _Delayed(), 30, 60, lr=0.05, batch_episodes=4, seed=0)
    assert isinstance(p1, nn.Module) and len(h1["episode_returns"]) == 30, (
        "train_reinforce returns (policy, history) and records exactly n_episodes returns (30 here, even though 30 "
        "isn't a multiple of batch_episodes = 4)")
    assert h1 == h2 and all(torch.equal(a, b) for a, b in zip(p1.state_dict().values(), p2.state_dict().values())), (
        "the same seed must give the same run whatever torch's global RNG state was: seed numpy AND torch from `seed`")
    p3, _ = _run(nb, _Delayed(), 0, 1, seed=1)
    p4, _ = _run(nb, _Delayed(), 0, 1, seed=2)
    assert not all(torch.equal(a, b) for a, b in zip(p3.state_dict().values(), p4.state_dict().values())), (
        "different seeds must give differently initialized policies: call torch.manual_seed(seed) before building it")


def test_stage4_learns_a_contextual_bandit(nb):
    finals = []
    for seed in range(5):
        _, h = _run(nb, _Contextual(), 400, 400, lr=0.05, batch_episodes=8, seed=seed)
        finals.append(np.mean(h["episode_returns"][-100:]))
    assert np.median(finals) > 0.85, (
        f"one-step episodes, reward 1 for matching the context: the success rate over the last 100 of 400 episodes "
        f"should approach 1 (median over 5 seeds > 0.85); got {np.round(finals, 2).tolist()}")


def test_stage4_credit_for_a_delayed_reward(nb):
    finals = []
    for seed in range(5):
        _, h = _run(nb, _Delayed(), 300, 600, lr=0.05, batch_episodes=8, gamma=1.0, seed=seed,
                    baseline=("value" if seed % 2 else None))
        finals.append(np.mean(h["episode_returns"][-80:]))
    assert np.median(finals) > 0.85, (
        f"the reward comes one step after the action that earned it: the return-to-go from step 0 must carry it back. "
        f"success over the last 80 of 300 episodes (median of 5 seeds, with and without a baseline) should be > 0.85; "
        f"got {np.round(finals, 2).tolist()}")


def test_stage4_value_baseline_bootstraps_truncation(nb):
    vals = []
    for seed in range(3):
        _, h = _run(nb, _Constant(), 400, 2000, gamma=0.8, lr=1e-3, lr_value=0.05, batch_episodes=1,
                    baseline="value", hidden_sizes=(8,), seed=seed)
        assert "value_net" in h, "with baseline='value', history['value_net'] holds the trained value network"
        with torch.no_grad():
            vals.append(float(h["value_net"](torch.ones(1, 1)).reshape(-1)[0]))
    v = float(np.median(vals))
    assert abs(v - 5.0) < 0.6, (
        f"one state, reward 1 per step, gamma 0.8, episodes TRUNCATED after 5 steps: bootstrapping the cut-off tail "
        f"with gamma V(s_T) makes every return-to-go equal V, so V -> 1 / (1 - 0.8) = 5. median learned V over 3 "
        f"seeds: {v:.2f} (about 2.3 means the returns weren't bootstrapped at the time limit)")


def test_stage4_value_baseline_does_not_bootstrap_termination(nb):
    vals = []
    for seed in range(3):
        _, h = _run(nb, _Terminating(), 400, 2000, gamma=0.8, lr=1e-3, lr_value=0.05, batch_episodes=1,
                    baseline="value", hidden_sizes=(8,), seed=seed)
        with torch.no_grad():
            vals.append(float(h["value_net"](torch.ones(1, 1)).reshape(-1)[0]))
    v = float(np.median(vals))
    want = float(np.mean([(1 - 0.8**m) / 0.2 for m in range(1, 6)]))
    assert abs(v - want) < 0.4, (
        f"the same one-state task, but the episode TERMINATES after 5 steps: nothing follows, so no bootstrap, and V "
        f"(one obs for all 5 steps) learns the average return-to-go, {want:.2f}. median learned V: {v:.2f} (about 5 "
        f"means you bootstrapped a terminated episode: only truncation gets gamma V(s_T))")


def test_stage4_uses_reward_to_go(nb):
    finals = []
    for seed in range(5):
        _, h = _run(nb, _NoisyStart(), 400, 800, lr=0.05, batch_episodes=8, gamma=1.0, hidden_sizes=(16,), seed=seed)
        # returns are +-10 + (0 or 1): an odd return means the step-1 action was right
        finals.append(np.mean([round(r) % 2 == 1 for r in h["episode_returns"][-100:]]))
    assert np.median(finals) > 0.85, (
        f"step 0 pays +-10 at random, then the step-1 action decides a +1. with reward-to-go, step 1's action is "
        f"weighted by its own reward only; weighting it by the whole episode's return buries that +1 under +-10 of "
        f"noise. step-1 success over the last 100 of 400 episodes should be > 0.85 (median of 5 seeds); got "
        f"{np.round(finals, 2).tolist()}. weight each step by its OWN return-to-go G_t")
