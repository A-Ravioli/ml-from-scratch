"""Checks for exercise.py (offline RL and imitation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. behavior cloning is
checked on a synthetic classification task, DAgger by what it asks the expert and by its result
on a drifting tightrope where cloning alone keeps falling off, the CQL penalty exactly by hand,
and offline Q-learning on a fixed dataset that never tries one of the actions. Loops are capped
so a bug fails instead of hanging.
"""
import numpy as np
import torch

from mlfs.deeprl import CategoricalPolicy, ReplayBuffer
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
            f"more than {self.max_steps:,} env.step calls: does every loop stop after the episodes it was asked for, "
            f"and does each episode end when terminated or truncated is True?")
        return self.env.step(action)


class _BoxSpace:
    def __init__(self, n):
        self.shape = (n,)


class _Drift(Env):
    """a tightrope: position x in -3..3 (one-hot obs), actions left / stay / right, then a gust moves x by -1 or +1
    with probability 0.1 each. |x| = 3 is a fall (terminated). reward 1 per step; truncated after 200 steps."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(7), Discrete(3)

    def _obs(self):
        o = np.zeros(7)
        o[self.x + 3] = 1.0
        return o

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.x, self.t = 0, 0
        return self._obs(), {}

    def step(self, action):
        self.x += int(action) - 1
        u = self.np_random.random()
        self.x = int(np.clip(self.x + (-1 if u < 0.1 else (1 if u < 0.2 else 0)), -3, 3))
        self.t += 1
        term = abs(self.x) >= 3
        return self._obs(), 1.0, term, (not term and self.t >= 200), {}


class _Expert:
    """walks back to the middle; remembers every position it was asked about."""

    def __init__(self):
        self.asked = []

    def __call__(self, obs):
        x = int(np.argmax(obs)) - 3
        self.asked.append(x)
        return 0 if x > 0 else (2 if x < 0 else 1)


# ---------------------------------------------------------------- stage 1: behavior cloning


def test_stage1_behavior_cloning_fits_the_expert(nb):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 2))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    torch.manual_seed(0)
    pi = CategoricalPolicy(2, 3, hidden_sizes=(32,))
    losses = bounded(nb.behavior_cloning, pi, X, y, epochs=40, lr=1e-2, batch_size=64, seed=0, budget=20_000_000,
                     what="behavior_cloning")
    assert isinstance(losses, list) and len(losses) == 40, "behavior_cloning returns one (mean) loss per epoch"
    assert losses[-1] < 0.5 * losses[0], f"the cross-entropy should fall steadily; first {losses[0]:.3f}, last {losses[-1]:.3f}"
    with torch.no_grad():
        acc = (pi(torch.as_tensor(X, dtype=torch.float32)).probs.argmax(1).numpy() == y).mean()
    assert acc > 0.9, f"after 40 epochs the policy's most likely action should match the expert's on > 90%; got {acc:.2f}"
    assert abs(losses[0] - np.log(3)) < 0.5, (
        f"the loss is the mean negative log-likelihood of the expert's actions (about ln 3 = 1.10 at the start for 3 "
        f"actions); first epoch {losses[0]:.3f}")


def test_stage1_reproducible_minibatches(nb):
    X = np.random.default_rng(1).normal(size=(100, 2))
    y = (X[:, 0] > 0).astype(int)

    def run(seed):
        torch.manual_seed(3)
        pi = CategoricalPolicy(2, 2, hidden_sizes=(8,))
        return nb.behavior_cloning(pi, X, y, epochs=3, batch_size=16, seed=seed)

    assert run(0) == run(0), "the same seed must shuffle the minibatches the same way"
    assert run(0) != run(1), "different seeds should shuffle differently"


def test_stage1_run_policy(nb):
    env = _Capped(_Drift(), 1000)
    expert = _Expert()
    rets, visited = nb.run_policy(env, expert, 3, seed=4)
    assert rets == [200.0, 200.0, 200.0], f"the expert never falls: three returns of 200; got {rets}"
    assert len(visited) == 600 and env.steps == 600, (
        "visited holds the observation every action was taken in: 3 episodes x 200 steps")
    assert all(isinstance(o, np.ndarray) and o.shape == (7,) for o in visited[:5]), "observations as ndarrays"
    assert [int(np.argmax(o)) - 3 for o in visited] == expert.asked, (
        "visited[t] must be the observation act() was called with at step t: record obs BEFORE stepping")
    again = nb.run_policy(_Drift(), _Expert(), 3, seed=4)[1]
    assert all(np.array_equal(a, b) for a, b in zip(visited, again)), "the first reset is seeded with `seed`"
    pos = [int(np.argmax(o)) for o in visited]
    assert pos[:200] != pos[200:400] or pos[200:400] != pos[400:], (
        "only the FIRST reset takes the seed; later episodes continue the env's random stream. your three episodes "
        "were identical, so every reset was reseeded")


# ---------------------------------------------------------------- stage 2: DAgger


def test_stage2_dagger_asks_about_the_learners_states(nb):
    finals = []
    for seed in range(3):
        expert = _Expert()
        torch.manual_seed(seed)
        pi = CategoricalPolicy(7, 3, hidden_sizes=(16,))
        pi, hist = bounded(nb.dagger, _Capped(_Drift(), 5000), expert, pi, n_iterations=4, episodes_per_iteration=5,
                           epochs=10, seed=seed, budget=60_000_000, what="dagger")
        assert len(hist["returns"]) == 4 and hist["returns"][0] == 200.0, (
            "history['returns'] has one mean return per iteration; iteration 0 rolls out the EXPERT (200 here)")
        assert np.all(np.diff(hist["dataset_size"]) > 0) and hist["dataset_size"][0] == 1000, (
            f"the dataset is AGGREGATED: 1,000 expert states after iteration 0, growing every iteration; got "
            f"{hist['dataset_size']}")
        n_all, n0 = hist["dataset_size"][-1], hist["dataset_size"][0]
        assert len(expert.asked) in (n_all, n_all + n0), (
            f"the expert should label every state in the dataset ({n_all}) and act only in iteration 0 (another {n0} "
            f"calls, unless you reused its actions as labels): it was called {len(expert.asked)} times. from "
            f"iteration 1 on, the LEARNER chooses the actions; the expert only labels")
        finals.append(np.mean(nb.run_policy(_Drift(), pi.act, 20, seed=100 + seed)[0]))
    assert np.median(finals) >= 198, (
        f"after 4 DAgger iterations the learner should (nearly) never fall: median return over 3 seeds "
        f"{np.median(finals):.1f} (want >= 198 out of 200); per seed {np.round(finals, 1).tolist()}")


# ---------------------------------------------------------------- stage 3: offline Q-learning and CQL


def test_stage3_cql_penalty_by_hand(nb):
    q = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    a = torch.tensor([1, 2])
    pen = nb.cql_penalty(q, a, alpha=0.5)
    lse = torch.logsumexp(q.detach(), 1)
    want = 0.5 * ((lse[0] - 2.0) + (lse[1] - 0.0)) / 2
    assert pen.ndim == 0 and abs(pen.item() - want.item()) < 1e-6, (
        f"alpha * mean(logsumexp_a Q(s, a) - Q(s, a_data)) = {want.item():.5f}; got {pen.item():.5f}")
    pen.backward()
    grad = 0.5 * (torch.softmax(q.detach(), 1) - torch.eye(3)[a]) / 2
    assert torch.allclose(q.grad, grad, atol=1e-6), (
        "its gradient pushes every Q(s, a) down in proportion to softmax(Q) and the dataset action's Q up: "
        "alpha (softmax(Q) - onehot(a)) / B")
    assert nb.cql_penalty(torch.zeros(4, 2), torch.zeros(4, dtype=torch.int64)).item() > 0, (
        "with equal Q-values the penalty is log(n_actions) > 0 (default alpha = 1)")


def _dataset(seed):
    """one-step episodes in 3 states (one-hot obs); the data only ever takes action 0, which pays 0.5."""
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(500, (3,), seed=seed)
    for _ in range(500):
        s = np.eye(3)[rng.integers(3)]
        buf.add(s, 0, 0.5, s, True)
    return buf


def test_stage3_offline_q_and_cql(nb):
    gaps, fits = [], []
    for seed in range(5):
        q0, h0 = bounded(nb.train_offline_q, _dataset(seed), 3, 2, 300, lr=1e-2, batch_size=32, cql_alpha=0.0,
                         hidden_sizes=(16,), seed=seed, budget=30_000_000, what="train_offline_q")
        q1, h1 = nb.train_offline_q(_dataset(seed), 3, 2, 300, lr=1e-2, batch_size=32, cql_alpha=1.0,
                                    hidden_sizes=(16,), seed=seed)
        with torch.no_grad():
            Q0, Q1 = q0(torch.eye(3)), q1(torch.eye(3))
        fits.append(float((Q0[:, 0] - 0.5).abs().max()))
        gaps.append(float((Q1[:, 0] - Q1[:, 1]).min()))
        assert len(h1["td_loss"]) == 300 and len(h1["cql"]) == 300, "history has a td_loss and a cql entry per update"
        assert h1["cql"][0] > 0 and all(c == 0 for c in h0["cql"]), "the penalty is only added when cql_alpha > 0"
    assert np.median(fits) < 0.05, (
        f"plain offline Q-learning on the dataset's action: every transition terminates with reward 0.5, so "
        f"Q(s, 0) -> 0.5; median max error {np.median(fits):.3f}")
    assert sum(g > 0.5 for g in gaps) >= 4, (
        f"action 1 never appears in the data, so plain Q-learning leaves Q(s, 1) wherever the initialization put it. "
        f"CQL (alpha = 1) must push it well below the dataset action's value: min_s Q(s, 0) - Q(s, 1) should exceed "
        f"0.5 in at least 4 of 5 seeds; got {np.round(gaps, 2).tolist()}")


def test_stage3_offline_q_bootstraps(nb):
    buf = ReplayBuffer(200, (1,), seed=0)
    for _ in range(200):
        buf.add(np.ones(1), 0, 1.0, np.ones(1), False)
    vals = []
    for seed in range(3):
        q, _ = bounded(nb.train_offline_q, buf, 1, 2, 800, gamma=0.5, lr=1e-2, batch_size=32, target_update=20,
                       hidden_sizes=(8,), seed=seed, budget=40_000_000, what="train_offline_q")
        with torch.no_grad():
            vals.append(float(q(torch.ones(1, 1))[0, 0]))
    v = float(np.median(vals))
    assert abs(v - 2.0) < 0.2, (
        f"one state that never terminates, reward 1, gamma 0.5: Q = 1 / (1 - 0.5) = 2, reached by bootstrapping "
        f"through the target network. median learned Q over 3 seeds: {v:.2f} (about 1 means the target network is "
        f"never updated: copy the weights every target_update steps)")


def test_stage3_penalty_uses_the_dataset_states(nb):
    o1, o2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    buf = ReplayBuffer(50, (3,), seed=0)
    for _ in range(50):
        buf.add(o1, 0, 0.0, o2, False)
    q0, _ = nb.train_offline_q(buf, 3, 2, 0, hidden_sizes=(8,), seed=5)  # the network as initialized
    _, h = nb.train_offline_q(buf, 3, 2, 1, cql_alpha=0.7, hidden_sizes=(8,), seed=5)
    with torch.no_grad():
        q = q0(torch.as_tensor(o1, dtype=torch.float32)[None])
    want = 0.7 * (torch.logsumexp(q, 1) - q[:, 0]).item()
    assert abs(h["cql"][0] - want) < 1e-5, (
        f"the first update's penalty must be cql_penalty(q_net(obs), actions) on the batch's OBSERVATIONS (not "
        f"next_obs): {want:.5f} for the initial network; history['cql'][0] = {h['cql'][0]:.5f}")


def test_stage3_double_matters(nb):
    def make_buffer():  # a fresh buffer each time: sampling advances the buffer's own generator
        rng = np.random.default_rng(0)
        buf = ReplayBuffer(400, (5,), seed=0)
        for _ in range(400):
            s, s2 = np.eye(5)[rng.integers(5)], np.eye(5)[rng.integers(5)]
            buf.add(s, int(rng.integers(4)), float(rng.normal(0, 2)), s2, False)
        return buf

    kw = dict(gamma=0.9, lr=3e-2, target_update=10_000, hidden_sizes=(16,), seed=1)
    a, _ = nb.train_offline_q(make_buffer(), 5, 4, 200, double=True, **kw)
    c, _ = nb.train_offline_q(make_buffer(), 5, 4, 200, double=True, **kw)
    b, _ = nb.train_offline_q(make_buffer(), 5, 4, 200, double=False, **kw)
    same = lambda x, y: all(torch.equal(p, q) for p, q in zip(x.state_dict().values(), y.state_dict().values()))
    assert same(a, c), "the same buffer contents and seed must give the same network"
    assert not same(a, b), "double=True and double=False gave identical networks: pass `double` on to dqn_loss"
