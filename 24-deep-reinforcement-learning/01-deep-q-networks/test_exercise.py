"""Checks for exercise.py (deep Q-networks).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. CartPole is checked
against an independent implementation of the published equations; the loss is checked exactly on
hand-built networks; learning is checked on tiny tasks with known optima, over several seeds.
Every environment is wrapped so a loop that never ends fails after a fixed number of steps.
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
            f"more than {self.max_steps:,} env.step calls: does your loop stop after n_steps environment steps?")
        return self.env.step(action)


class _BoxSpace:
    def __init__(self, n):
        self.shape = (n,)


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


class _Corridor(Env):
    """positions 0..5 (one-hot obs), start 1; 0 is a terminal worth +0.1, 5 a terminal worth +1; time limit 20."""

    def __init__(self):
        self.observation_space, self.action_space = _BoxSpace(6), Discrete(2)

    def _obs(self):
        o = np.zeros(6)
        o[self.pos] = 1.0
        return o

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.pos, self.t = 1, 0
        return self._obs(), {}

    def step(self, action):
        self.pos += 1 if action == 1 else -1
        self.t += 1
        terminated = self.pos in (0, 5)
        reward = {0: 0.1, 5: 1.0}.get(self.pos, 0.0)
        return self._obs(), reward, terminated, (not terminated and self.t >= 20), {}


def _cartpole_step(state, action):
    """the published frictionless equations (Barto, Sutton & Anderson 1983; Florian 2007), Euler, tau 0.02."""
    g, mc, mp, l, fm, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x, xd, th, thd = state
    f = fm if action == 1 else -fm
    c, s = np.cos(th), np.sin(th)
    tmp = (f + mp * l * thd**2 * s) / (mc + mp)
    thacc = (g * s - c * tmp) / (l * (4 / 3 - mp * c**2 / (mc + mp)))
    xacc = tmp - mp * l * thacc * c / (mc + mp)
    return np.array([x + tau * xd, xd + tau * xacc, th + tau * thd, thd + tau * thacc])


def _greedy_corridor(q_net):
    env = _Corridor()
    obs, _ = env.reset(seed=0)
    for t in range(20):
        with torch.no_grad():
            a = int(q_net(torch.as_tensor(obs, dtype=torch.float32)[None])[0].argmax())
        obs, r, term, trunc, _ = env.step(a)
        if term or trunc:
            return r, t + 1
    return 0.0, 20


# ---------------------------------------------------------------- stage 1: CartPole


def test_stage1_spaces_and_reset(nb):
    env = nb.CartPole()
    assert tuple(env.observation_space.shape) == (4,), "obs is (x, x_dot, theta, theta_dot): shape (4,)"
    assert env.action_space.n == 2, "two actions: 0 push left, 1 push right"
    assert env.max_steps == 500, "the default time limit is 500 steps (CartPole-v1)"
    out = env.reset(seed=3)
    assert isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict), "reset returns (obs, info dict)"
    obs = np.asarray(out[0])
    assert obs.shape == (4,) and np.all(np.abs(obs) <= 0.05), f"the start state is U(-0.05, 0.05)^4; got {obs}"
    again, _ = nb.CartPole().reset(seed=3)
    assert np.allclose(obs, again), "reset(seed=3) twice must give the same start state (draw it from self.np_random)"
    other, _ = nb.CartPole().reset(seed=4)
    assert not np.allclose(obs, other), "different seeds should give different start states"
    out[0][:] = 99.0
    assert np.all(np.abs(env.state) <= 0.05), "return a COPY of the state: changing the obs mustn't change the env"


def test_stage1_dynamics_match_the_equations(nb):
    env = nb.CartPole()
    env.reset(seed=0)
    for start, action in [([0.0, 0.0, 0.05, 0.0], 1), ([0.1, -0.3, -0.08, 0.6], 0), ([-1.0, 1.2, 0.15, -1.1], 1)]:
        env.state = np.array(start, dtype=float)
        env.t = 0
        obs, r, term, trunc, info = env.step(action)
        want = _cartpole_step(np.array(start), action)
        assert np.allclose(obs, want, rtol=1e-9, atol=1e-12), (
            f"from {start} with action {action}: expected {np.round(want, 6).tolist()}, got "
            f"{np.round(np.asarray(obs), 6).tolist()}. check the angular acceleration (the 4/3 term), that the force "
            f"is +10 for action 1, and the Euler order: x += tau * x_dot uses the OLD velocity")
        assert type(r) is float and r == 1.0, "every step, including the last, pays reward 1.0 (a Python float)"
        assert type(term) is bool and type(trunc) is bool and isinstance(info, dict), "flags are Python bools"


def test_stage1_termination_and_truncation(nb):
    env = nb.CartPole()
    env.reset(seed=0)
    env.state, env.t = np.array([2.39, 1.0, 0.0, 0.0]), 0
    _, r, term, trunc, _ = env.step(1)
    assert term and not trunc, "the cart moving past x = 2.4 terminates the episode"
    assert r == 1.0, f"the terminating step pays reward 1.0 too (as in CartPole-v1); got {r}"
    short = nb.CartPole(max_steps=1)
    short.reset(seed=0)
    short.state = np.array([2.39, 1.0, 0.0, 0.0])
    _, _, term, trunc, _ = short.step(1)
    assert term and not trunc, (
        "a step that terminates is never also truncated, even if it hits max_steps: truncated means the episode was "
        "cut off while it could have gone on")
    env.state, env.t = np.array([0.0, 0.0, 0.205, 1.0]), 0
    _, _, term, _, _ = env.step(1)
    assert term, "the pole tipping past 12 degrees (0.2094 rad) terminates the episode"
    env.state, env.t = np.array([0.0, 0.0, 0.1, 0.0]), 0
    _, _, term, _, _ = env.step(1)
    assert not term, "theta = 0.1 rad is inside the 12-degree limit: not terminated"

    env = nb.CartPole(max_steps=10)
    env.reset(seed=1)
    flags = []
    for t in range(10):
        _, _, term, trunc, _ = env.step(t % 2)
        flags.append((term, trunc))
    assert flags[:-1] == [(False, False)] * 9 and flags[-1] == (False, True), (
        f"alternating pushes keeps the pole up for 10 steps; with max_steps=10 the 10th step is truncated (and only "
        f"that one). got (terminated, truncated) = {flags}")


def test_stage1_episode_length_matches_the_equations(nb):
    env = nb.CartPole()
    obs, _ = env.reset(seed=7)
    state, n = np.array(obs, dtype=float), 0
    while True:
        state = _cartpole_step(state, 1)
        n += 1
        if abs(state[0]) > 2.4 or abs(state[2]) > 12 * 2 * np.pi / 360:
            break
    steps = 0
    for _ in range(100):
        _, _, term, trunc, _ = env.step(1)
        steps += 1
        if term or trunc:
            break
    assert term and steps == n, f"always pushing right from reset(seed=7) should fall after {n} steps; got {steps}"


# ---------------------------------------------------------------- stage 2: replay buffer


def test_stage2_add_len_and_wraparound(nb):
    buf = nb.ReplayBuffer(5, (2,), seed=0)
    assert len(buf) == 0, "a new buffer is empty"
    for i in range(3):
        buf.add(np.array([i, i]), i % 2, float(i), np.array([i + 1, i + 1]), False)
    assert len(buf) == 3, f"after 3 adds len is 3; got {len(buf)}"
    for i in range(3, 12):
        buf.add(np.array([i, i]), i % 2, float(i), np.array([i + 1, i + 1]), i == 11)
    assert len(buf) == 5, f"len never exceeds capacity (5); got {len(buf)}"
    seen = set()
    for _ in range(40):
        b = buf.sample(16)
        seen |= set(b["obs"][:, 0].tolist())
    assert seen == {7.0, 8.0, 9.0, 10.0, 11.0}, (
        f"after 12 adds into capacity 5, only the 5 newest transitions (7..11) remain; samples contained "
        f"{sorted(seen)}. the oldest entry must be overwritten first (a ring buffer)")


def test_stage2_sample_types_and_alignment(nb):
    buf = nb.ReplayBuffer(100, (3,), seed=1)
    for i in range(50):
        buf.add(np.full(3, i), i % 4, 0.5 * i, np.full(3, i + 100), i % 7 == 0)
    b = buf.sample(32)
    assert set(b) == {"obs", "actions", "rewards", "next_obs", "terminated"}, f"keys: got {sorted(b)}"
    assert all(isinstance(v, torch.Tensor) for v in b.values()), "sample returns torch tensors"
    assert b["obs"].shape == (32, 3) and b["next_obs"].shape == (32, 3), "obs and next_obs: (batch, *obs_shape)"
    assert b["actions"].shape == (32,) and b["rewards"].shape == (32,) and b["terminated"].shape == (32,), (
        "actions, rewards, terminated: (batch,)")
    assert b["obs"].dtype == torch.float32 and b["rewards"].dtype == torch.float32, "obs/rewards are float32"
    assert b["terminated"].dtype == torch.float32, "terminated is stored as float32 (0. or 1.) to use in the target"
    assert b["actions"].dtype == torch.int64, "discrete actions are int64 (for gather)"
    i = b["obs"][:, 0].long()
    assert torch.equal(b["actions"], i % 4) and torch.allclose(b["rewards"], 0.5 * i.float()), (
        "every field of a sampled row must come from the same transition")
    assert torch.allclose(b["next_obs"][:, 0], i.float() + 100) and torch.equal(
        b["terminated"], (i % 7 == 0).float()), "next_obs and terminated must line up with obs too"

    cont = nb.ReplayBuffer(10, (2,), action_shape=(2,), action_dtype="float32", seed=0)
    cont.add(np.zeros(2), np.array([0.5, -0.25]), 1.0, np.ones(2), False)
    c = cont.sample(4)
    assert c["actions"].shape == (4, 2) and c["actions"].dtype == torch.float32, (
        "continuous actions: action_shape=(2,), action_dtype='float32' gives actions of shape (batch, 2), float32")


def test_stage2_uniform_reproducible_and_copies(nb):
    buf = nb.ReplayBuffer(10, (1,), seed=2)
    obs = np.zeros(1)
    for i in range(10):
        obs[0] = i
        buf.add(obs, 0, 0.0, obs, False)  # the same array object, mutated: the buffer must have copied it
    idx = np.concatenate([buf.sample(1000)["obs"][:, 0].numpy() for _ in range(20)]).astype(int)
    assert set(idx.tolist()) == set(range(10)), (
        "mutating the array you passed to add() changed the stored obs: store a copy (write into preallocated "
        f"arrays). distinct stored obs: {sorted(set(idx.tolist()))}")
    freq = np.bincount(idx, minlength=10) / len(idx)
    assert np.all(np.abs(freq - 0.1) < 5 * np.sqrt(0.09 / len(idx))), f"sampling should be uniform; got {freq}"

    def draws(seed):
        b = nb.ReplayBuffer(50, (1,), seed=seed)
        for i in range(50):
            b.add(np.array([i]), 0, 0.0, np.array([i]), False)
        return b.sample(20)["obs"][:, 0].tolist()

    assert draws(5) == draws(5), "the same seed must give the same samples"
    assert draws(5) != draws(6), "different seeds should give different samples"


def test_stage2_empty_buffer_raises(nb):
    try:
        nb.ReplayBuffer(4, (1,), seed=0).sample(2)
    except ValueError:
        return
    raise AssertionError("sampling from an empty buffer should raise ValueError")


# ---------------------------------------------------------------- stage 3: Q-network and DQN loss


def _table_net(table):
    """a linear net on one-hot obs: Q(one_hot(s)) = table[s]."""
    table = torch.tensor(table, dtype=torch.float32)
    net = nn.Linear(table.shape[0], table.shape[1], bias=False)
    with torch.no_grad():
        net.weight.copy_(table.T)
    return net


def _batch(states, actions, rewards, next_states, terminated, n=3):
    eye = torch.eye(n)
    return {"obs": eye[states], "actions": torch.tensor(actions, dtype=torch.int64),
            "rewards": torch.tensor(rewards, dtype=torch.float32), "next_obs": eye[next_states],
            "terminated": torch.tensor(terminated, dtype=torch.float32)}


def test_stage3_qnetwork_shapes_and_gradients(nb):
    for dueling in (False, True):
        torch.manual_seed(0)
        net = nb.QNetwork(4, 3, hidden_sizes=(16, 16), dueling=dueling)
        assert isinstance(net, nn.Module), "QNetwork is a torch.nn.Module"
        q = net(torch.randn(5, 4))
        assert q.shape == (5, 3), f"dueling={dueling}: Q has shape (batch, n_actions) = (5, 3); got {tuple(q.shape)}"
        q.sum().backward()
        unused = [n for n, p in net.named_parameters() if p.grad is None]
        assert not unused, f"dueling={dueling}: every parameter should get a gradient; unused: {unused}"
    plain = sum(p.numel() for p in nb.QNetwork(4, 3, hidden_sizes=(16, 16)).parameters())
    duel = sum(p.numel() for p in nb.QNetwork(4, 3, hidden_sizes=(16, 16), dueling=True).parameters())
    assert duel > plain, (
        f"the dueling net has an extra value stream, so more parameters than the plain one ({plain}); got {duel}. "
        f"did dueling=True change anything?")
    default = nb.QNetwork(4, 2)
    assert sum(p.numel() for p in default.parameters()) == 4 * 128 + 128 + 128 * 128 + 128 + 128 * 2 + 2, (
        "the default hidden_sizes are (128, 128)")


def test_stage3_dueling_aggregation(nb):
    torch.manual_seed(1)
    net = nb.QNetwork(3, 4, hidden_sizes=(8,), dueling=True)
    x = torch.randn(6, 3)
    q = net(x)
    heads = [m for m in net.modules() if isinstance(m, nn.Linear) and m.out_features == 4 and m.bias is not None]
    assert heads, "a dueling net needs an advantage head: an nn.Linear with n_actions outputs"
    with torch.no_grad():  # shifting every advantage by the same constant mustn't change Q
        heads[-1].bias += 3.0
    assert torch.allclose(net(x), q, atol=1e-5), (
        "Q = V + A - mean_a A: adding a constant to every advantage must leave Q unchanged. yours changed, so the "
        "mean advantage isn't subtracted (without it V and A aren't identifiable)")
    with torch.no_grad():  # raising ONE advantage by c raises its Q by c (1 - 1/n) and lowers the others by c / n
        heads[-1].bias[0] += 2.0
    delta = net(x) - q
    want = torch.tensor([2.0 * 0.75, -0.5, -0.5, -0.5]).expand(6, 4)
    assert torch.allclose(delta, want, atol=1e-5), (
        f"raising the advantage of action 0 by 2 should change Q by (1.5, -0.5, -0.5, -0.5) with mean subtraction "
        f"(4 actions); got {delta[0].round(decimals=3).tolist()}. subtract the MEAN advantage, as in the spec")


def test_stage3_dqn_loss_hand_example(nb):
    q_tab = [[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]]
    t_tab = [[0.0, 4.0], [2.0, 1.0], [-1.0, 0.5]]
    batch = _batch([0, 1, 2], [1, 0, 0], [1.0, -1.0, 0.5], [1, 2, 0], [0.0, 0.0, 1.0])
    # targets: 1 + 0.5 * max(2, 1) = 2;  -1 + 0.5 * max(-1, 0.5) = -0.75;  0.5 (terminated)
    # td errors: Q(s, a) - y = 2 - 2 = 0;  0.5 + 0.75 = 1.25;  3 - 0.5 = 2.5
    mse = nb.dqn_loss(_table_net(q_tab), _table_net(t_tab), batch, gamma=0.5, huber=False)
    assert torch.is_tensor(mse) and mse.ndim == 0, "dqn_loss returns a scalar tensor"
    want = (0 + 1.25**2 + 2.5**2) / 3
    assert abs(mse.item() - want) < 1e-5, (
        f"MSE: targets (2, -0.75, 0.5), TD errors (0, 1.25, 2.5), mean of squares {want:.5f}; got {mse.item():.5f}. "
        f"the third transition is terminated: its target is just r = 0.5 (no bootstrap)")
    hub = nb.dqn_loss(_table_net(q_tab), _table_net(t_tab), batch, gamma=0.5, huber=True)
    want_h = (0 + (1.25 - 0.5) + (2.5 - 0.5)) / 3
    assert abs(hub.item() - want_h) < 1e-5, (
        f"Huber (smooth L1, delta 1): 0.5 e^2 for |e| <= 1, |e| - 0.5 beyond: mean {want_h:.5f}; got {hub.item():.5f}")


def test_stage3_double_dqn_target(nb):
    q_tab = [[0.0, 0.0], [5.0, 1.0], [0.0, 0.0]]
    t_tab = [[0.0, 0.0], [2.0, 3.0], [0.0, 0.0]]
    batch = _batch([0], [0], [0.0], [1], [0.0])
    plain = nb.dqn_loss(_table_net(q_tab), _table_net(t_tab), batch, gamma=1.0, double=False, huber=False)
    double = nb.dqn_loss(_table_net(q_tab), _table_net(t_tab), batch, gamma=1.0, double=True, huber=False)
    assert abs(plain.item() - 9.0) < 1e-5, f"plain DQN: target max_a Q_target(s', a) = 3, loss (0 - 3)^2 = 9; got {plain.item()}"
    assert abs(double.item() - 4.0) < 1e-5, (
        f"double DQN: the ONLINE net picks a* = argmax Q(s', .) = 0, the target net evaluates it: target 2, loss 4; "
        f"got {double.item()} (9 means you used the target net's own argmax, 25 means you evaluated with the online net)")


def test_stage3_semi_gradient(nb):
    tab = [[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]]
    batch = _batch([0, 1], [1, 0], [1.0, 0.0], [1, 2], [0.0, 0.0])
    tnet = _table_net(tab)
    nb.dqn_loss(_table_net(tab), tnet, batch, gamma=0.9, huber=False).backward()
    assert tnet.weight.grad is None or torch.all(tnet.weight.grad == 0), (
        "no gradient may flow into the target network: compute the target under torch.no_grad() (or .detach())")
    net = _table_net(tab)
    nb.dqn_loss(net, net, batch, gamma=0.9, huber=False).backward()  # the same net as its own target
    # y = (1 + 0.9 * 0.5, 0 + 0.9 * 3) = (1.45, 2.7); e = Q(s, a) - y = (0.55, -2.2); dL/dQ(s, a) = e (mean of 2 squares)
    want = torch.zeros(2, 3)
    want[1, 0], want[0, 1] = 0.55, -2.2
    assert torch.allclose(net.weight.grad, want, atol=1e-5), (
        f"with the online net as its own target, the gradient must only flow through Q(s, a) (a semi-gradient): "
        f"expected {want.tolist()}, got {net.weight.grad.tolist()}")


# ---------------------------------------------------------------- stage 4: the DQN agent


def test_stage4_linear_schedule(nb):
    got = [nb.linear_schedule(1.0, 0.1, 100, t) for t in (0, 50, 100, 1000)]
    assert np.allclose(got, [1.0, 0.55, 0.1, 0.1]), f"1 -> 0.1 over 100 steps, then held: expected (1, .55, .1, .1); got {got}"


def _run(nb, env, n_steps, **kw):
    return bounded(nb.train_dqn, _Capped(env, n_steps), n_steps, budget=20_000_000, what="train_dqn",
                   hint="the loop should run exactly n_steps environment steps.", **kw)


def test_stage4_interface_and_reproducibility(nb):
    kw = dict(learning_starts=50, batch_size=16, target_update=20, eps_decay_steps=100, hidden_sizes=(16,))
    q1, h1 = _run(nb, _Corridor(), 300, seed=3, **kw)
    q2, h2 = _run(nb, _Corridor(), 300, seed=3, **kw)
    assert isinstance(q1, nn.Module), "train_dqn returns (q_net, history)"
    assert set(h1) >= {"episode_returns", "episode_ends"}, "history has 'episode_returns' and 'episode_ends'"
    ends = h1["episode_ends"]
    assert len(ends) == len(h1["episode_returns"]) > 0 and all(np.diff(ends) > 0) and ends[-1] <= 300, (
        "episode_ends: the (1-based) env step at which each finished episode ended, increasing, at most n_steps")
    assert h1 == h2 and all(torch.equal(a, b) for a, b in zip(q1.state_dict().values(), q2.state_dict().values())), (
        "the same seed must give the same run (seed numpy AND torch from `seed`)")
    _, h3 = _run(nb, _Corridor(), 300, seed=4, **kw)
    assert h3 != h1, "different seeds should give different runs"


def test_stage4_learning_starts_and_seeding(nb):
    kw = dict(learning_starts=1000, hidden_sizes=(8,))
    torch.manual_seed(123)
    a, _ = _run(nb, _Corridor(), 40, seed=0, **kw)
    torch.manual_seed(456)
    b, _ = _run(nb, _Corridor(), 120, seed=0, **kw)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())), (
        "no gradient step may happen before learning_starts env steps, and the network's initialization must depend "
        "only on `seed` (not on torch's global RNG state): runs of 40 and 120 steps with seed 0 (and different global "
        "torch states) must return the same, untrained network")
    c, _ = _run(nb, _Corridor(), 40, seed=1, **kw)
    assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values())), (
        "different seeds must give differently initialized networks: call torch.manual_seed(seed) before building them")


def test_stage4_bootstraps_through_truncation(nb):
    vals = []
    for seed in range(3):
        q, h = _run(nb, _Constant(), 1500, gamma=0.8, lr=1e-2, learning_starts=50, batch_size=32, target_update=25,
                    eps_decay_steps=1, hidden_sizes=(8,), seed=seed)
        assert h["episode_ends"] == list(range(5, 1501, 5)), (
            "the single-state task truncates every 5 steps: an episode ends (and the env is reset) on `truncated` too")
        with torch.no_grad():
            vals.append(q(torch.ones(1, 1)).mean().item())
    v = float(np.median(vals))
    assert abs(v - 5.0) < 0.5, (
        f"one state, reward 1 per step, gamma 0.8, episodes TRUNCATED after 5 steps: a time limit isn't a terminal "
        f"state, so Q = 1 / (1 - 0.8) = 5. median learned Q over 3 seeds: {v:.2f}. (about 2.8 means you stored "
        f"`terminated or truncated` in the buffer: store `terminated` only)")


def test_stage4_learns_the_corridor(nb):
    solved = []
    for seed in range(5):
        q, h = _run(nb, _Corridor(), 1500, gamma=0.9, lr=3e-3, learning_starts=100, batch_size=32,
                    target_update=50, eps_decay_steps=600, hidden_sizes=(32,), seed=seed)
        solved.append(_greedy_corridor(q) == (1.0, 4))
    assert sum(solved) >= 4, (
        f"on a 6-cell corridor (a +0.1 exit one step left, a +1 exit four steps right, gamma 0.9) DQN should learn to "
        f"walk right in at least 4 of 5 seeds within 1,500 steps; solved: {solved}")


def test_stage4_learning_starts_boundary(nb):
    kw = dict(learning_starts=40, train_freq=1, batch_size=8, hidden_sizes=(8,), seed=2)
    a, _ = _run(nb, _Corridor(), 39, **kw)
    b, _ = _run(nb, _Corridor(), 40, **kw)
    c, _ = _run(nb, _Corridor(), 10, learning_starts=1000, train_freq=1, batch_size=8, hidden_sizes=(8,), seed=2)
    same = lambda x, y: all(torch.equal(p, q) for p, q in zip(x.state_dict().values(), y.state_dict().values()))
    assert same(a, c), "with learning_starts=40, 39 env steps must not take any gradient step"
    assert not same(b, c), (
        "the first gradient step happens after env step t (0-based) with t + 1 >= learning_starts: with "
        "learning_starts=40, exactly 40 env steps take one step")


class _Recorder:
    def __init__(self, env):
        self.env, self.actions, self.first_obs = env, [], None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *, seed=None):
        obs, info = self.env.reset(seed=seed)
        if self.first_obs is None:
            self.first_obs = obs
        return obs, info

    def step(self, action):
        self.actions.append(action)
        return self.env.step(action)


def test_stage4_epsilon_at_the_first_step(nb):
    agree = []
    for seed in range(10):
        env = _Recorder(_Corridor())
        q, _ = nb.train_dqn(env, 1, learning_starts=1000, eps_start=0.0, eps_end=1.0, eps_decay_steps=1,
                            hidden_sizes=(8,), seed=seed)
        with torch.no_grad():
            greedy = int(q(torch.as_tensor(env.first_obs, dtype=torch.float32)[None])[0].argmax())
        agree.append(env.actions[0] == greedy)
    assert all(agree), (
        "epsilon at env step t (0-based) is linear_schedule(eps_start, eps_end, eps_decay_steps, t): with eps_start=0 "
        "the FIRST action must be greedy. it wasn't in some seeds, so your schedule starts one step late (t + 1)")
