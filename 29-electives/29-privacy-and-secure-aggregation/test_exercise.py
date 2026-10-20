"""Checks for exercise.py (privacy and secure aggregation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Secure
aggregation is checked for exact cancellation in modular arithmetic (with and without
dropouts); Shamir sharing against a hand-computed polynomial; the DP aggregate's noise scale
statistically; the accountant against the closed form at q = 1, against values of the RDP
definition computed by numerical integration, against Abadi et al.'s published example, and
for validity against the exact privacy curve of the Gaussian mechanism; DP-FedAvg with
stand-in client updates that make the server's arithmetic visible.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mlfs.testing import bounded

torch.set_num_threads(1)

P61 = 2**61 - 1
M32 = 2**32


# ---------------------------------------------------------------- stage 1: fixed-point encoding


def test_stage1_encode_hand_example(nb):
    z = nb.encode(np.array([1.5, -0.25, 0.0]))
    assert isinstance(z, np.ndarray) and z.dtype == np.int64, f"encode returns an int64 NumPy array, got {type(z)} {getattr(z, 'dtype', '')}"
    assert z.tolist() == [98304, M32 - 16384, 0], (
        f"1.5 * 2^16 = 98304; -0.25 * 2^16 = -16384, which wraps to 2^32 - 16384 = {M32 - 16384}; got {z.tolist()}")
    assert nb.encode(np.array([-1.23]), scale=10, modulus=1000).tolist() == [988], (
        "scale=10, modulus=1000: round(-12.3) = -12, and -12 mod 1000 = 988")
    assert np.allclose(nb.decode(np.array([988]), scale=10, modulus=1000), [-1.2]), (
        "decode: 988 >= 1000 / 2, so it's the negative number 988 - 1000 = -12, divided by the scale: -1.2")
    zt = nb.encode(torch.tensor([0.5, -0.5]))
    assert zt.tolist() == [32768, M32 - 32768], "encode should also accept a torch tensor"


def test_stage1_sums_survive_encoding(nb):
    g = np.random.default_rng(0)
    xs = g.normal(scale=3.0, size=(20, 500))
    total = np.zeros(500, dtype=np.int64)
    for x in xs:
        z = nb.encode(x)
        assert z.min() >= 0 and z.max() < M32, "encoded entries live in [0, modulus)"
        total = (total + z) % M32
    got = nb.decode(total)
    assert got.dtype == np.float64, "decode returns float64"
    err = np.abs(got - xs.sum(0)).max()
    assert err <= 20 * 0.5 / 2**16 + 1e-12, (
        f"decode(sum of encodings mod 2^32) should equal the real sum to within n * 0.5 / scale = "
        f"{20 * 0.5 / 2**16:.2e}; the largest error is {err:.2e}. negative values must wrap around the modulus")


# ---------------------------------------------------------------- stage 2: pairwise masks


def test_stage2_key_agreement(nb):
    a, b = 123456789, 987654321
    A, B = nb.public_key(a), nb.public_key(b)
    assert A == pow(3, a, P61) and B == pow(3, b, P61), "public_key = generator ** secret mod prime (generator 3, prime 2^61 - 1)"
    sab, sba = nb.shared_secret(a, B), nb.shared_secret(b, A)
    assert sab == sba == pow(3, a * b, P61), (
        "(g^b)^a = (g^a)^b mod p: both ends of a pair must compute the same shared secret")
    r1, r2 = nb.prg(sab, 1000), nb.prg(sab, 1000)
    assert r1.dtype == np.int64 and r1.shape == (1000,), "prg returns an int64 array of length dim"
    assert np.array_equal(r1, r2), "prg must be deterministic in its seed: both clients of a pair need the same mask"
    assert r1.min() >= 0 and r1.max() < M32 and abs(r1.mean() / M32 - 0.5) < 0.05, "prg is uniform on [0, modulus)"
    assert not np.array_equal(nb.prg(sab + 1, 1000), r1), "different seeds give different masks"


def _clients(ids, dim, seed):
    g = np.random.default_rng(seed)
    secrets = {i: int(g.integers(2, P61 - 1)) for i in ids}
    publics = {i: pow(3, s, P61) for i, s in secrets.items()}
    zs = {i: g.integers(0, M32, dim) for i in ids}
    return secrets, publics, zs


def test_stage2_two_clients_by_hand(nb):
    secrets, publics, zs = _clients([0, 1], 50, 1)
    m = nb.prg(pow(publics[1], secrets[0], P61), 50)
    y0 = nb.mask(0, zs[0], secrets[0], publics)
    y1 = nb.mask(1, zs[1], secrets[1], publics)
    assert np.array_equal(y0, (zs[0] + m) % M32), "the lower id ADDS the pair's mask: y_0 = z_0 + m_01 mod 2^32"
    assert np.array_equal(y1, (zs[1] - m) % M32), "the higher id SUBTRACTS it: y_1 = z_1 - m_01 mod 2^32"


def test_stage2_masks_cancel_exactly(nb):
    ids = [3, 7, 10, 42, 99]
    secrets, publics, zs = _clients(ids, 300, 2)
    ys = {i: nb.mask(i, zs[i], secrets[i], publics) for i in ids}
    total_masked = sum(ys.values()) % M32
    total = sum(zs.values()) % M32
    assert np.array_equal(total_masked, total), (
        "the masked inputs must sum to exactly the true sum mod 2^32: every pair's mask is added once (by the lower "
        "id) and subtracted once (by the higher id). skip j == i, and compare ids, not list positions")
    zero = np.zeros(2000, dtype=np.int64)
    y = nb.mask(3, zero, secrets[3], publics)
    assert abs(y.mean() / M32 - 0.5) < 0.03 and (y != 0).mean() > 0.99, (
        "a single masked input should look uniformly random, even when the input is all zeros")


# ---------------------------------------------------------------- stage 3: Shamir and dropouts


def test_stage3_shamir_hand_example(nb):
    # f(x) = 5 + 3x + 2x^2 mod 13: f(1) = 10, f(2) = 19 = 6, f(3) = 32 = 6
    assert nb.shamir_reconstruct([(1, 10), (2, 6), (3, 6)], prime=13) == 5, (
        "Lagrange interpolation at 0 of (1, 10), (2, 6), (3, 6) mod 13 is f(0) = 5")
    assert nb.shamir_reconstruct([(3, 6), (1, 10), (2, 6)], prime=13) == 5, "the order of the shares doesn't matter"


def test_stage3_any_t_shares(nb):
    rng = np.random.default_rng(0)
    secret = 1234567890123
    shares = nb.shamir_share(secret, 7, 4, rng=rng)
    assert [x for x, _ in shares] == list(range(1, 8)), "shares are evaluated at x = 1, ..., n"
    for subset in ([0, 1, 2, 3], [6, 2, 4, 0], [1, 3, 5, 6], list(range(7))):
        got = nb.shamir_reconstruct([shares[k] for k in subset])
        assert got == secret, f"any {len(subset)} >= t = 4 shares must give back the secret; shares {subset} gave {got}"
    wrong = [nb.shamir_reconstruct(nb.shamir_share(secret, 7, 4, rng=np.random.default_rng(s))[:3]) for s in range(3)]
    assert all(w != secret for w in wrong), (
        "3 < t shares should NOT reveal the secret: use t - 1 random coefficients (a degree t - 1 polynomial)")
    for bad in ((3, 4), (3, 0)):
        try:
            nb.shamir_share(secret, *bad, rng=rng)
        except ValueError:
            pass
        else:
            raise AssertionError(f"n={bad[0]}, t={bad[1]} should raise ValueError (need 1 <= t <= n)")


def test_stage3_secure_aggregate(nb):
    g = np.random.default_rng(3)
    ids = [2, 5, 6, 11, 17, 30]
    ups = {i: g.normal(size=40) for i in ids}
    got = bounded(nb.secure_aggregate, ups, threshold=4, seed=0, what="secure_aggregate")
    want = sum(ups.values())
    assert got.shape == (40,) and np.abs(got - want).max() <= 6 / 2**16, (
        "with no dropouts, the secure sum should equal the plain sum up to fixed-point rounding")
    got_d = bounded(nb.secure_aggregate, ups, threshold=4, dropped=[5, 17], seed=1, what="secure_aggregate")
    want_d = sum(v for i, v in ups.items() if i not in (5, 17))
    assert np.abs(got_d - want_d).max() <= 6 / 2**16, (
        "clients 5 and 17 dropped after the masks were set up: the server must reconstruct their secrets from the "
        "survivors' shares, recompute their pairwise masks with every survivor, and remove them. "
        f"largest error {np.abs(got_d - want_d).max():.3g}")
    try:
        nb.secure_aggregate(ups, threshold=4, dropped=[2, 5, 6], seed=2)
    except ValueError:
        pass
    else:
        raise AssertionError("3 survivors < threshold 4: the sum can't be recovered; raise ValueError")


def test_stage3_upload_bytes(nb):
    got = nb.secagg_upload_bytes(100, 1000, n_dropped=3)
    assert got == 4 * 1000 + 8 * (1 + 99 + 3), (
        f"masked input 4 * 1000, one public key (8), 99 shares of its secret (8 each), and 3 shares of the dropped "
        f"clients' secrets: {4000 + 8 * 103}; got {got}")
    assert nb.secagg_upload_bytes(10, 50) == 200 + 80


# ---------------------------------------------------------------- stage 4: clipping and the Gaussian mechanism


def test_stage4_clip(nb):
    d = torch.tensor([3.0, 4.0])
    c = nb.clip_update(d, 1.0)
    assert torch.allclose(c, torch.tensor([0.6, 0.8])), f"(3, 4) has norm 5: clipped to norm 1 it is (0.6, 0.8); got {c.tolist()}"
    small = torch.tensor([0.1, -0.2])
    assert torch.allclose(nb.clip_update(small, 1.0), small), "updates already inside the ball are unchanged"
    z = nb.clip_update(torch.zeros(3), 1.0)
    assert torch.isfinite(z).all() and (z == 0).all(), "a zero update stays zero (no division by zero)"


def test_stage4_aggregate_hand_example_and_noise(nb):
    deltas = [torch.tensor([3.0, 4.0]), torch.tensor([0.1, 0.0])]
    out = nb.dp_aggregate(deltas, clip=1.0, noise_multiplier=0.0, expected_clients=2)
    assert torch.allclose(out, torch.tensor([0.35, 0.4])), (
        f"noise 0: (clip((3,4)) + (0.1, 0)) / 2 = ((0.6, 0.8) + (0.1, 0)) / 2 = (0.35, 0.4); got {out.tolist()}")
    zeros = [torch.zeros(40_000, dtype=torch.float64)] * 3
    noisy = nb.dp_aggregate(zeros, clip=0.5, noise_multiplier=2.0, expected_clients=10,
                            generator=torch.Generator().manual_seed(0))
    sd = noisy.std().item()
    assert abs(sd - 0.1) < 0.003 and abs(noisy.mean().item()) < 0.003, (
        f"noise on the SUM has std noise_multiplier * clip = 1.0; divided by expected_clients = 10 that's 0.1 per "
        f"coordinate; got std {sd:.4f}")
    again = nb.dp_aggregate(zeros, clip=0.5, noise_multiplier=2.0, expected_clients=10,
                            generator=torch.Generator().manual_seed(0))
    assert torch.equal(noisy, again), "draw the noise from `generator` (torch.randn(..., generator=generator))"


# ---------------------------------------------------------------- stage 5: the accountant


def test_stage5_closed_forms(nb):
    orders = [2, 3, 10, 64]
    full = nb.rdp_sampled_gaussian(1.0, 2.0, orders)
    assert np.allclose(full, np.array(orders) / 8.0), (
        f"q = 1 is the plain Gaussian mechanism: RDP(alpha) = alpha / (2 sigma^2); got {full}")
    assert np.allclose(nb.rdp_sampled_gaussian(0.0, 1.0, orders), 0.0), "q = 0: nothing is ever released, RDP 0"
    assert np.all(np.isinf(nb.rdp_sampled_gaussian(0.1, 0.0, [2, 5]))), "sigma = 0 (no noise) has infinite RDP"
    eps, order = nb.rdp_to_dp(np.array([1.0, 0.5]), [2, 3], math.exp(-2))
    assert np.isclose(eps, 1.5) and order == 3, (
        f"eps = min(1 + 2/1, 0.5 + 2/2) = 1.5 at order 3 (log(1/delta) = 2); got ({eps}, {order})")


def test_stage5_matches_the_definition(nb):
    # values of log(E_{z ~ N(0, s^2)}[((1-q) + q exp((2z-1)/(2s^2)))^a]) / (a-1), by numerical integration
    cases = {(0.1, 1.5): [0.005581, 0.008873, 0.016983, 0.03748, 1.112258],
             (0.02, 0.8): [0.001507, 0.002641, 0.012494, 1.780124, 8.327175],
             (0.5, 2.0): [0.068599, 0.110023, 0.211605, 0.423748, 1.286773]}
    for (q, s), want in cases.items():
        got = nb.rdp_sampled_gaussian(q, s, [2, 3, 5, 8, 16])
        assert np.allclose(got, want, rtol=2e-4, atol=2e-6), (
            f"q={q}, sigma={s}: expected RDP {want} at orders (2, 3, 5, 8, 16), got {np.round(got, 6).tolist()}. "
            "A_alpha = sum_k C(alpha, k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 sigma^2))")
    big = nb.rdp_sampled_gaussian(0.01, 0.5, [256])
    assert np.isfinite(big).all() and big[0] > 100, (
        f"order 256 with sigma 0.5 has terms like exp(65,000): sum them in log space (log-sum-exp); got {big}")


def test_stage5_published_example_and_validity(nb):
    eps = nb.compute_epsilon(0.01, 4.0, 10_000, 1e-5)
    assert 1.20 < eps < 1.27, (
        f"Abadi et al. (2016) report eps ~ 1.26 for q = 0.01, sigma = 4, 10,000 steps, delta = 1e-5 with the moments "
        f"accountant; got {eps:.3f}")
    # exact eps of ONE Gaussian mechanism (sensitivity 1) at delta = 1e-5, from its privacy curve (Balle & Wang, 2018)
    for sigma, exact in ((1.0, 4.3772), (2.0, 1.9931), (5.0, 0.7255)):
        got = nb.compute_epsilon(1.0, sigma, 1, 1e-5)
        assert exact <= got <= 1.4 * exact, (
            f"sigma={sigma}: the exact eps is {exact}; an accountant must never report less (that would be a false "
            f"privacy claim), and RDP is within ~35% here. got {got:.4f}")
    e = [nb.compute_epsilon(0.05, 1.0, T, 1e-5) for T in (10, 100, 1000)]
    assert e[0] < e[1] < e[2], f"more steps cost more privacy: {e}"
    assert nb.compute_epsilon(0.05, 2.0, 100, 1e-5) < e[1] < nb.compute_epsilon(0.1, 1.0, 100, 1e-5), (
        "more noise lowers eps; sampling more clients raises it")


# ---------------------------------------------------------------- stage 6: DP-FedAvg


def _lin():
    torch.manual_seed(0)
    return nn.Linear(2, 1)


def _flat(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).clone()


def _push(direction, record=None):
    """a stand-in client update that moves the model by `direction` from wherever it starts."""

    def upd(model, X, y, *, seed, **kw):
        w = _flat(model)
        if record is not None:
            record.append(w)
        out = w + direction
        torch.nn.utils.vector_to_parameters(out, model.parameters())
        return out, 1

    return upd


def test_stage6_noise_free_arithmetic(nb):
    data = [(torch.zeros(1, 2), torch.zeros(1))] * 4
    m = _lin()
    w0 = _flat(m)
    d = torch.tensor([3.0, 0.0, 4.0])  # norm 5
    entry = []
    hist = bounded(nb.dp_fedavg, m, data, _push(d, entry), rounds=2, sampling_rate=1.0, clip=1.0,
                   noise_multiplier=0.0, delta=1e-5, server_lr=0.5, seed=0, what="dp_fedavg")
    assert hist["clients"] == [4, 4], f"sampling_rate=1 selects every client every round; got {hist['clients']}"
    assert all(torch.allclose(e, w0 + (r // 4) * 0.5 * d / 5) for r, e in enumerate(entry)), (
        "every client must start from the current global model (reset the weights before each local_update)")
    assert torch.allclose(_flat(m), w0 + 2 * 0.5 * d / 5), (
        "each round: clip every delta to norm 1 -> (0.6, 0, 0.8), sum over 4 clients, divide by q N = 4, scale by "
        "server_lr = 0.5. two rounds move the model by (0.6, 0, 0.8)")
    P = 3
    per = 4 * P + 8 * (1 + 3)
    assert hist["bytes_up"] == [4 * per, 8 * per], (
        f"bytes_up is cumulative: each of the 4 clients uploads secagg_upload_bytes(4, {P}) = {per}; got {hist['bytes_up']}")


def test_stage6_poisson_sampling_fixed_denominator(nb):
    N = 50
    data = [(torch.zeros(1, 2), torch.zeros(1))] * N
    m = _lin()
    d = torch.tensor([0.1, 0.0, 0.0])
    hist = bounded(nb.dp_fedavg, m, data, _push(d), rounds=40, sampling_rate=0.3, clip=10.0, noise_multiplier=0.0,
                   delta=1e-5, eval_fn=lambda model: _flat(model)[0].item(), seed=1, what="dp_fedavg")
    counts = np.array(hist["clients"])
    assert 10 < counts.mean() < 20 and counts.std() > 1, (
        f"Poisson sampling: each client joins independently with probability 0.3, so the count varies around 15; "
        f"got counts {counts.tolist()}")
    moves = np.diff(hist["eval"])
    assert np.allclose(moves, counts * 0.1 / (0.3 * N), atol=1e-6), (
        "divide the sum by the EXPECTED number of clients q N (a fixed denominator), not by the number sampled: a "
        "round with m clients moves the model by m * delta / (q N). the fixed denominator is what bounds one "
        "client's influence")


def test_stage6_noise_epsilon_and_empty_rounds(nb):
    N = 10
    data = [(torch.zeros(1, 2), torch.zeros(1))] * N
    m = _lin()
    hist = bounded(nb.dp_fedavg, m, data, _push(torch.zeros(3)), rounds=30, sampling_rate=0.05, clip=1.0,
                   noise_multiplier=1.0, delta=1e-5, eval_fn=lambda model: _flat(model).clone(), seed=2,
                   what="dp_fedavg")
    empty = [r for r, c in enumerate(hist["clients"]) if c == 0]
    assert empty, "with q = 0.05 and 10 clients most rounds sample nobody"
    for r in empty[:3]:
        assert not torch.equal(hist["eval"][r + 1], hist["eval"][r]), (
            "rounds that sample no client must still add noise: the privacy analysis assumes a noisy release every "
            "round (otherwise an empty round reveals that nobody was sampled)")
    steps = np.diff(torch.stack(hist["eval"]).numpy(), axis=0)
    assert 1.4 < steps.std() < 2.6, (
        f"noise std per coordinate: noise_multiplier * clip / (q N) = 1 / 0.5 = 2; got {steps.std():.2f}")
    want = [nb.compute_epsilon(0.05, 1.0, r, 1e-5) for r in (1, 10, 30)]
    got = [hist["epsilon"][r - 1] for r in (1, 10, 30)]
    assert np.allclose(got, want, rtol=1e-6), f"history['epsilon'][r-1] = compute_epsilon(q, sigma, r, delta); got {got}, want {want}"


def test_stage6_learns(nb):
    g = np.random.default_rng(5)

    def blob(n):
        y = g.integers(0, 2, n)
        X = g.normal(size=(n, 2)) + np.where(y[:, None] == 1, 2.0, -2.0)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y)

    data = [blob(10) for _ in range(200)]
    Xt, yt = blob(500)

    def sgd(model, X, y, *, seed, lr=0.5, epochs=1):
        for _ in range(epochs):
            model.zero_grad()
            F.cross_entropy(model(X), y).backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
        return _flat(model), epochs

    accs = []
    for s in range(3):
        torch.manual_seed(s)
        m = nn.Linear(2, 2)
        bounded(nb.dp_fedavg, m, data, sgd, rounds=15, sampling_rate=0.25, clip=1.0, noise_multiplier=1.0,
                delta=1e-4, local_kwargs={"epochs": 2}, seed=s, what="dp_fedavg", budget=5_000_000)
        with torch.no_grad():
            accs.append((m(Xt).argmax(1) == yt).float().mean().item())
    assert np.median(accs) > 0.9, f"DP-FedAvg with 50 clients per round should still learn separable blobs; got {accs}"


# ---------------------------------------------------------------- one test per spec clause


def test_stage1_encode_rounds_not_truncates(nb):
    z = nb.encode(np.array([0.6 / 2**16, -0.6 / 2**16, 0.4 / 2**16]))
    assert z.tolist() == [1, M32 - 1, 0], (
        f"encode ROUNDS x * scale to the nearest integer: 0.6 -> 1, -0.6 -> -1 = 2^32 - 1, 0.4 -> 0; got {z.tolist()}. "
        "astype(int64) alone truncates toward zero (np.round first)")


def _secagg_setup(dim=2000, ids=(1, 4, 6, 9, 13), seed=7):
    g = np.random.default_rng(seed)
    return {i: g.normal(scale=0.3, size=dim) for i in ids}


def test_stage3_server_sees_only_masked_uploads(nb):
    ups = _secagg_setup()
    for dropped in ([], [6]):
        uploads = {}
        got = bounded(nb.secure_aggregate, ups, threshold=3, dropped=dropped, seed=3, uploads=uploads,
                      what="secure_aggregate")
        survivors = [i for i in ups if i not in dropped]
        assert sorted(uploads) == survivors, (
            f"`uploads` should hold exactly the survivors' masked vectors (keys {survivors}); got keys {sorted(uploads)}")
        for i in survivors:
            y = np.asarray(uploads[i])
            assert not np.array_equal(y, nb.encode(ups[i])), f"client {i} uploaded its raw encoding: it must be masked"
            assert abs(y.mean() / M32 - 0.5) < 0.03, (
                f"client {i}'s upload should look uniform on [0, 2^32) (mean/2^32 = 0.5); got {y.mean() / M32:.3f}. "
                "an unmasked update encoded around 0 sits near 0 and near 2^32")
        want = sum(ups[i] for i in survivors)
        naive = nb.decode(sum(np.asarray(uploads[i]) for i in survivors) % M32)
        if dropped:
            assert np.abs(naive - want).max() > 1.0, (
                "with a dropout, the survivors' uploads alone must NOT sum to their total: the masks shared with the "
                "dropped client are still in there until the server removes them")
        else:
            assert np.abs(naive - want).max() <= 5 / 2**16, "with no dropouts the uploads alone sum to the total"
        assert np.abs(got - want).max() <= 5 / 2**16, "the returned value is the survivors' sum"


def test_stage3_recovery_uses_survivors_shares(nb):
    ups = _secagg_setup(dim=50)
    revealed = {}
    bounded(nb.secure_aggregate, ups, threshold=3, dropped=[4, 13], seed=5, revealed=revealed, what="secure_aggregate")
    assert sorted(revealed) == [4, 13], f"`revealed` should have one entry per dropped client; got keys {sorted(revealed)}"
    for d, shares in revealed.items():
        holders = set(shares)
        assert holders <= {1, 6, 9} and len(holders) >= 3, (
            f"the dropped client {d}'s secret must be rebuilt from >= threshold shares held by SURVIVORS; holders "
            f"used: {sorted(holders)} (dropped clients can't send anything)")
    import types
    if isinstance(nb, types.ModuleType):  # under pytest: break reconstruction and check that the result notices
        orig = nb.shamir_reconstruct
        try:
            nb.shamir_reconstruct = lambda shares, prime=P61: 12345
            bad = nb.secure_aggregate(ups, threshold=3, dropped=[4], seed=5)
        finally:
            nb.shamir_reconstruct = orig
        want = sum(v for i, v in ups.items() if i != 4)
        assert np.abs(bad - want).max() > 1.0, (
            "secure_aggregate must remove a dropped client's masks using the secret rebuilt with shamir_reconstruct, "
            "not the client's own secret (the server never has it)")


def test_stage3_shamir_coefficients_are_random(nb):
    secret = 987654321
    a = nb.shamir_share(secret, 5, 3, rng=np.random.default_rng(1))
    b = nb.shamir_share(secret, 5, 3, rng=np.random.default_rng(2))
    assert a != b, "the t - 1 non-constant coefficients must come from rng: different rngs give different shares"
    assert all(y != secret for _, y in a), "with t >= 2 no single share equals the secret"
    one = [nb.shamir_share(secret, 5, 1, rng=np.random.default_rng(s)) for s in range(2)]
    assert all(y == secret for sh in one for _, y in sh), "t = 1 is a constant polynomial: every share is the secret"


def test_stage5_default_orders(nb):
    want = min(a / 800 + math.log(1e5) / (a - 1) for a in range(2, 257))
    got = nb.compute_epsilon(1.0, 20.0, 1, 1e-5)
    assert np.isclose(got, want, rtol=1e-9), (
        f"the default orders are 2..256: for sigma = 20, q = 1 the best order is 97, eps = {want:.6f}; got {got:.6f}")


def test_stage6_kwargs_eval_every_bytes_and_seeding(nb):
    N = 40
    data = [(torch.zeros(1, 2), torch.zeros(1))] * N
    seen = []

    def upd(model, X, y, *, seed, **kw):
        seen.append((seed, kw))
        return _flat(model) + 0.01, 1

    hist = bounded(nb.dp_fedavg, _lin(), data, upd, rounds=6, sampling_rate=0.25, clip=1.0, noise_multiplier=0.5,
                   delta=1e-5, local_kwargs={"epochs": 3, "lr": 0.2}, eval_fn=lambda m: 0.0, eval_every=4, seed=3,
                   what="dp_fedavg")
    assert seen and all(kw == {"epochs": 3, "lr": 0.2} for _, kw in seen), (
        f"pass local_kwargs on to every local_update call; got {seen[0][1] if seen else None}")
    assert len({s for s, _ in seen}) == len(seen), "every client call gets its own seed from rng"
    assert hist["eval_round"] == [0, 4, 6], f"eval at round 0, every eval_every = 4 rounds, and the last; got {hist['eval_round']}"
    P = 3
    want, up = [], 0
    for m in hist["clients"]:
        up += m * nb.secagg_upload_bytes(m, P) if m else 0
        want.append(up)
    assert hist["bytes_up"] == want and max(hist["clients"]) < N, (
        f"each of the m sampled clients uploads secagg_upload_bytes(m, P), with m the number SAMPLED that round "
        f"(not N); expected {want}, got {hist['bytes_up']}")
    runs = []
    for s in (11, 11, 12):
        m = _lin()
        nb.dp_fedavg(m, data, _push(torch.zeros(3)), rounds=3, sampling_rate=0.25, clip=1.0, noise_multiplier=1.0,
                     delta=1e-5, seed=s)
        runs.append(_flat(m))
    assert torch.equal(runs[0], runs[1]), "the same seed must give the same run: seed the noise generator from rng"
    assert not torch.equal(runs[0], runs[2]), "a different seed should give different noise"
