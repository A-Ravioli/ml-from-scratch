import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # privacy & secure aggregation · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `encode`, `decode`: real vectors as integers mod $2^{32}$ |
    | 2 | `public_key`, `shared_secret`, `prg`, `mask`: pairwise masks that cancel in the sum |
    | 3 | `shamir_share`, `shamir_reconstruct`, `secure_aggregate`, `secagg_upload_bytes`: surviving dropouts |
    | 4 | `clip_update`, `dp_aggregate`: the Gaussian mechanism on a sum of client updates |
    | 5 | `rdp_sampled_gaussian`, `rdp_to_dp`, `compute_epsilon`: a privacy accountant |
    | 6 | `dp_fedavg`: user-level differentially private federated averaging, with ε and bytes per round |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def encode(x, *, scale=2**16, modulus=2**32):
    """stage 1: round(x * scale) mod modulus, as an int64 NumPy array with entries in [0, modulus)."""
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return np.mod(np.round(x.astype(np.float64) * scale).astype(np.int64), modulus)


@app.function
def decode(z, *, scale=2**16, modulus=2**32):
    """stage 1: invert encode: entries >= modulus / 2 are negative; returns float64."""
    z = np.mod(np.asarray(z, dtype=np.int64), modulus)
    signed = np.where(z >= modulus // 2, z - modulus, z)
    return signed.astype(np.float64) / scale


@app.cell(hide_code=True)
def _():
    checks.run(1, encode=encode, decode=decode)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def public_key(secret, *, prime=2**61 - 1, generator=3):
    """stage 2: generator ** secret mod prime (a toy Diffie-Hellman public key)."""
    return pow(generator, int(secret), prime)


@app.function
def shared_secret(my_secret, their_public, *, prime=2**61 - 1):
    """stage 2: their_public ** my_secret mod prime; the same number from both ends."""
    return pow(int(their_public), int(my_secret), prime)


@app.function
def prg(seed, dim, *, modulus=2**32):
    """stage 2: dim pseudorandom integers, uniform in [0, modulus), from np.random.default_rng(seed)."""
    return np.random.default_rng(int(seed)).integers(0, modulus, size=dim, dtype=np.int64)


@app.function
def mask(i, z, my_secret, publics, *, modulus=2**32, prime=2**61 - 1):
    """stage 2: z + sum_{j > i} prg(s_ij) - sum_{j < i} prg(s_ij) mod modulus, over the other ids in publics."""
    z = np.asarray(z, dtype=np.int64)
    out = np.mod(z, modulus)
    for j, pub in publics.items():
        if j == i:
            continue
        m = prg(shared_secret(my_secret, pub, prime=prime), z.shape[0], modulus=modulus)
        out = np.mod(out + m if j > i else out - m, modulus)
    return out


@app.cell(hide_code=True)
def _():
    checks.run(2, public_key=public_key, shared_secret=shared_secret, prg=prg, mask=mask)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def shamir_share(secret, n, t, *, rng, prime=2**61 - 1):
    """stage 3: n shares (x, f(x)), x = 1..n, of a random degree-(t-1) polynomial f mod prime with f(0) = secret."""
    if not 1 <= t <= n:
        raise ValueError(f"need 1 <= t <= n, got t={t}, n={n}")
    coeffs = [int(secret) % prime] + [int(c) for c in rng.integers(0, prime, size=t - 1)]
    shares = []
    for x in range(1, n + 1):
        y = 0
        for c in reversed(coeffs):  # Horner
            y = (y * x + c) % prime
        shares.append((x, y))
    return shares


@app.function
def shamir_reconstruct(shares, *, prime=2**61 - 1):
    """stage 3: f(0) by Lagrange interpolation mod prime from a list of (x, y) shares."""
    secret = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, (xj, _) in enumerate(shares):
            if j != i:
                num = num * xj % prime
                den = den * (xj - xi) % prime
        secret = (secret + yi * num * pow(den, -1, prime)) % prime
    return secret


@app.function
def secure_aggregate(updates, *, threshold, dropped=(), scale=2**16, modulus=2**32, prime=2**61 - 1, seed=None,
                     uploads=None, revealed=None):
    """stage 3: simulate one round of secure aggregation; returns the float64 sum of the surviving clients' updates."""
    rng = np.random.default_rng(seed)
    ids = sorted(updates)
    n = len(ids)
    dropped = set(dropped)
    survivors = [i for i in ids if i not in dropped]
    if len(survivors) < threshold:
        raise ValueError(f"only {len(survivors)} survivors, fewer than the threshold {threshold}: the sum is unrecoverable")
    # round 0: keys
    secrets = {i: int(rng.integers(2, prime - 1)) for i in ids}
    publics = {i: public_key(secrets[i], prime=prime) for i in ids}
    # round 1: every client Shamir-shares its secret; holder ids[k] keeps share k
    held = {j: {} for j in ids}
    for i in ids:
        for (x, y), j in zip(shamir_share(secrets[i], n, threshold, rng=rng, prime=prime), ids):
            held[j][i] = (x, y)
    # round 2: survivors upload masked inputs; dropped clients never do
    total = np.zeros(len(np.asarray(updates[ids[0]]).reshape(-1)), dtype=np.int64)
    for i in survivors:
        z = encode(np.asarray(updates[i]).reshape(-1), scale=scale, modulus=modulus)
        y = mask(i, z, secrets[i], publics, modulus=modulus, prime=prime)
        if uploads is not None:
            uploads[i] = y
        total = np.mod(total + y, modulus)
    # round 3: survivors reveal shares of each dropped client's secret; the server removes its masks
    for d in sorted(dropped):
        shares = [held[j][d] for j in survivors[:threshold]]
        if revealed is not None:
            revealed[d] = {j: held[j][d] for j in survivors[:threshold]}
        s_d = shamir_reconstruct(shares, prime=prime)
        for j in survivors:
            m = prg(shared_secret(s_d, publics[j], prime=prime), total.shape[0], modulus=modulus)
            # survivor j added +m if d > j, -m if d < j: undo it
            total = np.mod(total - m if d > j else total + m, modulus)
    return decode(total, scale=scale, modulus=modulus)


@app.function
def secagg_upload_bytes(n_clients, dim, *, n_dropped=0, entry_bytes=4, key_bytes=8):
    """stage 3: bytes one surviving client uploads in a secure-aggregation round (masked input, key, shares)."""
    return entry_bytes * dim + key_bytes * (1 + (n_clients - 1) + n_dropped)


@app.cell(hide_code=True)
def _():
    checks.run(3, shamir_share=shamir_share, shamir_reconstruct=shamir_reconstruct,
               secure_aggregate=secure_aggregate, secagg_upload_bytes=secagg_upload_bytes)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def clip_update(delta, clip):
    """stage 4: delta scaled by min(1, clip / ||delta||_2)."""
    norm = delta.norm()
    return delta * torch.clamp(clip / (norm + 1e-12), max=1.0)


@app.function
def dp_aggregate(deltas, *, clip, noise_multiplier, expected_clients, generator=None):
    """stage 4: (sum_k clip(delta_k) + N(0, (noise_multiplier * clip)^2 I)) / expected_clients."""
    total = torch.stack([clip_update(d, clip) for d in deltas]).sum(0)
    noise = torch.randn(total.shape, generator=generator, dtype=total.dtype) * (noise_multiplier * clip)
    return (total + noise) / expected_clients


@app.cell(hide_code=True)
def _():
    checks.run(4, clip_update=clip_update, dp_aggregate=dp_aggregate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def rdp_sampled_gaussian(q, sigma, orders):
    """stage 5: RDP epsilon(alpha) of one Poisson-subsampled Gaussian step (sensitivity 1), for integer orders >= 2."""
    out = []
    for a in orders:
        a = int(a)
        if q == 0:
            out.append(0.0)
            continue
        if sigma == 0:
            out.append(math.inf)
            continue
        if q == 1:
            out.append(a / (2 * sigma**2))
            continue
        terms = [math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)
                 + (a - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma**2)
                 for k in range(a + 1)]
        m = max(terms)
        log_A = m + math.log(sum(math.exp(t - m) for t in terms))
        out.append(log_A / (a - 1))
    return np.array(out)


@app.function
def rdp_to_dp(rdp, orders, delta):
    """stage 5: (epsilon, best order) = min over orders of rdp + log(1 / delta) / (order - 1)."""
    orders = np.asarray(orders, dtype=float)
    eps = np.asarray(rdp, dtype=float) + math.log(1 / delta) / (orders - 1)
    k = int(np.argmin(eps))
    return float(eps[k]), int(orders[k])


@app.function
def compute_epsilon(q, sigma, steps, delta, orders=tuple(range(2, 257))):
    """stage 5: epsilon after `steps` compositions of the subsampled Gaussian, at this delta."""
    return rdp_to_dp(steps * rdp_sampled_gaussian(q, sigma, orders), orders, delta)[0]


@app.cell(hide_code=True)
def _():
    checks.run(5, rdp_sampled_gaussian=rdp_sampled_gaussian, rdp_to_dp=rdp_to_dp, compute_epsilon=compute_epsilon)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 6")
    return


@app.function
def dp_fedavg(model, client_data, local_update, *, rounds, sampling_rate, clip, noise_multiplier, delta,
              server_lr=1.0, local_kwargs=None, eval_fn=None, eval_every=1, seed=None):
    """stage 6: DP-FedAvg (McMahan et al., 2018) with Poisson client sampling; trains model in place, returns history."""
    local_kwargs = {} if local_kwargs is None else dict(local_kwargs)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    params = list(model.parameters())
    N = len(client_data)
    expected = sampling_rate * N
    orders = tuple(range(2, 257))
    rdp_step = rdp_sampled_gaussian(sampling_rate, noise_multiplier, orders)
    hist = {"round": [], "clients": [], "bytes_up": [], "epsilon": [], "eval_round": [], "eval": []}
    up = 0
    if eval_fn is not None:
        hist["eval_round"].append(0)
        hist["eval"].append(eval_fn(model))
    for r in range(1, rounds + 1):
        chosen = np.flatnonzero(rng.random(N) < sampling_rate)
        w = torch.cat([p.detach().reshape(-1) for p in params]).clone()
        deltas = []
        for k in chosen:
            X, y = client_data[k]
            torch.nn.utils.vector_to_parameters(w, params)
            w_k, _ = local_update(model, X, y, seed=int(rng.integers(2**31)), **local_kwargs)
            deltas.append(w_k - w)
        if not deltas:
            deltas = [torch.zeros_like(w)]
        update = dp_aggregate(deltas, clip=clip, noise_multiplier=noise_multiplier, expected_clients=expected,
                              generator=gen)
        torch.nn.utils.vector_to_parameters(w + server_lr * update, params)
        up += len(chosen) * secagg_upload_bytes(len(chosen), w.numel()) if len(chosen) else 0
        hist["round"].append(r)
        hist["clients"].append(len(chosen))
        hist["bytes_up"].append(up)
        hist["epsilon"].append(rdp_to_dp(r * rdp_step, orders, delta)[0])
        if eval_fn is not None and (r % eval_every == 0 or r == rounds):
            hist["eval_round"].append(r)
            hist["eval"].append(eval_fn(model))
    return hist


@app.cell(hide_code=True)
def _():
    checks.run(6, dp_fedavg=dp_fedavg)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`): it checks your secure
    aggregation at scale, compares your accountant with naive composition, and ends with the
    block's project, the privacy–accuracy–communication triangle.
    """)
    return


if __name__ == "__main__":
    app.run()
